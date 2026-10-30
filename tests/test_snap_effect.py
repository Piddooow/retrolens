import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import snap_effect as se  # noqa: E402

CFG = se.CONFIG
PL, PT = CFG["proses_lebar"], CFG["proses_tinggi"]
SATU_FRAME = 1.0 / 30


def frame_sintetis():
    """Background gradasi dan 'orang' (kepala + badan) bertekstur di tengah frame proses."""
    yy, xx = np.mgrid[0:PT, 0:PL]
    frame = np.zeros((PT, PL, 3), dtype=np.uint8)
    frame[..., 0] = (xx * 255 // PL).astype(np.uint8)
    frame[..., 1] = 120
    frame[..., 2] = (yy * 255 // PT).astype(np.uint8)
    kepala = (xx - 320) ** 2 + (yy - 110) ** 2 < 45 ** 2
    badan = (abs(xx - 320) < 90) & (yy > 150)
    orang = kepala | badan
    tekstur = ((xx // 3 + yy // 3) % 2 * 60 + 80).astype(np.uint8)
    for c in range(3):
        frame[..., c][orang] = tekstur[orang]
    return frame, orang.astype(np.float32)


def controller_siap(cfg=CFG):
    frame, mask = frame_sintetis()
    ctrl = se.SnapController(cfg)
    ctrl.amati(0.0, frame, mask)
    return ctrl, frame, mask


class TesConfig(unittest.TestCase):
    def test_default_valid(self):
        self.assertEqual(se.validasi_config(CFG), [])

    def test_tolak_nilai_buruk(self):
        for kunci, nilai in (("flash_turun_detik", 0), ("debu_umur_detik", -1), ("mode_trigger", "X"),
                             ("abu_warna", "merah"), ("cubit_masuk", 0.9), ("mask_ambang", 1.5),
                             ("flash_tunda_ujung_detik", 2.0), ("debu_ukuran_butir", 0)):
            cfg = dict(CFG)
            cfg[kunci] = nilai
            self.assertTrue(se.validasi_config(cfg), kunci)

    def test_tolak_kunci_hilang(self):
        cfg = dict(CFG)
        del cfg["kosong_detik"]
        self.assertTrue(se.validasi_config(cfg))


class TesFlash(unittest.TestCase):
    def test_kurva_sesuai_tabel(self):
        naik = CFG["flash_naik_detik"]
        for u, s in zip(*CFG["flash_naik_tabel"]):
            self.assertAlmostEqual(float(se.kurva_flash(u * naik, CFG)), s, places=4)
        awal_turun = naik + CFG["flash_tahan_detik"]
        for u, s in zip(*CFG["flash_turun_tabel"]):
            t = awal_turun + u * CFG["flash_turun_detik"]
            if u < 1.0:
                self.assertAlmostEqual(float(se.kurva_flash(t, CFG)), s, places=3)
        self.assertEqual(float(se.kurva_flash(-0.01, CFG)), 0.0)
        self.assertEqual(float(se.kurva_flash(awal_turun + CFG["flash_turun_detik"], CFG)), 0.0)

    def test_putih_penuh_di_asal_lalu_seluruh_layar(self):
        flash = se.FlashOverlay(CFG, PL, PT)
        flash.mulai((0.1, 0.9))
        s = flash.kekuatan(CFG["flash_naik_detik"])
        self.assertAlmostEqual(float(s[int(0.9 * PT), int(0.1 * PL)]), 1.0, places=3)
        self.assertLess(float(s[0, PL - 1]), 1.0)
        semua = flash.kekuatan(CFG["flash_naik_detik"] + CFG["flash_tunda_ujung_detik"] + 1e-4)
        self.assertEqual(float(semua.min()), 1.0)

    def test_asal_lebih_terang_saat_naik(self):
        flash = se.FlashOverlay(CFG, PL, PT)
        flash.mulai((0.1, 0.9))
        s = flash.kekuatan(0.2)
        self.assertGreater(float(s[int(0.9 * PT), int(0.1 * PL)]), float(s[0, PL - 1]))

    def test_komposit_resolusi_tampilan(self):
        flash = se.FlashOverlay(CFG, PL, PT)
        flash.mulai((0.5, 0.5))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        hasil = flash.komposit(frame, 0.2)
        self.assertEqual(hasil.shape, frame.shape)
        self.assertEqual(hasil.dtype, np.uint8)
        self.assertGreater(int(hasil[360, 640].sum()), 0)
        self.assertTrue(np.all(flash.komposit(frame, 0.5) == 255))


class TesLinimasa(unittest.TestCase):
    def test_state_di_tiap_batas(self):
        ctrl, _, _ = controller_siap()
        tampil = np.full((360, 640, 3), 90, dtype=np.uint8)
        self.assertTrue(ctrl.picu(0.0))
        debu_akhir = CFG["debu_mulai_detik"] + CFG["debu_sapuan_detik"] + CFG["debu_umur_detik"]
        kosong_akhir = debu_akhir + CFG["kosong_detik"]
        kembali_akhir = kosong_akhir + CFG["kembali_detik"]
        self.assertAlmostEqual(debu_akhir, 4.33, places=6)
        self.assertAlmostEqual(kembali_akhir, 6.83, places=6)
        harapan = ((0.1, "SNAP"), (debu_akhir - SATU_FRAME, "SNAP"), (debu_akhir + SATU_FRAME, "KOSONG"),
                   (kosong_akhir - SATU_FRAME, "KOSONG"), (kosong_akhir + SATU_FRAME, "KEMBALI"),
                   (kembali_akhir + SATU_FRAME, "LIVE"))
        t = 0.0
        for target, state in harapan:
            while t < target:
                t = min(target, t + SATU_FRAME)
                ctrl.render(t, tampil)
            self.assertEqual(ctrl.state, state, "t=%.3f" % t)

    def test_live_tampil_sebelum_ganti_lalu_putih(self):
        ctrl, _, _ = controller_siap()
        tampil = np.full((360, 640, 3), 90, dtype=np.uint8)
        ctrl.picu(0.0)
        self.assertTrue(np.array_equal(ctrl.render(0.0, tampil), tampil))
        self.assertTrue(np.all(ctrl.render(0.45, tampil) == 255))
        self.assertTrue(np.all(ctrl.render(1.3, tampil) == 255))

    def test_cooldown_dan_reset(self):
        ctrl, _, _ = controller_siap()
        self.assertTrue(ctrl.picu(0.0))
        self.assertFalse(ctrl.picu(0.5))
        ctrl.reset(1.0)
        self.assertEqual(ctrl.state, "LIVE")
        self.assertFalse(ctrl.picu(1.0 + CFG["cooldown_detik"] - 0.01))
        ctrl.amati(2.5, *frame_sintetis())
        self.assertTrue(ctrl.picu(1.0 + CFG["cooldown_detik"] + 0.01))

    def test_batal_tanpa_orang(self):
        ctrl = se.SnapController(CFG)
        frame, _ = frame_sintetis()
        ctrl.amati(0.0, frame, np.zeros((PT, PL), dtype=np.float32))
        self.assertFalse(ctrl.picu(0.0))
        self.assertEqual(ctrl.pesan(0.1), "Tidak ada orang terdeteksi")
        self.assertEqual(ctrl.pesan(0.1 + CFG["pesan_detik"]), "")


class TesDebu(unittest.TestCase):
    def setUp(self):
        frame, mask = frame_sintetis()
        self.frame = frame
        self.debu = se.Disintegration(CFG, PL, PT)
        self.debu.siapkan(frame, mask > 0.5, CFG["debu_ukuran_butir"])

    def test_jumlah_butir_masuk_akal(self):
        self.assertGreater(self.debu.jumlah, 5000)

    def test_kepala_lepas_lebih_dulu(self):
        y = self.debu.y0
        atas = self.debu.lepas[y < y.min() + 0.2 * (y.max() - y.min())]
        bawah = self.debu.lepas[y > y.max() - 0.2 * (y.max() - y.min())]
        self.assertLess(float(np.median(atas)), float(np.median(bawah)))

    def test_ada_lubang_di_badan(self):
        # Di tengah sapuan, sebagian butir bawah sudah lepas sementara butir di atasnya belum.
        t = 0.5 * CFG["debu_sapuan_detik"]
        y = self.debu.y0
        tengah = (y > np.percentile(y, 55)) & (y < np.percentile(y, 75))
        lepas = self.debu.lepas[tengah] < t
        self.assertTrue(lepas.any() and not lepas.all())

    def test_bergerak_ke_kiri_atas_lalu_hilang(self):
        semua = np.arange(self.debu.jumlah)
        umur = np.full(self.debu.jumlah, 0.6 * CFG["debu_umur_detik"], dtype=np.float32)
        x, y = self.debu.posisi(semua, umur)
        dx, dy = x - self.debu.x0, y - self.debu.y0
        self.assertLess(float(dx.max()), 0.0)
        self.assertLess(float(dy.mean()), 0.0)
        # Jitter membuat debu menyebar, bukan bergerak sebagai satu blok.
        self.assertGreater(float(dx.std()), 5.0)
        kanvas = np.zeros((PT * PL, 3), dtype=np.float32)
        akhir = CFG["debu_sapuan_detik"] + CFG["debu_umur_detik"]
        self.assertEqual(self.debu.render(kanvas, akhir), 0)
        self.assertTrue(self.debu.selesai(akhir))
        self.assertEqual(float(kanvas.max()), 0.0)

    def test_butir_belum_lepas_pakai_piksel_asli(self):
        kanvas = np.zeros((PT * PL, 3), dtype=np.float32)
        self.debu.render(kanvas, -1.0)
        kanvas = kanvas.reshape(PT, PL, 3)
        self.assertTrue(np.array_equal(kanvas[200, 320], self.frame[200, 320].astype(np.float32)))

    def test_seed_bisa_diulang(self):
        frame, mask = frame_sintetis()
        lain = se.Disintegration(CFG, PL, PT)
        lain.siapkan(frame, mask > 0.5, CFG["debu_ukuran_butir"])
        self.assertTrue(np.array_equal(lain.lepas, self.debu.lepas))


class TesBackground(unittest.TestCase):
    def test_plate_berjalan_dan_inpaint(self):
        frame, mask = frame_sintetis()
        bg = se.BackgroundModel(CFG, PL, PT)
        bg.perbarui(frame, mask)
        self.assertFalse(bg.terlihat[200, 320])
        self.assertTrue(bg.terlihat[10, 10])
        plate, lengkap, dibuang = bg.plate_untuk_snap(frame, mask)
        self.assertFalse(lengkap)
        self.assertFalse(dibuang)
        self.assertEqual(plate.dtype, np.uint8)
        self.assertTrue(bg.status.startswith("BG OTOMATIS"))

    def test_manual_ditolak_jika_ada_orang(self):
        frame, mask = frame_sintetis()
        bg = se.BackgroundModel(CFG, PL, PT)
        bg.mulai_manual()
        hasil = None
        for _ in range(CFG["plate_jumlah_frame"]):
            hasil = bg.tambah_manual(frame, mask)
        self.assertEqual(hasil, se.BackgroundModel.ADA_ORANG)
        self.assertTrue(bg.status.startswith("BG OTOMATIS"))

    def test_manual_diterima_dan_eksposur_dibatasi(self):
        frame, mask = frame_sintetis()
        bg = se.BackgroundModel(CFG, PL, PT)
        bg.mulai_manual()
        kosong = np.zeros((PT, PL), dtype=np.float32)
        for i in range(CFG["plate_jumlah_frame"]):
            hasil = bg.tambah_manual(frame, kosong)
            self.assertEqual(hasil, None if i < CFG["plate_jumlah_frame"] - 1 else se.BackgroundModel.OK)
        self.assertEqual(bg.status, "BG TERSIMPAN")
        terang = cv2.convertScaleAbs(frame, alpha=1.15)
        plate, lengkap, dibuang = bg.plate_untuk_snap(terang, mask)
        self.assertTrue(lengkap)
        self.assertFalse(dibuang)
        rasio = plate[10:50, 10:50].astype(np.float32).mean() / frame[10:50, 10:50].astype(np.float32).mean()
        self.assertAlmostEqual(rasio, 1.15, delta=0.02)

    def test_manual_dibuang_jika_kamera_bergeser(self):
        frame, mask = frame_sintetis()
        bg = se.BackgroundModel(CFG, PL, PT)
        bg.set_manual(frame)
        geser = np.ascontiguousarray(np.roll(frame, 40, axis=1)[:, :, ::-1])
        plate, lengkap, dibuang = bg.plate_untuk_snap(geser, mask)
        self.assertTrue(dibuang)
        self.assertIsNone(bg.manual)

    def test_manual_ditolak_jika_kamera_tertutup(self):
        frame, mask = frame_sintetis()
        bg = se.BackgroundModel(CFG, PL, PT)
        bg.perbarui(frame, mask)
        tertutup = np.full_like(frame, (40, 60, 150))
        kosong = np.zeros((PT, PL), dtype=np.float32)
        bg.mulai_manual()
        hasil = None
        for _ in range(CFG["plate_jumlah_frame"]):
            hasil = bg.tambah_manual(tertutup, kosong)
        self.assertEqual(hasil, se.BackgroundModel.TIDAK_COCOK)
        self.assertIsNone(bg.manual)


class TesMaskSnap(unittest.TestCase):
    def test_buang_bintik_kecil(self):
        _, mask = frame_sintetis()
        mask[5:8, 5:8] = 1.0
        biner, luas = se.PersonSegmenter.mask_snap(mask, CFG)
        self.assertFalse(biner[6, 6])
        self.assertTrue(biner[200, 320])
        self.assertGreater(luas, CFG["mask_luas_min_orang"])


if __name__ == "__main__":
    unittest.main()
