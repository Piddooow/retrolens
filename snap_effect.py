"""Mode SNAP untuk Retrolens: orang di depan kamera hancur jadi debu setelah jentikan jari."""

import argparse
import collections
import contextlib
import json
import math
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    import sounddevice as sd
except Exception:  # PortAudio tidak terpasang: mode suara dimatikan, sisanya tetap jalan.
    sd = None

from snap.background import CAPTURE_MESSAGES, CAPTURE_OK, BackgroundEngine, dilate_to  # noqa: E402,F401
from snap.config import CONFIG, MODE_TRIGGER, validasi_config  # noqa: E402,F401
from snap.ash import AshAmbient  # noqa: E402
from snap.config import hex_ke_bgr  # noqa: E402,F401
from snap.dissolve import DissolveRenderer, noise_fraktal, normalisasi, smoothstep  # noqa: E402,F401
from snap.flash import FlashOverlay, kurva_flash  # noqa: E402,F401
from snap.gestures import SnapGestureDetector  # noqa: E402,F401
from snap.matte import MatteEngine  # noqa: E402

MSG_NEED_ROOM = "I still need to see the room behind you. Step out for 2 seconds, then snap again"
MSG_ESTIMATED = "Background estimated, result may not be perfect"


class PersonSegmenter:
    """Pembungkus ImageSegmenter selfie. Nilai mask tinggi berarti orang."""

    def __init__(self, path_model: str, cfg: Dict[str, Any]):
        vision = mp.tasks.vision
        opsi = vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=path_model),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.cfg = cfg
        self.mask = None  # type: Optional[np.ndarray]
        self._seg = vision.ImageSegmenter.create_from_options(opsi)

    def confidence(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        """Mask float32 (tinggi, lebar) seukuran input, 0 background sampai 1 orang."""
        gambar = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hasil = self._seg.segment_for_video(gambar, ts_ms)
        # category_mask model ini terbalik (0 = orang), jadi hanya confidence mask yang dipakai.
        return np.squeeze(hasil.confidence_masks[0].numpy_view()).astype(np.float32, copy=True)

    def perbarui(self, bgr: np.ndarray, ts_ms: int) -> np.ndarray:
        """Mask yang dihaluskan antar frame untuk frame BGR resolusi proses."""
        baru = self.confidence(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts_ms)
        if self.mask is None or self.mask.shape != baru.shape:
            self.mask = baru
        else:
            e = self.cfg["mask_ema_baru"]
            self.mask = cv2.addWeighted(baru, e, self.mask, 1.0 - e, 0.0)
        return self.mask

    def reset(self) -> None:
        self.mask = None

    @staticmethod
    def mask_snap(mask: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Mask biner bersih untuk efek dan fraksi luas orang terhadap frame."""
        kernel = np.ones((cfg["mask_kernel"], cfg["mask_kernel"]), dtype=np.uint8)
        biner = (mask > cfg["mask_ambang"]).astype(np.uint8)
        biner = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel)
        jumlah, label, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
        simpan = np.zeros(jumlah, dtype=np.uint8)
        simpan[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg["mask_luas_min_komponen"] * biner.size
        biner = cv2.erode(simpan[label], kernel, iterations=1).astype(bool)
        return biner, float(biner.mean())

    def close(self) -> None:
        self._seg.close()


class SnapController:
    """State machine SNAP. Semua waktu (now) berupa detik monotonic yang diberikan pemanggil."""

    LIVE = "LIVE"
    PLATE_CAPTURE = "PLATE_CAPTURE"
    SNAP = "SNAP"
    KOSONG = "KOSONG"
    KEMBALI = "KEMBALI"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        lebar, tinggi = cfg["proses_lebar"], cfg["proses_tinggi"]
        self.lebar, self.tinggi = lebar, tinggi
        self.bg = BackgroundEngine(cfg)
        self.matte = MatteEngine(cfg)
        # Optional snap.profiler.Profiler; the app sets it to time the matte and background stages.
        self.prof = None  # type: Any
        self.bg_siap = False
        self.bg_persen = 0
        self.flash = FlashOverlay(cfg, lebar, tinggi)
        self.debu = DissolveRenderer(cfg)
        self.abu = AshAmbient(cfg)
        self.ukuran_butir = cfg["particle_cell_px"]

        self.t_ganti = self.flash.mulai_putih
        self.debu_akhir = cfg["debu_mulai_detik"] + cfg["debu_sapuan_detik"] + cfg["debu_umur_detik"]
        self.kosong_akhir = self.debu_akhir + cfg["kosong_detik"]
        self.kembali_akhir = self.kosong_akhir + cfg["kembali_detik"]

        self.state = self.LIVE
        self.t0 = 0.0
        self.cooldown_sampai = -math.inf
        self.t_rekam = 0.0
        self.butir_tampil = 0
        self.waktu_efek_terakhir = 0.0
        self._pesan = ""
        self._pesan_sampai = -math.inf
        self._frame = None  # type: Optional[np.ndarray]
        self._frame_tampil = None  # type: Optional[np.ndarray]
        self._mask = None  # type: Optional[np.ndarray]
        self._tint = None  # type: Optional[np.ndarray]
        self._tint_t = -math.inf
        self._kosongkan_snapshot()

    def _kosongkan_snapshot(self) -> None:
        self._beku_tampil = None  # type: Optional[np.ndarray]
        self._matte_beku = None  # type: Optional[np.ndarray]
        self._zona = None  # type: Optional[np.ndarray]
        self._zf = None  # type: Optional[np.ndarray]
        # Pixels the experimental fallback estimated into the plate.
        self._terisi = None  # type: Optional[np.ndarray]
        self._perkiraan = False
        self._siap = False
        # Base layer data (union zone, plate view) cached for one live frame and matte.
        self._dasar_kunci = None  # type: Optional[Tuple[np.ndarray, Optional[np.ndarray]]]
        self._dasar_hasil = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]
        # Base layer data used by the last rendered frame, for the quality tools.
        self._info_dasar = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]

    def set_pesan(self, teks: str, now: float, durasi: Optional[float] = None) -> None:
        self._pesan = teks
        self._pesan_sampai = now + (self.cfg["pesan_detik"] if durasi is None else durasi)

    def pesan(self, now: float) -> str:
        return self._pesan if now < self._pesan_sampai else ""

    def waktu_efek(self, now: float) -> float:
        return now - self.t0 if self.state in (self.SNAP, self.KOSONG, self.KEMBALI) else 0.0

    def siap_dipicu(self, now: float) -> bool:
        return self.state == self.LIVE and now >= self.cooldown_sampai

    def sisa_hitung_mundur(self, now: float) -> int:
        return max(0, int(math.ceil(self.t_rekam - now)))

    def _tahap(self, nama: str) -> Any:
        return self.prof.stage(nama) if self.prof is not None else contextlib.nullcontext()

    def amati(self, now: float, frame_tampil: np.ndarray, frame_proses: np.ndarray, mask: np.ndarray) -> None:
        """Latest display frame, its process size copy, and the person mask (every state)."""
        self._frame_tampil = frame_tampil
        self._frame = frame_proses
        self._mask = mask
        with self._tahap("matte"):
            self.matte.update(frame_tampil, mask)
        if self.state not in (self.LIVE, self.PLATE_CAPTURE):
            t = now - self.t0
            # The base layer also hides the live person; preparing it here keeps it out of the render stage.
            if t >= self.t_ganti and not self.flash.putih_penuh(t):
                self._dasar(frame_tampil)
            return
        with self._tahap("matte"):
            zona = self.matte.core()
        with self._tahap("background"):
            if self.state == self.LIVE:
                for teks in self.bg.update(now, frame_tampil, mask):
                    self.set_pesan(teks, now, self.cfg["pesan_detik"] * 2)
            elif self.state == self.PLATE_CAPTURE:
                if now < self.t_rekam:
                    self.bg.update(now, frame_tampil, mask, allow_auto=False)
                else:
                    hasil = self.bg.capture_frame(frame_tampil, mask)
                    if hasil is not None:
                        self.set_pesan(CAPTURE_MESSAGES[hasil], now, self.cfg["pesan_detik"] * 2)
                        self._masuk_live(now)
            # The shadow part never lowers coverage (it is only added where known), so the core is enough here.
            cakupan = self.bg.coverage(zona) if zona.any() else self.bg.known_fraction()
            self.bg_siap = cakupan >= self.cfg["bg_coverage_min"]
            self.bg_persen = int(math.floor(100.0 * cakupan))

    def zona_hapus(self) -> Tuple[np.ndarray, np.ndarray]:
        """Removal zone of the latest frame: feathered Zf (float 0..1) and its shadow part (bool)."""
        frame = self._frame_tampil
        return self.matte.zone(frame, self.bg.plate_adj(frame), self.bg.known())

    def peta_gain(self) -> np.ndarray:
        """Exposure and color gain map (grid size, 3 channels) for the latest frame."""
        self.bg.plate_adj(self._frame_tampil)
        return self.bg.gain_small

    def status_bg(self) -> str:
        return "BG READY" if self.bg_siap else "BG LEARNING %d%%" % self.bg_persen

    def tint(self, now: float) -> Optional[Tuple[np.ndarray, float]]:
        """Soft blue overlay on the unknown background after a refused snap: (mask, opacity) or None."""
        if self._tint is None:
            return None
        hold = self.cfg["pesan_detik"]
        fade = self.cfg["coverage_tint_fade_s"]
        alpha = self.cfg["coverage_tint_alpha"] * (1.0 - float(smoothstep((now - self._tint_t - hold) / fade)))
        if alpha <= 0.0:
            self._tint = None
            return None
        return self._tint, alpha

    def mulai_rekam_plate(self, now: float) -> None:
        if self.state != self.LIVE:
            return
        self.state = self.PLATE_CAPTURE
        self.t_rekam = now + self.cfg["plate_hitung_mundur"]
        self.bg.cancel_capture()

    def picu(self, now: float, asal: Optional[Tuple[float, float]] = None) -> bool:
        """Mulai efek dari frame dan mask terakhir. asal = titik jentik ternormalisasi 0..1."""
        if not self.siap_dipicu(now):
            return False
        if self._frame_tampil is None or self._mask is None or self.matte.matte is None:
            self.set_pesan("Tidak ada orang terdeteksi", now)
            return False
        _, luas = PersonSegmenter.mask_snap(self._mask, self.cfg)
        if luas < self.cfg["mask_luas_min_orang"]:
            self.set_pesan("Tidak ada orang terdeteksi", now)
            return False
        zf, _ = self.zona_hapus()
        zona = zf > 0.0
        # Coverage gate: never vanish over background that was not seen.
        perkiraan = self.bg.coverage(zona) < self.cfg["bg_coverage_min"]
        if perkiraan and not self.cfg["allow_inpaint_fallback"]:
            self.set_pesan(MSG_NEED_ROOM, now, self.cfg["pesan_detik"] * 2)
            self._tint = zona & ~self.bg.known()
            self._tint_t = now
            return False
        self._kosongkan_snapshot()
        # Snapshot dibekukan di sini; dissolve memakai frame dan matte ini sampai selesai.
        self._beku_tampil = self._frame_tampil.copy()
        self._matte_beku = self.matte.matte.copy()
        self._zona = zona
        self._zf = zf
        self._perkiraan = perkiraan
        self.flash.mulai(asal if asal is not None else tuple(self.cfg["flash_asal_cadangan"]))
        self.state = self.SNAP
        self.t0 = now
        return True

    def reset(self, now: float) -> None:
        self._masuk_live(now)

    def _masuk_live(self, now: float) -> None:
        self.state = self.LIVE
        self.cooldown_sampai = now + self.cfg["cooldown_detik"]
        self.butir_tampil = 0
        self._kosongkan_snapshot()

    def _siapkan(self) -> None:
        """Dipanggil sekali saat layar putih penuh, jadi jeda persiapan tidak terlihat."""
        if self._perkiraan:
            # Experimental: estimate the unseen room once and keep it in the plate for every frame.
            # The pixels stay unknown, so the room learning replaces them later.
            hole = self._zona & ~self.bg.known()
            self.bg.plate[:] = self.bg.fill_unknown(self.bg.plate, self._zona)
            self._terisi = hole
        self.debu.siapkan(self._beku_tampil, self._matte_beku, self._zf, self.ukuran_butir)
        tinggi, lebar = self._beku_tampil.shape[:2]
        self.abu.siapkan((lebar, tinggi), np.random.default_rng(self.cfg["debu_seed"] + 1))
        self._siap = True

    def _dasar(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Union removal zone (float 0..1) and plate view for one live frame."""
        kunci = self._dasar_kunci
        if kunci is not None and kunci[0] is frame and kunci[1] is self.matte.matte:
            return self._dasar_hasil
        known = self.bg.known()
        matte_ok = self.matte.matte is not None and self.matte.matte.shape == frame.shape[:2]
        with self._tahap("matte"):
            orang = self.matte.core() if matte_ok else None
        with self._tahap("background"):
            # Exposure is matched only on known room pixels away from both the snapshot and the live person.
            area = known & ~self._zona
            if orang is not None:
                area &= ~orang
            adj = self.bg.plate_adj(frame, area)
        zf = self._zf
        if matte_ok:
            with self._tahap("matte"):
                zf_live, _ = self.matte.zone(frame, adj, known)
                zf = np.maximum(zf, zf_live)
        # A pixel of the room that was never seen keeps the live frame instead of a made up plate value.
        seen = self.bg.count > 0
        if self._terisi is not None:
            seen |= self._terisi
        zf = zf * seen
        with self._tahap("background"):
            pv = self.bg.plate_view(adj, zf)
        self._dasar_kunci = (frame, self.matte.matte)
        self._dasar_hasil = (zf, pv)
        return zf, pv

    def _komposisi(self, t: float, frame_tampil: np.ndarray) -> np.ndarray:
        zf, pv = self._dasar(frame_tampil)
        self._info_dasar = (zf, pv)
        hasil = frame_tampil.copy()
        # Base layer: live frame outside the zone, plate view inside, feathered at the edge.
        x, y, w, h = cv2.boundingRect((zf > 0.0).view(np.uint8))
        if w > 0 and h > 0:
            kotak = (slice(y, y + h), slice(x, x + w))
            bobot = np.ascontiguousarray(zf[kotak])
            hasil[kotak] = cv2.blendLinear(frame_tampil[kotak], pv[kotak], 1.0 - bobot, bobot)
        if t < self.debu_akhir:
            self.debu.render(hasil, t, self.abu.specks(t))
            self.butir_tampil = self.debu.visible
        else:
            self.butir_tampil = 0
        return hasil

    def info_kualitas(self, lebar: int, tinggi: int) -> Dict[str, Any]:
        """Data for tools/measure_quality.py at display size; entries are None when not available.

        zone is the part of the removal zone that is fully replaced by the plate view.
        """
        info = {"zone": None, "plate_view": None, "mass": None, "step_px": None}  # type: Dict[str, Any]
        if self._info_dasar is not None:
            zf, pv = self._info_dasar
            penuh = (zf >= 1.0).view(np.uint8) * np.uint8(255)
            if penuh.shape != (tinggi, lebar):
                penuh = cv2.resize(penuh, (lebar, tinggi), interpolation=cv2.INTER_NEAREST)
                pv = cv2.resize(pv, (lebar, tinggi), interpolation=cv2.INTER_LINEAR)
            info["zone"] = penuh
            info["plate_view"] = pv
        if self._siap:
            effect_t = self.waktu_efek_terakhir
            active = self.cfg["debu_mulai_detik"] <= effect_t < self.debu_akhir
            info["mass"] = self.debu.mass if active else (1.0 if effect_t < self.cfg["debu_mulai_detik"] else 0.0)
            info["step_px"] = self.debu.step_max * lebar / float(self.debu.size[0]) if active else 0.0
        return info

    def render(self, now: float, frame_tampil: np.ndarray) -> np.ndarray:
        """Frame akhir (uint8, resolusi tampilan) tanpa teks HUD."""
        if self.state in (self.LIVE, self.PLATE_CAPTURE):
            return frame_tampil
        t = now - self.t0
        self.waktu_efek_terakhir = t
        self._info_dasar = None
        if t >= self.kembali_akhir:
            self._masuk_live(now)
            return frame_tampil
        if t >= self.kosong_akhir:
            self.state = self.KEMBALI
        elif t >= self.debu_akhir and self.state == self.SNAP:
            self.state = self.KOSONG
            if self._perkiraan:
                self.set_pesan(MSG_ESTIMATED, now, self.cfg["kosong_detik"])

        if t < self.t_ganti:
            hasil = frame_tampil
        else:
            if not self._siap:
                self._siapkan()
            if self.flash.putih_penuh(t):
                return np.full_like(frame_tampil, 255)
            hasil = self._komposisi(t, frame_tampil)
        if self.state == self.KEMBALI:
            # Outside the zone the base already equals the live frame, so only the person fades back in.
            k = float(smoothstep((t - self.kosong_akhir) / self.cfg["kembali_detik"]))
            hasil = cv2.addWeighted(hasil, 1.0 - k, frame_tampil, k, 0.0)
        return self.flash.komposit(hasil, t)


class OnsetAudio:
    """Deteksi onset per blok audio; terpisah dari thread supaya bisa diuji dengan sinyal sintetis."""

    def __init__(self, cfg: Dict[str, Any], samplerate: int):
        n = cfg["audio_blocksize"]
        self.cfg = cfg
        self.n = n
        self.jendela = np.hanning(n).astype(np.float32)
        self.pita = np.fft.rfftfreq(n, 1.0 / samplerate) >= cfg["audio_highpass_hz"]
        panjang = max(1, int(cfg["audio_jendela_noise_detik"] * samplerate / n))
        self.riwayat = collections.deque(maxlen=panjang)  # type: collections.deque
        self.sebelum = 0.0
        self.noise = 0.0
        self.ambang_kalibrasi = None  # type: Optional[float]

    def energi(self, blok: np.ndarray) -> float:
        if len(blok) != self.n:
            blok = np.resize(blok, self.n) if len(blok) else np.zeros(self.n, dtype=np.float32)
        return float(np.mean(np.abs(np.fft.rfft(blok * self.jendela)[self.pita]) ** 2))

    def ambang(self) -> float:
        if self.ambang_kalibrasi is not None:
            return self.ambang_kalibrasi
        return self.noise * self.cfg["audio_ambang_awal_kali_noise"]

    def proses(self, blok: np.ndarray) -> Tuple[float, bool]:
        e = self.energi(blok)
        self.noise = float(np.median(self.riwayat)) if self.riwayat else e
        # Blok awal stream sering nol, jadi tunggu riwayat noise penuh sebelum menandai onset.
        siap = len(self.riwayat) == self.riwayat.maxlen
        onset = siap and e > self.ambang() and e > self.cfg["audio_onset_kali"] * self.sebelum
        self.riwayat.append(e)
        self.sebelum = e
        return e, onset


class AudioSnapDetector:
    """Bunyi jentikan dari mikrofon. Callback hanya menyalin ke ring buffer; analisis di thread pekerja."""

    def __init__(self, cfg: Dict[str, Any], path_kalibrasi: str):
        self.cfg = cfg
        self.path_kalibrasi = path_kalibrasi
        self.aktif = False
        self.error = ""
        self.diam = False
        self.level = 0.0
        self.ambang = 0.0
        self.onset = None  # type: Optional[OnsetAudio]
        self.ambang_tersimpan = self._muat_kalibrasi()
        self._event = queue.Queue()  # type: queue.Queue
        self._pesan = queue.Queue()  # type: queue.Queue
        self._antrean = queue.Queue()  # type: queue.Queue
        self._stop = threading.Event()
        self._kunci = threading.Lock()
        self._kal = None  # type: Optional[Dict[str, Any]]
        self._stream = None
        self._thread = None  # type: Optional[threading.Thread]
        self._ring = np.zeros((cfg["audio_ring_blok"], cfg["audio_blocksize"]), dtype=np.float32)
        self._waktu = np.zeros(cfg["audio_ring_blok"], dtype=np.float64)
        self._tulis = 0

    def _muat_kalibrasi(self) -> Optional[float]:
        try:
            with open(self.path_kalibrasi) as f:
                ambang = float(json.load(f)["ambang"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return ambang if ambang > 0 else None

    def mulai(self) -> bool:
        if sd is None:
            self.error = "Modul sounddevice tidak tersedia"
            return False
        try:
            info = sd.query_devices(kind="input")
            samplerate = int(info["default_samplerate"])
            self.onset = OnsetAudio(self.cfg, samplerate)
            self.onset.ambang_kalibrasi = self.ambang_tersimpan
            self._stream = sd.InputStream(samplerate=samplerate, blocksize=self.cfg["audio_blocksize"],
                                          channels=1, dtype="float32", callback=self._callback)
            self._stream.start()
        except Exception as e:
            self.error = "Mikrofon tidak bisa dibuka (%s)" % e
            self._tutup_stream()
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._kerja, name="audio-snap", daemon=True)
        self._thread.start()
        self.aktif = True
        return True

    def _callback(self, indata: np.ndarray, frames: int, waktu: Any, status: Any) -> None:
        # Berjalan di thread audio real-time: tanpa alokasi array baru, hanya salin ke slot ring buffer.
        i = self._tulis % len(self._ring)
        self._ring[i, :frames] = indata[:frames, 0]
        self._waktu[i] = time.monotonic()
        self._tulis += 1
        self._antrean.put_nowait(i)

    def _kerja(self) -> None:
        cfg = self.cfg
        terakhir = time.monotonic()
        t_bersuara = terakhir
        try:
            while not self._stop.is_set():
                try:
                    i = self._antrean.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - terakhir > cfg["audio_putus_detik"]:
                        self.error = "Mikrofon terputus"
                        self.aktif = False
                        return
                    continue
                blok = self._ring[i].copy()
                t = float(self._waktu[i])
                terakhir = time.monotonic()
                # Izin mikrofon yang ditolak di macOS tidak memunculkan error, hanya sampel nol.
                if np.any(blok):
                    t_bersuara = t
                    self.diam = False
                elif t - t_bersuara > cfg["audio_diam_detik"]:
                    self.diam = True
                e, onset = self.onset.proses(blok)
                self.level = e
                self.ambang = self.onset.ambang()
                with self._kunci:
                    kal = self._kal
                if kal is not None:
                    self._langkah_kalibrasi(kal, t, e)
                elif onset:
                    self._event.put(t)
        except Exception as e:
            self.error = "Audio berhenti (%s)" % e
            self.aktif = False

    def mulai_kalibrasi(self, now: float) -> bool:
        if not self.aktif:
            return False
        with self._kunci:
            self._kal = {"mulai": now, "hening": [], "jentik": []}
        return True

    def sedang_kalibrasi(self) -> bool:
        with self._kunci:
            return self._kal is not None

    def teks_kalibrasi(self, now: float) -> str:
        with self._kunci:
            kal = self._kal
        if kal is None:
            return ""
        dt = now - kal["mulai"]
        hening = self.cfg["audio_kalibrasi_hening_detik"]
        if dt < hening:
            return "Kalibrasi suara: diam sebentar... %d" % int(math.ceil(hening - dt))
        sisa = hening + self.cfg["audio_kalibrasi_jentik_detik"] - dt
        return "Kalibrasi suara: jentikkan jari %d kali... %d" % (
            self.cfg["audio_kalibrasi_jumlah"], max(1, int(math.ceil(sisa))))

    def _langkah_kalibrasi(self, kal: Dict[str, Any], t: float, e: float) -> None:
        dt = t - kal["mulai"]
        hening = self.cfg["audio_kalibrasi_hening_detik"]
        if dt < 0:
            return
        if dt < hening:
            kal["hening"].append(e)
        elif dt < hening + self.cfg["audio_kalibrasi_jentik_detik"]:
            kal["jentik"].append((t, e))
        else:
            with self._kunci:
                self._kal = None
            self._pesan.put(self._selesaikan_kalibrasi(kal))

    def _selesaikan_kalibrasi(self, kal: Dict[str, Any]) -> str:
        cfg = self.cfg
        if not kal["hening"] or not kal["jentik"]:
            return "Kalibrasi gagal, tekan K lagi"
        # Puncak saat hening, bukan median: median terlalu rendah sehingga suara bicara ikut memicu.
        noise = max(kal["hening"])
        puncak = []  # type: List[Tuple[float, float]]
        for t, e in sorted(kal["jentik"], key=lambda p: -p[1]):
            if e <= noise:
                break
            if all(abs(t - tp) >= cfg["audio_kalibrasi_jarak_detik"] for tp, _ in puncak):
                puncak.append((t, e))
            if len(puncak) == cfg["audio_kalibrasi_jumlah"]:
                break
        if len(puncak) < cfg["audio_kalibrasi_jumlah"]:
            return "Kalibrasi gagal: hanya %d jentikan terdengar, tekan K lagi" % len(puncak)
        ambang = math.sqrt(max(noise, 1e-12) * min(e for _, e in puncak))
        self.onset.ambang_kalibrasi = ambang
        self.ambang_tersimpan = ambang
        try:
            with open(self.path_kalibrasi, "w") as f:
                json.dump({"ambang": ambang, "noise": noise, "jentik_terlemah": min(e for _, e in puncak),
                           "dibuat": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        except OSError as e:
            return "Kalibrasi dipakai tapi gagal disimpan (%s)" % e
        return "Kalibrasi suara tersimpan"

    def ambil_event(self) -> List[float]:
        return self._kuras(self._event)

    def ambil_pesan(self) -> List[str]:
        return self._kuras(self._pesan)

    @staticmethod
    def _kuras(q: queue.Queue) -> List[Any]:
        isi = []
        while True:
            try:
                isi.append(q.get_nowait())
            except queue.Empty:
                return isi

    def _tutup_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as e:
            print("Gagal menutup stream audio:", e)
        self._stream = None

    def hentikan(self) -> None:
        self._stop.set()
        self._tutup_stream()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.aktif = False


class PemicuGabungan:
    """Gabungkan event gesture dan suara sesuai mode trigger."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._gestur = []  # type: List[Tuple[float, Tuple[float, float]]]
        self._suara = []  # type: List[float]

    def reset(self) -> None:
        self._gestur = []
        self._suara = []

    def periksa(self, now: float, mode: str, gestur: List[Tuple[float, Tuple[float, float]]],
                suara: List[float]) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """(dipicu, titik asal flash). Titik asal None berarti pakai posisi cadangan."""
        if mode == "GESTUR":
            return (True, gestur[-1][1]) if gestur else (False, None)
        if mode == "SUARA":
            return (True, None) if suara else (False, None)
        if mode == "ATAU":
            if gestur:
                return True, gestur[-1][1]
            return (True, None) if suara else (False, None)
        # DAN: suara biasanya tiba lebih dulu daripada gambar kamera, jadi keduanya ditahan sebentar.
        jendela = self.cfg["audio_dan_jendela_detik"]
        self._gestur = [g for g in self._gestur + gestur if now - g[0] <= jendela]
        self._suara = [t for t in self._suara + suara if now - t <= jendela]
        for t_g, asal in self._gestur:
            if any(abs(t_g - t_s) <= jendela for t_s in self._suara):
                self.reset()
                return True, asal
        return False, None


def buat_lembar_uji(path_gambar: str, path_bg: Optional[str], path_keluar: Optional[str],
                    cfg: Dict[str, Any]) -> int:
    gambar = cv2.imread(path_gambar)
    if gambar is None:
        print("Gambar tidak bisa dibaca: %s" % path_gambar)
        return 1
    path_seg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfie_segmenter.tflite")
    if not os.path.isfile(path_seg):
        print("File model 'selfie_segmenter.tflite' tidak ditemukan. Lokasi yang diharapkan: %s" % path_seg)
        return 1

    ukuran_proses = (cfg["proses_lebar"], cfg["proses_tinggi"])
    if not path_bg:
        # One photo never shows the room behind the person, so the preview estimates it.
        print("No --bg given: the background behind the person is estimated (FSR inpainting).")
        cfg = dict(cfg, allow_inpaint_fallback=True)
    ctrl = SnapController(cfg)
    seg = PersonSegmenter(path_seg, cfg)
    try:
        kecil = cv2.resize(gambar, ukuran_proses, interpolation=cv2.INTER_AREA)
        mask = seg.perbarui(kecil, 0)
    finally:
        seg.close()
    if path_bg:
        bg = cv2.imread(path_bg)
        if bg is None:
            print("Gambar background tidak bisa dibaca: %s" % path_bg)
            return 1
        ctrl.bg.set_plate(cv2.resize(bg, (gambar.shape[1], gambar.shape[0]), interpolation=cv2.INTER_AREA))
    # Show the still photo long enough for every visible background pixel to become known.
    langkah = cfg["bg_known_samples"] + cfg["bg_safe_frames"] + 1
    for i in range(langkah):
        ctrl.amati(i / 30.0, gambar, kecil, mask)
    t0 = langkah / 30.0
    if not ctrl.picu(t0):
        print(ctrl.pesan(t0))
        return 1

    tinggi, lebar = gambar.shape[:2]
    kolom = int(math.ceil(math.sqrt(cfg["uji_jumlah_frame"])))
    baris = int(math.ceil(cfg["uji_jumlah_frame"] / float(kolom)))
    ubin_l, ubin_t = lebar // kolom, tinggi // kolom
    lembar = np.zeros((ubin_t * baris, ubin_l * kolom, 3), dtype=np.uint8)
    for i in range(cfg["uji_jumlah_frame"]):
        t = i * cfg["uji_selang_detik"]
        ubin = cv2.resize(ctrl.render(t0 + t, gambar), (ubin_l, ubin_t), interpolation=cv2.INTER_AREA)
        teks = "t = %.1f s  %s" % (t, ctrl.state)
        cv2.putText(ubin, teks, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ubin, teks, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        b, k = divmod(i, kolom)
        lembar[b * ubin_t:(b + 1) * ubin_t, k * ubin_l:(k + 1) * ubin_l] = ubin

    if not path_keluar:
        dasar, _ = os.path.splitext(path_gambar)
        path_keluar = dasar + "_lembar.png"
    if not cv2.imwrite(path_keluar, lembar):
        print("Gagal menyimpan: %s" % path_keluar)
        return 1
    print("Lembar contoh disimpan ke %s (%d butir debu)" % (path_keluar, ctrl.debu.count))
    return 0


def main_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Uji efek SNAP tanpa webcam.")
    parser.add_argument("--uji", required=True, metavar="GAMBAR", help="foto orang yang akan dihancurkan")
    parser.add_argument("--bg", metavar="GAMBAR", help="foto background kosong dari sudut kamera yang sama")
    parser.add_argument("--keluar", metavar="PATH", help="lokasi file lembar contoh")
    args = parser.parse_args(argv)
    salah = validasi_config(CONFIG)
    if salah:
        print("CONFIG tidak valid:\n  - " + "\n  - ".join(salah))
        return 1
    return buat_lembar_uji(args.uji, args.bg, args.keluar, CONFIG)


if __name__ == "__main__":
    sys.exit(main_cli())
