import argparse
import collections
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from snap.profiler import Profiler
from snap_effect import (CONFIG, MODE_TRIGGER, AudioSnapDetector, PemicuGabungan, PersonSegmenter,
                         SnapController, SnapGestureDetector, validasi_config)

FOLDER = os.path.dirname(os.path.abspath(__file__))
JENDELA = 'RETROLENS Pake Python'
PATH_KALIBRASI = os.path.join(FOLDER, "snap_kalibrasi.json")
FOLDER_REKAMAN = os.path.join(FOLDER, "rekaman")
BANTUAN_SNAP = "M mode, B background, S snap, T trigger, K kalibrasi, D debug, V rekam, R reset, Q keluar"
BANTUAN_RETROLENS = "M mode SNAP, V rekam, Q keluar"
WARNA_TINT = (235, 175, 90)
# The D key cycles through these views.
DEBUG_VIEWS = ("off", "stats", "matte", "zone", "coverage", "gain")
PETUNJUK_TRIGGER = {
    "GESTUR": "Jentikkan jari (jempol + jari tengah) di depan kamera, atau tekan S",
    "SUARA": "Jentikkan jari sampai terdengar, atau tekan S",
    "ATAU": "Jentikkan jari (gerakan atau bunyi), atau tekan S",
    "DAN": "Jentikkan jari: gerakan dan bunyi harus terdeteksi, atau tekan S",
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

filters = ["MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA", "BLUR", "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY"]

def apply_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
    if filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_name == "INVERT":
        return cv2.bitwise_not(roi)
    elif filter_name == "BLUR":
        return cv2.GaussianBlur(roi, (25, 25), 0)
    elif filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered = cv2.transform(roi, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    elif filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255] # Orange
        filtered[mask_c == 0] = [147, 20, 255]  # Pink
        return filtered
    elif filter_name == "PIXELATE":
        h_r, w_r = roi.shape[:2]
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (w_r//10, h_r//10), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
    elif filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    elif filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif filter_name == "GLITCH":
        h_r, w_r = roi.shape[:2]
        shift = max(5, w_r // 20)
        glitch_roi = roi.copy()
        if w_r > shift:
            glitch_roi[:, :-shift, 2] = roi[:, shift:, 2]
            glitch_roi[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch_roi
    elif filter_name == "NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)
    elif filter_name == "GALAXY" and mask_person is not None and frame_galaxy is not None:
        bh, bw = roi.shape[:2]
        roi_mask = mask_person[y:y+bh, x:x+bw]
        roi_galaxy = frame_galaxy[y:y+bh, x:x+bw]

        # mask_person berisi confidence orang (0..1), jadi background = nilai rendah.
        bg_condition = (roi_mask < CONFIG["mask_ambang"])
        filtered = roi.copy()
        filtered[bg_condition] = roi_galaxy[bg_condition]
        return filtered

    return roi


def buat_galaxy_bg():
    galaxy_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    galaxy_bg[:] = (30, 10, 40) # Space purple
    for _ in range(800):
        sx = np.random.randint(0, 1920)
        sy = np.random.randint(0, 1080)
        galaxy_bg[sy, sx] = (255, 255, 255)
    for _ in range(100):
        sx = np.random.randint(0, 1920)
        sy = np.random.randint(0, 1080)
        cv2.circle(galaxy_bg, (sx, sy), np.random.randint(2, 6), (np.random.randint(150, 255), np.random.randint(100, 255), 255), -1)
    return galaxy_bg


class Retrolens:
    """Mode lama: portal empat jari dan 11 filter."""

    def __init__(self):
        self.galaxy_bg = buat_galaxy_bg()
        self.current_filter = 0
        self.gesture_triggered = False

    def butuh_mask(self):
        return filters[self.current_filter] == "GALAXY"

    def proses(self, img, results, mask_person):
        h, w, c = img.shape
        # Crop galaxy bg to match frame size dynamically
        frame_galaxy = self.galaxy_bg[:h, :w]
        filter_name = filters[self.current_filter]

        pts_portal = []
        change_filter = False

        if results.hand_landmarks:
            # Cek gestur untuk ganti filter
            if len(results.hand_landmarks) >= 2:
                idx0 = results.hand_landmarks[0][8]
                idx1 = results.hand_landmarks[1][8]
                pt0 = (int(idx0.x * w), int(idx0.y * h))
                pt1 = (int(idx1.x * w), int(idx1.y * h))
                if math.hypot(pt0[0]-pt1[0], pt0[1]-pt1[1]) < 40:
                    change_filter = True

            # Cek gestur cubit jempol & telunjuk untuk GAMBAR
            for hand_lms in results.hand_landmarks:
                thumb = hand_lms[4]
                index = hand_lms[8]
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)

                # Gestur ganti filter cadangan (Jempol & Kelingking)
                pinky = hand_lms[20]
                px, py = int(pinky.x * w), int(pinky.y * h)
                if math.hypot(tx - px, ty - py) < 40:
                    change_filter = True

            if change_filter:
                if not self.gesture_triggered:
                    self.current_filter = (self.current_filter + 1) % len(filters)
                    self.gesture_triggered = True
            else:
                self.gesture_triggered = False

            for hand_lms in results.hand_landmarks:
                for id, lm in enumerate(hand_lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id in [4, 8]:
                        pts_portal.append([cx, cy])
                        cv2.circle(img, (cx, cy), 8, (255, 255, 0), cv2.FILLED)

            # 4 POIN -> PORTAL MIRING (Polygon)
            if len(pts_portal) == 4:
                pts_portal.sort(key=lambda p: p[1])
                top_pts = pts_portal[:2]
                bottom_pts = pts_portal[2:]
                top_pts.sort(key=lambda p: p[0])
                bottom_pts.sort(key=lambda p: p[0])

                poly_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.int32)

                x, y, bw, bh = cv2.boundingRect(poly_pts)
                x, y = max(0, x), max(0, y)
                bw, bh = min(w - x, bw), min(h - y, bh)

                if bw > 0 and bh > 0:
                    roi = img[y:y+bh, x:x+bw].copy()
                    filtered_roi = apply_filter(roi, filter_name, x, y, mask_person, frame_galaxy)

                    mask = np.zeros((bh, bw), dtype=np.uint8)
                    poly_roi = poly_pts - [x, y]
                    cv2.fillPoly(mask, [poly_roi], 255)
                    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                    img[y:y+bh, x:x+bw] = np.where(mask_3ch == 255, filtered_roi, roi)

                    cv2.polylines(img, [poly_pts], True, (255, 255, 255), 2)

                    # --- PARTIKEL GLOW DI TEPI PORTAL ---
                    for i in range(4):
                        pt1 = poly_pts[i]
                        pt2 = poly_pts[(i+1)%4]
                        for _ in range(5):
                            alpha = np.random.random()
                            px = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + np.random.randint(-15, 15)
                            py = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + np.random.randint(-15, 15)
                            cv2.circle(img, (px, py), np.random.randint(1, 4), (0, 255, 255), -1)

                    cv2.putText(img, f"PORTAL: {filter_name}", (top_pts[0][0], top_pts[0][1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        # Overlay Teks
        cv2.putText(img, "Ganti Filter: Jempol & Kelingking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Filter Aktif: {filters[self.current_filter]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


class Timestamp:
    """Timestamp ms untuk MediaPipe VIDEO: wajib naik terus, jadi pakai jam monotonic."""

    def __init__(self):
        self.terakhir = -1

    def berikut(self) -> int:
        self.terakhir = max(self.terakhir + 1, time.monotonic_ns() // 1_000_000)
        return self.terakhir


def tulis(img, teks: str, pos: Tuple[int, int], skala: float = 0.6, warna=(255, 255, 255),
          tebal: int = 1, tengah: bool = False) -> None:
    """Teks di atas kotak gelap supaya terbaca di latar terang maupun gelap."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (lebar_teks, tinggi_teks), dasar = cv2.getTextSize(teks, font, skala, tebal)
    x, y = pos
    if tengah:
        x -= lebar_teks // 2
    # Di OpenCV 5 ketebalan mengubah lebar huruf, jadi garis tepi tebal tidak bisa dipakai sebagai outline.
    pad = max(3, tinggi_teks // 3)
    h, w = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - tinggi_teks - pad)
    x1, y1 = min(w, x + lebar_teks + pad), min(h, y + dasar + pad)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = cv2.convertScaleAbs(img[y0:y1, x0:x1], alpha=0.35)
    cv2.putText(img, teks, (x, y), font, skala, warna, tebal, cv2.LINE_AA)


def tumpuk(img, peta, warna, opacity: float) -> None:
    """Blend a solid color over img in place, weighted per pixel by peta (0..1 or bool)."""
    a = peta.astype(np.float32) * np.float32(opacity)
    img[:] = cv2.blendLinear(img, np.full_like(img, warna), 1.0 - a, a)


def skala_muat(teks: str, lebar_maks: int, skala: float, tebal: int = 1) -> float:
    (lebar_teks, _), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, skala, tebal)
    return skala if lebar_teks <= lebar_maks else skala * lebar_maks / float(lebar_teks)


def tulis_bawah(img, teks: str, baris_dari_bawah: int, warna=(255, 255, 255)) -> None:
    h, w = img.shape[:2]
    skala = skala_muat(teks, w - 20, 0.55)
    tulis(img, teks, (w // 2, h - 14 - int(34 * baris_dari_bawah)), skala, warna, tengah=True)


class PengukurFPS:
    def __init__(self, jendela: int):
        self.waktu = collections.deque(maxlen=jendela + 1)  # type: collections.deque

    def tik(self, now: float) -> None:
        self.waktu.append(now)

    @property
    def penuh(self) -> bool:
        return len(self.waktu) == self.waktu.maxlen

    @property
    def fps(self) -> float:
        if len(self.waktu) < 2 or self.waktu[-1] <= self.waktu[0]:
            return 0.0
        return (len(self.waktu) - 1) / (self.waktu[-1] - self.waktu[0])


class PerekamVideo:
    """Rekam frame tampilan ke MP4. Frame ditulis menurut jam supaya durasi file sama dengan aslinya."""

    def __init__(self):
        self.writer = None
        self.path = ""

    @property
    def aktif(self) -> bool:
        return self.writer is not None

    def mulai(self, frame, fps: float, now: float) -> str:
        os.makedirs(FOLDER_REKAMAN, exist_ok=True)
        self.path = os.path.join(FOLDER_REKAMAN, time.strftime("snap_%Y%m%d_%H%M%S.mp4"))
        h, w = frame.shape[:2]
        self.fps = max(1.0, round(fps, 1))
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not writer.isOpened():
            writer.release()
            return "Gagal membuat file rekaman"
        self.writer = writer
        self.ukuran = (w, h)
        self.t0 = now
        self.ditulis = 0
        print("Merekam ke %s (%.1f fps)" % (self.path, self.fps))
        return "Mulai merekam"

    def tulis(self, frame, now: float) -> None:
        if self.writer is None or (frame.shape[1], frame.shape[0]) != self.ukuran:
            return
        target = int((now - self.t0) * self.fps) + 1
        while self.ditulis < target:
            self.writer.write(frame)
            self.ditulis += 1

    def berhenti(self) -> str:
        if self.writer is None:
            return ""
        self.writer.release()
        self.writer = None
        print("Rekaman disimpan: %s" % self.path)
        return "Rekaman disimpan: rekaman/%s" % os.path.basename(self.path)


class ModeSnap:
    """Lem antara kamera, detektor, dan SnapController, plus HUD dan overlay debug."""

    def __init__(self, cfg: Dict[str, Any], segmenter: PersonSegmenter, profiler: Profiler):
        self.cfg = cfg
        self.seg = segmenter
        self.ctrl = SnapController(cfg)
        self.ctrl.prof = profiler
        self.gestur = SnapGestureDetector(cfg)
        self.pemicu = PemicuGabungan(cfg)
        self.audio = None  # type: Optional[AudioSnapDetector]
        self.audio_gagal = False
        self.mode_trigger = cfg["mode_trigger"]
        self.debug = 0
        self.state_lalu = SnapController.LIVE
        self.prof = profiler
        self.t_fps_rendah = None  # type: Optional[float]

    def butuh_tangan(self) -> bool:
        return self.ctrl.state == SnapController.LIVE

    def masuk(self, now: float) -> None:
        self.ctrl.reset(now)
        self._mulai_ulang_input()
        if self.mode_trigger != "GESTUR" and not self._pastikan_audio(now):
            self.mode_trigger = "GESTUR"

    def keluar(self, now: float) -> None:
        self.ctrl.reset(now)

    def tutup(self) -> None:
        if self.audio is not None:
            self.audio.hentikan()

    def _mulai_ulang_input(self) -> None:
        self.gestur.reset()
        self.pemicu.reset()
        self.seg.reset()
        self.ctrl.matte.reset()

    def _pastikan_audio(self, now: float) -> bool:
        if self.audio is not None and self.audio.aktif:
            return True
        if not self.cfg["audio_aktif"]:
            self.ctrl.set_pesan("Audio dimatikan di CONFIG (audio_aktif)", now)
            return False
        if self.audio_gagal:
            self.ctrl.set_pesan("Mikrofon tidak tersedia, hanya mode GESTUR", now)
            return False
        self.audio = AudioSnapDetector(self.cfg, PATH_KALIBRASI)
        if self.audio.mulai():
            return True
        self.audio_gagal = True
        print(self.audio.error)
        self.ctrl.set_pesan(self.audio.error + ". Mode trigger: GESTUR", now, self.cfg["pesan_detik"] * 3)
        return False

    def _cek_audio(self, now: float) -> None:
        if self.audio is None or self.audio.aktif or self.audio_gagal:
            return
        # Mikrofon hilang di tengah jalan: matikan audio dan kembali ke gesture.
        self.audio_gagal = True
        print(self.audio.error)
        self.audio.hentikan()
        if self.mode_trigger != "GESTUR":
            self.mode_trigger = "GESTUR"
        self.ctrl.set_pesan(self.audio.error + ". Mode trigger: GESTUR", now, self.cfg["pesan_detik"] * 3)

    def _picu(self, now: float, asal: Optional[Tuple[float, float]]) -> None:
        if self.ctrl.picu(now, asal):
            self.pemicu.reset()

    def proses(self, now: float, frame, hasil_tangan, ts: int):
        """Frame akhir tanpa HUD. hasil_tangan None jika landmarker tidak dijalankan."""
        h, w = frame.shape[:2]
        # Runs in every state: during the effect the live person is hidden too.
        with self.prof.stage("segmenter"):
            kecil = cv2.resize(frame, (self.cfg["proses_lebar"], self.cfg["proses_tinggi"]),
                               interpolation=cv2.INTER_AREA)
            mask = self.seg.perbarui(kecil, ts)
        # Times its own matte and background stages.
        self.ctrl.amati(now, frame, kecil, mask)

        self._cek_audio(now)
        audio_aktif = self.audio is not None and self.audio.aktif
        suara = self.audio.ambil_event() if audio_aktif else []
        if audio_aktif:
            for pesan in self.audio.ambil_pesan():
                self.ctrl.set_pesan(pesan, now, self.cfg["pesan_detik"] * 2)
        gestur = []  # type: List[Tuple[float, Tuple[float, float]]]
        if hasil_tangan is not None:
            daftar = [np.array([(p.x * w, p.y * h) for p in tangan], dtype=np.float32)
                      for tangan in hasil_tangan.hand_landmarks]
            gestur = self.gestur.perbarui(now, daftar, w, h)
        kalibrasi = audio_aktif and self.audio.sedang_kalibrasi()
        if not kalibrasi and self.ctrl.siap_dipicu(now):
            dipicu, asal = self.pemicu.periksa(now, self.mode_trigger, gestur, suara)
            if dipicu:
                self._picu(now, asal)

        with self.prof.stage("render"):
            hasil = self.ctrl.render(now, frame)
        if self.ctrl.state == SnapController.LIVE and self.state_lalu != SnapController.LIVE:
            self._mulai_ulang_input()
        self.state_lalu = self.ctrl.state
        return hasil

    def tombol(self, huruf: str, now: float) -> None:
        if huruf == "b":
            if self.ctrl.state == SnapController.LIVE:
                self.ctrl.mulai_rekam_plate(now)
        elif huruf in ("s", " "):
            self._picu(now, None)
        elif huruf == "t":
            berikut = MODE_TRIGGER[(MODE_TRIGGER.index(self.mode_trigger) + 1) % len(MODE_TRIGGER)]
            self.pemicu.reset()
            if berikut != "GESTUR" and not self._pastikan_audio(now):
                # Pesan alasan dari _pastikan_audio dibiarkan tampil.
                self.mode_trigger = "GESTUR"
                return
            self.mode_trigger = berikut
            if berikut != "GESTUR" and self.audio is not None and self.audio.ambang_tersimpan is None:
                self.ctrl.set_pesan("Trigger %s. Tekan K untuk kalibrasi suara" % berikut, now)
            else:
                self.ctrl.set_pesan("Trigger: %s" % berikut, now)
        elif huruf == "k":
            if self.ctrl.state != SnapController.LIVE:
                return
            if self._pastikan_audio(now):
                self.audio.mulai_kalibrasi(now)
        elif huruf == "d":
            self.debug = (self.debug + 1) % len(DEBUG_VIEWS)
        elif huruf == "r":
            self.ctrl.reset(now)
            self._mulai_ulang_input()

    def periksa_fps(self, now: float, pengukur: PengukurFPS) -> None:
        """Butir debu membesar jika FPS rendah terus-menerus; berlaku mulai snap berikutnya."""
        turun = self.cfg["debu_ukuran_butir_turun"]
        if not pengukur.penuh or self.ctrl.ukuran_butir == turun or pengukur.fps >= self.cfg["fps_batas_turun"]:
            self.t_fps_rendah = None
            return
        if self.t_fps_rendah is None:
            self.t_fps_rendah = now
        elif now - self.t_fps_rendah >= self.cfg["fps_turun_detik"]:
            self.ctrl.ukuran_butir = turun
            print("FPS %.1f di bawah %d selama %.0f detik: ukuran butir debu naik ke %d piksel." % (
                pengukur.fps, self.cfg["fps_batas_turun"], self.cfg["fps_turun_detik"], turun))

    def gambar_debug(self, img, now: float) -> None:
        h, w = img.shape[:2]
        if self.seg.mask is not None:
            orang = (cv2.resize(self.seg.mask, (w, h), interpolation=cv2.INTER_LINEAR)
                     > self.cfg["mask_ambang"]).astype(np.uint8)
            hijau = cv2.addWeighted(img, 0.6, np.full_like(img, (0, 160, 0)), 0.4, 0.0)
            cv2.copyTo(hijau, orang, img)
        if self.butuh_tangan():
            for jejak in self.gestur.tangan:
                lm = jejak["lm"]
                for i in (0, 4, 9, 12):
                    cv2.circle(img, (int(lm[i][0]), int(lm[i][1])), 6, (0, 255, 255), -1)
                jentik = now - jejak["t_jentik"] < self.cfg["pesan_detik"]
                teks = "cubit %.2f  tengah %.2f  %s" % (jejak["c"], jejak["m"], "JENTIK" if jentik else jejak["state"])
                tulis(img, teks, (int(lm[0][0]) - 120, min(h - 70, int(lm[0][1]) + 30)), 0.55,
                      (0, 255, 0) if jejak["state"] == SnapGestureDetector.SIAP or jentik else (255, 255, 255))
        y = 90
        if self.audio is not None and self.audio.aktif:
            level, ambang = self.audio.level, max(self.audio.ambang, 1e-12)
            # Skala log: ambang di tengah bar, dua dekade ke kiri dan kanan.
            posisi = min(1.0, max(0.0, (math.log10(max(level, 1e-12)) - math.log10(ambang) + 2.0) / 4.0))
            cv2.rectangle(img, (10, y), (210, y + 14), (60, 60, 60), -1)
            cv2.rectangle(img, (10, y), (10 + int(200 * posisi), y + 14),
                          (0, 0, 255) if level > ambang else (0, 200, 0), -1)
            cv2.line(img, (110, y - 3), (110, y + 17), (255, 255, 255), 2)
            tulis(img, "suara %.1e / ambang %.1e" % (level, ambang), (220, y + 12), 0.5)
            y += 30
        tahap = "ms (avg %d frames): %s  total %.1f" % (
            self.prof.window, "  ".join(self.prof.summary()), self.prof.total())
        tulis(img, tahap, (10, y + 10), skala_muat(tahap, w - 20, 0.5))
        if self.ctrl.state not in (SnapController.LIVE, SnapController.PLATE_CAPTURE):
            tulis(img, "t efek = %.2f s  butir %d  ukuran %d px" % (
                self.ctrl.waktu_efek(now), self.ctrl.butir_tampil, self.ctrl.ukuran_butir), (10, y + 40), 0.5)

    def gambar_peta(self, img, view: str) -> None:
        """Semi transparent map of one engine stage over the frame."""
        h, w = img.shape[:2]
        alpha = self.cfg["debug_alpha"]
        matte = self.ctrl.matte.matte
        label = "Debug view: %s  (D for next)" % view
        if matte is None or matte.shape != (h, w):
            tulis(img, label + "  -  waiting for the camera", (10, 60), 0.55)
            return
        if view == "matte":
            tumpuk(img, matte, (0, 200, 0), alpha)
        elif view == "zone":
            zf, shadow = self.ctrl.zona_hapus()
            tumpuk(img, zf * ~shadow, (200, 0, 200), alpha)
            tumpuk(img, shadow, (0, 140, 255), alpha)
            label += "  (orange = shadow)"
        elif view == "coverage":
            tumpuk(img, ~self.ctrl.bg.known(), (0, 0, 220), alpha)
            label += "  (red = room not seen yet, %d%% known)" % int(100 * self.ctrl.bg.known_fraction())
        elif view == "gain":
            gain = self.ctrl.peta_gain().mean(axis=2)
            lo, hi = self.cfg["bg_gain_min"], self.cfg["bg_gain_max"]
            warna = cv2.applyColorMap(cv2.convertScaleAbs(gain - lo, alpha=255.0 / (hi - lo)), cv2.COLORMAP_VIRIDIS)
            warna = cv2.resize(warna, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(img, 1.0 - alpha, warna, alpha, 0.0, dst=img)
            label += "  (gain %.2f to %.2f)" % (float(gain.min()), float(gain.max()))
        tulis(img, label, (10, 60), skala_muat(label, w - 20, 0.55))

    def gambar_hud(self, img, now: float, fps: float) -> None:
        h, w = img.shape[:2]
        tint = self.ctrl.tint(now)
        if tint is not None and tint[0].shape == img.shape[:2]:
            # Show which part of the room is still unknown after a refused snap.
            mask = tint[0].view(np.uint8)
            biru = cv2.addWeighted(img, 1.0 - tint[1], np.full_like(img, WARNA_TINT), tint[1], 0.0)
            cv2.copyTo(biru, mask, img)
        view = DEBUG_VIEWS[self.debug]
        if view == "stats":
            self.gambar_debug(img, now)
        elif view != "off":
            self.gambar_peta(img, view)
        status = "Mode SNAP  |  State %s  |  Trigger %s  |  %s  |  %d FPS" % (
            self.ctrl.state, self.mode_trigger, self.ctrl.status_bg(), int(round(fps)))
        tulis(img, status, (10, 30), skala_muat(status, w - 120, 0.6), (0, 255, 255))

        tengah = ""
        if self.ctrl.state == SnapController.PLATE_CAPTURE:
            sisa = self.ctrl.sisa_hitung_mundur(now)
            tengah = "Step out of the frame: %d" % sisa if sisa > 0 else "Learning the room, stay out of the frame..."
        elif self.audio is not None and self.audio.aktif and self.audio.sedang_kalibrasi():
            tengah = self.audio.teks_kalibrasi(now)
        if tengah:
            tulis(img, tengah, (w // 2, h // 2), skala_muat(tengah, w - 40, 1.2, 2), (255, 255, 255), 2, tengah=True)

        pesan = self.ctrl.pesan(now)
        if pesan:
            tulis_bawah(img, pesan, 2, (0, 255, 255))
        elif self.ctrl.state == SnapController.LIVE and not tengah:
            if self.audio is not None and self.audio.aktif and self.audio.diam and self.mode_trigger != "GESTUR":
                tulis_bawah(img, "Mikrofon diam. Cek izin mikrofon terminal di System Settings", 2, (0, 165, 255))
            else:
                tulis_bawah(img, PETUNJUK_TRIGGER[self.mode_trigger], 2)
        tulis_bawah(img, BANTUAN_SNAP, 1, (200, 200, 200))


def path_model(nama: str) -> Optional[str]:
    path = os.path.join(FOLDER, nama)
    if not os.path.isfile(path):
        print("File model '%s' tidak ditemukan. Lokasi yang diharapkan: %s" % (nama, path))
        return None
    return path


def buka_kamera(cfg) -> Tuple[Optional[cv2.VideoCapture], int]:
    kandidat = [cfg["kamera_indeks"]]
    if cfg["kamera_indeks"] != 1:
        kandidat.append(1)
    for indeks in kandidat:
        cap = cv2.VideoCapture(indeks)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["kamera_lebar"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["kamera_tinggi"])
            return cap, indeks
        cap.release()
        print("Kamera indeks %d gagal dibuka." % indeks)
    return None, -1


def jendela_terbuka() -> bool:
    try:
        return cv2.getWindowProperty(JENDELA, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        # Backend Cocoa melempar error jika jendela sudah ditutup pengguna.
        return False


def baca_argumen():
    parser = argparse.ArgumentParser(description="Retrolens: filter portal dan efek SNAP dari webcam.")
    parser.add_argument("--sumber", help="pakai file video sebagai input, bukan webcam")
    parser.add_argument("--snap", action="store_true", help="langsung buka mode SNAP")
    return parser.parse_args()


def main() -> int:
    args = baca_argumen()
    salah = validasi_config(CONFIG)
    if salah:
        print("CONFIG tidak valid:")
        for s in salah:
            print("  -", s)
        return 1

    path_tangan = path_model('hand_landmarker.task')
    if path_tangan is None:
        return 1

    cap = None
    landmarker = None
    segmenter = None
    snap = None  # type: Optional[ModeSnap]
    perekam = PerekamVideo()
    profiler = Profiler(CONFIG["profiler_window"])
    try:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path_tangan),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        landmarker = HandLandmarker.create_from_options(options)

        path_seg = path_model('selfie_segmenter.tflite')
        if path_seg is not None:
            try:
                segmenter = PersonSegmenter(path_seg, CONFIG)
                snap = ModeSnap(CONFIG, segmenter, profiler)
            except Exception as e:
                print("Segmenter tidak tersedia:", e)

        if args.sumber:
            cap = cv2.VideoCapture(args.sumber)
            if not cap.isOpened():
                print("File video tidak bisa dibuka: %s" % args.sumber)
                return 1
            fps_file = cap.get(cv2.CAP_PROP_FPS)
            jeda_frame = 1.0 / fps_file if fps_file > 0 else 0.0
            cermin = False
        else:
            cap, indeks = buka_kamera(CONFIG)
            if cap is None:
                print("Tidak ada kamera yang bisa dibuka. Di macOS, izinkan aplikasi terminal di "
                      "System Settings > Privacy & Security > Camera.")
                return 1
            print("Kamera indeks %d, resolusi %dx%d." % (
                indeks, cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            jeda_frame = 0.0
            cermin = True

        retrolens = Retrolens()
        timestamp = Timestamp()
        pengukur = PengukurFPS(CONFIG["fps_jendela_frame"])
        mode = "RETROLENS"
        if args.snap:
            if snap is None:
                print("Mode SNAP butuh selfie_segmenter.tflite; membuka mode RETROLENS.")
            else:
                mode = "SNAP"
                snap.masuk(time.monotonic())
        pesan_umum = ""
        pesan_umum_sampai = 0.0
        gagal_baca = 0
        print("%s Tekan tombol 'q' di keyboard untuk keluar." % (
            "Memutar video..." if args.sumber else "Membuka Kamera..."))
        print("Mode %s. Tombol: %s" % (mode, BANTUAN_SNAP if mode == "SNAP" else BANTUAN_RETROLENS))

        while True:
            mulai = time.monotonic()
            with profiler.stage("capture"):
                success, img = cap.read()
                if success and cermin:
                    img = cv2.flip(img, 1)
            if not success:
                if args.sumber:
                    print("Video selesai.")
                    break
                gagal_baca += 1
                if gagal_baca >= CONFIG["kamera_gagal_maks"]:
                    print("Kamera gagal memberi gambar %d kali berturut-turut. Di macOS, cek izin kamera "
                          "untuk aplikasi terminal." % gagal_baca)
                    return 1
                cv2.waitKey(30)
                continue
            gagal_baca = 0
            h, w = img.shape[:2]
            now = time.monotonic()
            pengukur.tik(now)

            ts = timestamp.berikut()
            results = None
            img_rgb = None
            if mode == "RETROLENS" or snap.butuh_tangan():
                with profiler.stage("hand"):
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    results = landmarker.detect_for_video(mp_image, ts)

            if mode == "RETROLENS":
                mask_person = None
                if retrolens.butuh_mask() and segmenter is not None:
                    with profiler.stage("segmenter"):
                        mask_person = segmenter.confidence(img_rgb, ts)
                        if mask_person.shape != (h, w):
                            mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_LINEAR)
                with profiler.stage("render"):
                    retrolens.proses(img, results, mask_person)
                tampil = img
            else:
                tampil = snap.proses(now, img, results, ts)
                snap.periksa_fps(now, pengukur)

            # Rekaman diambil sebelum HUD supaya file bersih dari teks.
            perekam.tulis(tampil, now)
            with profiler.stage("hud"):
                layar = tampil.copy()
                if mode == "SNAP":
                    snap.gambar_hud(layar, now, pengukur.fps)
                else:
                    tulis_bawah(layar, BANTUAN_RETROLENS, 1, (200, 200, 200))
                if now < pesan_umum_sampai:
                    tulis_bawah(layar, pesan_umum, 3, (0, 255, 255))
                if perekam.aktif:
                    cv2.circle(layar, (w - 95, 25), 9, (0, 0, 255), -1)
                    tulis(layar, "REC", (w - 78, 33), 0.7, (0, 0, 255), 2)

            with profiler.stage("display"):
                cv2.imshow(JENDELA, layar)
                tunggu_ms = 1
                if jeda_frame > 0:
                    tunggu_ms = max(1, int((jeda_frame - (time.monotonic() - mulai)) * 1000))
                key = cv2.waitKey(tunggu_ms) & 0xFF
            profiler.end_frame()
            if not jendela_terbuka():
                break
            if key == 255:
                continue
            huruf = chr(key).lower()
            if huruf == "q":
                break
            elif huruf == "m":
                if snap is None:
                    pesan_umum = "Mode SNAP butuh selfie_segmenter.tflite"
                    pesan_umum_sampai = now + CONFIG["pesan_detik"] * 2
                elif mode == "RETROLENS":
                    mode = "SNAP"
                    snap.masuk(now)
                else:
                    snap.keluar(now)
                    mode = "RETROLENS"
            elif huruf == "v":
                if perekam.aktif:
                    pesan_umum = perekam.berhenti()
                else:
                    pesan_umum = perekam.mulai(tampil, pengukur.fps or CONFIG["fps_target"], now)
                pesan_umum_sampai = now + CONFIG["pesan_detik"] * 2
            elif mode == "SNAP":
                snap.tombol(huruf, now)
        return 0
    finally:
        perekam.berhenti()
        if snap is not None:
            snap.tutup()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        if segmenter is not None:
            segmenter.close()


if __name__ == "__main__":
    sys.exit(main())
