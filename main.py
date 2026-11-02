import argparse
import collections
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from snap import hud
from snap.gestures import HoldGestureDetector
from snap.profiler import Profiler
from snap.worker import InferenceWorker
from snap_effect import (CONFIG, MODE_TRIGGER, STYLE_NAMES, AudioSnapDetector, PemicuGabungan, PersonSegmenter,
                         SnapController, SnapGestureDetector, smoothstep, validasi_config)

FOLDER = os.path.dirname(os.path.abspath(__file__))
JENDELA = 'RETROLENS Pake Python'
PATH_KALIBRASI = os.path.join(FOLDER, "snap_kalibrasi.json")
FOLDER_REKAMAN = os.path.join(FOLDER, "rekaman")
BANTUAN_SNAP = "M mode, B learn room, S snap, X style, H help, T trigger, K sound check, D debug, V record, R reset, Q quit"
BANTUAN_RETROLENS = "M for snap mode, V to record, Q to quit"
WARNA_TINT = (235, 175, 90)
# The D key cycles through these views.
DEBUG_VIEWS = ("off", "stats", "matte", "zone", "coverage", "gain")
# Hold gestures with an action so far: label -> (CONFIG key of the hold time, states where it acts).
# Thumb_Down cancels anything that is not LIVE already.
_SEMUA = (SnapController.LIVE, SnapController.PLATE_CAPTURE) + SnapController.EFFECT_STATES
HOLD_GESTUR = {
    "Thumb_Up": ("hold_thumb_up_s", (SnapController.LIVE,)),
    "Thumb_Down": ("hold_thumb_down_s", _SEMUA[1:]),
    "Open_Palm": ("hold_palm_s", (SnapController.INVISIBLE,)),
    "Victory": ("hold_victory_s", (SnapController.LIVE, SnapController.INVISIBLE)),
    "ILoveYou": ("hold_love_s", _SEMUA),
}
# Gestures that act while held: label -> (CONFIG key of the minimum hold, or None for none, states).
# The controller ignores them during the flash.
TERUS_GESTUR = {
    "Closed_Fist": ("hold_fist_min_s", (SnapController.VANISH, SnapController.REAPPEAR)),
    "Pointing_Up": (None, (SnapController.VANISH, SnapController.REAPPEAR)),
}
BANTUAN_KARTU = (
    "How to use",
    "",
    "Snap: vanish, or come back when invisible",
    "Open palm, 1 s: come back",
    "Thumbs up, 1 s: learn the room again",
    "Peace sign, 1 s: change style (Dust, Ash, Ember)",
    "Fist, hold: freeze time during the effect",
    "Pointing up, hold: slow motion during the effect",
    "Rock sign, 1 s: show or hide this help",
    "Thumbs down, 1 s: cancel and go back to normal",
    "",
    "Keys: S snap, B learn room, X style, H help, R reset",
    "D debug, V record, Shift+V save frames, M mode, Q quit",
)
PETUNJUK_SNAP = {
    "GESTUR": "Snap your fingers to vanish",
    "SUARA": "Snap loud enough for me to hear, or press S",
    "ATAU": "Snap your fingers to vanish",
    "DAN": "Snap where I can see and hear it",
}
PETUNJUK_PORTAL = "Touch thumb and pinkie to change the filter. M for snap mode"
NAMA_TRIGGER = {"GESTUR": "gesture", "SUARA": "sound", "ATAU": "gesture or sound", "DAN": "gesture and sound"}
NAMA_STATE = {SnapController.LIVE: "Live", SnapController.PLATE_CAPTURE: "Learning room",
              SnapController.VANISH: "Vanishing", SnapController.INVISIBLE: "Invisible",
              SnapController.REAPPEAR: "Coming back"}


def bagian_rekam(rekam: Tuple[bool, int]) -> List[Tuple[str, Tuple[int, int, int]]]:
    """Status pill parts for recording: (recording video, frames saved or -1)."""
    bagian = [("REC", hud.MERAH)] if rekam[0] else []
    if rekam[1] >= 0:
        bagian.append(("PNG %d" % rekam[1], hud.MERAH))
    return bagian

MSG_GESTURE_MISSING = "Gesture model missing, only snap works. See README"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
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

        # The filter name and how to change it are drawn by the HUD, so recordings stay clean.


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
        self._penutup = []  # type: List[threading.Thread]

    @property
    def aktif(self) -> bool:
        return self.writer is not None

    def mulai(self, frame, fps: float, now: float) -> str:
        os.makedirs(FOLDER_REKAMAN, exist_ok=True)
        self.path = os.path.join(FOLDER_REKAMAN, time.strftime("snap_%Y%m%d_%H%M%S.mp4"))
        h, w = frame.shape[:2]
        self.fps = max(1.0, round(fps, 1))
        # H.264 when this OpenCV build has it, otherwise MPEG-4 Part 2.
        for kode in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*kode), self.fps, (w, h))
            if writer.isOpened():
                break
            writer.release()
        else:
            return "Cannot start recording"
        self.writer = writer
        self.ukuran = (w, h)
        self.t0 = now
        self.ditulis = 0
        print("Recording to %s (%.1f fps)" % (self.path, self.fps))
        return "Recording"

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
        writer, path = self.writer, self.path
        self.writer = None
        # Finishing an H.264 file takes a moment; doing it off the main thread keeps the picture moving.
        penutup = threading.Thread(target=self._tutup, args=(writer, path), name="recording-close")
        penutup.start()
        self._penutup.append(penutup)
        return "Saved to rekaman/%s" % os.path.basename(path)

    @staticmethod
    def _tutup(writer, path: str) -> None:
        writer.release()
        print("Recording saved: %s" % path)

    def selesai(self) -> None:
        """Wait until every stopped recording is completely written."""
        for penutup in self._penutup:
            penutup.join()
        self._penutup = []


class PerekamPNG:
    """Save every shown frame as a lossless PNG (Shift+V), for checking the effect frame by frame."""

    def __init__(self):
        self.folder = ""
        self.ditulis = 0

    @property
    def aktif(self) -> bool:
        return bool(self.folder)

    def mulai(self) -> str:
        folder = os.path.join(FOLDER_REKAMAN, time.strftime("frames_%Y%m%d_%H%M%S"))
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            print("Cannot create %s: %s" % (folder, e))
            return "Cannot save frames"
        self.folder, self.ditulis = folder, 0
        print("Saving frames to %s" % folder)
        return "Saving frames"

    def tulis(self, frame) -> None:
        if not self.folder:
            return
        path = os.path.join(self.folder, "%05d.png" % self.ditulis)
        # Compression level 1 keeps it lossless and fast enough to run with the camera.
        if not cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            print("Cannot write %s, stopping the frame saver" % path)
            self.berhenti()
            return
        self.ditulis += 1

    def berhenti(self) -> str:
        if not self.folder:
            return ""
        pesan = "Saved %d frames to rekaman/%s" % (self.ditulis, os.path.basename(self.folder))
        print(pesan)
        self.folder = ""
        return pesan


class HudPortal:
    """Status pill, hint and toast for portal mode; snap mode draws its own in ModeSnap."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.t_masuk = -math.inf
        self.pesan = ""
        self.sampai = -math.inf
        self._toast = hud.Pudar(cfg["hud_fade_s"])
        self._petunjuk = hud.Pudar(cfg["hud_fade_s"])

    def masuk(self, now: float) -> None:
        self.t_masuk = now

    def set_pesan(self, teks: str, now: float) -> None:
        self.pesan, self.sampai = teks, now + self.cfg["pesan_detik"] * 2

    def gambar(self, img, now: float, fps: float, nama_filter: str, rekam: Tuple[bool, int]) -> None:
        masuk = float(smoothstep((now - self.t_masuk) / self.cfg["hud_fade_s"]))
        bagian = [("Portal", hud.PUTIH), (nama_filter.title(), hud.PUTIH), ("%d fps" % int(round(fps)), hud.PUTIH)]
        hud.pil(img, bagian + bagian_rekam(rekam), masuk)
        for pudar, teks, gambar in ((self._toast, self.pesan if now < self.sampai else "", hud.toast),
                                    (self._petunjuk, PETUNJUK_PORTAL, hud.petunjuk)):
            pudar.set(teks, now)
            tampil = pudar.get(now)
            if tampil is not None:
                gambar(img, tampil[0], tampil[1])


class ModeSnap:
    """Lem antara kamera, detektor, dan SnapController, plus HUD dan overlay debug."""

    def __init__(self, cfg: Dict[str, Any], segmenter: PersonSegmenter, profiler: Profiler,
                 hold_aktif: bool = False, catatan_gestur: str = ""):
        """hold_aktif: the hand model is a GestureRecognizer; catatan_gestur: shown when SNAP mode opens."""
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
        self.hold_aktif = hold_aktif
        self.catatan_gestur = catatan_gestur
        self.tahan = HoldGestureDetector(cfg)
        self._tangan = []  # type: List[Tuple[np.ndarray, str, float, bool]]
        self._cincin = []  # type: List[Tuple[int, Tuple[float, float], float]]
        self._denyut = None  # type: Optional[Tuple[float, Tuple[float, float]]]
        self.bantuan = False
        self.t_bantuan = -math.inf
        self.t_masuk = -math.inf
        self._pudar_tengah = hud.Pudar(cfg["hud_fade_s"])
        self._pudar_toast = hud.Pudar(cfg["hud_fade_s"])
        self._pudar_petunjuk = hud.Pudar(cfg["hud_fade_s"])
        self._cincin_pudar = {}  # type: Dict[int, Tuple[Tuple[float, float], float, float]]

    def _holds(self) -> Dict[str, float]:
        """Hold gestures that have an action in the current state: label -> hold seconds."""
        if not self.hold_aktif:
            return {}
        return {label: self.cfg[kunci] for label, (kunci, states) in HOLD_GESTUR.items()
                if self.ctrl.state in states}

    def _terus(self) -> Dict[str, float]:
        """Held gestures active in the current state: label -> minimum hold seconds."""
        if not self.hold_aktif:
            return {}
        return {label: (self.cfg[kunci] if kunci else 0.0) for label, (kunci, states) in TERUS_GESTUR.items()
                if self.ctrl.state in states}

    def butuh_tangan(self) -> bool:
        # The snap works in LIVE (vanish) and INVISIBLE (reappear), with or without the gesture model.
        return (self.ctrl.state in (SnapController.LIVE, SnapController.INVISIBLE)
                or bool(self._holds()) or bool(self._terus()))

    def masuk(self, now: float) -> None:
        self.t_masuk = now
        self.ctrl.reset(now)
        self._mulai_ulang_input()
        if self.catatan_gestur:
            self.ctrl.set_pesan(self.catatan_gestur, now, self.cfg["pesan_detik"] * 3)
        if self.mode_trigger != "GESTUR" and not self._pastikan_audio(now):
            self.mode_trigger = "GESTUR"

    def keluar(self, now: float) -> None:
        self.ctrl.reset(now, pudar=False)
        self.bantuan = False

    def tutup(self) -> None:
        if self.audio is not None:
            self.audio.hentikan()

    def _mulai_ulang_input(self) -> None:
        self.gestur.reset()
        self.pemicu.reset()
        self.seg.reset()
        self.ctrl.matte.reset()
        self.tahan.reset()

    def _pastikan_audio(self, now: float) -> bool:
        if self.audio is not None and self.audio.aktif:
            return True
        if not self.cfg["audio_aktif"]:
            self.ctrl.set_pesan("Sound is turned off in the settings", now)
            return False
        if self.audio_gagal:
            self.ctrl.set_pesan("No microphone, using gestures only", now)
            return False
        self.audio = AudioSnapDetector(self.cfg, PATH_KALIBRASI)
        if self.audio.mulai():
            return True
        self.audio_gagal = True
        print(self.audio.error)
        self.ctrl.set_pesan(self.audio.error + ". Using gestures only", now, self.cfg["pesan_detik"] * 3)
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
        self.ctrl.set_pesan(self.audio.error + ". Using gestures only", now, self.cfg["pesan_detik"] * 3)

    def _picu(self, now: float, asal: Optional[Tuple[float, float]]) -> None:
        if self.ctrl.picu(now, asal):
            self.pemicu.reset()
            self.tahan.cooldown(now)

    def _aksi_tahan(self, label: str, pusat: Tuple[float, float], now: float) -> None:
        self._denyut = (now, pusat)
        if label == "Thumb_Up":
            self.ctrl.mulai_rekam_plate(now)
        elif label == "Thumb_Down":
            # Back in LIVE, proses resets the inputs like after any effect.
            self.ctrl.reset(now)
            self.ctrl.set_pesan("Cancelled", now)
        elif label == "Open_Palm":
            self._picu(now, None)
        elif label == "Victory":
            self.ctrl.ganti_gaya(now)
        elif label == "ILoveYou":
            self.ganti_bantuan(now)

    def ganti_bantuan(self, now: float) -> None:
        self.bantuan = not self.bantuan
        self.t_bantuan = now

    def _perbarui_tahan(self, now: float, daftar: List[np.ndarray], hasil_tangan, lebar: int) -> None:
        tangan = []  # type: List[Tuple[np.ndarray, str, float, bool]]
        for lm, kategori in zip(daftar, hasil_tangan.gestures):
            label, skor = (kategori[0].category_name, float(kategori[0].score)) if kategori else ("None", 0.0)
            # Hold gestures are ignored on a hand that is about to snap.
            siap = any(j["lm"] is lm and j["state"] == SnapGestureDetector.SIAP for j in self.gestur.tangan)
            tangan.append((lm, label, skor, siap))
        self._tangan = tangan
        for label, pusat in self.tahan.update(now, tangan, self._holds(), self._terus(), lebar):
            self._aksi_tahan(label, pusat, now)
        self._cincin = self.tahan.progress()
        self.ctrl.atur_waktu("Closed_Fist" in self.tahan.held, "Pointing_Up" in self.tahan.held)

    def proses(self, now: float, frame, hasil_tangan, ts: int,
               model: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Frame akhir tanpa HUD. hasil_tangan None jika landmarker tidak dijalankan.

        model = (process size frame, person mask) already computed for this frame on the inference
        thread; None runs the segmenter here.
        """
        h, w = frame.shape[:2]
        if model is not None:
            kecil, mask = model
        else:
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
        cincin_lama = {k: (pusat, bagian) for k, pusat, bagian in self._cincin}
        self._tangan, self._cincin = [], []
        # Time control only lasts while a fist or pointing hand is seen this frame.
        self.ctrl.atur_waktu(False, False)
        if hasil_tangan is not None:
            daftar = [np.array([(p.x * w, p.y * h) for p in tangan], dtype=np.float32)
                      for tangan in hasil_tangan.hand_landmarks]
            gestur = self.gestur.perbarui(now, daftar, w, h)
            if self.hold_aktif:
                self._perbarui_tahan(now, daftar, hasil_tangan, w)
        if self._denyut is None or self._denyut[0] != now:
            # A hold that stopped without firing fades its ring out; a fired one pulses instead.
            aktif = {k for k, _, _ in self._cincin}
            for k, (pusat, bagian) in cincin_lama.items():
                if k not in aktif:
                    self._cincin_pudar[k] = (pusat, bagian, now)
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
                self.ctrl.set_pesan("Trigger: %s. Press K for a sound check" % NAMA_TRIGGER[berikut], now)
            else:
                self.ctrl.set_pesan("Trigger: %s" % NAMA_TRIGGER[berikut], now)
        elif huruf == "k":
            if self.ctrl.state != SnapController.LIVE:
                return
            if self._pastikan_audio(now):
                self.audio.mulai_kalibrasi(now)
        elif huruf == "d":
            self.debug = (self.debug + 1) % len(DEBUG_VIEWS)
        elif huruf == "x":
            self.ctrl.ganti_gaya(now)
        elif huruf == "h":
            self.ganti_bantuan(now)
        elif huruf == "r":
            self.ctrl.reset(now)
            self._mulai_ulang_input()

    def periksa_fps(self, now: float, pengukur: PengukurFPS) -> None:
        """Feed the measured frame rate to the quality ladder; its level is used from the next effect on."""
        if not pengukur.penuh:
            return
        lama = self.ctrl.tangga.level
        self.ctrl.tangga.perbarui(now, pengukur.fps)
        if self.ctrl.tangga.level != lama:
            print("Quality level %d (fps %.1f); used from the next effect on." % (self.ctrl.tangga.level, pengukur.fps))

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
        for lm, label, skor, siap in self._tangan:
            teks = "%s %.2f%s" % (label, skor, "  (snap ready)" if siap else "")
            tulis(img, teks, (int(lm[0][0]) - 120, min(h - 48, int(lm[0][1]) + 54)), 0.55, (255, 220, 120))
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
            tulis(img, "t efek = %.2f s  rate %.2f  butir %d  ukuran %d px  style %s  quality %d" % (
                self.ctrl.waktu_efek(now), self.ctrl.jam.rate, self.ctrl.butir_tampil, self.ctrl.ukuran_butir,
                self.ctrl.gaya, self.ctrl.tangga.level), (10, y + 40), 0.5)

    def gambar_peta(self, img, view: str) -> None:
        """Semi transparent map of one engine stage over the frame."""
        h, w = img.shape[:2]
        alpha = self.cfg["debug_alpha"]
        matte = self.ctrl.matte.matte
        label = "Debug view: %s  (D for next)" % view
        y_label = 70
        if matte is None or matte.shape != (h, w):
            tulis(img, label + "  -  waiting for the camera", (10, y_label), 0.55)
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
        tulis(img, label, (10, y_label), skala_muat(label, w - 20, 0.55))

    def _petunjuk(self) -> str:
        """The one context hint for the current state."""
        st = self.ctrl.state
        if self.audio is not None and self.audio.aktif and self.audio.sedang_kalibrasi():
            return ""
        if st == SnapController.LIVE:
            if self.audio is not None and self.audio.aktif and self.audio.diam and self.mode_trigger != "GESTUR":
                return "The microphone hears nothing. Check microphone access in System Settings"
            if not self.ctrl.bg_siap:
                if self.hold_aktif:
                    return "Show a thumbs up (or press B), then step out so I can learn the room"
                return "Press B, then step out so I can learn the room"
            return PETUNJUK_SNAP[self.mode_trigger] + "  (H for help)"
        if st == SnapController.PLATE_CAPTURE:
            return "Stay out of the frame"
        if st == SnapController.INVISIBLE:
            return "Snap or show your open palm to come back" if self.hold_aktif else "Snap or press S to come back"
        if self.hold_aktif and self.ctrl.kontrol_waktu_aktif():
            return "Hold a fist to freeze time, point up for slow motion"
        return ""

    def _teks_tengah(self, now: float) -> str:
        if self.ctrl.state == SnapController.PLATE_CAPTURE:
            sisa = self.ctrl.sisa_hitung_mundur(now)
            return "Step out of the frame  %d" % sisa if sisa > 0 else "Learning the room..."
        if self.audio is not None and self.audio.aktif and self.audio.sedang_kalibrasi():
            return self.audio.teks_kalibrasi(now)
        return ""

    def _bagian_pil(self, fps: float, rekam: Tuple[bool, int]) -> List[Tuple[str, Tuple[int, int, int]]]:
        ctrl = self.ctrl
        siap = ("Ready", hud.HIJAU) if ctrl.bg_siap else ("Learning %d%%" % ctrl.bg_persen, hud.KUNING)
        bagian = [("Snap", hud.PUTIH), (NAMA_STATE[ctrl.state], hud.PUTIH), (STYLE_NAMES[ctrl.gaya], hud.PUTIH),
                  siap, ("%d fps" % int(round(fps)), hud.PUTIH)]
        return bagian + bagian_rekam(rekam)

    def gambar_hud(self, img, now: float, fps: float, rekam: Tuple[bool, int] = (False, -1)) -> None:
        """Everything drawn over the shown frame; never part of a recording.

        rekam = (recording video, frames saved by Shift+V or -1 when off).
        """
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
        masuk = float(smoothstep((now - self.t_masuk) / self.cfg["hud_fade_s"]))
        hud.pil(img, self._bagian_pil(fps, rekam), masuk)
        for pudar, teks, gambar in ((self._pudar_tengah, self._teks_tengah(now), hud.tengah),
                                    (self._pudar_toast, self.ctrl.pesan(now), hud.toast),
                                    (self._pudar_petunjuk, self._petunjuk(), hud.petunjuk)):
            pudar.set(teks, now)
            tampil = pudar.get(now)
            if tampil is not None:
                gambar(img, tampil[0], tampil[1])
        k = float(smoothstep((now - self.t_bantuan) / self.cfg["help_fade_s"]))
        alpha_bantuan = k if self.bantuan else 1.0 - k
        if alpha_bantuan > 0.0:
            hud.kartu_bantuan(img, BANTUAN_KARTU, alpha_bantuan, self.cfg)
        self._gambar_cincin(img, now)

    def _gambar_cincin(self, img, now: float) -> None:
        cfg = self.cfg
        for _, pusat, bagian in self._cincin:
            hud.cincin_tahan(img, pusat, bagian, cfg)
        # A hold that was let go fades out instead of disappearing.
        for kunci, (pusat, bagian, sejak) in list(self._cincin_pudar.items()):
            alpha = cfg["ring_alpha"] * (1.0 - float(smoothstep((now - sejak) / cfg["hud_fade_s"])))
            if alpha <= 0.0:
                del self._cincin_pudar[kunci]
                continue
            hud.cincin(img, pusat, bagian, cfg["ring_radius_px"], cfg["ring_stroke_px"], alpha)
        if self._denyut is not None and not hud.denyut(img, self._denyut[1], now - self._denyut[0], cfg):
            self._denyut = None


def path_model(nama: str) -> Optional[str]:
    path = os.path.join(FOLDER, nama)
    if not os.path.isfile(path):
        print("Model file '%s' not found. Put it here: %s" % (nama, path))
        return None
    return path


def buka_model_tangan(cfg) -> Tuple[Any, Any, bool, str]:
    """One hand model for both modes: (model, detect function, hold gestures available, note for the user).

    GestureRecognizer when its file is next to main.py, otherwise HandLandmarker without hold gestures.
    The model is never downloaded here. model is None when neither file can be used.
    """
    path = os.path.join(FOLDER, cfg["gesture_model"])
    if os.path.isfile(path):
        try:
            model = GestureRecognizer.create_from_options(GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=path), running_mode=VisionRunningMode.VIDEO,
                num_hands=2, min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.7,
                min_tracking_confidence=0.7))
            return model, model.recognize_for_video, True, ""
        except Exception as e:
            print("Gesture model %s could not be loaded: %s" % (path, e))
    path_tangan = path_model('hand_landmarker.task')
    if path_tangan is None:
        return None, None, False, ""
    model = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=path_tangan), running_mode=VisionRunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7))
    print(MSG_GESTURE_MISSING)
    return model, model.detect_for_video, False, MSG_GESTURE_MISSING


class TugasModel:
    """One captured frame for the inference thread, with what to compute for it."""

    def __init__(self, frame, now: float, ts: int, mode: str, tangan: bool, seg: Optional[str]):
        self.frame, self.now, self.ts, self.mode, self.tangan, self.seg = frame, now, ts, mode, tangan, seg


class HasilModel:
    """Model results for exactly one TugasModel frame, plus how long each model took (ms)."""

    def __init__(self):
        self.tangan = None  # type: Any
        self.model = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]
        self.mask_portal = None  # type: Optional[np.ndarray]
        self.ms = {}  # type: Dict[str, float]


def jalankan_model(tugas: TugasModel, deteksi_tangan, segmenter, cfg) -> HasilModel:
    """Run the hand model and the segmenter for one frame (on the inference thread)."""
    hasil = HasilModel()
    frame = tugas.frame
    rgb = None
    if tugas.tangan:
        mulai = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hasil.tangan = deteksi_tangan(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), tugas.ts)
        hasil.ms["hand"] = (time.perf_counter() - mulai) * 1000.0
    if tugas.seg is not None:
        mulai = time.perf_counter()
        if tugas.seg == "snap":
            kecil = cv2.resize(frame, (cfg["proses_lebar"], cfg["proses_tinggi"]), interpolation=cv2.INTER_AREA)
            hasil.model = (kecil, segmenter.perbarui(kecil, tugas.ts))
        else:
            if rgb is None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = segmenter.confidence(rgb, tugas.ts)
            h, w = frame.shape[:2]
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            hasil.mask_portal = mask
        hasil.ms["segmenter"] = (time.perf_counter() - mulai) * 1000.0
    return hasil


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
        print("Camera %d could not be opened." % indeks)
    return None, -1


def jendela_terbuka() -> bool:
    try:
        return cv2.getWindowProperty(JENDELA, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        # Backend Cocoa melempar error jika jendela sudah ditutup pengguna.
        return False


def baca_argumen():
    parser = argparse.ArgumentParser(description="Retrolens: portal filters and the SNAP effect from the webcam.")
    parser.add_argument("--sumber", help="use a video file instead of the webcam")
    parser.add_argument("--snap", action="store_true", help="start in SNAP mode")
    return parser.parse_args()


def main() -> int:
    args = baca_argumen()
    salah = validasi_config(CONFIG)
    if salah:
        print("Invalid CONFIG:")
        for s in salah:
            print("  -", s)
        return 1

    cap = None
    landmarker = None
    segmenter = None
    snap = None  # type: Optional[ModeSnap]
    perekam = PerekamVideo()
    perekam_png = PerekamPNG()
    pekerja = None  # type: Optional[InferenceWorker]
    profiler = Profiler(CONFIG["profiler_window"])
    try:
        landmarker, deteksi_tangan, hold_aktif, catatan_gestur = buka_model_tangan(CONFIG)
        if landmarker is None:
            return 1

        path_seg = path_model('selfie_segmenter.tflite')
        if path_seg is not None:
            try:
                segmenter = PersonSegmenter(path_seg, CONFIG)
                snap = ModeSnap(CONFIG, segmenter, profiler, hold_aktif, catatan_gestur)
            except Exception as e:
                print("Person segmenter not available, snap mode is off:", e)

        if args.sumber:
            cap = cv2.VideoCapture(args.sumber)
            if not cap.isOpened():
                print("Cannot open the video file: %s" % args.sumber)
                return 1
            fps_file = cap.get(cv2.CAP_PROP_FPS)
            jeda_frame = 1.0 / fps_file if fps_file > 0 else 0.0
            cermin = False
        else:
            cap, indeks = buka_kamera(CONFIG)
            if cap is None:
                print("No camera could be opened. On macOS, allow your terminal app in "
                      "System Settings > Privacy & Security > Camera.")
                return 1
            print("Camera %d, %dx%d." % (
                indeks, cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            jeda_frame = 0.0
            cermin = True

        retrolens = Retrolens()
        timestamp = Timestamp()
        pengukur = PengukurFPS(CONFIG["fps_jendela_frame"])
        mode = "RETROLENS"
        if args.snap:
            if snap is None:
                print("Snap mode needs selfie_segmenter.tflite; opening portal mode.")
            else:
                mode = "SNAP"
                snap.masuk(time.monotonic())
        portal = HudPortal(CONFIG)
        if mode == "RETROLENS":
            portal.masuk(time.monotonic())

        def kabar(teks: str, now: float) -> None:
            """One message at a time, shown as a toast in the current mode."""
            if not teks:
                return
            if mode == "SNAP":
                snap.ctrl.set_pesan(teks, now, CONFIG["pesan_detik"] * 2)
            else:
                portal.set_pesan(teks, now)

        gagal_baca = 0
        pekerja = InferenceWorker(lambda tugas: jalankan_model(tugas, deteksi_tangan, segmenter, CONFIG))
        terakhir = 0
        now = time.monotonic()
        jadwal = now
        tampil = None
        print("%s Press q in the window to quit." % ("Playing the video..." if args.sumber else "Opening the camera..."))
        print("Mode %s. Keys: %s" % (mode, BANTUAN_SNAP if mode == "SNAP" else BANTUAN_RETROLENS))

        while True:
            with profiler.stage("capture"):
                success, img = cap.read()
                if success and cermin:
                    img = cv2.flip(img, 1)
            if not success:
                if args.sumber:
                    print("The video has ended.")
                    break
                gagal_baca += 1
                if gagal_baca >= CONFIG["kamera_gagal_maks"]:
                    print("The camera gave no picture %d times in a row. On macOS, check the camera permission "
                          "for your terminal app." % gagal_baca)
                    return 1
                cv2.waitKey(30)
                continue
            gagal_baca = 0
            ts = timestamp.berikut()
            if mode == "SNAP":
                seg_mode = "snap"
            elif retrolens.butuh_mask() and segmenter is not None:
                seg_mode = "portal"
            else:
                seg_mode = None
            perlu_tangan = mode == "RETROLENS" or snap.butuh_tangan()
            pekerja.kirim(TugasModel(img, time.monotonic(), ts, mode, perlu_tangan, seg_mode))
            # The newest finished frame, together with the results computed from that very frame.
            jadi = pekerja.ambil(terakhir, 1.0)
            key = 255
            if jadi is not None and jadi[1].mode == mode:
                terakhir, tugas, hasil = jadi
                for nama, ms in hasil.ms.items():
                    profiler.add(nama, ms)
                img, now = tugas.frame, tugas.now
                pengukur.tik(now)
                if mode == "RETROLENS":
                    with profiler.stage("render"):
                        retrolens.proses(img, hasil.tangan, hasil.mask_portal)
                    tampil = img
                else:
                    tampil = snap.proses(now, img, hasil.tangan, tugas.ts, hasil.model)
                    snap.periksa_fps(now, pengukur)

                # Rekaman diambil sebelum HUD supaya file bersih dari teks.
                perekam.tulis(tampil, now)
                perekam_png.tulis(tampil)
                with profiler.stage("hud"):
                    layar = tampil.copy()
                    rekam = (perekam.aktif, perekam_png.ditulis if perekam_png.aktif else -1)
                    if mode == "SNAP":
                        snap.gambar_hud(layar, now, pengukur.fps, rekam)
                    else:
                        portal.gambar(layar, now, pengukur.fps, filters[retrolens.current_filter], rekam)
                with profiler.stage("display"):
                    cv2.imshow(JENDELA, layar)
            elif jadi is not None:
                # Inferred for the other mode just before a mode switch; the next frame fits again.
                terakhir = jadi[0]
            with profiler.stage("display"):
                tunggu_ms = 1
                if jeda_frame > 0:
                    # A video file plays on a fixed schedule, so waiting errors do not add up.
                    jadwal = max(jadwal + jeda_frame, time.monotonic() - jeda_frame)
                    tunggu_ms = max(1, int(round((jadwal - time.monotonic()) * 1000)))
                key = cv2.waitKey(tunggu_ms) & 0xFF
            profiler.end_frame()
            if not jendela_terbuka():
                break
            if key == 255:
                continue
            # Shift+V must be checked before lowercasing, or it would act like V.
            if chr(key) == "V":
                kabar(perekam_png.berhenti() if perekam_png.aktif else perekam_png.mulai(), now)
                continue
            huruf = chr(key).lower()
            if huruf == "q":
                break
            elif huruf == "m":
                if snap is None:
                    kabar("Snap mode needs the file selfie_segmenter.tflite", now)
                elif mode == "RETROLENS":
                    mode = "SNAP"
                    snap.masuk(now)
                else:
                    snap.keluar(now)
                    mode = "RETROLENS"
                    portal.masuk(now)
            elif huruf == "v":
                if perekam.aktif:
                    kabar(perekam.berhenti(), now)
                elif tampil is not None:
                    kabar(perekam.mulai(tampil, pengukur.fps or CONFIG["fps_target"], now), now)
            elif mode == "SNAP":
                snap.tombol(huruf, now)
        return 0
    finally:
        if pekerja is not None:
            # Stop inference before the models it uses are closed below.
            pekerja.tutup()
        perekam.berhenti()
        perekam.selesai()
        perekam_png.berhenti()
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
