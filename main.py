import argparse
import math
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from snap_effect import CONFIG, PersonSegmenter, validasi_config

FOLDER = os.path.dirname(os.path.abspath(__file__))
JENDELA = 'RETROLENS Pake Python'

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


def baca_argumen():
    parser = argparse.ArgumentParser(description="Retrolens: filter portal dan efek SNAP dari webcam.")
    parser.add_argument("--sumber", help="pakai file video sebagai input, bukan webcam")
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
        gagal_baca = 0
        print("%s Tekan tombol 'q' di keyboard untuk keluar." % (
            "Memutar video..." if args.sumber else "Membuka Kamera..."))

        while True:
            mulai = time.monotonic()
            success, img = cap.read()
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

            if cermin:
                img = cv2.flip(img, 1)
            h, w = img.shape[:2]

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            ts = timestamp.berikut()
            results = landmarker.detect_for_video(mp_image, ts)

            mask_person = None
            if retrolens.butuh_mask() and segmenter is not None:
                mask_person = segmenter.confidence(img_rgb, ts)
                if mask_person.shape != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_LINEAR)

            retrolens.proses(img, results, mask_person)
            cv2.imshow(JENDELA, img)

            tunggu_ms = 1
            if jeda_frame > 0:
                tunggu_ms = max(1, int((jeda_frame - (time.monotonic() - mulai)) * 1000))
            key = cv2.waitKey(tunggu_ms) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
        return 0
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        if segmenter is not None:
            segmenter.close()


if __name__ == "__main__":
    sys.exit(main())
