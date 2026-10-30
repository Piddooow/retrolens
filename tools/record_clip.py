"""Record a raw test clip from the camera, mirrored like the app, with no HUD and no effect.

Example:
    python tools/record_clip.py room_empty 10

Shows a 3 second countdown, records for SECONDS, then saves uji_lokal/v2/NAME.mp4 at the
measured camera fps, plus NAME.json with the fps and frame times.
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snap_effect import CONFIG  # noqa: E402

WINDOW = "Record test clip"
COUNTDOWN_S = 3.0
JPEG_QUALITY = 95


def label(img: np.ndarray, text: str, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, 0.9, 2)
    x, y = 20, 20 + th
    img[y - th - 8:y + base + 8, x - 8:x + tw + 8] = cv2.convertScaleAbs(
        img[y - th - 8:y + base + 8, x - 8:x + tw + 8], alpha=0.35)
    cv2.putText(img, text, (x, y), font, 0.9, color, 2, cv2.LINE_AA)


def open_camera() -> Optional[cv2.VideoCapture]:
    for index in (CONFIG["kamera_indeks"], 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["kamera_lebar"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["kamera_tinggi"])
            return cap
        cap.release()
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record a raw mirrored test clip.")
    parser.add_argument("name", help="file name without extension, for example room_empty")
    parser.add_argument("seconds", type=float, help="recording length in seconds")
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        print("SECONDS must be more than 0")
        return 1
    folder = os.path.join(ROOT, "uji_lokal", "v2")
    os.makedirs(folder, exist_ok=True)

    cap = open_camera()
    if cap is None:
        print("No camera could be opened. On macOS allow your Terminal app in "
              "System Settings > Privacy & Security > Camera.")
        return 1
    # Frames are kept as JPEG in memory so the fps is known before the file is written.
    encoded = []  # type: List[bytes]
    times = []  # type: List[float]
    start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok:
                if now - start > COUNTDOWN_S + args.seconds + 5.0:
                    print("The camera stopped sending frames.")
                    return 1
                continue
            frame = cv2.flip(frame, 1)
            elapsed = now - start
            view = frame.copy()
            if elapsed < COUNTDOWN_S:
                label(view, "%s: starting in %d" % (args.name, int(COUNTDOWN_S - elapsed) + 1), (0, 200, 255))
            else:
                ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    print("Could not encode a frame.")
                    return 1
                encoded.append(data.tobytes())
                times.append(now)
                left = COUNTDOWN_S + args.seconds - elapsed
                label(view, "REC %s: %.0f s left" % (args.name, max(0.0, left)), (0, 0, 255))
                if left <= 0:
                    break
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                print("Stopped, nothing saved.")
                return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(times) < 2:
        print("Not enough frames recorded.")
        return 1
    fps = (len(times) - 1) / (times[-1] - times[0])
    first = cv2.imdecode(np.frombuffer(encoded[0], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    path = os.path.join(folder, args.name + ".mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print("Could not create %s" % path)
        return 1
    for data in encoded:
        writer.write(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
    writer.release()
    with open(os.path.join(folder, args.name + ".json"), "w") as f:
        json.dump({"fps": fps, "frames": len(times), "size": [w, h],
                   "times": [round(t - times[0], 4) for t in times]}, f, indent=1)
    print("Saved %s: %d frames, %.1f fps, %dx%d" % (path, len(times), fps, w, h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
