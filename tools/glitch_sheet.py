"""Side by side sheets of the Glitch score: each reference frame next to the same step drawn on a camera frame.

Example:
    python tools/glitch_sheet.py uji_lokal/orang.png --out uji_lokal/render/glitch_sheet

Writes OUT/sheet_NN.jpg (8 steps per sheet, reference left, ours right) and OUT/side_by_side.mp4 at the
reference's 13 fps. Without --bg the room behind the person is a stand-in made by inpainting, so slice gaps
show a guessed room there.
"""

import argparse
import os
import sys
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from snap.glitch import GlitchRenderer  # noqa: E402
from snap.glitch_score import SCORE  # noqa: E402
from snap.matte import MatteEngine  # noqa: E402
from snap_effect import CONFIG, PersonSegmenter  # noqa: E402

DEFAULT_REF = os.path.join(os.path.dirname(ROOT), "reference-effect-spider-verse-1.mp4")
TILE_H = 270
PER_SHEET = 8


def read_all(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("Cannot open video: %s" % path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def first_frame(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        frames = read_all(path)
        if not frames:
            raise SystemExit("No picture in %s" % path)
        img = frames[0]
    return img


def tile(img: np.ndarray, label: str) -> np.ndarray:
    w = int(round(img.shape[1] * TILE_H / img.shape[0]))
    t = cv2.resize(img, (w, TILE_H), interpolation=cv2.INTER_AREA)
    cv2.putText(t, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(t, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def describe(step: dict) -> str:
    return " ".join(sorted(k for k in step if k != "cover")) or "calm"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reference vs Glitch step sheets.")
    parser.add_argument("frame", help="photo or video of the person (its first frame is used)")
    parser.add_argument("--bg", help="photo of the empty room from the same spot")
    parser.add_argument("--ref", default=DEFAULT_REF, help="reference-effect-spider-verse-1.mp4")
    parser.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args(argv)

    frame = first_frame(args.frame)
    ref = read_all(args.ref)
    if len(ref) != len(SCORE):
        print("The reference has %d frames, the score %d steps" % (len(ref), len(SCORE)))
        return 1
    cfg = dict(CONFIG)
    seg = PersonSegmenter(os.path.join(ROOT, "selfie_segmenter.tflite"), cfg)
    small = cv2.resize(frame, (cfg["proses_lebar"], cfg["proses_tinggi"]), interpolation=cv2.INTER_AREA)
    mask = seg.perbarui(small, 1)
    seg.close()
    matte = MatteEngine(cfg)
    matte.update(frame, mask)
    alpha = matte.matte
    if args.bg:
        room = cv2.imread(args.bg)
        if room is None or room.shape != frame.shape:
            print("--bg must be a photo with the same size as the frame")
            return 1
    else:
        hole = cv2.dilate((alpha > 0.0).astype(np.uint8) * 255, np.ones((41, 41), np.uint8))
        room = cv2.inpaint(frame, hole, 9, cv2.INPAINT_TELEA)
    a = cv2.dilate(alpha, np.ones((5, 5), np.uint8))[..., None]
    clean = np.clip(frame * (1.0 - a) + room * a + 0.5, 0, 255).astype(np.uint8)

    os.makedirs(args.out, exist_ok=True)
    renderer = GlitchRenderer()
    pairs = []
    for i, step in enumerate(SCORE):
        ours = renderer.render(frame, alpha, clean, step, seed=i)
        pair = np.hstack([tile(ref[i], "ref #%d" % i), tile(ours, "ours: %s" % describe(step))])
        pairs.append(pair)
    writer = cv2.VideoWriter(os.path.join(args.out, "side_by_side.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 13.0,
                             (pairs[0].shape[1], pairs[0].shape[0]))
    for p in pairs:
        writer.write(p)
    writer.release()
    for k in range(0, len(pairs), PER_SHEET):
        group = pairs[k:k + PER_SHEET]
        while len(group) % 2:
            group.append(np.zeros_like(group[0]))
        rows = [np.hstack([group[j], np.full((TILE_H, 12, 3), 40, np.uint8), group[j + 1]])
                for j in range(0, len(group), 2)]
        cv2.imwrite(os.path.join(args.out, "sheet_%02d.jpg" % (k // PER_SHEET)), np.vstack(rows),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("Wrote %d sheets and side_by_side.mp4 to %s" % ((len(pairs) + PER_SHEET - 1) // PER_SHEET, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
