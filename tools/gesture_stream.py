"""Record what the GestureRecognizer sees in a clip: landmarks, top label and score per hand.

The result feeds the gesture tests and tuning without running the model again.

Example (the old jentik and biasa clips were saved unmirrored, the app mirrors the camera):
    python tools/gesture_stream.py --mirror uji_lokal/jentik.mp4 uji_lokal/gestur_jentik.npz

Saved arrays: lm (frames, 2, 21, 2) pixels with NaN for a missing hand, label (frames, 2) with ""
for a missing hand, score (frames, 2), plus fps, w and h.
"""

import argparse
import os
import sys
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snap_effect import CONFIG  # noqa: E402

MAX_HANDS = 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record GestureRecognizer output of a clip.")
    parser.add_argument("video")
    parser.add_argument("out", help=".npz file to write")
    parser.add_argument("--mirror", action="store_true",
                        help="flip frames like the app does with the camera (for clips saved unmirrored)")
    args = parser.parse_args(argv)
    model = os.path.join(ROOT, CONFIG["gesture_model"])
    if not os.path.isfile(model):
        print("Gesture model not found: %s (see README, one time setup)" % model)
        return 1
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Cannot open video: %s" % args.video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vision = mp.tasks.vision
    options = vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS, min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7)
    lms, labels, scores = [], [], []
    w = h = 0
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Timestamps start at 1 ms and follow the clip's own frame rate.
            result = recognizer.recognize_for_video(image, 1 + int(round(index * 1000.0 / fps)))
            lm = np.full((MAX_HANDS, 21, 2), np.nan, dtype=np.float32)
            label = np.full(MAX_HANDS, "", dtype="<U16")
            score = np.zeros(MAX_HANDS, dtype=np.float32)
            for k, hand in enumerate(result.hand_landmarks[:MAX_HANDS]):
                lm[k] = [(p.x * w, p.y * h) for p in hand]
                if result.gestures[k]:
                    label[k] = result.gestures[k][0].category_name
                    score[k] = result.gestures[k][0].score
            lms.append(lm)
            labels.append(label)
            scores.append(score)
            index += 1
    cap.release()
    if not lms:
        print("No frames read from %s" % args.video)
        return 1
    np.savez(args.out, lm=np.stack(lms), label=np.stack(labels), score=np.stack(scores), fps=fps, w=w, h=h)
    print("Saved %d frames to %s" % (len(lms), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
