"""Render the SNAP pipeline from a video file with a fixed clock, without a camera.

Example:
    python tools/render_offline.py uji_lokal/v2/sit_snap.mp4 --bg uji_lokal/v2/room_empty.mp4 \
        --snap-at 2.0 --out uji_lokal/render/sit_snap

Writes OUT/render.mp4, OUT/frames/NNNNN.png, OUT/meta.json, and for effect frames the
removal zone (OUT/zone) and expected background (OUT/plate) used by tools/measure_quality.py.
Input frames are used as they are (no mirroring), like `main.py --sumber`.
"""

import argparse
import json
import os
import sys
import time
from typing import Iterator, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import main as app  # noqa: E402
from snap.profiler import Profiler  # noqa: E402
from snap_effect import CONFIG, PersonSegmenter, SnapController, validasi_config  # noqa: E402

PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 3]


def read_frames(path: str) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("Cannot open video: %s" % path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline SNAP render at a fixed clock.")
    parser.add_argument("input", help="video with the person")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--bg", help="video of the empty room, fed first so the background is learned")
    parser.add_argument("--snap-at", type=float, action="append", default=[],
                        help="snap time in seconds from the start of INPUT (repeatable)")
    parser.add_argument("--origin", help="flash origin as x,y in 0..1 (default: CONFIG fallback)")
    parser.add_argument("--seed", type=int, help="override debu_seed")
    parser.add_argument("--fps", type=float, default=30.0, help="fixed clock rate (default 30)")
    parser.add_argument("--no-png", action="store_true", help="skip the PNG sequence")
    parser.add_argument("--allow-inpaint", action="store_true",
                        help="estimate unseen background (experimental) instead of refusing the snap")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = dict(CONFIG)
    if args.seed is not None:
        cfg["debu_seed"] = args.seed
    if args.allow_inpaint:
        cfg["allow_inpaint_fallback"] = True
    errors = validasi_config(cfg)
    if errors:
        print("Invalid CONFIG:\n  - " + "\n  - ".join(errors))
        return 1
    origin = None
    if args.origin:
        origin = tuple(float(v) for v in args.origin.split(","))
        if len(origin) != 2 or not all(0.0 <= v <= 1.0 for v in origin):
            print("--origin must be x,y with values from 0 to 1")
            return 1
    model = os.path.join(ROOT, "selfie_segmenter.tflite")
    if not os.path.isfile(model):
        print("Model file not found: %s" % model)
        return 1

    for sub in ("frames", "zone", "plate"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    dt = 1.0 / args.fps
    segmenter = PersonSegmenter(model, cfg)
    snap = app.ModeSnap(cfg, segmenter, Profiler(cfg["profiler_window"]))
    writer = None
    meta = {"input": os.path.abspath(args.input), "bg": os.path.abspath(args.bg) if args.bg else None,
            "fps": args.fps, "seed": cfg["debu_seed"], "config": cfg, "bg_frames": 0,
            "triggers": [], "frames": []}
    pending = sorted(args.snap_at)
    tick = 0
    try:
        snap.masuk(0.0)
        if args.bg:
            for frame in read_frames(args.bg):
                snap.proses(tick * dt, frame, None, tick + 1)
                tick += 1
            meta["bg_frames"] = tick

        for index, frame in enumerate(read_frames(args.input)):
            now = tick * dt
            t_input = index * dt
            start = time.perf_counter()
            out = snap.proses(now, frame, None, tick + 1)
            ms = (time.perf_counter() - start) * 1000.0
            tick += 1
            h, w = out.shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(args.out, "render.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                if not writer.isOpened():
                    print("Cannot create render.mp4")
                    return 1
            writer.write(out)
            name = "%05d.png" % index
            if not args.no_png:
                cv2.imwrite(os.path.join(args.out, "frames", name), out, PNG_PARAMS)

            ctrl = snap.ctrl
            info = ctrl.info_kualitas(w, h) if ctrl.state != SnapController.LIVE else {}
            if info.get("zone") is not None:
                cv2.imwrite(os.path.join(args.out, "zone", name), info["zone"], PNG_PARAMS)
            if info.get("plate_view") is not None and ctrl.waktu_efek(now) >= ctrl.debu_akhir:
                cv2.imwrite(os.path.join(args.out, "plate", name), info["plate_view"], PNG_PARAMS)
            meta["frames"].append({
                "index": index, "t": round(t_input, 6), "clock": round(now, 6), "state": ctrl.state,
                "effect_t": round(ctrl.waktu_efek(now), 6), "mass": info.get("mass"),
                "step_px": info.get("step_px"), "ms": round(ms, 3),
                "zone": info.get("zone") is not None,
                "plate": info.get("plate_view") is not None and ctrl.waktu_efek(now) >= ctrl.debu_akhir,
            })

            # The key is handled after the frame is shown, exactly like the live loop.
            while pending and t_input + 1e-9 >= pending[0]:
                at = pending.pop(0)
                started = ctrl.picu(now, origin) if origin is not None else None
                if started is None:
                    snap.tombol("s", now)
                    started = ctrl.state == SnapController.SNAP
                meta["triggers"].append({"requested_t": at, "t": round(t_input, 6), "clock": round(now, 6),
                                         "started": bool(started), "message": ctrl.pesan(now)})
                print("snap at %.3f s: %s %s" % (t_input, "started" if started else "refused", ctrl.pesan(now)))
    finally:
        if writer is not None:
            writer.release()
        segmenter.close()

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    times = [fr["ms"] for fr in meta["frames"]]
    print("Rendered %d frames to %s (process median %.1f ms)" % (
        len(times), args.out, float(np.median(times)) if times else 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
