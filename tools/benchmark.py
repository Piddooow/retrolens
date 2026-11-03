"""Measure SNAP mode frame rate per state, running the models on the inference thread like the app.

Replays a clip as a camera (paced at --camera-fps, or as fast as possible with 0), with the real hand
model and segmenter, snaps to vanish and again to reappear, and prints per state the achieved frame rate,
the slowest 2 second window, and the main thread time per stage. The window draw is not included
(it needs a screen). --serial runs the models on the main thread instead, the way the app did before.

Example:
    python tools/benchmark.py uji_lokal/orang.mp4 --bg uji_lokal/v2/room_empty.mp4
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main as app  # noqa: E402
from snap.profiler import Profiler  # noqa: E402
from snap.worker import InferenceWorker  # noqa: E402
from snap.config import STYLES  # noqa: E402
from snap_effect import CONFIG, PersonSegmenter  # noqa: E402

STATES = ("LIVE", "VANISH", "INVISIBLE", "REAPPEAR")
STAGES = ("hand", "segmenter", "matte", "background", "render", "hud")


def read(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame if frame.shape[1] == 1280 else cv2.resize(frame, (1280, 720)))
    cap.release()
    return frames


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SNAP mode frame rate per state.")
    parser.add_argument("input", help="clip with a person")
    parser.add_argument("--bg", help="clip of the empty room, played first so the room is learned")
    parser.add_argument("--seconds", type=float, default=16.0, help="how long to run (the clip loops)")
    parser.add_argument("--camera-fps", type=float, default=29.0, help="capture pace; 0 = as fast as possible")
    parser.add_argument("--serial", action="store_true", help="run the models on the main thread")
    parser.add_argument("--style", choices=STYLES, help="effect style (default: CONFIG style_default)")
    parser.add_argument("--live-only", action="store_true", help="never snap, measure the LIVE state only")
    args = parser.parse_args(argv)
    frames = read(args.input)
    if not frames:
        print("Cannot read %s" % args.input)
        return 1
    model, detect, holds, _ = app.buka_model_tangan(CONFIG)
    if model is None:
        return 1
    seg = PersonSegmenter(os.path.join(ROOT, "selfie_segmenter.tflite"), CONFIG)
    prof = Profiler(1)
    snap = app.ModeSnap(CONFIG, seg, prof, holds)
    snap.masuk(0.0)
    if args.style:
        snap.ctrl.pilih_gaya(args.style, 0.0)
    ts = 0
    if args.bg:
        for frame in read(args.bg):
            ts += 1
            snap.proses(ts / 30.0, frame, None, ts)
            prof.end_frame()
    else:
        snap.ctrl.bg.set_plate(frames[0])

    def run_models(tugas):
        return app.jalankan_model(tugas, detect, seg, CONFIG)

    worker = None if args.serial else InferenceWorker(run_models)
    times = {}  # type: Dict[str, List[float]]
    stages = {}  # type: Dict[str, List[Dict[str, float]]]
    last = 0
    start = time.monotonic() + 0.1
    snaps = [] if args.live_only else [start + 1.5, start + 8.5]
    k = 0
    try:
        while True:
            if args.camera_fps > 0:
                wait = start + k / args.camera_fps - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            captured = time.monotonic()
            if captured - start > args.seconds:
                break
            ts += 1
            job = app.TugasModel(frames[k % len(frames)], captured, ts, "SNAP", snap.butuh_tangan(), "snap")
            k += 1
            if worker is None:
                done = (k, job, run_models(job))
            else:
                worker.kirim(job)
                done = worker.ambil(last, 1.0)
                if done is None:
                    continue
            last, job, result = done
            for name, ms in result.ms.items():
                prof.add(name, ms)
            if snaps and job.now >= snaps[0]:
                snaps.pop(0)
                snap.tombol("s", job.now)
            out = snap.proses(job.now, job.frame, result.tangan, job.ts, result.model)
            with prof.stage("hud"):
                screen = out.copy()
                snap.gambar_hud(screen, job.now, 30.0)
            state = snap.ctrl.state
            times.setdefault(state, []).append(job.now)
            stages.setdefault(state, []).append(dict(prof._frame))
            prof.end_frame()
    finally:
        if worker is not None:
            worker.tutup()
        model.close()
        seg.close()
    print("%s, camera %s" % ("serial" if args.serial else "inference thread",
                             "%.0f fps" % args.camera_fps if args.camera_fps > 0 else "unpaced"))
    print("%-10s %6s %6s %9s  " % ("state", "frames", "fps", "worst 2s") + " ".join("%10s" % s for s in STAGES)
          + " %10s %10s" % ("render p95", "max gap ms"))
    for state in STATES:
        if len(times.get(state, [])) < 3:
            continue
        t = np.array(times[state])
        gaps = np.diff(t)
        # Only gaps inside the state; a jump means the state was left and entered again.
        gaps = gaps[gaps < 0.5]
        fps = 1.0 / float(np.mean(gaps))
        window = max(1, int(round(2.0 * fps)))
        worst = fps if len(gaps) < window else 1.0 / float(np.convolve(gaps, np.ones(window) / window, "valid").max())
        med = [float(np.median([s.get(n, 0.0) for s in stages[state]])) for n in STAGES]
        render_p95 = float(np.percentile([s.get("render", 0.0) for s in stages[state]], 95))
        print("%-10s %6d %6.1f %9.1f  " % (state, len(t), fps, worst) + " ".join("%10.1f" % v for v in med)
              + " %10.1f %10.1f" % (render_p95, 1000.0 * float(gaps.max())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
