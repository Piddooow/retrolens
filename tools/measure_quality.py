"""Compute the SNAP acceptance metrics (spec section 6) from a tools/render_offline.py output folder.

Example:
    python tools/measure_quality.py uji_lokal/render/sit_snap

Prints PASS, FAIL, or N/A per metric. Exit code is 0 only when no metric fails.
N/A means the render does not contain the data the metric needs (for example no INVISIBLE frames).
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snap.glitch_score import REAPPEAR_SCORE, VANISH_SCORE  # noqa: E402

# Old state names map to the v2 names so baseline renders can be measured too.
INVISIBLE_STATES = ("INVISIBLE", "KOSONG")
EFFECT_STATES = ("VANISH", "SNAP", "INVISIBLE", "KOSONG", "REAPPEAR", "KEMBALI")
SEAM_BAND_PX = 20
NOISE_FRAMES = 60
FLICKER_PERCENTILE = 99.9
LIMITS = {"residue": 0.0, "seam": 2.0, "noise": 0.15, "flicker_ratio": 1.0, "continuity": 0.04,
          "step": 12.0, "fps_avg": 27.0, "fps_min": 24.0, "fps_min_window_s": 2.0}


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


def load(folder: str, sub: str, index: int, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    path = os.path.join(folder, sub, "%05d.png" % index)
    return cv2.imread(path, flags) if os.path.isfile(path) else None


def expected_boundaries(cfg: Dict) -> Dict[Tuple[str, str], float]:
    if cfg.get("style_default") == "GLITCH":
        # The glitch vanish and reappear last as many glitch steps as their scores have.
        hz = float(cfg["glitch_step_hz"])
        return {("VANISH", "INVISIBLE"): len(VANISH_SCORE) / hz, ("REAPPEAR", "LIVE"): len(REAPPEAR_SCORE) / hz}
    end = cfg["debu_mulai_detik"] + cfg["debu_sapuan_detik"] + cfg["debu_umur_detik"]
    table = {("SNAP", "KOSONG"): end, ("VANISH", "INVISIBLE"): end}
    # Renders made before the v2 state machine had a fixed empty and return time.
    if "kosong_detik" in cfg:
        table[("KOSONG", "KEMBALI")] = end + cfg["kosong_detik"]
        table[("KEMBALI", "LIVE")] = end + cfg["kosong_detik"] + cfg["kembali_detik"]
    if "reappear_sweep_s" in cfg:
        table[("REAPPEAR", "LIVE")] = (cfg["reappear_sweep_s"] + cfg["reappear_life_s"]
                                       + cfg["reappear_blend_s"])
    return table


class Report:
    def __init__(self):
        self.rows = []  # type: List[Tuple[str, str, str]]

    def add(self, name: str, ok: Optional[bool], detail: str) -> None:
        self.rows.append((name, "N/A" if ok is None else ("PASS" if ok else "FAIL"), detail))

    def show(self) -> int:
        for name, verdict, detail in self.rows:
            print("%-12s %-4s  %s" % (name, verdict, detail))
        return 1 if any(v == "FAIL" for _, v, _ in self.rows) else 0


def measure(folder: str) -> int:
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    cfg, frames, fps = meta["config"], meta["frames"], meta["fps"]
    report = Report()
    band_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SEAM_BAND_PX + 1, 2 * SEAM_BAND_PX + 1))

    residue, seams = [], []
    inside_diff, outside_diff, flicker = [], [], []
    prev_out = prev_in = None
    prev_state = None
    for fr, live in zip(frames, read_frames(meta["input"])):
        i, state = fr["index"], fr["state"]
        out = zone = None
        if state in EFFECT_STATES and fr["zone"]:
            out = load(folder, "frames", i)
            zone = load(folder, "zone", i, cv2.IMREAD_GRAYSCALE)
        if out is None or zone is None:
            prev_out = prev_in = None
            prev_state = state
            continue
        inside = zone > 0
        # The seam is where camera pixels meet plate pixels, so it is measured once only the base
        # layer is drawn; during the dissolve the dust drifts across the band by design.
        if state in INVISIBLE_STATES:
            band = (cv2.dilate(inside.astype(np.uint8), band_kernel) > 0) & ~inside
            if band.any():
                seams.append(float(np.abs(out[band].astype(np.int16) - live[band]).mean()))
        if state in INVISIBLE_STATES and fr["plate"]:
            plate = load(folder, "plate", i)
            if plate is not None:
                residue.append(int(np.abs(out[inside].astype(np.int16) - plate[inside]).max()))
            if prev_out is not None and prev_state == state and len(inside_diff) < NOISE_FRAMES:
                d_out = out.astype(np.int16) - prev_out
                d_in = live.astype(np.int16) - prev_in
                inside_diff.append(d_out[inside])
                outside_diff.append(d_out[~inside])
                # The zone may change no more than the live camera changes outside it on the same
                # frame pair. The 99.9th percentile keeps one hot pixel from deciding, and keeps the
                # much larger outside area from winning just by having more pixels.
                flicker.append((float(np.percentile(np.abs(d_out[inside]), FLICKER_PERCENTILE)),
                                float(np.percentile(np.abs(d_in[~inside]), FLICKER_PERCENTILE))))
        prev_out, prev_in, prev_state = out, live, state

    if residue:
        report.add("residue", max(residue) <= LIMITS["residue"],
                   "max %d over %d INVISIBLE frames (limit 0, noise off)" % (max(residue), len(residue)))
    else:
        report.add("residue", None, "no INVISIBLE frames with a background reference")
    if seams:
        report.add("seam", max(seams) <= LIMITS["seam"],
                   "worst frame %.2f, mean %.2f levels in the %d px band (limit %.1f)" % (
                       max(seams), float(np.mean(seams)), SEAM_BAND_PX, LIMITS["seam"]))
    else:
        report.add("seam", None, "no INVISIBLE frames")
    if inside_diff:
        s_in = float(np.concatenate(inside_diff).std())
        s_out = float(np.concatenate(outside_diff).std())
        if s_out == 0.0:
            report.add("noise", False, "output outside the zone does not change at all (frozen frame)")
        else:
            ratio = s_in / s_out
            report.add("noise", abs(ratio - 1.0) <= LIMITS["noise"],
                       "temporal std inside %.2f vs outside %.2f (ratio %.2f, allowed 0.85 to 1.15)" % (
                           s_in, s_out, ratio))
        worst = max(flicker, key=lambda p: p[0] - LIMITS["flicker_ratio"] * p[1])
        ok = all(inside_change <= LIMITS["flicker_ratio"] * cam_change for inside_change, cam_change in flicker)
        report.add("flicker", ok, "worst pair: inside change %.1f levels vs live camera %.1f levels "
                   "(99.9th percentile, limit: not above the camera)" % worst)
    else:
        report.add("noise", None, "fewer than 2 consecutive INVISIBLE frames")
        report.add("flicker", None, "fewer than 2 consecutive INVISIBLE frames")

    masses = [fr["mass"] for fr in frames if fr["state"] in EFFECT_STATES and fr["mass"] is not None]
    if len(masses) > 1:
        jump = float(np.max(np.abs(np.diff(masses))))
        report.add("continuity", jump <= LIMITS["continuity"],
                   "largest visible mass change %.2f%% per frame (limit 4%%)" % (100 * jump))
    else:
        report.add("continuity", None, "no mass data")
    steps = [fr["step_px"] for fr in frames if fr["step_px"] is not None]
    if steps:
        report.add("step", max(steps) <= LIMITS["step"],
                   "largest particle move %.1f px per frame (limit 12)" % max(steps))
    else:
        report.add("step", None, "no particle data")

    table = expected_boundaries(cfg)
    starts = [tr["clock"] for tr in meta["triggers"] if tr["started"]]
    errors = []
    for a, b in zip(frames, frames[1:]):
        key = (a["state"], b["state"])
        if key in table and starts:
            start = max(s for s in starts if s <= b["clock"])
            errors.append((abs((b["clock"] - start) - table[key]), "%s->%s" % key))
    if errors:
        worst = max(errors)
        report.add("timing", worst[0] <= 1.0 / fps + 1e-6,
                   "worst boundary %s off by %.1f ms (limit one frame, %.1f ms)" % (
                       worst[1], 1000 * worst[0], 1000.0 / fps))
    else:
        report.add("timing", None, "no state boundaries")

    ms = np.array([fr["ms"] for fr in frames], dtype=np.float64)
    if len(ms):
        window = max(1, int(round(LIMITS["fps_min_window_s"] * fps)))
        rolling = np.convolve(ms, np.ones(window) / window, mode="valid") if len(ms) >= window else ms
        avg_fps, worst_fps = 1000.0 / ms.mean(), 1000.0 / rolling.max()
        report.add("speed", avg_fps >= LIMITS["fps_avg"] and worst_fps >= LIMITS["fps_min"],
                   "processing only, no camera: average %.0f fps, worst 2 s window %.0f fps" % (
                       avg_fps, worst_fps))
    print("Quality report for %s" % folder)
    return report.show()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure SNAP quality metrics from an offline render.")
    parser.add_argument("folder", help="output folder of tools/render_offline.py")
    return measure(parser.parse_args(argv).folder)


if __name__ == "__main__":
    sys.exit(main())
