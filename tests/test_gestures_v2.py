import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import snap_effect as se  # noqa: E402
from snap import hud  # noqa: E402
from snap.gestures import HoldGestureDetector  # noqa: E402

CFG = se.CONFIG
FPS = 30.0
UP = {"Thumb_Up": CFG["hold_thumb_up_s"]}
FIST = {"Closed_Fist": CFG["hold_fist_min_s"]}
UJI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uji_lokal")


def hand_at(x, y=400.0):
    """A still hand: 21 landmarks around (x, y) in pixels."""
    lm = np.zeros((21, 2), dtype=np.float32)
    lm[:, 0] = x + np.arange(21)
    lm[:, 1] = y - np.arange(21)
    return lm


def run(stream, holds=UP, continuous=None, det=None, start=0.0):
    """stream: per frame a list of (x, label, score, siap) hands. Returns (fired frames, held per frame)."""
    det = det or HoldGestureDetector(CFG)
    fired, held = [], []
    for i, hands in enumerate(stream):
        now = start + i / FPS
        frame = [(hand_at(x), label, score, siap) for x, label, score, siap in hands]
        for label, _ in det.update(now, frame, holds, continuous or {}, 1280):
            fired.append((i, label))
        held.append(set(det.held))
    return fired, held


def frames(label, seconds, x=400.0, score=0.9, siap=False):
    return [[(x, label, score, siap)] for _ in range(int(round(seconds * FPS)))]


def none(seconds):
    return [[] for _ in range(int(round(seconds * FPS)))]


class TestHoldGestures(unittest.TestCase):
    def test_fires_once_after_the_hold_time(self):
        fired, _ = run(frames("Thumb_Up", 3.0))
        self.assertEqual(fired, [(int(round(CFG["hold_thumb_up_s"] * FPS)), "Thumb_Up")])

    def test_low_score_does_not_count(self):
        fired, _ = run(frames("Thumb_Up", 2.0, score=CFG["hold_min_score"] - 0.05))
        self.assertEqual(fired, [])

    def test_thirty_percent_disagreeing_frames_do_not_fire(self):
        stream = []
        for i in range(int(3.0 * FPS)):
            stream.append([(400.0, "None" if i % 10 in (2, 5, 8) else "Thumb_Up", 0.9, False)])
        self.assertEqual(run(stream)[0], [])
        # One frame in ten disagreeing is fine.
        stream = [[(400.0, "None" if i % 10 == 5 else "Thumb_Up", 0.9, False)] for i in range(int(2.0 * FPS))]
        self.assertEqual(len(run(stream)[0]), 1)

    def test_gap_longer_than_limit_restarts_the_hold(self):
        gap_long = frames("Thumb_Up", 0.6) + none(0.2) + frames("Thumb_Up", 0.6)
        self.assertEqual(run(gap_long)[0], [])
        gap_short = frames("Thumb_Up", 0.6) + none(0.1) + frames("Thumb_Up", 0.6)
        self.assertEqual(len(run(gap_short)[0]), 1)

    def test_release_required_before_firing_again(self):
        stream = frames("Thumb_Up", 3.0) + none(0.2) + frames("Thumb_Up", 1.5)
        self.assertEqual(len(run(stream)[0]), 1)
        stream = frames("Thumb_Up", 1.2) + none(0.4) + frames("Thumb_Up", 1.2)
        fired, _ = run(stream)
        self.assertEqual(len(fired), 2)
        second = int(round(1.2 * FPS)) + int(round(0.4 * FPS)) + int(round(CFG["hold_thumb_up_s"] * FPS))
        self.assertEqual(fired[1][0], second)

    def test_global_cooldown_after_a_gesture(self):
        holds = {"Thumb_Up": 1.0, "Thumb_Down": 1.0}
        # Thumb_Down starts 0.5 s later on another hand and would finish at 1.5 s.
        stream = []
        for i in range(int(2.5 * FPS)):
            hands = [(300.0, "Thumb_Up", 0.9, False)]
            if i >= 0.5 * FPS:
                hands.append((1000.0, "Thumb_Down", 0.9, False))
            stream.append(hands)
        fired, _ = run(stream, holds)
        self.assertEqual([label for _, label in fired], ["Thumb_Up", "Thumb_Down"])
        self.assertEqual(fired[0][0], 30)
        self.assertEqual(fired[1][0], 30 + int(round(CFG["gesture_cooldown_s"] * FPS)))

    def test_cooldown_after_a_snap(self):
        det = HoldGestureDetector(CFG)
        det.cooldown(0.9)
        fired, _ = run(frames("Thumb_Up", 2.0), det=det)
        self.assertEqual(fired[0][0], int(round((0.9 + CFG["gesture_cooldown_s"]) * FPS)))

    def test_ignored_while_the_same_hand_is_ready_to_snap(self):
        self.assertEqual(run(frames("Thumb_Up", 2.0, siap=True))[0], [])
        # SIAP on another hand does not block this one.
        stream = [[(300.0, "Thumb_Up", 0.9, False), (1000.0, "None", 0.9, True)] for _ in range(int(1.5 * FPS))]
        self.assertEqual(len(run(stream)[0]), 1)

    def test_each_hand_holds_on_its_own(self):
        stream = frames("Thumb_Up", 0.6, x=300.0) + frames("Thumb_Up", 0.6, x=1000.0)
        self.assertEqual(run(stream)[0], [])

    def test_hand_order_swapping_keeps_tracks(self):
        stream = []
        for i in range(int(1.5 * FPS)):
            hands = [(300.0, "Thumb_Up", 0.9, False), (1000.0, "None", 0.9, False)]
            stream.append(hands if i % 2 else hands[::-1])
        self.assertEqual(len(run(stream)[0]), 1)

    def test_disabled_labels_do_nothing(self):
        self.assertEqual(run(frames("Victory", 3.0))[0], [])

    def test_fist_counts_only_after_its_minimum(self):
        snap_release = none(0.5) + frames("Closed_Fist", 2 / FPS) + none(0.5)
        _, held = run(snap_release, {}, FIST)
        self.assertFalse(any(held))
        _, held = run(frames("Closed_Fist", 0.5) + none(0.3), {}, FIST)
        first = next(i for i, h in enumerate(held) if "Closed_Fist" in h)
        self.assertEqual(first, int(round(CFG["hold_fist_min_s"] * FPS)))
        # Held through the short gap tolerance, then let go.
        self.assertIn("Closed_Fist", held[int(round(0.5 * FPS))])
        self.assertNotIn("Closed_Fist", held[-1])

    def test_progress_runs_from_zero_to_one(self):
        det = HoldGestureDetector(CFG)
        seen = []
        for i in range(int(1.2 * FPS)):
            det.update(i / FPS, [(hand_at(400.0), "Thumb_Up", 0.9, False)], UP, {}, 1280)
            seen.append([p for _, p in det.progress()])
        self.assertEqual(seen[0], [0.0])
        self.assertAlmostEqual(seen[15][0], 0.5, places=6)
        # After firing the label is released only when it disappears, so no new ring starts.
        self.assertEqual(seen[-1], [])


class TestRing(unittest.TestCase):
    def test_ring_fills_clockwise_from_the_top(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        hud.cincin(img, (100.0, 100.0), 0.25, 40, 3, 1.0)
        ys, xs = np.nonzero(img[..., 0] > 128)
        self.assertGreater(len(xs), 0)
        # A quarter ring runs from 12 to 3 o'clock: the top right quadrant only.
        self.assertTrue(np.all(xs >= 98) and np.all(ys <= 102))

    def test_pulse_grows_then_fades_out(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        self.assertTrue(hud.denyut(img, (100.0, 100.0), 0.05, CFG))
        self.assertFalse(hud.denyut(img, (100.0, 100.0), CFG["ring_pulse_s"] + CFG["hud_fade_s"], CFG))


class FakePoint:
    def __init__(self, x, y):
        self.x, self.y = x, y


class FakeCategory:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


class FakeResult:
    """Shaped like a GestureRecognizerResult for one hand at the center of the frame."""

    def __init__(self, label, score=0.9):
        lm = hand_at(640.0, 400.0)
        self.hand_landmarks = [[FakePoint(x / 1280.0, y / 720.0) for x, y in lm]]
        self.gestures = [[FakeCategory(label, score)]]


class TestModeSnapHolds(unittest.TestCase):
    def mode(self, hold_aktif):
        import main as app
        seg = se.PersonSegmenter(os.path.join(os.path.dirname(UJI), "selfie_segmenter.tflite"), CFG)
        self.addCleanup(seg.close)
        snap = app.ModeSnap(CFG, seg, app.Profiler(30), hold_aktif)
        snap.masuk(0.0)
        return snap

    def feed(self, snap, label, start, seconds, frame):
        tick = int(round(start * FPS))
        for i in range(int(round(seconds * FPS))):
            snap.proses((tick + i) / FPS, frame, FakeResult(label), tick + i + 1)
        return (tick + int(round(seconds * FPS))) / FPS

    def test_thumb_up_learns_the_room_and_thumb_down_cancels(self):
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        snap = self.mode(True)
        t = self.feed(snap, "Thumb_Up", 1.0, 0.5, frame)
        self.assertEqual(snap.ctrl.state, "LIVE")
        self.assertTrue(snap._cincin)
        t = self.feed(snap, "Thumb_Up", t, 0.6, frame)
        self.assertEqual(snap.ctrl.state, "PLATE_CAPTURE")
        # Release, then hold thumbs down to cancel the capture.
        t = self.feed(snap, "None", t, 0.7, frame)
        self.feed(snap, "Thumb_Down", t, 1.1, frame)
        self.assertEqual(snap.ctrl.state, "LIVE")
        self.assertEqual(snap.ctrl.pesan(t + 1.2), "Cancelled")

    def test_no_holds_without_the_gesture_model(self):
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        snap = self.mode(False)
        self.feed(snap, "Thumb_Up", 1.0, 2.0, frame)
        self.assertEqual(snap.ctrl.state, "LIVE")
        self.assertEqual(snap._cincin, [])


@unittest.skipUnless(os.path.isfile(os.path.join(UJI, "gestur_jentik.npz")),
                     "recorded gesture streams missing (tools/gesture_stream.py --mirror)")
class TestRecordedStreams(unittest.TestCase):
    def replay(self, name, holds, continuous):
        data = np.load(os.path.join(UJI, "gestur_%s.npz" % name))
        lm, label, score, fps = data["lm"], data["label"], data["score"], float(data["fps"])
        snap = se.SnapGestureDetector(CFG)
        hold = HoldGestureDetector(CFG)
        snaps, fired, held = [], [], set()
        for i in range(len(lm)):
            now = i / fps
            present = [k for k in range(lm.shape[1]) if not np.isnan(lm[i, k, 0, 0])]
            daftar = [lm[i, k] for k in present]
            snaps += [i for _ in snap.perbarui(now, daftar, int(data["w"]), int(data["h"]))]
            hands = []
            for k, arr in zip(present, daftar):
                siap = any(j["lm"] is arr and j["state"] == se.SnapGestureDetector.SIAP for j in snap.tangan)
                hands.append((arr, str(label[i, k]) or "None", float(score[i, k]), siap))
            fired += [(i, lab) for lab, _ in hold.update(now, hands, holds, continuous, int(data["w"]))]
            held |= hold.held
        return snaps, fired, held

    def test_snaps_found_with_gesture_model_landmarks(self):
        snaps, _, held = self.replay("jentik", UP, FIST)
        self.assertEqual(len(snaps), 8)
        # The snap release reads as a fist for a frame or two; it must never count as held.
        self.assertNotIn("Closed_Fist", held)

    def test_no_snap_or_hold_on_ordinary_hand_movement(self):
        snaps, fired, held = self.replay("biasa", {"Thumb_Up": 1.0, "Thumb_Down": 1.0}, FIST)
        self.assertEqual(snaps, [])
        self.assertEqual(fired, [])
        self.assertEqual(held, set())


@unittest.skipUnless(os.path.isfile(os.path.join(UJI, "lm_jentik.npz")), "recorded landmark streams missing")
class TestRecordedLandmarks(unittest.TestCase):
    def count(self, name):
        data = np.load(os.path.join(UJI, "lm_%s.npz" % name))
        lm, fps = data["lm"], float(data["fps"])
        det = se.SnapGestureDetector(CFG)
        total = 0
        for i in range(len(lm)):
            daftar = [lm[i, k] for k in range(lm.shape[1]) if not np.isnan(lm[i, k, 0, 0])]
            total += len(det.perbarui(i / fps, daftar, int(data["w"]), int(data["h"])))
        return total

    def test_snap_detection_on_recorded_landmarks(self):
        self.assertEqual(self.count("jentik"), 8)
        self.assertEqual(self.count("biasa"), 0)


if __name__ == "__main__":
    unittest.main()
