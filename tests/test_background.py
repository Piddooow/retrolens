import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import snap_effect as se  # noqa: E402
from snap.background import (CAPTURE_MISMATCH, CAPTURE_OK, CAPTURE_PERSON, MSG_LEARNED, MSG_MOVED,  # noqa: E402
                             BackgroundEngine, dilate_to)

CFG = se.CONFIG
W, H = 1280, 720
LW, LH = 640, 360
DT = 1.0 / 30


def room(seed=0):
    """Textured synthetic room with values 40..180, so gains up to 1.4 never clip."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(40, 180, (45, 80, 3)).astype(np.uint8)
    img = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.int16)
    img += rng.integers(-10, 11, (H, W, 3)).astype(np.int16)
    return np.clip(img, 40, 180).astype(np.uint8)


def person_lo():
    yy, xx = np.mgrid[0:LH, 0:LW]
    head = (xx - 320) ** 2 + (yy - 110) ** 2 < 45 ** 2
    body = (np.abs(xx - 320) < 90) & (yy > 150)
    return (head | body).astype(np.float32)


def with_person(img, mask_lo):
    out = img.copy()
    big = cv2.resize(mask_lo, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
    out[big] = (70, 60, 50)
    return out


def noisy(img, rng, sigma):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).round().astype(np.uint8)


def learn_empty(eng, img, seconds=2.5, rng=None, sigma=0.0):
    empty = np.zeros((LH, LW), dtype=np.float32)
    messages = []
    for i in range(int(seconds / DT)):
        frame = noisy(img, rng, sigma) if rng is not None else img
        messages += eng.update(i * DT, frame, empty)
    return messages


class TestBackgroundEngine(unittest.TestCase):
    def test_flicker_never_writes_person_pixels(self):
        eng = BackgroundEngine(CFG)
        scene = room(1)
        rng = np.random.default_rng(5)
        learn_empty(eng, scene, rng=rng, sigma=2.0)
        before = eng.plate.copy()
        mask = person_lo()
        empty = np.zeros_like(mask)
        frame = with_person(scene, mask)
        for i in range(300):
            # The segmenter loses the person for one frame in every ten.
            eng.update(10.0 + i * DT, noisy(frame, rng, 2.0), empty if i % 10 == 9 else mask)
        area = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR) > 0
        self.assertEqual(int(np.abs(eng.plate[area].astype(np.int16) - before[area]).max()), 0)

    def test_local_exposure_match_at_the_seam(self):
        eng = BackgroundEngine(CFG)
        scene = room(2)
        eng.set_plate(scene)
        mask = person_lo()
        gradient = np.linspace(0.9, 1.1, H, dtype=np.float32)[:, None, None]
        live_room = np.clip(scene.astype(np.float32) * 1.2 * gradient, 0, 255).round().astype(np.uint8)
        live = with_person(live_room, mask)
        safe = ~dilate_to(mask >= CFG["bg_safe_conf"], CFG["bg_safe_margin_px"], (W, H))
        adj = eng.plate_adj(live, eng.known() & safe)
        zone = dilate_to(mask > 0.5, CFG["zone_dilate_px"], (W, H))
        ring = cv2.dilate(zone.view(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0
        band = ring & ~zone
        seam = float(np.abs(live[band].astype(np.float32) - adj[band]).mean())
        self.assertLessEqual(seam, 2.0, "seam %.2f" % seam)

    def run_frames(self, eng, frame, mask, start, seconds):
        messages = []
        for i in range(int(seconds / DT)):
            messages += eng.update(start + i * DT, frame, mask)
        return messages

    def test_camera_shift_invalidates_but_brightness_does_not(self):
        scene = room(3)
        mask = person_lo()
        eng = BackgroundEngine(CFG)
        eng.set_plate(scene)
        shifted = cv2.warpAffine(scene, np.float32([[1, 0, 3], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        frame = with_person(shifted, mask)
        for i in range(int(2.0 / DT)):
            if MSG_MOVED in eng.update(i * DT, frame, mask):
                break
        else:
            self.fail("camera move was not detected")
        self.assertEqual(eng.known_fraction(), 0.0)
        # Afterwards it relearns the new view, but never the old room behind the person.
        self.run_frames(eng, frame, mask, (i + 1) * DT, 1.0)
        behind = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR) > 0
        self.assertFalse(eng.known()[behind].any())

        eng = BackgroundEngine(CFG)
        eng.set_plate(scene)
        brighter = cv2.convertScaleAbs(scene, alpha=1.25)
        messages = self.run_frames(eng, with_person(brighter, mask), mask, 0.0, 2.0)
        self.assertNotIn(MSG_MOVED, messages)
        self.assertGreater(eng.known_fraction(), 0.99)

    def test_noise_matches_live_sigma(self):
        # 2.0 catches 8 bit rounding of the estimate, which snaps it to 1.48 or 2.97.
        for sigma in (2.0, 3.0):
            with self.subTest(sigma=sigma):
                eng = BackgroundEngine(CFG)
                scene = room(4)
                eng.set_plate(scene)
                mask = person_lo()
                rng = np.random.default_rng(9)
                frame = with_person(scene, mask)
                for i in range(10):
                    live = noisy(frame, rng, sigma)
                    eng.update(i * DT, live, mask)
                adj = eng.plate_adj(live)
                zone = dilate_to(mask > 0.5, CFG["zone_dilate_px"], (W, H))
                view = eng.plate_view(adj, zone)
                added = view[zone].astype(np.float32) - adj[zone]
                area = eng.safe
                live_sigma = float((live[area].astype(np.float32) - scene[area]).std())
                self.assertAlmostEqual(float(added.std()), live_sigma, delta=0.15 * live_sigma)
                self.assertTrue(np.array_equal(view[~zone], adj[~zone]))

    def test_pause_restarts_the_frame_history(self):
        eng = BackgroundEngine(CFG)
        scene = room(4)
        eng.set_plate(scene)
        rng = np.random.default_rng(3)
        empty = np.zeros((LH, LW), dtype=np.float32)
        t = 0.0
        for _ in range(40):
            t += DT
            eng.update(t, noisy(scene, rng, 2.0), empty)
        before = eng.sigma.copy()
        self.assertTrue(eng.safe.all())
        # After a pause the scene looks different (the person moved, light changed); that pair is not noise.
        t += 3.0
        other = cv2.convertScaleAbs(scene, alpha=1.3)
        eng.update(t, noisy(other, rng, 2.0), empty)
        self.assertTrue(np.array_equal(eng.sigma, before))
        self.assertFalse(eng.safe.any())
        for _ in range(CFG["bg_safe_frames"]):
            t += DT
            eng.update(t, noisy(other, rng, 2.0), empty)
        self.assertTrue(eng.safe.all())
        self.assertAlmostEqual(float(eng.sigma.mean()), 2.0, delta=0.3)

    def test_noise_off_returns_plate_exactly(self):
        eng = BackgroundEngine(dict(CFG, bg_noise_on=False))
        scene = room(4)
        eng.set_plate(scene)
        eng.sigma[:] = 3.0
        zone = dilate_to(person_lo() > 0.5, CFG["zone_dilate_px"], (W, H))
        self.assertTrue(np.array_equal(eng.plate_view(scene, zone), scene))

    def test_noise_is_reproducible(self):
        views = []
        for _ in range(2):
            eng = BackgroundEngine(CFG)
            eng.set_plate(room(4))
            eng.sigma[:] = 2.5
            zone = dilate_to(person_lo() > 0.5, CFG["zone_dilate_px"], (W, H))
            views.append(eng.plate_view(eng.plate, zone))
        self.assertTrue(np.array_equal(views[0], views[1]))

    def test_auto_capture_when_room_is_empty(self):
        eng = BackgroundEngine(CFG)
        messages = learn_empty(eng, room(5), seconds=2.0)
        self.assertIn(MSG_LEARNED, messages)
        self.assertEqual(eng.known_fraction(), 1.0)

    def test_no_auto_capture_with_person(self):
        eng = BackgroundEngine(CFG)
        mask = person_lo()
        messages = self.run_frames(eng, with_person(room(5), mask), mask, 0.0, 2.0)
        self.assertNotIn(MSG_LEARNED, messages)
        area = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR) > 0
        self.assertFalse(eng.known()[area].any())

    def manual(self, eng, frame, mask):
        eng.start_capture()
        result = None
        for _ in range(CFG["plate_jumlah_frame"]):
            result = eng.capture_frame(frame, mask)
        return result

    def test_manual_capture_rejects_person(self):
        eng = BackgroundEngine(CFG)
        mask = person_lo()
        self.assertEqual(self.manual(eng, with_person(room(6), mask), mask), CAPTURE_PERSON)
        self.assertEqual(eng.known_fraction(), 0.0)

    def test_manual_capture_accepts_empty_room(self):
        eng = BackgroundEngine(CFG)
        scene = room(6)
        self.assertEqual(self.manual(eng, scene, np.zeros((LH, LW), np.float32)), CAPTURE_OK)
        self.assertEqual(eng.known_fraction(), 1.0)
        self.assertTrue(np.array_equal(eng.plate, scene))

    def test_manual_capture_rejects_covered_lens(self):
        eng = BackgroundEngine(CFG)
        scene = room(7)
        eng.set_plate(scene)
        covered = np.full((H, W, 3), (40, 60, 150), dtype=np.uint8)
        self.assertEqual(self.manual(eng, covered, np.zeros((LH, LW), np.float32)), CAPTURE_MISMATCH)
        self.assertTrue(np.array_equal(eng.plate, scene))

    def test_fallback_fills_only_unknown_pixels(self):
        eng = BackgroundEngine(CFG)
        scene = room(8)
        eng.set_plate(scene)
        zone = dilate_to(person_lo() > 0.5, CFG["zone_dilate_px"], (W, H))
        hole = np.zeros((H, W), dtype=bool)
        hole[300:340, 600:680] = True
        eng.count[hole] = 0
        filled = eng.fill_unknown(scene, zone)
        self.assertTrue(np.array_equal(filled[~hole], scene[~hole]))
        # Random texture inside the hole cannot be recovered, so the estimate only has to beat
        # a flat patch in the mean color of the surrounding room.
        flat_err = float(np.abs(scene[hole] - scene[~hole & zone].mean(axis=0)).mean())
        self.assertLess(float(np.abs(filled[hole].astype(np.float32) - scene[hole]).mean()), flat_err)


class TestCoverageGate(unittest.TestCase):
    def controller(self, coverage):
        ctrl = se.SnapController(CFG)
        scene = room(10)
        mask = person_lo()
        ctrl.bg.set_plate(scene)
        zone = dilate_to(mask > CFG["mask_ambang"], CFG["zone_dilate_px"], (W, H))
        ys, xs = np.nonzero(zone)
        unknown = int(round(len(ys) * (1.0 - coverage)))
        ctrl.bg.count[ys[:unknown], xs[:unknown]] = 0
        frame = with_person(scene, mask)
        ctrl.amati(0.0, frame, cv2.resize(frame, (LW, LH), interpolation=cv2.INTER_AREA), mask)
        return ctrl

    def test_refuses_at_99_percent(self):
        ctrl = self.controller(0.99)
        self.assertFalse(ctrl.picu(0.1))
        self.assertEqual(ctrl.pesan(0.2), se.MSG_NEED_ROOM)
        tint = ctrl.tint(0.2)
        self.assertIsNotNone(tint)
        self.assertAlmostEqual(tint[1], CFG["coverage_tint_alpha"])
        self.assertTrue(tint[0].any())
        self.assertFalse(ctrl.bg_siap)
        end = 0.1 + CFG["pesan_detik"] + CFG["coverage_tint_fade_s"] + 0.01
        self.assertIsNone(ctrl.tint(end))

    def test_proceeds_at_100_percent(self):
        ctrl = self.controller(1.0)
        self.assertTrue(ctrl.bg_siap)
        self.assertTrue(ctrl.picu(0.1))
        self.assertEqual(ctrl.state, ctrl.VANISH)


if __name__ == "__main__":
    unittest.main()
