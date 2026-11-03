import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snap.config import hex_ke_bgr  # noqa: E402
from snap.glitch import GlitchRenderer, Tempelan  # noqa: E402
from snap.glitch_score import GHOST_ALPHA, PALETTE, SCORE, validate_score  # noqa: E402
from test_background import H, W, room  # noqa: E402

# Frames of reference-effect-spider-verse-1.mp4 without any glitch, checked by eye frame by frame.
CALM_FRAMES = {0, 1, 2, 3, 4, 5, 32, 41, 48, 49, 91, 92, 93}
BODY_GRAY = 120


def person():
    """A flat gray person (head and torso) on a textured room, with its soft matte."""
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (640, 260), 90, 255, -1)
    cv2.rectangle(mask, (500, 340), (780, 719), 255, -1)
    alpha = cv2.GaussianBlur(mask, (0, 0), 2.0).astype(np.float32) / 255.0
    scene = room(5)
    frame = scene.copy()
    frame[mask > 127] = BODY_GRAY
    return frame, alpha, scene, mask > 127


def bgr(name):
    return hex_ke_bgr(PALETTE[name])


class ScoreCase(unittest.TestCase):
    def test_score_matches_the_reference(self):
        self.assertEqual(len(SCORE), 94)
        self.assertEqual(validate_score(SCORE), [])
        self.assertEqual({i for i, step in enumerate(SCORE) if not step}, CALM_FRAMES)

    def test_validator_catches_mistakes(self):
        bad = [{"cut": 4}, {"fill": ("NOPE",), "cover": 0.5}, {"fill": ("RED",)}, {"bg": "SKY"}, {"zoom": 1}]
        self.assertEqual(len(validate_score(bad)), 5)

    def test_see_through_copies_give_the_reference_overlaps(self):
        yellow = bgr("YELLOW")
        # Measured over the reference's yellow figure (#44, #40); RGB hex.
        for name, measured in (("MAGENTA", "#FC7C29"), ("CYAN", "#A6D66A"), ("GREEN_GHOST", "#8EEA1E")):
            mixed = (1.0 - GHOST_ALPHA) * yellow + GHOST_ALPHA * bgr(name)
            self.assertLessEqual(float(np.abs(mixed - hex_ke_bgr(measured)).max()), 25.0, name)


class RendererCase(unittest.TestCase):
    def setUp(self):
        self.frame, self.alpha, self.scene, self.body = person()
        self.r = GlitchRenderer()

    def test_calm_step_and_empty_frame_return_the_frame(self):
        self.assertIs(self.r.render(self.frame, self.alpha, self.scene, {}, 1), self.frame)
        empty = np.zeros_like(self.alpha)
        self.assertIs(self.r.render(self.frame, empty, self.scene, {"cut": 3}, 1), self.frame)

    def test_same_seed_same_picture(self):
        for i, step in enumerate(SCORE):
            if not step:
                continue
            a = self.r.render(self.frame, self.alpha, self.scene, step, i)
            b = GlitchRenderer().render(self.frame, self.alpha, self.scene, step, i)
            np.testing.assert_array_equal(a, b, err_msg="step %d" % i)

    def test_a_held_step_keeps_its_geometry(self):
        step = SCORE[19]
        self.r.render(self.frame, self.alpha, self.scene, step, 7)
        plan = self.r._rencana
        moved = np.roll(self.alpha, 6, axis=1)
        self.r.render(self.frame, moved, self.scene, step, 7)
        self.assertIs(self.r._rencana, plan)
        self.r.render(self.frame, self.alpha, self.scene, step, 8)
        self.assertIsNot(self.r._rencana, plan)

    def test_nothing_changes_outside_the_effect_region(self):
        for i, step in enumerate(SCORE):
            if not step or "bg" in step:
                continue
            out = self.r.render(self.frame, self.alpha, self.scene, step, i)
            x0, y0, x1, y1 = self.r._rencana.e
            outside = np.ones((H, W), dtype=bool)
            outside[y0:y1, x0:x1] = False
            np.testing.assert_array_equal(out[outside], self.scene[outside], err_msg="step %d" % i)

    def test_full_frame_background_replaces_the_room(self):
        out = self.r.render(self.frame, self.alpha, self.scene, {"bg": "REGIONS"}, 3)
        layer = self.r.latar("REGIONS", 3, W, H)
        far = cv2.dilate(self.body.view(np.uint8), np.ones((9, 9), np.uint8)) == 0
        np.testing.assert_array_equal(out[far], layer[far])

    def test_flat_fill_uses_the_exact_palette_color(self):
        out = self.r.render(self.frame, self.alpha, self.scene, {"fill": ("RED",), "cover": 1.0}, 4)
        inner = cv2.erode(self.body.view(np.uint8), np.ones((9, 9), np.uint8)) > 0
        np.testing.assert_array_equal(out[inner], np.broadcast_to(bgr("RED").astype(np.uint8), out[inner].shape))

    def test_see_through_copy_over_the_yellow_body(self):
        out = self.r.render(self.frame, self.alpha, self.scene, {"copies": (("MAGENTA", 0),)}, 5)
        inner = cv2.erode(self.body.view(np.uint8), np.ones((9, 9), np.uint8)) > 0
        expected = (1.0 - GHOST_ALPHA) * bgr("YELLOW") + GHOST_ALPHA * bgr("MAGENTA")
        self.assertLessEqual(float(np.abs(out[inner].astype(np.float32) - expected).max()), 2.0)

    def test_cuts_slide_pieces_without_losing_the_body(self):
        out = self.r.render(self.frame, self.alpha, self.scene, {"cut": 3}, 6)
        self.assertGreater(int((out != self.frame).any(axis=2).sum()), 1000)
        still_body = (np.abs(out.astype(np.int16) - BODY_GRAY) <= 2).all(axis=2)
        self.assertGreater(still_body.sum(), 0.85 * self.body.sum())

    def test_without_the_room_the_body_stays_under(self):
        out = self.r.render(self.frame, self.alpha, None, {"cut": 3}, 6)
        inner = cv2.erode(self.body.view(np.uint8), np.ones((9, 9), np.uint8)) > 0
        # Gaps show the original frame (the body) instead of a room nobody has seen; only the soft edge of a
        # slid piece lands inside the body.
        off = (np.abs(out[inner].astype(np.int16) - BODY_GRAY) > 2).any(axis=1)
        self.assertLess(off.mean(), 0.01)

    def test_tempelan_composites_like_over(self):
        base = self.scene.copy()
        layer = np.zeros((H, W, 4), dtype=np.uint8)
        layer[100:200, 100:200] = (40, 80, 120, 255)
        layer[300:400, 300:400] = (20, 40, 60, 128)
        from snap.glitch import tempel
        tempel(base, (0, 0, W, H), Tempelan(layer))
        np.testing.assert_array_equal(base[150, 150], (40, 80, 120))
        half = self.scene[350, 350].astype(np.float32) * (127.0 / 255.0) + (20, 40, 60)
        self.assertLessEqual(float(np.abs(base[350, 350] - half).max()), 1.0)
        np.testing.assert_array_equal(base[600, 1000], self.scene[600, 1000])


if __name__ == "__main__":
    unittest.main()
