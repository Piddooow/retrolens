import itertools
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snap.config import hex_ke_bgr  # noqa: E402
from snap.glitch import GlitchRenderer, Tempelan  # noqa: E402
from snap.glitch_score import GHOST_ALPHA, PALETTE, REAPPEAR_SCORE, SCORE, VANISH_SCORE, validate_score  # noqa: E402
from test_background import H, W, room  # noqa: E402
from test_controller_v2 import CFG as CTRL_CFG, FakeSegmenter, Rig  # noqa: E402

DT = 1.0 / 30

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

    def test_transition_scores(self):
        # About 0.8 s each in the reference (frames 296 to 342 and 8 to 56 at 60 fps): 10 steps at 12 Hz.
        for score in (VANISH_SCORE, REAPPEAR_SCORE):
            self.assertEqual(len(score), 10)
            self.assertEqual(validate_score(score), [])
            self.assertTrue(all(step for step in score))
        keep_out = [step.get("keep", 1.0) for step in VANISH_SCORE]
        keep_in = [step.get("keep", 1.0) for step in REAPPEAR_SCORE]
        self.assertEqual(keep_out, sorted(keep_out, reverse=True))
        self.assertEqual(keep_in, sorted(keep_in))
        self.assertEqual(keep_in[-1], 1.0)

    def test_validator_catches_mistakes(self):
        bad = [{"cut": 4}, {"fill": ("NOPE",), "cover": 0.5}, {"fill": ("RED",)}, {"bg": "SKY"}, {"zoom": 1},
               {"keep": 0.0}]
        self.assertEqual(len(validate_score(bad)), 6)

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

    def test_keep_shows_part_of_the_body_and_the_room_elsewhere(self):
        inner = cv2.erode(self.body.view(np.uint8), np.ones((9, 9), np.uint8)) > 0
        for keep in (0.5, 0.08):
            out = self.r.render(self.frame, self.alpha, self.scene, {"keep": keep}, 9)
            shown = (np.abs(out[inner].astype(np.int16) - BODY_GRAY) <= 2).all(axis=1).mean()
            self.assertAlmostEqual(float(shown), keep, delta=0.2 if keep > 0.2 else 0.05)
            hidden = ~(np.abs(out.astype(np.int16) - BODY_GRAY) <= 2).all(axis=2) & inner
            np.testing.assert_array_equal(out[hidden], self.scene[hidden])

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


class LiveGlitchCase(unittest.TestCase):
    """The Glitch style while visible, through the real SnapController."""

    def setUp(self):
        self.cfg = dict(CTRL_CFG, style_default="GLITCH")
        self.rig = Rig(self.cfg)
        self.hz = self.cfg["glitch_step_hz"]

    def test_steps_follow_the_clock_at_12_hz(self):
        rig = self.rig
        t0 = rig.t
        seen = []
        while rig.t < t0 + 2.0 - 1e-9:
            rig.step(rig.t + DT)
            seen.append(rig.ctrl.glitch_langkah)
        # A step holds for 1/12 s, so on a 30 fps camera it shows on 2 or 3 frames in turn.
        runs = [len(list(g)) for _, g in itertools.groupby(seen)]
        self.assertTrue(set(runs[1:-1]) <= {2, 3}, runs)
        self.assertEqual(seen[-1], int((rig.t - t0) * self.hz + 1e-6))

    def test_calm_opening_then_the_glitch(self):
        rig = self.rig
        t0 = rig.t
        # Steps 0 to 5 of the score are calm: the camera frame shows as it is.
        rig.run_until(t0 + 5.5 / self.hz)
        self.assertTrue(np.array_equal(rig.out, rig.frame))
        rig.run_until(t0 + 6.5 / self.hz)
        self.assertEqual(rig.ctrl.glitch_langkah, 6)
        self.assertFalse(np.array_equal(rig.out, rig.frame))

    def test_a_held_step_keeps_its_shapes(self):
        rig = self.rig
        t0 = rig.t
        rig.run_until(t0 + 8.1 / self.hz)
        plan = rig.ctrl.glitch._rencana
        rig.step(t0 + 8.6 / self.hz)
        self.assertEqual(rig.ctrl.glitch_langkah, 8)
        self.assertIs(rig.ctrl.glitch._rencana, plan)

    def test_moved_pieces_uncover_the_room(self):
        rig = self.rig
        clean = rig.ctrl._tanpa_orang(rig.frame)
        person = rig.ctrl.matte.core()
        # With the room learned, the person is replaced by the room (noise synthesis off in these tests).
        self.assertLessEqual(int(np.abs(clean[person].astype(np.int16) - rig.scene[person]).max()), 2)

    def test_no_glitch_while_learning_the_room_or_with_other_styles(self):
        rig = self.rig
        rig.run_until(rig.t + 9.0 / self.hz)
        rig.ctrl.mulai_rekam_plate(rig.t)
        rig.step(rig.t + DT)
        self.assertTrue(np.array_equal(rig.out, rig.frame))
        self.assertEqual(rig.ctrl.glitch_langkah, -1)

        other = Rig(self.cfg)
        self.assertTrue(other.ctrl.pilih_gaya("DUST", other.t))
        other.run_until(other.t + 1.0)
        self.assertTrue(np.array_equal(other.out, other.frame))

    def test_picking_styles(self):
        ctrl = self.rig.ctrl
        self.assertTrue(ctrl.pilih_gaya("ASH", self.rig.t))
        self.assertEqual(ctrl.gaya, "ASH")
        self.assertEqual(ctrl.pesan(self.rig.t + 0.1), "Ash")
        for expected in ("EMBER", "GLITCH", "DUST"):
            ctrl.ganti_gaya(self.rig.t)
            self.assertEqual(ctrl.gaya, expected)
        self.rig.vanish()
        self.assertFalse(ctrl.pilih_gaya("GLITCH", self.rig.t))
        self.assertEqual(ctrl.gaya, "DUST")
        self.assertIn("finished", ctrl.pesan(self.rig.t))

    def test_snap_still_works_with_glitch_and_the_score_restarts_after(self):
        rig = self.rig
        rig.invisible()
        rig.ctrl.picu(rig.t)
        while rig.ctrl.state != "LIVE":
            rig.step(rig.t + DT)
        rig.step(rig.t + DT)
        self.assertEqual(rig.ctrl.glitch_langkah, 0)


class GlitchTransitionCase(unittest.TestCase):
    """Glitch vanish and reappear through the real SnapController."""

    def setUp(self):
        self.cfg = dict(CTRL_CFG, style_default="GLITCH")
        self.rig = Rig(self.cfg)
        self.length = len(VANISH_SCORE) / float(self.cfg["glitch_step_hz"])

    def shown(self):
        """Pixels that differ from the room: what is visible of the person."""
        return int((self.rig.out != self.rig.scene).any(axis=2).sum())

    def test_vanish_glitches_out_without_a_flash_and_leaves_only_the_room(self):
        rig = self.rig
        rig.vanish()
        start = rig.t
        self.assertEqual(rig.ctrl.state, "VANISH")
        self.assertTrue(rig.ctrl.kontrol_waktu_aktif())
        visible = []
        while rig.ctrl.state == "VANISH":
            rig.step(rig.t + DT)
            self.assertLess(float(rig.out.mean()), 200.0)
            if rig.ctrl.state == "VANISH":
                visible.append(self.shown())
        self.assertAlmostEqual(rig.t - start, self.length, delta=DT + 1e-6)
        self.assertEqual(rig.ctrl.state, "INVISIBLE")
        self.assertTrue(np.array_equal(rig.out, rig.scene))
        # The body drops away: the last steps show far less of it than the first.
        self.assertLess(np.mean(visible[-5:]), 0.5 * np.mean(visible[:5]))

    def test_reappear_glitches_in_and_ends_on_the_camera_frame(self):
        rig = self.rig
        rig.invisible()
        rig.ctrl.picu(rig.t)
        start = rig.t
        rig.step(rig.t + DT)
        first = self.shown()
        while rig.ctrl.state == "REAPPEAR":
            last = self.shown()
            rig.step(rig.t + DT)
        self.assertAlmostEqual(rig.t - start, self.length, delta=DT + 1e-6)
        self.assertEqual(rig.ctrl.state, "LIVE")
        self.assertTrue(np.array_equal(rig.out, rig.frame))
        self.assertLess(first, 0.5 * last)

    def test_freeze_holds_a_glitch_step_and_cancel_works(self):
        rig = self.rig
        rig.vanish()
        rig.run_until(rig.t + 0.25)
        # The clock eases to a stop, then the step holds.
        for _ in range(30):
            rig.ctrl.atur_waktu(True, False)
            rig.step(rig.t + DT)
        step = rig.ctrl.glitch_langkah
        for _ in range(30):
            rig.ctrl.atur_waktu(True, False)
            rig.step(rig.t + DT)
        self.assertEqual(rig.ctrl.glitch_langkah, step)
        self.assertEqual(rig.ctrl.state, "VANISH")
        rig.ctrl.reset(rig.t)
        self.assertEqual(rig.ctrl.state, "LIVE")
        # Cancel during a glitch reappear (a new rig: a cancel starts the snap cooldown).
        rig = Rig(self.cfg)
        rig.invisible()
        rig.ctrl.picu(rig.t)
        rig.step(rig.t + DT)
        rig.ctrl.reset(rig.t)
        self.assertEqual(rig.ctrl.state, "LIVE")

    def test_style_picked_while_invisible_sets_the_reappear(self):
        rig = Rig(CTRL_CFG)
        rig.invisible()
        self.assertTrue(rig.ctrl.pilih_gaya("GLITCH", rig.t))
        rig.ctrl.picu(rig.t)
        self.assertTrue(rig.ctrl._glitch_efek)
        self.assertAlmostEqual(rig.ctrl.akhir_kembali, self.length)


class EffectKeysCase(unittest.TestCase):
    def setUp(self):
        import main as app
        self.app = app
        self.snap = app.ModeSnap(CTRL_CFG, FakeSegmenter(), app.Profiler(30), hold_aktif=True)
        self.snap.masuk(0.0)
        self.t = 1.0

    def test_number_keys_pick_directly(self):
        self.snap.tombol("4", self.t)
        self.assertEqual(self.snap.ctrl.gaya, "GLITCH")
        self.snap.tombol("2", self.t)
        self.assertEqual(self.snap.ctrl.gaya, "ASH")
        self.snap.tombol("1", self.t)
        self.assertEqual(self.snap.ctrl.gaya, "DUST")

    def test_effects_menu(self):
        snap = self.snap
        snap.tombol("h", self.t)
        snap.tombol("e", self.t)
        self.assertTrue(snap.menu)
        self.assertFalse(snap.bantuan)
        self.assertIn("4  Glitch", self.app.baris_menu("DUST"))
        self.assertIn("1  Dust   (on)", self.app.baris_menu("DUST"))
        img = room(2)
        before = img.copy()
        snap.gambar_hud(img, self.t + 1.0, 30.0)
        self.assertFalse(np.array_equal(img, before))
        snap.tombol("3", self.t + 1.0)
        self.assertEqual(snap.ctrl.gaya, "EMBER")
        self.assertFalse(snap.menu)
        snap.tombol("e", self.t + 2.0)
        snap.tombol("e", self.t + 2.1)
        self.assertFalse(snap.menu)


if __name__ == "__main__":
    unittest.main()
