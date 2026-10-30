import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import snap_effect as se  # noqa: E402
from snap.clock import EffectClock  # noqa: E402
from test_background import H, LH, LW, W, room  # noqa: E402
from test_dissolve import scene_with_person, soft_person  # noqa: E402
from test_gestures_v2 import FakeResult  # noqa: E402

CFG = dict(se.CONFIG, bg_noise_on=False)
DT = 1.0 / 30
END = CFG["debu_mulai_detik"] + CFG["debu_sapuan_detik"] + CFG["debu_umur_detik"]
FLASH_END = CFG["flash_naik_detik"] + CFG["flash_tahan_detik"] + CFG["flash_turun_detik"] + CFG["flash_tunda_ujung_detik"]
REAPPEAR_END = CFG["reappear_sweep_s"] + CFG["reappear_life_s"] + CFG["reappear_blend_s"]


class Rig:
    """A controller with a learned synthetic room and a person standing in it."""

    def __init__(self, cfg=CFG):
        self.scene = room(31)
        self.frame = scene_with_person(self.scene)
        self.mask = soft_person()
        self.ctrl = se.SnapController(cfg)
        self.ctrl.bg.set_plate(self.scene)
        self.t = 0.0
        self.out = self.step(0.0)

    def step(self, t, frame=None, mask=None):
        frame = self.frame if frame is None else frame
        mask = self.mask if mask is None else mask
        self.t = t
        self.ctrl.amati(t, frame, cv2.resize(frame, (LW, LH), interpolation=cv2.INTER_AREA), mask)
        self.out = self.ctrl.render(t, frame)
        return self.out

    def run_until(self, t_end, **kw):
        while self.t < t_end - 1e-9:
            self.step(min(t_end, self.t + DT), **kw)
        return self.out

    def vanish(self, at=0.1):
        self.run_until(at)
        assert self.ctrl.picu(self.t)

    def invisible(self):
        self.vanish()
        self.run_until(self.t + END + 0.2)
        assert self.ctrl.state == "INVISIBLE"


class TestStates(unittest.TestCase):
    def test_full_cycle_ends_exactly_on_the_live_frame(self):
        rig = Rig()
        rig.invisible()
        self.assertTrue(np.array_equal(rig.out, rig.scene))
        rig.run_until(rig.t + 10.0)
        self.assertEqual(rig.ctrl.state, "INVISIBLE")
        start = rig.t
        self.assertTrue(rig.ctrl.picu(start))
        self.assertEqual(rig.ctrl.state, "REAPPEAR")
        # Nothing of the person shows at the start; the first dust fades in from zero.
        first = rig.step(start + DT)
        self.assertLess(rig.ctrl.info_kualitas(W, H)["mass"], 0.001)
        self.assertLessEqual(int(np.abs(first.astype(np.int16) - rig.scene).max()), 1)
        blend_start = start + CFG["reappear_sweep_s"] + CFG["reappear_life_s"]
        rig.run_until(blend_start)
        gaps = [float(np.abs(rig.out.astype(np.int16) - rig.frame).mean())]
        while rig.t < start + REAPPEAR_END - DT - 1e-9:
            rig.step(rig.t + DT)
            gaps.append(float(np.abs(rig.out.astype(np.int16) - rig.frame).mean()))
        self.assertEqual(rig.ctrl.state, "REAPPEAR")
        # The crossfade closes the gap to the live frame every frame; at the last frame only the eased
        # remainder is left, so the switch to LIVE is the smallest step of all.
        self.assertTrue(all(b <= a + 1e-6 for a, b in zip(gaps, gaps[1:])))
        rest = 1.0 - float(se.smoothstep((rig.t - blend_start) / CFG["reappear_blend_s"]))
        self.assertLessEqual(gaps[-1], rest * gaps[0] + 0.5)
        rig.run_until(start + REAPPEAR_END + DT)
        self.assertEqual(rig.ctrl.state, "LIVE")
        self.assertTrue(np.array_equal(rig.out, rig.frame))
        self.assertFalse(rig.ctrl.picu(rig.t + CFG["cooldown_detik"] - 2 * DT))

    def test_auto_reappear(self):
        rig = Rig(dict(CFG, auto_reappear_s=3.0))
        rig.vanish()
        start = rig.t
        rig.run_until(start + END + 3.0 - DT)
        self.assertEqual(rig.ctrl.state, "INVISIBLE")
        rig.run_until(start + END + 3.0 + DT)
        self.assertEqual(rig.ctrl.state, "REAPPEAR")

    def test_snap_does_nothing_during_vanish_and_reappear(self):
        rig = Rig()
        rig.vanish()
        rig.run_until(rig.t + 2.0)
        self.assertFalse(rig.ctrl.picu(rig.t))
        self.assertEqual(rig.ctrl.state, "VANISH")
        rig.run_until(rig.t + END)
        rig.ctrl.picu(rig.t)
        rig.step(rig.t + DT)
        self.assertFalse(rig.ctrl.picu(rig.t))
        self.assertEqual(rig.ctrl.state, "REAPPEAR")

    def test_plate_capture_returns_to_live(self):
        rig = Rig()
        rig.ctrl.mulai_rekam_plate(rig.t)
        self.assertEqual(rig.ctrl.state, "PLATE_CAPTURE")
        empty = np.zeros((LH, LW), dtype=np.float32)
        rig.run_until(rig.t + CFG["plate_hitung_mundur"] + (CFG["plate_jumlah_frame"] + 2) * DT,
                      frame=rig.scene, mask=empty)
        self.assertEqual(rig.ctrl.state, "LIVE")


class TestCancel(unittest.TestCase):
    def cancel_at(self, setup, label):
        rig = Rig()
        setup(rig)
        before = rig.out.copy()
        rig.ctrl.reset(rig.t)
        self.assertEqual(rig.ctrl.state, "LIVE", label)
        gaps = []
        for _ in range(int(round(CFG["cancel_fade_s"] / DT)) + 2):
            out = rig.step(rig.t + DT)
            gaps.append(float(np.abs(out.astype(np.int16) - rig.frame).mean()))
        self.assertTrue(np.array_equal(rig.out, rig.frame), label)
        self.assertTrue(all(b <= a + 1e-6 for a, b in zip(gaps, gaps[1:])), label)
        if not np.array_equal(before, rig.frame):
            # Eased, not a cut: the first frame after the cancel still looks mostly like before.
            start = float(np.abs(before.astype(np.int16) - rig.frame).mean())
            self.assertGreater(gaps[0], 0.8 * start, label)

    def test_cancel_from_every_state(self):
        def capture(rig):
            rig.ctrl.mulai_rekam_plate(rig.t)
            rig.step(rig.t + DT)

        def vanish_at(t):
            def go(rig):
                rig.vanish()
                rig.run_until(rig.t + t)
            return go

        def reappear(rig):
            rig.invisible()
            rig.ctrl.picu(rig.t)
            rig.run_until(rig.t + 1.5)

        for label, setup in (("LIVE", lambda rig: None), ("PLATE_CAPTURE", capture), ("VANISH flash", vanish_at(0.2)),
                             ("VANISH white", vanish_at(0.8)), ("VANISH dust", vanish_at(2.2)),
                             ("INVISIBLE", Rig.invisible), ("REAPPEAR", reappear)):
            with self.subTest(state=label):
                self.cancel_at(setup, label)


class TestClock(unittest.TestCase):
    def test_rate_eases_without_steps(self):
        clock = EffectClock(CFG)
        clock.start(0.0)
        rng = np.random.default_rng(4)
        targets = rng.choice([0.0, CFG["slow_rate"], 1.0], size=40).repeat(rng.integers(1, 40, size=40))
        prev_rate, prev_t = clock.rate, clock.t
        for i, target in enumerate(targets):
            clock.update((i + 1) * DT, float(target))
            self.assertLessEqual(abs(clock.rate - prev_rate), 0.1)
            self.assertTrue(0.0 <= clock.rate <= 1.0)
            self.assertGreaterEqual(clock.t, prev_t)
            prev_rate, prev_t = clock.rate, clock.t

    def test_freeze_stops_time_and_release_resumes(self):
        clock = EffectClock(CFG)
        clock.start(0.0)
        for i in range(90):
            clock.update((i + 1) * DT, 0.0)
        frozen = clock.t
        clock.update(91 * DT, 0.0)
        self.assertLess(clock.t - frozen, 0.001 * DT)
        for i in range(91, 200):
            clock.update((i + 1) * DT, 1.0)
        self.assertAlmostEqual(clock.rate, 1.0, places=3)

    def test_freeze_is_ignored_during_the_flash(self):
        rig = Rig()
        rig.vanish()
        start = rig.t
        while rig.t < start + FLASH_END - DT:
            rig.ctrl.atur_waktu(True, False)
            rig.step(rig.t + DT)
        self.assertAlmostEqual(rig.ctrl.jam.t, rig.t - start, places=6)
        self.assertEqual(rig.ctrl.jam.rate, 1.0)
        # After the flash the fist slows the dissolve down to a stop.
        while rig.t < start + END + 1.0:
            rig.ctrl.atur_waktu(True, False)
            rig.step(rig.t + DT)
        self.assertEqual(rig.ctrl.state, "VANISH")
        self.assertLess(rig.ctrl.jam.rate, 0.05)

    def test_slow_motion_during_reappear(self):
        rig = Rig()
        rig.invisible()
        rig.ctrl.picu(rig.t)
        start = rig.t
        while rig.t < start + 2.0:
            rig.ctrl.atur_waktu(False, True)
            rig.step(rig.t + DT)
        self.assertAlmostEqual(rig.ctrl.jam.rate, CFG["slow_rate"], places=2)
        self.assertLess(rig.ctrl.jam.t, 1.0)


class TestStyles(unittest.TestCase):
    def snapshot_at(self, style, t):
        rig = Rig(dict(CFG, style_default=style))
        rig.vanish()
        rig.run_until(rig.t + t)
        return rig

    def test_styles_keep_zero_residue_and_an_exact_return(self):
        for style in se.STYLE_NAMES:
            with self.subTest(style=style):
                rig = Rig(dict(CFG, style_default=style))
                rig.invisible()
                self.assertTrue(np.array_equal(rig.out, rig.scene))
                rig.ctrl.picu(rig.t)
                rig.run_until(rig.t + REAPPEAR_END + DT)
                self.assertEqual(rig.ctrl.state, "LIVE")
                self.assertTrue(np.array_equal(rig.out, rig.frame))

    def test_styles_change_only_colors_and_motion(self):
        dust = self.snapshot_at("DUST", 2.6)
        ash = self.snapshot_at("ASH", 2.6)
        ember = self.snapshot_at("EMBER", 2.6)
        self.assertFalse(np.array_equal(dust.out, ash.out))
        self.assertFalse(np.array_equal(dust.out, ember.out))
        # Same release map, same particles; ASH only moves slower and sinks.
        self.assertTrue(np.array_equal(dust.ctrl.debu.p_release, ash.ctrl.debu.p_release))
        both = ~np.isnan(dust.ctrl.debu._prev_xy[:, 1]) & ~np.isnan(ash.ctrl.debu._prev_xy[:, 1])
        self.assertGreater(float(ash.ctrl.debu._prev_xy[both, 1].mean()), float(dust.ctrl.debu._prev_xy[both, 1].mean()))
        # EMBER adds a warm glow at the dissolve front: more red than DUST on the remaining body.
        box = dust.ctrl.debu.box
        sl = (slice(box[1], box[1] + box[3]), slice(box[0], box[0] + box[2]))
        self.assertGreater(float(ember.out[sl][..., 2].astype(np.int32).sum()),
                           float(dust.out[sl][..., 2].astype(np.int32).sum()))

    def test_style_switch_from_live_and_invisible_only(self):
        rig = Rig()
        self.assertTrue(rig.ctrl.ganti_gaya(rig.t))
        self.assertEqual(rig.ctrl.gaya, "ASH")
        self.assertEqual(rig.ctrl.toast(rig.t + 0.5)[0], "Ash")
        self.assertIsNone(rig.ctrl.toast(rig.t + CFG["style_toast_s"]))
        rig.vanish()
        self.assertFalse(rig.ctrl.ganti_gaya(rig.t))
        rig.run_until(rig.t + END + 0.1)
        self.assertTrue(rig.ctrl.ganti_gaya(rig.t))
        self.assertEqual(rig.ctrl.gaya, "EMBER")


class TestReappear(unittest.TestCase):
    def test_mass_and_steps_are_smooth(self):
        rig = Rig()
        rig.invisible()
        rig.ctrl.picu(rig.t)
        start = rig.t
        masses, steps = [], []
        while rig.ctrl.state == "REAPPEAR":
            rig.step(rig.t + DT)
            if rig.ctrl.state == "REAPPEAR":
                info = rig.ctrl.info_kualitas(W, H)
                masses.append(info["mass"])
                steps.append(info["step_px"])
        self.assertAlmostEqual(rig.t - start, REAPPEAR_END, delta=DT + 1e-6)
        self.assertLess(masses[0], 0.01)
        self.assertGreater(masses[-1], 0.99)
        self.assertLessEqual(float(np.max(np.abs(np.diff(masses)))), 0.04)
        self.assertLessEqual(max(steps), CFG["max_particle_step_px"])

    def test_bottom_assembles_before_the_head(self):
        rig = Rig()
        rig.invisible()
        rig.ctrl.picu(rig.t)
        r = rig.ctrl.debu
        y = r.p_y0
        span = y.max() - y.min()
        head = r.p_release[y < y.min() + 0.2 * span]
        feet = r.p_release[y > y.max() - 0.2 * span]
        # Played backward, the dust released last (the bottom) comes back first.
        self.assertGreater(float(np.median(feet)), float(np.median(head)))

    def test_crossfade_stays_inside_the_zone(self):
        rig = Rig()
        rig.invisible()
        rig.ctrl.picu(rig.t)
        start = rig.t
        rig.run_until(start + CFG["reappear_sweep_s"] + CFG["reappear_life_s"] + 0.15)
        x0, y0, x1, y1 = rig.ctrl.debu.roi
        outside = np.ones((H, W), dtype=bool)
        outside[y0:y1, x0:x1] = False
        self.assertTrue(np.array_equal(rig.out[outside], rig.frame[outside]))


class FakeSegmenter:
    """Stands in for PersonSegmenter: always the same soft person mask."""

    def __init__(self):
        self.mask = soft_person()

    def perbarui(self, bgr, ts):
        return self.mask

    def reset(self):
        pass


class TestModeSnapV2(unittest.TestCase):
    def setUp(self):
        import main as app
        self.app = app
        self.snap = app.ModeSnap(CFG, FakeSegmenter(), app.Profiler(30), hold_aktif=True)
        self.snap.masuk(0.0)
        self.scene = room(31)
        self.frame = scene_with_person(self.scene)
        self.snap.ctrl.bg.set_plate(self.scene)
        self.t = 1.2
        self.feed("None", 0.2)

    def feed(self, label, seconds):
        for _ in range(int(round(seconds / DT))):
            self.t += DT
            self.snap.proses(self.t, self.frame, FakeResult(label), int(self.t * 1000))

    def test_open_palm_brings_you_back(self):
        self.snap.tombol("s", self.t)
        self.assertEqual(self.snap.ctrl.state, "VANISH")
        self.feed("None", END + 0.2)
        self.assertEqual(self.snap.ctrl.state, "INVISIBLE")
        self.feed("Open_Palm", CFG["hold_palm_s"] + 0.1)
        self.assertEqual(self.snap.ctrl.state, "REAPPEAR")

    def test_peace_sign_changes_style_and_rock_sign_toggles_help(self):
        self.feed("Victory", CFG["hold_victory_s"] + 0.1)
        self.assertEqual(self.snap.ctrl.gaya, "ASH")
        self.feed("None", 0.7)
        self.feed("ILoveYou", CFG["hold_love_s"] + 0.1)
        self.assertTrue(self.snap.bantuan)

    def test_fist_freezes_the_dissolve_after_the_flash_only(self):
        self.snap.tombol("s", self.t)
        self.feed("Closed_Fist", 1.0)
        self.assertEqual(self.snap.ctrl.jam.rate, 1.0)
        self.feed("None", FLASH_END - 1.0 + 0.1)
        self.feed("Closed_Fist", 1.5)
        self.assertLess(self.snap.ctrl.jam.rate, 0.2)
        self.assertEqual(self.snap.ctrl.state, "VANISH")
        self.feed("None", 1.5)
        self.assertGreater(self.snap.ctrl.jam.rate, 0.8)

    def test_keys_x_h_and_r(self):
        self.snap.tombol("x", self.t)
        self.assertEqual(self.snap.ctrl.gaya, "ASH")
        self.snap.tombol("h", self.t)
        self.assertTrue(self.snap.bantuan)
        img = self.frame.copy()
        self.snap.gambar_hud(img, self.t + 1.0, 30.0)
        self.assertFalse(np.array_equal(img, self.frame))
        self.snap.tombol("s", self.t)
        self.feed("None", 0.5)
        self.snap.tombol("r", self.t)
        self.assertEqual(self.snap.ctrl.state, "LIVE")


class TestRecorders(unittest.TestCase):
    def setUp(self):
        import main as app
        self.app = app
        self.tmp = tempfile.mkdtemp()
        self.old = app.FOLDER_REKAMAN
        app.FOLDER_REKAMAN = self.tmp
        self.addCleanup(setattr, app, "FOLDER_REKAMAN", self.old)
        rng = np.random.default_rng(2)
        self.frames = [rng.integers(0, 256, (360, 640, 3), dtype=np.uint8) for _ in range(3)]

    def test_png_frames_are_lossless(self):
        rec = self.app.PerekamPNG()
        rec.mulai()
        for f in self.frames:
            rec.tulis(f)
        folder = rec.folder
        self.assertIn("3 frames", rec.berhenti())
        for i, f in enumerate(self.frames):
            self.assertTrue(np.array_equal(cv2.imread(os.path.join(folder, "%05d.png" % i)), f))

    def test_video_opens_with_a_working_codec(self):
        rec = self.app.PerekamVideo()
        self.assertEqual(rec.mulai(self.frames[0], 30.0, 0.0), "Mulai merekam")
        for i, f in enumerate(self.frames):
            rec.tulis(f, i / 30.0)
        rec.berhenti()
        self.assertGreater(os.path.getsize(rec.path), 0)


if __name__ == "__main__":
    unittest.main()
