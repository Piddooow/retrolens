import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import snap_effect as se  # noqa: E402
from snap.ash import AshAmbient  # noqa: E402
from snap.dissolve import DissolveRenderer  # noqa: E402
from snap.matte import MatteEngine  # noqa: E402
from test_background import H, LH, LW, W, person_lo, room, with_person  # noqa: E402

CFG = se.CONFIG
DT = 1.0 / 30
START = CFG["debu_mulai_detik"]
END = CFG["debu_mulai_detik"] + CFG["debu_sapuan_detik"] + CFG["debu_umur_detik"]
ALL_KNOWN = np.ones((H, W), dtype=bool)


def soft_person(dx=0):
    mask = person_lo()
    if dx:
        mask = np.roll(mask, dx, axis=1)
    return cv2.GaussianBlur(mask, (0, 0), 2.0)


def scene_with_person(scene, dx=0):
    mask = person_lo()
    if dx:
        mask = np.roll(mask, dx, axis=1)
    return with_person(scene, mask)


class DissolveCase(unittest.TestCase):
    def setUp(self):
        self.scene = room(21)
        self.frame = scene_with_person(self.scene)
        eng = MatteEngine(CFG)
        eng.update(self.frame, soft_person())
        self.matte = eng.matte
        self.zone, _ = eng.zone(self.frame, self.scene, ALL_KNOWN)

    def renderer(self, cfg=CFG):
        r = DissolveRenderer(cfg)
        r.siapkan(self.frame, self.matte, self.zone, cfg["particle_cell_px"])
        return r

    def draw(self, r, t, extra=None):
        out = self.scene.copy()
        r.render(out, t, extra)
        return out


class TestDissolveRenderer(DissolveCase):
    def test_particle_count_and_mass_match_the_matte(self):
        r = self.renderer()
        cell = CFG["particle_cell_px"]
        self.assertGreater(r.count, 5000)
        # Particles carry the whole matte except the tiny edge cells below matte_floor.
        carried = float(r.p_mass.sum()) * cell * cell / float(self.matte.sum())
        self.assertGreater(carried, 0.995)
        self.assertLessEqual(carried, 1.0 + 1e-4)

    def test_unreleased_body_is_the_sharp_snapshot(self):
        r = self.renderer()
        out = self.draw(r, 1.0)
        full = self.matte >= 1.0
        self.assertTrue(full.any())
        self.assertTrue(np.array_equal(out[full], self.frame[full]))
        self.assertEqual(r.visible, 0)
        self.assertAlmostEqual(r.mass, 1.0, places=5)

    def test_zero_residue_after_the_dust(self):
        r = self.renderer()
        # The last dust fades with zero slope, so it is still visible shortly before the end.
        self.assertFalse(np.array_equal(self.draw(r, END - 0.3), self.scene))
        for t in (END, END + DT, END + 1.0):
            self.assertTrue(np.array_equal(self.draw(r, t), self.scene), "t=%.3f" % t)
            self.assertEqual(r.mass, 0.0)

    def test_mass_continuity_and_particle_step(self):
        r = self.renderer()
        masses, steps = [], []
        t = 1.2
        while t < END + 0.1:
            self.draw(r, t)
            masses.append(r.mass)
            steps.append(r.step_max)
            t += DT
        self.assertAlmostEqual(masses[0], 1.0, places=5)
        self.assertEqual(masses[-1], 0.0)
        jump = float(np.max(np.abs(np.diff(masses))))
        self.assertLessEqual(jump, 0.04, "mass jump %.4f" % jump)
        self.assertLessEqual(max(steps), CFG["max_particle_step_px"])
        self.assertGreater(max(steps), 5.0)

    def test_head_releases_before_chest(self):
        r = self.renderer()
        y = r.p_y0
        span = y.max() - y.min()
        head = r.p_release[y < y.min() + 0.2 * span]
        chest = r.p_release[y > y.max() - 0.2 * span]
        self.assertLess(float(np.median(head)), float(np.median(chest)))
        self.assertGreaterEqual(float(r.p_release.min()), START - 1e-5)
        self.assertLessEqual(float(r.p_release.max()), START + CFG["debu_sapuan_detik"] + 1e-5)

    def test_body_has_holes_in_the_middle_of_the_sweep(self):
        r = self.renderer()
        t = START + 0.5 * CFG["debu_sapuan_detik"]
        band = (r.p_y0 > np.percentile(r.p_y0, 55)) & (r.p_y0 < np.percentile(r.p_y0, 75))
        released = r.p_release[band] < t
        self.assertTrue(released.any() and not released.all())

    def test_dust_moves_with_the_wind_and_spreads(self):
        for flip in (False, True):
            with self.subTest(flip=flip):
                r = self.renderer(dict(CFG, debu_balik_horizontal=flip))
                t = START + 0.6 * CFG["debu_sapuan_detik"]
                self.draw(r, t)
                xy = r._prev_xy
                old = (t - r.p_release) > 0.4
                old &= ~np.isnan(xy[:, 0])
                self.assertGreater(int(old.sum()), 500)
                dx = xy[old, 0] - r.p_x0[old]
                dy = xy[old, 1] - r.p_y0[old]
                if flip:
                    self.assertGreater(float(dx.min()), 0.0)
                else:
                    self.assertLess(float(dx.max()), 0.0)
                self.assertLess(float(dy.mean()), 0.0)
                # Jitter spreads the dust instead of moving it as one block.
                self.assertGreater(float(dx.std()), 5.0)

    def test_seed_reproducible(self):
        t = START + 1.0
        a = self.draw(self.renderer(), t)
        b = self.draw(self.renderer(), t)
        self.assertTrue(np.array_equal(a, b))
        c = self.draw(self.renderer(dict(CFG, debu_seed=CFG["debu_seed"] + 1)), t)
        self.assertFalse(np.array_equal(a, c))

    def test_nothing_changes_outside_the_roi(self):
        r = self.renderer()
        x0, y0, x1, y1 = r.roi
        self.assertLess((x1 - x0) * (y1 - y0), W * H)
        inside = np.zeros((H, W), dtype=bool)
        inside[y0:y1, x0:x1] = True
        half = np.zeros(((H + 1) // 2, (W + 1) // 2), dtype=bool)
        half[y0 // 2:(y1 + 1) // 2, x0 // 2:(x1 + 1) // 2] = True
        for t in (1.4, 2.0, 2.6, 3.2, 3.8, 4.0, 4.3):
            out = self.draw(r, t)
            changed = np.any(out != self.scene, axis=2)
            if t <= 4.0:
                self.assertTrue(changed.any(), "t=%.2f" % t)
            self.assertFalse(changed[~inside].any(), "t=%.2f" % t)
            # The dust itself never reaches past the ROI, so the crop hides nothing.
            if r.last_alpha is not None:
                self.assertEqual(float(np.abs(r.last_alpha[~half]).max()), 0.0, "t=%.2f" % t)

    def test_dust_is_soft_not_blocky(self):
        r = self.renderer()
        self.draw(r, START + 0.5 * CFG["debu_sapuan_detik"])
        alpha = cv2.resize(r.last_alpha, (W, H), interpolation=cv2.INTER_LINEAR)
        alpha = alpha[self.matte == 0]
        dust = alpha > 0.02
        partial = dust & (alpha < 0.98)
        self.assertGreater(int(dust.sum()), 1000)
        self.assertGreater(float(partial.sum()) / float(dust.sum()), 0.5)

    def test_ash_is_soft_and_ends_with_the_dust(self):
        ash = AshAmbient(CFG)
        ash.siapkan((W, H), np.random.default_rng(1))
        r = self.renderer()
        x0, y0, x1, y1 = r.roi
        outside = np.ones((H, W), dtype=bool)
        outside[y0:y1, x0:x1] = False
        t = START + 0.5
        out = self.draw(r, t, ash.specks(t))
        self.assertTrue(np.any(out != self.scene, axis=2)[outside].any())
        self.assertLessEqual(CFG["abu_selesai_detik"], END)
        self.assertIsNone(ash.specks(CFG["abu_selesai_detik"]))
        self.assertIsNone(ash.specks(START - DT))


class TestControllerDissolve(unittest.TestCase):
    def run_snap(self, cfg, frames_for):
        """Snap at 0.1 s and render at 30 fps until 2 s into INVISIBLE. frames_for(t) -> (frame, mask_lo)."""
        ctrl = se.SnapController(cfg)
        scene = room(22)
        ctrl.bg.set_plate(scene)
        frame, mask = frames_for(0.0)
        ctrl.amati(0.0, frame, cv2.resize(frame, (LW, LH), interpolation=cv2.INTER_AREA), mask)
        self.assertTrue(ctrl.picu(0.1))
        shots = []
        t = 0.1
        while t < 0.1 + ctrl.debu_akhir + 2.0:
            t += DT
            frame, mask = frames_for(t - 0.1)
            ctrl.amati(t, frame, cv2.resize(frame, (LW, LH), interpolation=cv2.INTER_AREA), mask)
            out = ctrl.render(t, frame)
            shots.append((t - 0.1, ctrl.state, out, frame, ctrl.info_kualitas(W, H)))
        return ctrl, scene, shots

    def test_static_scene_ends_as_the_empty_room(self):
        cfg = dict(CFG, bg_noise_on=False)
        scene = room(22)
        frame = scene_with_person(scene)
        ctrl, _, shots = self.run_snap(cfg, lambda t: (frame, soft_person()))
        empty = [s for s in shots if s[1] == "INVISIBLE"]
        self.assertGreater(len(empty), 30)
        for t, _, out, _, info in empty:
            self.assertTrue(np.array_equal(out, scene), "t=%.3f" % t)
            inside = info["zone"] > 0
            self.assertTrue(inside.any())
            self.assertTrue(np.array_equal(out[inside], info["plate_view"][inside]))

    def test_residue_zero_with_noise_on(self):
        scene = room(22)
        frame = scene_with_person(scene)
        ctrl, _, shots = self.run_snap(CFG, lambda t: (frame, soft_person()))
        empty = [s for s in shots if s[1] == "INVISIBLE"]
        self.assertGreater(len(empty), 30)
        for t, _, out, _, info in empty:
            inside = info["zone"] > 0
            self.assertEqual(int(np.abs(out[inside].astype(np.int16) - info["plate_view"][inside]).max()), 0)

    def test_person_who_moves_stays_hidden(self):
        cfg = dict(CFG, bg_noise_on=False)
        scene = room(22)

        def frames_for(t):
            dx = 0 if t < 1.0 else 100
            return scene_with_person(scene, dx), soft_person(dx)

        _, _, shots = self.run_snap(cfg, frames_for)
        empty = [s for s in shots if s[1] == "INVISIBLE"]
        self.assertGreater(len(empty), 30)
        for t, _, out, _, _ in empty:
            self.assertTrue(np.array_equal(out, scene), "t=%.3f" % t)

    def test_never_seen_room_is_revealed_softly(self):
        cfg = dict(CFG, bg_noise_on=False)
        scene = room(22)
        ctrl = se.SnapController(cfg)
        ctrl.bg.set_plate(scene)
        frame = scene_with_person(scene)
        mask = soft_person()

        def step(t, img, m):
            ctrl.amati(t, img, cv2.resize(img, (LW, LH), interpolation=cv2.INTER_AREA), m)
            return ctrl.render(t, img)

        step(0.0, frame, mask)
        self.assertTrue(ctrl.picu(0.1))
        t = 0.1
        while t < 0.1 + ctrl.debu_akhir + 0.2:
            t += DT
            step(t, frame, mask)
        self.assertEqual(ctrl.state, "INVISIBLE")
        # The person steps right, onto room the camera never saw.
        moved = scene_with_person(scene, 200)
        moved_mask = soft_person(200)
        unseen = np.zeros((H, W), dtype=bool)
        unseen[400:720, 1100:1200] = True
        ctrl.bg.count[unseen] = 0
        gaps = []
        for _ in range(int(round(0.6 / DT))):
            t += DT
            out = step(t, moved, moved_mask)
            gaps.append(float(np.abs(out[unseen].astype(np.int16) - moved[unseen]).mean()))
            self.assertTrue(np.array_equal(out[~unseen], scene[~unseen]), "t=%.3f" % t)
        self.assertEqual(ctrl.state, "INVISIBLE")
        self.assertEqual(ctrl.pesan(t), se.MSG_CANT_SEE)
        # A soft reveal: still mostly hidden on the first frame, monotonic, complete after reveal_fade_s.
        hidden = float(np.abs(scene[unseen].astype(np.int16) - moved[unseen]).mean())
        self.assertGreater(gaps[0], 0.8 * hidden)
        self.assertTrue(all(b <= a + 1e-6 for a, b in zip(gaps, gaps[1:])))
        done = int(np.ceil(CFG["reveal_fade_s"] / DT))
        self.assertEqual(gaps[done], 0.0)
        self.assertEqual(gaps[-1], 0.0)

if __name__ == "__main__":
    unittest.main()
