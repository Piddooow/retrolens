import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import snap_effect as se  # noqa: E402
from snap.matte import MatteEngine  # noqa: E402
from test_background import H, LH, LW, W, person_lo, room, with_person  # noqa: E402

CFG = se.CONFIG
ALL_KNOWN = np.ones((H, W), dtype=bool)


def soft_person():
    """Person confidence with a soft edge, like the segmenter output around hair."""
    return cv2.GaussianBlur(person_lo(), (0, 0), 2.0)


def shade(img, box, factors):
    out = img.copy()
    y0, y1, x0, x1 = box
    out[y0:y1, x0:x1] = np.clip(out[y0:y1, x0:x1].astype(np.float32) * np.float32(factors), 0, 255).astype(np.uint8)
    return out


class TestMatte(unittest.TestCase):
    def setUp(self):
        self.scene = room(11)
        self.frame = with_person(self.scene, person_lo())
        self.eng = MatteEngine(CFG)
        self.eng.update(self.frame, soft_person())

    def test_guided_filter_accepts_keyword_depth_and_scale(self):
        guide = self.frame.astype(np.float32) / 255.0
        src = np.zeros((H, W), dtype=np.float32)
        out = cv2.ximgproc.guidedFilter(guide, src, 8, 0.001, dDepth=-1, scale=0.5)
        self.assertEqual(out.shape, (H, W))
        self.assertEqual(out.dtype, np.float32)

    def test_matte_is_display_size_and_clean(self):
        matte = self.eng.matte
        self.assertEqual(matte.shape, (H, W))
        self.assertGreaterEqual(float(matte.min()), 0.0)
        self.assertLessEqual(float(matte.max()), 1.0)
        self.assertFalse(((matte > 0) & (matte < CFG["matte_floor"])).any())
        # Values between floor and 1 exist at the edge, so it is a soft matte, not a binary mask.
        self.assertTrue(((matte > 0.1) & (matte < 0.9)).any())

    def test_zone_covers_every_matte_pixel(self):
        zf, _ = self.eng.zone(self.frame, self.scene, ALL_KNOWN)
        visible = self.eng.matte > CFG["matte_floor"]
        self.assertTrue(visible.any())
        self.assertTrue(np.all(zf[visible] == 1.0))

    def test_feather_is_monotonic_from_inside_to_outside(self):
        zf, _ = self.eng.zone(self.frame, self.scene, ALL_KNOWN)
        for y in (300, 500, 700):
            row = zf[y, W // 2:]
            self.assertEqual(float(row[0]), 1.0)
            self.assertTrue(np.all(np.diff(row) <= 1e-6), "row %d" % y)
            self.assertTrue(((row > 0) & (row < 1)).any(), "row %d has no feather" % y)

    def test_hysteresis_keeps_person_down_to_keep_threshold(self):
        frame = self.scene
        weak = np.zeros((LH, LW), dtype=np.float32)
        weak[100:300, 200:440] = 0.4
        strong = weak / 0.4 * 0.9
        center = (H // 2 + 50, W // 2)

        fresh = MatteEngine(CFG)
        fresh.update(frame, weak)
        self.assertFalse(fresh.person[center])

        held = MatteEngine(CFG)
        held.update(frame, strong)
        self.assertTrue(held.person[center])
        held.update(frame, weak)
        self.assertTrue(held.person[center])
        held.reset()
        self.assertIsNone(held.person)


class TestShadow(unittest.TestCase):
    def setUp(self):
        self.scene = room(12)
        self.mask = person_lo()
        self.eng = MatteEngine(CFG)
        # The body spans x 460..820 at display size; these patches sit right of it and far away.
        self.near = (450, 600, 840, 900)
        self.far = (60, 160, 60, 160)

    def zone_for(self, live, known=ALL_KNOWN, cfg=CFG):
        eng = MatteEngine(cfg)
        eng.update(live, self.mask)
        return eng.zone(live, self.scene, known)

    def test_dark_patch_next_to_person_joins_the_zone(self):
        live = with_person(shade(self.scene, self.near, 0.7), self.mask)
        zf, shadow = self.zone_for(live)
        y0, y1, x0, x1 = self.near
        self.assertTrue(np.all(zf[y0:y1, x0:x1] == 1.0))
        # The part of the patch closest to the person is already core zone; the rest is shadow.
        eng = MatteEngine(CFG)
        eng.update(live, self.mask)
        outside_core = ~eng.core()[y0:y1, x0:x1]
        self.assertGreater(float(outside_core.mean()), 0.5)
        self.assertTrue(np.all(shadow[y0:y1, x0:x1][outside_core]))

    def test_same_patch_far_from_person_is_ignored(self):
        live = with_person(shade(self.scene, self.far, 0.7), self.mask)
        zf, shadow = self.zone_for(live)
        y0, y1, x0, x1 = self.far
        self.assertFalse(shadow.any())
        self.assertEqual(float(zf[y0:y1, x0:x1].max()), 0.0)

    def test_colored_patch_is_not_a_shadow(self):
        live = with_person(shade(self.scene, self.near, (0.5, 0.8, 0.8)), self.mask)
        _, shadow = self.zone_for(live)
        y0, y1, x0, x1 = self.near
        self.assertLess(float(shadow[y0:y1, x0:x1].mean()), 0.05)

    def test_shadow_only_where_room_is_known(self):
        live = with_person(shade(self.scene, self.near, 0.7), self.mask)
        known = ALL_KNOWN.copy()
        y0, y1, x0, x1 = self.near
        known[y0:y1, x0:x1] = False
        _, shadow = self.zone_for(live, known)
        self.assertFalse(shadow[y0:y1, x0:x1].any())

    def test_shadow_off(self):
        live = with_person(shade(self.scene, self.near, 0.7), self.mask)
        _, shadow = self.zone_for(live, cfg=dict(CFG, shadow_on=False))
        self.assertFalse(shadow.any())


class TestControllerZone(unittest.TestCase):
    def test_snap_keeps_the_feathered_zone(self):
        scene = room(13)
        mask = soft_person()
        frame = with_person(scene, person_lo())
        ctrl = se.SnapController(CFG)
        ctrl.bg.set_plate(scene)
        ctrl.amati(0.0, frame, cv2.resize(frame, (LW, LH), interpolation=cv2.INTER_AREA), mask)
        self.assertTrue(ctrl.bg_siap)
        self.assertTrue(ctrl.picu(0.1))
        self.assertEqual(ctrl._zf.shape, (H, W))
        self.assertTrue(np.all(ctrl._zf[ctrl.matte.matte > CFG["matte_floor"]] == 1.0))
        self.assertTrue(np.array_equal(ctrl._zona, ctrl._zf > 0))


if __name__ == "__main__":
    unittest.main()
