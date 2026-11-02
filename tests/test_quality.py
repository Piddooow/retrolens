import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snap.quality import LEVEL_MAX, QualityLadder  # noqa: E402
from test_controller_v2 import CFG, END, REAPPEAR_END, Rig  # noqa: E402

DT = 1.0 / 30


def feed(ladder, start, seconds, fps):
    t = start
    while t < start + seconds - 1e-9:
        t += DT
        ladder.perbarui(t, fps)
    return t


class TestLadder(unittest.TestCase):
    def test_steps_down_once_per_bad_hold(self):
        q = QualityLadder(CFG)
        t = feed(q, 0.0, CFG["fps_bad_hold_s"] - 2 * DT, CFG["fps_bad"] - 1)
        self.assertEqual(q.level, 0)
        t = feed(q, t, 3 * DT, CFG["fps_bad"] - 1)
        self.assertEqual(q.level, 1)
        t = feed(q, t, CFG["fps_bad_hold_s"] * 10, CFG["fps_bad"] - 1)
        self.assertEqual(q.level, LEVEL_MAX)

    def test_steps_back_up_slowly_and_ignores_the_middle(self):
        q = QualityLadder(CFG)
        t = feed(q, 0.0, CFG["fps_bad_hold_s"] * 2 + 0.2, CFG["fps_bad"] - 1)
        self.assertEqual(q.level, 2)
        t = feed(q, t, 10.0, (CFG["fps_bad"] + CFG["fps_good"]) / 2.0)
        self.assertEqual(q.level, 2)
        t = feed(q, t, CFG["fps_good_hold_s"] - 2 * DT, CFG["fps_good"] + 1)
        self.assertEqual(q.level, 2)
        t = feed(q, t, 3 * DT, CFG["fps_good"] + 1)
        self.assertEqual(q.level, 1)
        feed(q, t, CFG["fps_good_hold_s"] * 3, CFG["fps_good"] + 1)
        self.assertEqual(q.level, 0)

    def test_short_dips_do_nothing(self):
        q = QualityLadder(CFG)
        t = 0.0
        for _ in range(10):
            t = feed(q, t, CFG["fps_bad_hold_s"] / 2, CFG["fps_bad"] - 5)
            t = feed(q, t, 0.1, CFG["fps_bad"] + 1)
        self.assertEqual(q.level, 0)


class TestLadderInController(unittest.TestCase):
    def test_level_is_used_only_when_an_effect_starts(self):
        rig = Rig()
        rig.ctrl.tangga.level = 1
        rig.step(rig.t + DT)
        self.assertEqual(rig.ctrl.matte.scale, CFG["matte_scale"])
        rig.vanish()
        self.assertEqual(rig.ctrl.matte.scale, CFG["ladder_matte_scale"])
        rig.ctrl.tangga.level = 2
        rig.run_until(rig.t + END + 0.2)
        self.assertEqual(rig.ctrl.ukuran_butir, CFG["particle_cell_px"])
        rig.ctrl.picu(rig.t)
        self.assertEqual(rig.ctrl.ukuran_butir, CFG["debu_ukuran_butir_turun"])

    def test_cheapest_level_keeps_zero_residue_and_an_exact_return(self):
        rig = Rig()
        rig.ctrl.tangga.level = LEVEL_MAX
        rig.invisible()
        self.assertEqual(rig.ctrl.debu.blur, 0.0)
        self.assertFalse(rig.ctrl.matte.bayangan)
        self.assertTrue(np.array_equal(rig.out, rig.scene))
        rig.ctrl.picu(rig.t)
        rig.run_until(rig.t + REAPPEAR_END + DT)
        self.assertEqual(rig.ctrl.state, "LIVE")
        self.assertTrue(np.array_equal(rig.out, rig.frame))


if __name__ == "__main__":
    unittest.main()
