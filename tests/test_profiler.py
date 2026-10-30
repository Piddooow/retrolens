import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snap.profiler import STAGES, Profiler  # noqa: E402


class TestProfiler(unittest.TestCase):
    def test_skipped_stage_counts_as_zero(self):
        prof = Profiler(4)
        for i in range(4):
            with prof.stage("hand"):
                pass
            if i == 0:
                prof._frame["render"] = 8.0
            prof.end_frame()
        self.assertAlmostEqual(prof.averages()["render"], 2.0)

    def test_window_drops_old_frames(self):
        prof = Profiler(2)
        for ms in (100.0, 1.0, 3.0):
            prof._frame["capture"] = ms
            prof.end_frame()
        self.assertAlmostEqual(prof.averages()["capture"], 2.0)

    def test_stage_adds_up_within_frame(self):
        prof = Profiler(3)
        with prof.stage("render"):
            pass
        first = prof._frame["render"]
        with prof.stage("render"):
            pass
        self.assertGreaterEqual(prof._frame["render"], first)
        prof.end_frame()
        self.assertEqual(list(prof.averages())[:len(STAGES)], list(STAGES))


if __name__ == "__main__":
    unittest.main()
