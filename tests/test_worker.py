import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snap.worker import InferenceWorker  # noqa: E402


class Recorder:
    def __init__(self, delay=0.0, gate=None):
        self.delay = delay
        self.gate = gate
        self.seen = []
        self.started = threading.Event()

    def __call__(self, job):
        self.started.set()
        if self.gate is not None:
            self.gate.wait(2.0)
        self.seen.append(job)
        time.sleep(self.delay)
        if job == "boom":
            raise ValueError("bad frame")
        return ("result of", job)


class TestInferenceWorker(unittest.TestCase):
    def worker(self, fn):
        w = InferenceWorker(fn)
        self.addCleanup(w.tutup)
        return w

    def test_result_stays_with_its_job(self):
        rec = Recorder(delay=0.05)
        w = self.worker(rec)
        n = w.kirim("frame 1")
        rec.started.wait(1.0)
        w.kirim("frame 2")
        nomor, job, hasil = w.ambil(0, 2.0)
        self.assertEqual(nomor, n)
        self.assertEqual(hasil, ("result of", job))
        self.assertEqual(job, "frame 1")

    def test_does_not_wait_for_the_newest_job(self):
        gate = threading.Event()
        w = self.worker(Recorder(gate=gate))
        w.kirim("frame 1")
        start = time.monotonic()
        self.assertIsNone(w.ambil(0, 1.0))
        self.assertLess(time.monotonic() - start, 0.2)
        gate.set()

    def test_waits_for_the_older_job_in_flight(self):
        rec = Recorder(delay=0.2)
        w = self.worker(rec)
        w.kirim("frame 1")
        rec.started.wait(1.0)
        w.kirim("frame 2")
        start = time.monotonic()
        got = w.ambil(0, 2.0)
        self.assertIsNotNone(got)
        self.assertEqual(got[1], "frame 1")
        self.assertGreater(time.monotonic() - start, 0.1)
        # Frame 2 is now the newest one: it is not waited for.
        start = time.monotonic()
        self.assertIsNone(w.ambil(got[0], 1.0))
        self.assertLess(time.monotonic() - start, 0.1)

    def test_latest_frame_wins_and_order_is_kept(self):
        gate = threading.Event()
        rec = Recorder(gate=gate)
        w = self.worker(rec)
        w.kirim(1)
        time.sleep(0.05)
        for job in (2, 3, 4, 5):
            w.kirim(job)
        gate.set()
        deadline = time.monotonic() + 2.0
        while len(rec.seen) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
        # Job 1 was running; 2, 3 and 4 were replaced while waiting; only 5 ran after it.
        self.assertEqual(rec.seen, [1, 5])
        nomor, job, _ = w.ambil(0, 1.0)
        self.assertEqual((nomor, job), (5, 5))

    def test_error_reaches_the_caller(self):
        rec = Recorder(delay=0.05)
        w = self.worker(rec)
        w.kirim("boom")
        rec.started.wait(1.0)
        w.kirim("next")
        with self.assertRaises(RuntimeError):
            w.ambil(0, 2.0)

    def test_close_stops_the_thread(self):
        w = InferenceWorker(Recorder())
        w.tutup()
        self.assertFalse(w._thread.is_alive())
        self.assertIsNone(w.ambil(0, 0.1))


if __name__ == "__main__":
    unittest.main()
