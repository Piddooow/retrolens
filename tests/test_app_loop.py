import os
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as app  # noqa: E402
from test_background import person_lo, room, with_person  # noqa: E402


class FakeWindow:
    """Behaves like the macOS window: it only exists after the first imshow, and can be closed."""

    def __init__(self, keys=None, close_after=None):
        self.shown = 0
        self.calls = 0
        self.keys = keys or {}
        self.close_after = close_after

    def imshow(self, name, img):
        self.shown += 1

    def waitKey(self, ms):
        self.calls += 1
        return self.keys.get(self.calls, 255)

    def getWindowProperty(self, name, prop):
        if self.shown == 0:
            raise cv2.error("window does not exist yet")
        if self.close_after is not None and self.shown >= self.close_after:
            return 0.0
        return 1.0


class TestAppLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.video = os.path.join(cls.tmp, "clip.mp4")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
        frame = with_person(room(5), person_lo())
        for _ in range(90):
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_app(self, window):
        saved = (app.cv2.imshow, app.cv2.waitKey, app.cv2.getWindowProperty, app.cv2.destroyAllWindows, sys.argv)
        app.cv2.imshow, app.cv2.waitKey = window.imshow, window.waitKey
        app.cv2.getWindowProperty, app.cv2.destroyAllWindows = window.getWindowProperty, lambda: None
        sys.argv = ["main.py", "--sumber", self.video, "--snap"]
        try:
            return app.main()
        finally:
            (app.cv2.imshow, app.cv2.waitKey, app.cv2.getWindowProperty, app.cv2.destroyAllWindows,
             sys.argv) = saved

    def test_keeps_running_until_q(self):
        window = FakeWindow(keys={30: ord("q")})
        self.assertEqual(self.run_app(window), 0)
        # It must not stop on the first loop, before the first frame could be shown.
        self.assertEqual(window.calls, 30)
        self.assertGreater(window.shown, 20)

    def test_closing_the_window_quits(self):
        window = FakeWindow(close_after=10)
        self.assertEqual(self.run_app(window), 0)
        self.assertEqual(window.shown, 10)


if __name__ == "__main__":
    unittest.main()
