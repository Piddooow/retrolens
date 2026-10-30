import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import snap_effect as se  # noqa: E402
from snap import hud  # noqa: E402
from snap.background import CAPTURE_MESSAGES  # noqa: E402
from test_background import room  # noqa: E402
from test_controller_v2 import CFG, END, FLASH_END, FakeSegmenter  # noqa: E402
from test_dissolve import scene_with_person  # noqa: E402
from test_gestures_v2 import FakeResult  # noqa: E402

DT = 1.0 / 30
FADE = CFG["hud_fade_s"]


class TestPudar(unittest.TestCase):
    def test_fades_in_and_out(self):
        p = hud.Pudar(FADE)
        p.set("Hello", 0.0)
        self.assertIsNone(p.get(0.0))
        self.assertAlmostEqual(p.get(FADE / 2)[1], 0.5)
        self.assertEqual(p.get(FADE), ("Hello", 1.0))
        p.set("", 1.0)
        self.assertAlmostEqual(p.get(1.0 + FADE / 2)[1], 0.5)
        self.assertIsNone(p.get(1.0 + FADE))

    def test_new_text_waits_for_the_old_one(self):
        p = hud.Pudar(FADE)
        p.set("A", 0.0)
        p.set("B", 1.0)
        self.assertEqual(p.get(1.0 + FADE / 2)[0], "A")
        self.assertIsNone(p.get(1.0 + FADE))
        self.assertEqual(p.get(1.0 + 2 * FADE), ("B", 1.0))

    def test_never_pops(self):
        p = hud.Pudar(FADE)
        rng = np.random.default_rng(6)
        prev = ("", 0.0)
        for i in range(600):
            now = i * DT
            if rng.random() < 0.08:
                p.set(str(rng.choice(["", "A", "B", "C"])), now)
            cur = p.get(now) or ("", 0.0)
            # smoothstep over FADE moves at most 1.5 * DT / FADE per frame.
            self.assertLessEqual(abs(cur[1] - prev[1]), 1.5 * DT / FADE + 1e-6, "frame %d" % i)
            if cur[0] != prev[0] and cur[1] > 0.0 and prev[1] > 0.0:
                self.assertLess(cur[1] + prev[1], 0.15, "text swapped while visible at frame %d" % i)
            prev = cur


class TestDrawing(unittest.TestCase):
    def test_pill_sits_top_left_and_hides_at_zero(self):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        hud.pil(img, [("Snap", hud.PUTIH), ("Ready", hud.HIJAU)], 0.0)
        self.assertFalse(img.any())
        hud.pil(img, [("Snap", hud.PUTIH), ("Ready", hud.HIJAU)], 1.0)
        ys, xs = np.nonzero(img.max(axis=2))
        self.assertLess(int(ys.max()), 60)
        self.assertLess(int(xs.max()), 320)

    def test_long_text_shrinks_to_fit(self):
        img = np.zeros((360, 320, 3), dtype=np.uint8)
        hud.toast(img, se.MSG_NEED_ROOM, 1.0)
        ys, xs = np.nonzero(img.max(axis=2) > 200)
        self.assertGreater(len(xs), 0)
        self.assertGreater(int(xs.min()), 0)
        self.assertLess(int(xs.max()), 319)


class TestSnapHud(unittest.TestCase):
    def setUp(self):
        import main as app
        self.app = app
        self.snap = app.ModeSnap(CFG, FakeSegmenter(), app.Profiler(30), hold_aktif=True)
        self.snap.masuk(0.0)
        self.scene = room(31)
        self.frame = scene_with_person(self.scene)
        self.t = 1.2

    def feed(self, label, seconds):
        out = None
        for _ in range(int(round(seconds / DT))):
            self.t += DT
            out = self.snap.proses(self.t, self.frame, FakeResult(label), int(self.t * 1000))
        return out

    def test_hint_follows_the_state(self):
        self.feed("None", 0.2)
        self.assertIn("learn the room", self.snap._petunjuk())
        self.snap.ctrl.bg.set_plate(self.scene)
        self.feed("None", 0.2)
        self.assertTrue(self.snap._petunjuk().startswith("Snap your fingers"))
        self.snap.tombol("s", self.t)
        self.feed("None", 0.5)
        self.assertEqual(self.snap._petunjuk(), "")
        self.feed("None", FLASH_END)
        self.assertIn("fist", self.snap._petunjuk())
        self.feed("None", END)
        self.assertIn("come back", self.snap._petunjuk())

    def test_hud_never_touches_the_recorded_frame(self):
        out = self.feed("Thumb_Up", 0.5)
        kept = out.copy()
        layar = out.copy()
        self.snap.gambar_hud(layar, self.t, 30.0, (True, 12))
        self.assertTrue(np.array_equal(out, kept))
        self.assertFalse(np.array_equal(layar, kept))

    def test_abandoned_hold_ring_fades_out(self):
        self.feed("Thumb_Up", 0.5)
        self.assertTrue(self.snap._cincin)
        # The hold survives short gaps; once it has really ended the ring starts to fade.
        for _ in range(30):
            self.feed("None", DT)
            if not self.snap._cincin:
                break
        self.assertEqual(self.snap._cincin, [])
        self.assertEqual(len(self.snap._cincin_pudar), 1)
        img = self.frame.copy()
        self.snap.gambar_hud(img, self.t + FADE / 2, 30.0)
        self.assertEqual(len(self.snap._cincin_pudar), 1)
        self.snap.gambar_hud(img, self.t + FADE + DT, 30.0)
        self.assertEqual(self.snap._cincin_pudar, {})


class TestScreenTextLanguage(unittest.TestCase):
    MARKERS = ("tidak", "jentik", "tekan", "mikrofon", "kalibrasi", "merekam", "gagal", "keluar", "ganti", "orang")

    def test_screen_texts_are_english(self):
        import main as app
        texts = list(app.PETUNJUK_SNAP.values()) + list(app.NAMA_STATE.values()) + list(app.NAMA_TRIGGER.values())
        texts += list(app.BANTUAN_KARTU) + [app.PETUNJUK_PORTAL, app.MSG_GESTURE_MISSING]
        texts += [se.MSG_NEED_ROOM, se.MSG_ESTIMATED, se.MSG_CANT_SEE, se.MSG_NO_PERSON]
        texts += list(CAPTURE_MESSAGES.values()) + list(se.STYLE_NAMES.values())
        for teks in texts:
            for marker in self.MARKERS:
                self.assertNotIn(marker, teks.lower(), teks)


if __name__ == "__main__":
    unittest.main()
