"""SNAP mode for Retrolens: the person in front of the camera turns to dust after a finger snap.

The code lives in the snap package; this module keeps the old public names importable and runs the
--uji preview sheet from a photo.
"""

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from snap.background import CAPTURE_MESSAGES, CAPTURE_OK, BackgroundEngine, dilate_to  # noqa: E402,F401
from snap.config import CONFIG, MODE_TRIGGER, validasi_config  # noqa: E402,F401
from snap.ash import AshAmbient  # noqa: E402,F401
from snap.config import hex_ke_bgr  # noqa: E402,F401
from snap.dissolve import DissolveRenderer, noise_fraktal, normalisasi, smoothstep  # noqa: E402,F401
from snap.flash import FlashOverlay, kurva_flash  # noqa: E402,F401
from snap.gestures import SnapGestureDetector  # noqa: E402,F401
from snap.matte import MatteEngine  # noqa: E402,F401
from snap.controller import (MSG_CANT_SEE, MSG_ESTIMATED, MSG_NEED_ROOM, MSG_NO_PERSON, STYLE_NAMES,  # noqa: E402,F401
                             SnapController)  # noqa: E402,F401
from snap.segmenter import PersonSegmenter  # noqa: E402,F401
from snap.clock import EffectClock  # noqa: E402,F401
from snap.audio import AudioSnapDetector, OnsetAudio, PemicuGabungan  # noqa: E402,F401


def buat_lembar_uji(path_gambar: str, path_bg: Optional[str], path_keluar: Optional[str],
                    cfg: Dict[str, Any]) -> int:
    gambar = cv2.imread(path_gambar)
    if gambar is None:
        print("Cannot read the photo: %s" % path_gambar)
        return 1
    path_seg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfie_segmenter.tflite")
    if not os.path.isfile(path_seg):
        print("Model file 'selfie_segmenter.tflite' not found. Put it here: %s" % path_seg)
        return 1

    ukuran_proses = (cfg["proses_lebar"], cfg["proses_tinggi"])
    if not path_bg:
        # One photo never shows the room behind the person, so the preview estimates it.
        print("No --bg given: the background behind the person is estimated (FSR inpainting).")
        cfg = dict(cfg, allow_inpaint_fallback=True)
    ctrl = SnapController(cfg)
    seg = PersonSegmenter(path_seg, cfg)
    try:
        kecil = cv2.resize(gambar, ukuran_proses, interpolation=cv2.INTER_AREA)
        mask = seg.perbarui(kecil, 0)
    finally:
        seg.close()
    if path_bg:
        bg = cv2.imread(path_bg)
        if bg is None:
            print("Cannot read the background photo: %s" % path_bg)
            return 1
        ctrl.bg.set_plate(cv2.resize(bg, (gambar.shape[1], gambar.shape[0]), interpolation=cv2.INTER_AREA))
    # Show the still photo long enough for every visible background pixel to become known.
    langkah = cfg["bg_known_samples"] + cfg["bg_safe_frames"] + 1
    for i in range(langkah):
        ctrl.amati(i / 30.0, gambar, kecil, mask)
    t0 = langkah / 30.0
    if not ctrl.picu(t0):
        print(ctrl.pesan(t0))
        return 1

    tinggi, lebar = gambar.shape[:2]
    kolom = int(math.ceil(math.sqrt(cfg["uji_jumlah_frame"])))
    baris = int(math.ceil(cfg["uji_jumlah_frame"] / float(kolom)))
    ubin_l, ubin_t = lebar // kolom, tinggi // kolom
    lembar = np.zeros((ubin_t * baris, ubin_l * kolom, 3), dtype=np.uint8)
    for i in range(cfg["uji_jumlah_frame"]):
        t = i * cfg["uji_selang_detik"]
        ubin = cv2.resize(ctrl.render(t0 + t, gambar), (ubin_l, ubin_t), interpolation=cv2.INTER_AREA)
        teks = "t = %.1f s  %s" % (t, ctrl.state)
        cv2.putText(ubin, teks, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ubin, teks, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        b, k = divmod(i, kolom)
        lembar[b * ubin_t:(b + 1) * ubin_t, k * ubin_l:(k + 1) * ubin_l] = ubin

    if not path_keluar:
        dasar, _ = os.path.splitext(path_gambar)
        path_keluar = dasar + "_lembar.png"
    if not cv2.imwrite(path_keluar, lembar):
        print("Cannot save: %s" % path_keluar)
        return 1
    print("Preview sheet saved to %s (%d dust particles)" % (path_keluar, ctrl.debu.count))
    return 0


def main_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Try the SNAP effect without a webcam.")
    parser.add_argument("--uji", required=True, metavar="PHOTO", help="photo of the person who vanishes")
    parser.add_argument("--bg", metavar="PHOTO", help="photo of the empty room from the same camera spot")
    parser.add_argument("--keluar", metavar="PATH", help="where to save the preview sheet")
    args = parser.parse_args(argv)
    salah = validasi_config(CONFIG)
    if salah:
        print("Invalid CONFIG:\n  - " + "\n  - ".join(salah))
        return 1
    return buat_lembar_uji(args.uji, args.bg, args.keluar, CONFIG)


if __name__ == "__main__":
    sys.exit(main_cli())
