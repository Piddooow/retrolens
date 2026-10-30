"""Radial flash from the snap point, with the curve measured from the reference snap."""

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from snap.config import hex_ke_bgr


def kurva_flash(t: Any, cfg: Dict[str, Any]) -> np.ndarray:
    """Kekuatan flash 0..1 pada t detik sejak jentikan; t boleh skalar atau array."""
    t = np.asarray(t, dtype=np.float32)
    naik, tahan, turun = cfg["flash_naik_detik"], cfg["flash_tahan_detik"], cfg["flash_turun_detik"]
    xn, yn = cfg["flash_naik_tabel"]
    xt, yt = cfg["flash_turun_tabel"]
    # np.interp menahan nilai di ujung tabel, jadi cukup dipotong per fase.
    s_naik = np.interp(t / naik, xn, yn)
    s_turun = np.interp((t - naik - tahan) / turun, xt, yt)
    hasil = np.where(t < naik, s_naik, np.where(t < naik + tahan, 1.0, s_turun))
    hasil = np.where((t < 0) | (t >= naik + tahan + turun), 0.0, hasil)
    return hasil.astype(np.float32)


class FlashOverlay:
    """Flash radial dari titik jentikan, kurva dari pengukuran reference snap."""

    def __init__(self, cfg: Dict[str, Any], lebar: int, tinggi: int):
        self.cfg = cfg
        self.lebar, self.tinggi = lebar, tinggi
        self.dekat = hex_ke_bgr(cfg["flash_warna_dekat"])
        self.jauh = hex_ke_bgr(cfg["flash_warna_jauh"])
        naik = cfg["flash_naik_detik"]
        self.mulai_putih = naik + cfg["flash_tunda_ujung_detik"]
        self.akhir_putih = naik + cfg["flash_tahan_detik"]
        self.akhir = self.akhir_putih + cfg["flash_turun_detik"] + cfg["flash_tunda_ujung_detik"]
        self.jarak = np.zeros((tinggi, lebar), dtype=np.float32)
        self.warna_dasar = np.zeros((tinggi, lebar, 3), dtype=np.float32)

    def mulai(self, asal: Tuple[float, float]) -> None:
        """asal dalam koordinat ternormalisasi 0..1."""
        ys, xs = np.mgrid[0:self.tinggi, 0:self.lebar].astype(np.float32) + 0.5
        d = np.hypot(xs - asal[0] * self.lebar, ys - asal[1] * self.tinggi)
        self.jarak = (d / d.max()).astype(np.float32)
        j = self.jarak[..., None]
        self.warna_dasar = (self.dekat * (1.0 - j) + self.jauh * j).astype(np.float32)

    def kekuatan(self, t: float) -> np.ndarray:
        return kurva_flash(t - self.jarak * self.cfg["flash_tunda_ujung_detik"], self.cfg)

    def putih_penuh(self, t: float) -> bool:
        return self.mulai_putih <= t < self.akhir_putih

    def komposit(self, frame: np.ndarray, t: float) -> np.ndarray:
        if t <= 0 or t >= self.akhir:
            return frame
        if self.putih_penuh(t):
            return np.full_like(frame, 255)
        s = self.kekuatan(t)
        if float(s.max()) <= 0.0:
            return frame
        ambang = self.cfg["flash_ambang_putih"]
        ke_putih = np.clip((s - ambang) / (1.0 - ambang), 0.0, 1.0)[..., None]
        warna = self.warna_dasar * (1.0 - ke_putih) + 255.0 * ke_putih
        tinggi, lebar = frame.shape[:2]
        # Lapisan flash halus, jadi dihitung kecil lalu diperbesar supaya frame tetap tajam.
        if (lebar, tinggi) != (self.lebar, self.tinggi):
            warna = cv2.resize(warna, (lebar, tinggi), interpolation=cv2.INTER_LINEAR)
            s = cv2.resize(s, (lebar, tinggi), interpolation=cv2.INTER_LINEAR)
        # frame x (1 - s) + flash color x s, done by OpenCV at display resolution.
        return cv2.blendLinear(frame, cv2.convertScaleAbs(warna), 1.0 - s, s)
