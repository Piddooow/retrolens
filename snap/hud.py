"""HUD drawing helpers. The HUD is drawn last and never enters recordings."""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from snap.dissolve import smoothstep

PUTIH = (255, 255, 255)


def cincin(img: np.ndarray, pusat: Tuple[float, float], bagian: float, radius: int, tebal: int,
           alpha: float) -> None:
    """Ring arc that fills clockwise from the top, blended over img at alpha (in place)."""
    if bagian <= 0.0 or alpha <= 0.0:
        return
    h, w = img.shape[:2]
    cx, cy = int(round(pusat[0])), int(round(pusat[1]))
    m = radius + tebal + 2
    x0, y0, x1, y1 = max(0, cx - m), max(0, cy - m), min(w, cx + m + 1), min(h, cy + m + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    lapisan = roi.copy()
    # Rotated by -90 degrees, angle 0 points up; OpenCV angles grow clockwise on screen.
    cv2.ellipse(lapisan, (cx - x0, cy - y0), (radius, radius), -90.0, 0.0, 360.0 * min(1.0, bagian),
                PUTIH, tebal, cv2.LINE_AA)
    roi[:] = cv2.addWeighted(lapisan, alpha, roi, 1.0 - alpha, 0.0)


def cincin_tahan(img: np.ndarray, pusat: Tuple[float, float], bagian: float, cfg: Dict[str, Any]) -> None:
    """Progress ring of a hold gesture in progress."""
    cincin(img, pusat, bagian, cfg["ring_radius_px"], cfg["ring_stroke_px"], cfg["ring_alpha"])


def denyut(img: np.ndarray, pusat: Tuple[float, float], dt: float, cfg: Dict[str, Any]) -> bool:
    """Full ring that grows to ring_pulse_px after a hold fired dt seconds ago, then fades out.

    Returns False once it has faded completely.
    """
    naik, pudar = cfg["ring_pulse_s"], cfg["hud_fade_s"]
    if dt >= naik + pudar:
        return False
    besar = cfg["ring_pulse_px"] - cfg["ring_radius_px"]
    radius = cfg["ring_radius_px"] + besar * float(smoothstep(dt / naik))
    alpha = cfg["ring_alpha"] * (1.0 - float(smoothstep((dt - naik) / pudar)))
    cincin(img, pusat, 1.0, int(round(radius)), cfg["ring_stroke_px"], alpha)
    return True


FONT = cv2.FONT_HERSHEY_SIMPLEX
GELAP = (24, 24, 24)
TITIK = (150, 150, 150)
PILL_ALPHA = 0.55
HIJAU = (150, 215, 150)
KUNING = (90, 195, 245)
MERAH = (80, 80, 235)


def kotak_bulat(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, radius: int, warna: Tuple[int, int, int]) -> None:
    """Filled rectangle with anti aliased rounded corners (in place)."""
    r = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), warna, -1)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), warna, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, warna, -1, cv2.LINE_AA)


def _panel(img: np.ndarray, baris: Tuple[str, ...], pusat_x: int, atas: int, skala: float, jarak: int, pad: int,
           radius: int, alpha_kartu: float, alpha_teks: float) -> None:
    """Dark rounded panel with white lines of text, blended over img."""
    if alpha_teks <= 0.0:
        return
    h, w = img.shape[:2]
    lebar = max(cv2.getTextSize(b, FONT, skala, 1)[0][0] for b in baris if b) if any(baris) else 0
    if lebar > w - 2 * pad - 24:
        # Shrink the text to fit a narrow frame instead of cutting it off.
        skala *= (w - 2 * pad - 24) / float(lebar)
        lebar = max(cv2.getTextSize(b, FONT, skala, 1)[0][0] for b in baris if b)
    tinggi_baris = cv2.getTextSize("Ag", FONT, skala, 1)[0][1]
    x0 = max(0, pusat_x - lebar // 2 - pad)
    x1 = min(w - 1, pusat_x + lebar // 2 + pad)
    y0 = max(0, atas)
    y1 = min(h - 1, atas + 2 * pad + tinggi_baris + (len(baris) - 1) * jarak)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1 + 1, x0:x1 + 1]
    kartu = roi.copy()
    kotak_bulat(kartu, 0, 0, x1 - x0, y1 - y0, radius, GELAP)
    roi[:] = cv2.addWeighted(kartu, alpha_kartu, roi, 1.0 - alpha_kartu, 0.0)
    teks = roi.copy()
    for i, b in enumerate(baris):
        if b:
            cv2.putText(teks, b, (pad, pad + tinggi_baris + i * jarak), FONT, skala, PUTIH, 1, cv2.LINE_AA)
    roi[:] = cv2.addWeighted(teks, alpha_teks, roi, 1.0 - alpha_teks, 0.0)


def toast(img: np.ndarray, teks: str, alpha: float) -> None:
    """Message centered above the hint line, at opacity alpha (0..1)."""
    h, w = img.shape[:2]
    _panel(img, (teks,), w // 2, h - 112, 0.65, 0, 12, 14, PILL_ALPHA * alpha, alpha)


def petunjuk(img: np.ndarray, teks: str, alpha: float) -> None:
    """The one context hint line at the bottom, at opacity alpha (0..1)."""
    h, w = img.shape[:2]
    _panel(img, (teks,), w // 2, h - 54, 0.55, 0, 10, 12, PILL_ALPHA * alpha, alpha)


def tengah(img: np.ndarray, teks: str, alpha: float) -> None:
    """Large centered text (countdowns and calibration steps), at opacity alpha (0..1)."""
    h, w = img.shape[:2]
    _panel(img, (teks,), w // 2, h // 2 - 30, 1.0, 0, 18, 18, PILL_ALPHA * alpha, alpha)


def pil(img: np.ndarray, bagian: List[Tuple[str, Tuple[int, int, int]]], alpha: float) -> None:
    """Status pill at the top left: colored text parts separated by small dots, at opacity alpha."""
    if alpha <= 0.0 or not bagian:
        return
    skala, pad_x, pad_y, jarak = 0.55, 12, 9, 18
    lebar = [cv2.getTextSize(t, FONT, skala, 1)[0][0] for t, _ in bagian]
    tinggi = cv2.getTextSize("Ag", FONT, skala, 1)[0][1]
    x0, y0 = 12, 12
    x1 = x0 + 2 * pad_x + sum(lebar) + jarak * (len(bagian) - 1)
    y1 = y0 + 2 * pad_y + tinggi + 4
    h, w = img.shape[:2]
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    roi = img[y0:y1 + 1, x0:x1 + 1]
    kartu = roi.copy()
    kotak_bulat(kartu, 0, 0, x1 - x0, y1 - y0, (y1 - y0) // 2, GELAP)
    a = PILL_ALPHA * alpha
    roi[:] = cv2.addWeighted(kartu, a, roi, 1.0 - a, 0.0)
    teks = roi.copy()
    x = pad_x
    base = pad_y + tinggi
    for i, ((t, warna), lb) in enumerate(zip(bagian, lebar)):
        if i:
            cv2.circle(teks, (x - jarak // 2, base - tinggi // 2), 2, TITIK, -1, cv2.LINE_AA)
        cv2.putText(teks, t, (x, base), FONT, skala, warna, 1, cv2.LINE_AA)
        x += lb + jarak
    roi[:] = cv2.addWeighted(teks, alpha, roi, 1.0 - alpha, 0.0)


class Pudar:
    """One line of text that fades in and out over fade_s with an ease in out curve.

    A new text waits until the old one has faded out, then fades in, so text never swaps while visible.
    Setting "" fades the current text out.
    """

    def __init__(self, fade_s: float):
        self.fade = fade_s
        self.teks = ""
        self.t_masuk = -math.inf
        self.lama = None  # type: Optional[Tuple[str, float, float]]

    def _tampak(self, now: float) -> Tuple[str, float]:
        """(text, linear level 0..1) that is visible now."""
        if self.lama is not None:
            teks, level, sejak = self.lama
            sisa = level - (now - sejak) / self.fade
            if sisa > 1e-9:
                return teks, sisa
            self.lama = None
        if not self.teks:
            return "", 0.0
        return self.teks, min(1.0, max(0.0, (now - self.t_masuk) / self.fade))

    def set(self, teks: str, now: float) -> None:
        if teks == self.teks and self.lama is None:
            return
        tampak, level = self._tampak(now)
        if teks and teks == tampak:
            # Back to the text that is fading out: fade it up again from where it is.
            self.teks, self.t_masuk, self.lama = teks, now - level * self.fade, None
            return
        self.lama = (tampak, level, now) if level > 0.0 else None
        self.teks = teks
        self.t_masuk = now + level * self.fade

    def get(self, now: float) -> Optional[Tuple[str, float]]:
        """(text, opacity) to draw now, or None."""
        teks, level = self._tampak(now)
        if not teks or level <= 0.0:
            return None
        return teks, float(smoothstep(level))


def kartu_bantuan(img: np.ndarray, baris: Tuple[str, ...], alpha: float, cfg: Dict[str, Any]) -> None:
    """Centered help card at opacity alpha (0..1)."""
    h, w = img.shape[:2]
    skala, jarak, pad = 0.6, 30, 24
    tinggi = 2 * pad + (len(baris) - 1) * jarak + 14
    _panel(img, baris, w // 2, max(0, (h - tinggi) // 2), skala, jarak, pad, cfg["help_radius_px"],
           cfg["help_alpha"] * alpha, alpha)
