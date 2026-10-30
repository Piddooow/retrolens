"""HUD drawing helpers. The HUD is drawn last and never enters recordings."""

from typing import Any, Dict, Tuple

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
