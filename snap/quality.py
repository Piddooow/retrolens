"""Adaptive quality ladder: cheaper settings while the frame rate stays low, back again once it recovers."""

from typing import Any, Dict, Optional

# Steps in the order they are taken: matte refined at quarter size, larger dust cells, no splat blur,
# no shadow zone.
LEVEL_MAX = 4


class QualityLadder:
    """Picks a quality level from the measured frame rate.

    A step down happens after fps stays below fps_bad for fps_bad_hold_s; a step back up after fps stays
    above fps_good for fps_good_hold_s. One step at a time. The controller applies the level only when
    an effect starts, so nothing changes in the middle of one.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.level = 0
        self._buruk_sejak = None  # type: Optional[float]
        self._baik_sejak = None  # type: Optional[float]

    def perbarui(self, now: float, fps: float) -> None:
        cfg = self.cfg
        if fps < cfg["fps_bad"]:
            self._baik_sejak = None
            if self._buruk_sejak is None:
                self._buruk_sejak = now
            elif now - self._buruk_sejak >= cfg["fps_bad_hold_s"] and self.level < LEVEL_MAX:
                self.level += 1
                self._buruk_sejak = now
        elif fps > cfg["fps_good"]:
            self._buruk_sejak = None
            if self._baik_sejak is None:
                self._baik_sejak = now
            elif now - self._baik_sejak >= cfg["fps_good_hold_s"] and self.level > 0:
                self.level -= 1
                self._baik_sejak = now
        else:
            self._buruk_sejak = None
            self._baik_sejak = None

    def settings(self) -> Dict[str, Any]:
        """Engine settings for the current level."""
        cfg, level = self.cfg, self.level
        return {
            "matte_scale": cfg["ladder_matte_scale"] if level >= 1 else cfg["matte_scale"],
            "cell": cfg["debu_ukuran_butir_turun"] if level >= 2 else cfg["particle_cell_px"],
            "dust_blur": 0.0 if level >= 3 else cfg["dust_blur"],
            "shadow": False if level >= 4 else cfg["shadow_on"],
        }
