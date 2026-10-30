"""EffectClock: effect time that runs at an eased rate (1 normal, 0 frozen, slow_rate for slow motion)."""

import math
from typing import Any, Dict


class EffectClock:
    """Effect seconds driven by time.monotonic() and a smoothly changing rate.

    The rate follows its target as a critically damped system with time constant rate_tau_s, so it
    starts and stops moving with zero slope and never jumps (at 30 fps no step is above 0.07 of the
    full rate). The update uses the exact solution, so it does not depend on the frame rate.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.start(0.0)

    def start(self, now: float) -> None:
        self.t = 0.0
        self.rate = 1.0
        self._speed = 0.0
        self._last = now

    def update(self, now: float, target: float) -> float:
        """Advance to now with the rate easing toward target; returns the effect time."""
        dt = max(0.0, now - self._last)
        self._last = now
        if dt == 0.0:
            return self.t
        omega = 1.0 / self.cfg["rate_tau_s"]
        e0, v0 = self.rate - target, self._speed
        decay = math.exp(-omega * dt)
        drift = (v0 + omega * e0) * dt
        e1 = (e0 + drift) * decay
        self._speed = (v0 - omega * drift) * decay
        rate = min(1.0, max(0.0, target + e1))
        # Trapezoid rule, so slowing down and speeding up shift the time symmetrically.
        self.t += 0.5 * (self.rate + rate) * dt
        self.rate = rate
        return self.t
