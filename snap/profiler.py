"""Per stage timing with a rolling average, shown in the debug overlay."""

import collections
import contextlib
import time
from typing import Deque, Dict, Iterator, List

STAGES = ("capture", "hand", "segmenter", "matte", "background", "render", "hud", "display")


class Profiler:
    def __init__(self, window: int):
        self.window = window
        self._history = {}  # type: Dict[str, Deque[float]]
        self._frame = {}  # type: Dict[str, float]
        for name in STAGES:
            self._history[name] = collections.deque(maxlen=window)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._frame[name] = self._frame.get(name, 0.0) + (time.perf_counter() - start) * 1000.0

    def add(self, name: str, ms: float) -> None:
        """Record a stage measured elsewhere, for example on the inference thread."""
        self._frame[name] = self._frame.get(name, 0.0) + ms

    def end_frame(self) -> None:
        # A stage that did not run this frame counts as 0 ms, so averages show the real per frame cost.
        for name in set(self._history) | set(self._frame):
            if name not in self._history:
                self._history[name] = collections.deque(maxlen=self.window)
            self._history[name].append(self._frame.get(name, 0.0))
        self._frame = {}

    def averages(self) -> Dict[str, float]:
        order = list(STAGES) + sorted(n for n in self._history if n not in STAGES)
        return {n: sum(self._history[n]) / len(self._history[n]) for n in order if self._history[n]}

    def total(self) -> float:
        return sum(self.averages().values())

    def summary(self) -> List[str]:
        return ["%s %.1f" % (name, ms) for name, ms in self.averages().items()]
