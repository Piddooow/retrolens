"""BackgroundEngine: learns the empty room at display resolution so the user can vanish cleanly.

The engine never guesses: pixels it has not seen without the person stay unknown, and the
controller refuses to vanish until the room behind the user is known (coverage gate).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

CAPTURE_OK = "ok"
CAPTURE_PERSON = "person"
CAPTURE_MOVING = "moving"
CAPTURE_MISMATCH = "mismatch"

MSG_LEARNED = "Background learned"
MSG_MOVED = "Camera moved. Step out for a moment so I can relearn the room"
CAPTURE_MESSAGES = {
    CAPTURE_OK: "Background learned",
    CAPTURE_PERSON: "Someone is still in the frame. Step out and press B again",
    CAPTURE_MOVING: "The picture kept moving. Keep the camera still and press B again",
    CAPTURE_MISMATCH: "That did not look like the room. Do not cover the camera, then press B again",
}


def ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def dilate_to(mask_lo: np.ndarray, radius_px: float, size: Tuple[int, int]) -> np.ndarray:
    """Dilate a low resolution mask and upsample it to `size` (width, height) as bool.

    Dilating at low resolution is about 10x cheaper. One extra low resolution pixel covers
    everything a bilinear upsample of the source could mark, so the result is never smaller
    than a full resolution dilation by `radius_px`.
    """
    scale = size[0] / float(mask_lo.shape[1])
    radius = int(math.ceil(radius_px / scale)) + 1
    grown = cv2.dilate(mask_lo.astype(np.uint8), ellipse(radius))
    return cv2.resize(grown, size, interpolation=cv2.INTER_NEAREST) > 0


class BackgroundEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.size = None  # type: Optional[Tuple[int, int]]
        self._rng = np.random.default_rng(cfg["debu_seed"] + 101)
        self._dither = np.random.default_rng(cfg["debu_seed"] + 102)
        self._hann = cv2.createHanningWindow(tuple(cfg["bg_moved_grid"]), cv2.CV_32F)
        # Unit noise tiles are drawn first, here, so the first vanish frame does not pay for them.
        tile_px = cfg["bg_noise_tile_px"]
        self._unit_tiles = self._rng.standard_normal((cfg["bg_noise_tiles"], tile_px, tile_px, 3)).astype(np.float32)
        self.sigma = np.zeros(3, dtype=np.float32)
        self.gain_small = None  # type: Optional[np.ndarray]
        self.last_check = (0.0, 1.0)

    def _allocate(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.plate = np.zeros((height, width, 3), dtype=np.uint8)
        self.count = np.zeros((height, width), dtype=np.uint8)
        self.safe = np.zeros((height, width), dtype=bool)
        self._safe_run = np.zeros((height, width), dtype=np.uint8)
        self._step = np.full((height, width, 3), self.cfg["bg_median_step"], dtype=np.uint8)
        self._prev_small = None  # type: Optional[np.ndarray]
        self._prev_sub = None  # type: Optional[np.ndarray]
        self._last_update = None  # type: Optional[float]
        self._absent_since = None  # type: Optional[float]
        self._auto_armed = True
        self._capture = None  # type: Optional[Dict[str, Any]]
        self._next_check = -math.inf
        self._moved_votes = 0
        # Plate snapshot from the last passing move check. The running median would follow a
        # moved camera before the next check, so moves are judged against this instead.
        self._anchor = None  # type: Optional[np.ndarray]
        self._next_noise = -math.inf

    def _ensure(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self.size != (w, h):
            self._allocate(w, h)

    # Knowledge

    def known(self) -> np.ndarray:
        return self.count >= self.cfg["bg_known_samples"]

    def known_fraction(self) -> float:
        return 0.0 if self.size is None else float(np.count_nonzero(self.known())) / self.count.size

    def coverage(self, zone: np.ndarray) -> float:
        """Fraction of the zone whose background is known (1.0 for an empty zone)."""
        if self.size is None:
            return 0.0
        total = int(np.count_nonzero(zone))
        return 1.0 if total == 0 else float(np.count_nonzero(zone & self.known())) / total

    def set_plate(self, image: np.ndarray) -> None:
        """Use a known empty room image (for example --uji --bg) as the fully known plate."""
        self._ensure(image)
        self.plate[:] = image
        np.maximum(self.count, self.cfg["bg_known_samples"], out=self.count)
        self._anchor = self._grid(self.plate)

    def invalidate(self) -> None:
        self.count[:] = 0
        self._safe_run[:] = 0
        self._moved_votes = 0
        self._anchor = None
        self._auto_armed = True

    # Per frame learning

    def _grid(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, tuple(self.cfg["bg_moved_grid"]), interpolation=cv2.INTER_AREA)

    def _observe(self, frame: np.ndarray, matte_lo: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        # Motion is judged on a smaller copy so sensor noise does not count as movement.
        small = self._grid(frame)
        still = False
        if self._prev_small is not None:
            still = float(np.mean(cv2.mean(cv2.absdiff(small, self._prev_small))[:3])) < self.cfg["bg_still_diff"]
        self._prev_small = small
        absent = float(np.mean(matte_lo > self.cfg["mask_ambang"])) < self.cfg["bg_absent_frac"]
        return small, still, absent

    def update(self, now: float, frame: np.ndarray, matte_lo: np.ndarray, allow_auto: bool = True) -> List[str]:
        """Learn from one display frame and the low resolution person matte. Returns user messages."""
        cfg = self.cfg
        self._ensure(frame)
        messages = []  # type: List[str]
        if self._last_update is not None and now - self._last_update > cfg["bg_frame_gap_s"]:
            # Updates paused (for example during a vanish). Frames from before the pause are not
            # consecutive, so the safe frame count and the noise frame pair start over.
            self._safe_run[:] = 0
            self._prev_sub = None
            self._prev_small = None
        self._last_update = now
        small, still, absent = self._observe(frame, matte_lo)

        unsafe = dilate_to(matte_lo >= cfg["bg_safe_conf"], cfg["bg_safe_margin_px"], self.size)
        safe_now = np.where(unsafe, np.uint8(0), np.uint8(255))
        # Consecutive safe frames per pixel; one flickering frame resets it.
        self._safe_run = cv2.bitwise_and(cv2.add(self._safe_run, 1), safe_now)
        self.safe = self._safe_run > cfg["bg_safe_frames"]
        self._update_noise(now, frame[::2, ::2].copy())
        safe_u8 = self.safe.view(np.uint8) * np.uint8(255)
        # While a camera move is suspected, do not blend the moved view into the plate.
        if self.safe.any() and self._moved_votes == 0:
            first = (self.safe & (self.count == 0)).view(np.uint8) * np.uint8(255)
            cv2.copyTo(frame, first, self.plate)
            # Approximate running median: step toward the frame by at most bg_median_step.
            up = cv2.min(cv2.subtract(frame, self.plate), self._step)
            down = cv2.min(cv2.subtract(self.plate, frame), self._step)
            cv2.add(self.plate, up, dst=self.plate, mask=safe_u8)
            cv2.subtract(self.plate, down, dst=self.plate, mask=safe_u8)
            cv2.add(self.count, 1, dst=self.count, mask=safe_u8)

        if self._capture is not None and self._capture["auto"]:
            if absent and still:
                result = self._add_capture_frame(frame)
                if result is not None:
                    self._auto_armed = False
                    if result == CAPTURE_OK:
                        messages.append(MSG_LEARNED)
                    else:
                        messages.append(CAPTURE_MESSAGES[result])
            else:
                self._capture = None
                self._absent_since = None
        elif self._capture is None:
            if not absent:
                self._auto_armed = True
            if absent and still:
                if self._absent_since is None:
                    self._absent_since = now
                elif allow_auto and self._auto_armed and now - self._absent_since >= cfg["bg_auto_capture_s"]:
                    self._capture = {"auto": True, "sum": np.zeros(frame.shape, np.float32), "n": 0, "bad": None}
            else:
                self._absent_since = None

        # A suspected move is confirmed on the next frame instead of waiting a full interval.
        if self._capture is None and (now >= self._next_check or self._moved_votes > 0):
            self._next_check = now + cfg["bg_moved_check_s"]
            if self._camera_moved(small):
                self.invalidate()
                messages.append(MSG_MOVED)
        return messages

    # Camera moved check

    def _signal(self, small: np.ndarray, weight: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        # Zero mean inside the soft mask, so the shared mask edge cannot pull the peak to zero shift.
        mag -= float((mag * weight).sum()) / max(float(weight.sum()), 1e-6)
        return mag * weight

    def compare(self, small_a: np.ndarray, small_b: np.ndarray, area: np.ndarray) -> Tuple[float, float]:
        """(shift in grid px, response) between two grid size images over a full size bool area."""
        grid = tuple(self.cfg["bg_moved_grid"])
        weight = cv2.resize(area.astype(np.float32), grid, interpolation=cv2.INTER_AREA)
        weight = cv2.GaussianBlur(weight, (0, 0), self.cfg["bg_moved_mask_blur"])
        (dx, dy), response = cv2.phaseCorrelate(self._signal(small_a, weight), self._signal(small_b, weight),
                                                self._hann)
        return float(math.hypot(dx, dy)), float(response)

    def _mismatch(self, shift: float, response: float) -> bool:
        return shift > self.cfg["bg_shift_max_px"] or response < self.cfg["bg_phase_min_response"]

    def _camera_moved(self, small: np.ndarray) -> bool:
        cfg = self.cfg
        area = self.known() & self.safe
        if float(np.count_nonzero(area)) / area.size < cfg["bg_moved_min_known"]:
            self._moved_votes = 0
            return False
        plate_small = self._grid(self.plate)
        if self._anchor is None:
            self._anchor = plate_small
        self.last_check = self.compare(self._anchor, small, area)
        # Two checks in a row must agree before the plate is thrown away.
        if self._mismatch(*self.last_check):
            self._moved_votes += 1
        else:
            self._moved_votes = 0
            self._anchor = plate_small
        return self._moved_votes >= cfg["bg_moved_confirm"]

    # Capture (automatic and manual)

    def start_capture(self) -> None:
        if self.size is None:
            self._capture = None
            return
        self._capture = {"auto": False, "sum": None, "n": 0, "bad": None}

    def cancel_capture(self) -> None:
        self._capture = None
        self._absent_since = None

    def capturing(self) -> bool:
        return self._capture is not None

    def capture_frame(self, frame: np.ndarray, matte_lo: np.ndarray) -> Optional[str]:
        """Feed one manual capture frame. None while recording, then a CAPTURE_* result."""
        self._ensure(frame)
        if self._capture is None:
            self.start_capture()
        small, still, absent = self._observe(frame, matte_lo)
        if not absent:
            self._capture["bad"] = CAPTURE_PERSON
        elif not still and self._capture["n"] > 0 and self._capture["bad"] is None:
            self._capture["bad"] = CAPTURE_MOVING
        return self._add_capture_frame(frame)

    def _add_capture_frame(self, frame: np.ndarray) -> Optional[str]:
        cap = self._capture
        if cap["sum"] is None:
            cap["sum"] = np.zeros(frame.shape, np.float32)
        cv2.accumulate(frame, cap["sum"])
        cap["n"] += 1
        needed = self.cfg["bg_auto_capture_frames"] if cap["auto"] else self.cfg["plate_jumlah_frame"]
        if cap["n"] < needed:
            return None
        self._capture = None
        if cap["bad"] is not None:
            return cap["bad"]
        average = cv2.convertScaleAbs(cap["sum"], alpha=1.0 / cap["n"])
        return self._accept(average)

    def _accept(self, average: np.ndarray) -> str:
        known = self.known()
        if float(np.count_nonzero(known)) / known.size >= self.cfg["bg_moved_min_known"]:
            shift, response = self.compare(self._grid(self.plate), self._grid(average), known)
            # A clear shift means the camera really moved, so the new picture is the right one.
            # A weak response means it does not look like the room at all (covered lens).
            if response < self.cfg["bg_phase_min_response"]:
                return CAPTURE_MISMATCH
        self.plate[:] = average
        np.maximum(self.count, self.cfg["bg_known_samples"], out=self.count)
        self._moved_votes = 0
        self._anchor = self._grid(self.plate)
        return CAPTURE_OK

    # Exposure, color, and noise matching

    def plate_adj(self, frame: np.ndarray, area: Optional[np.ndarray] = None) -> np.ndarray:
        """Plate with a smooth local gain map so it matches the live exposure and color."""
        cfg = self.cfg
        if area is None:
            area = self.known() & self.safe
        grid = tuple(cfg["bg_gain_grid"])
        w, h = self.size
        half = ((w + 1) // 2, (h + 1) // 2)
        mask = area.astype(np.uint8) * np.uint8(255)

        def cell_mean(img: np.ndarray) -> np.ndarray:
            # Halving the masked 8 bit image first gives the same cell means up to rounding, and is
            # much cheaper than float math at display size.
            small = cv2.resize(cv2.bitwise_and(img, img, mask=mask), half, interpolation=cv2.INTER_AREA)
            return cv2.resize(small.astype(np.float32), grid, interpolation=cv2.INTER_AREA)

        live = cell_mean(frame)
        plate = cell_mean(self.plate)
        weight = cv2.resize(mask, half, interpolation=cv2.INTER_AREA).astype(np.float32) * np.float32(1.0 / 255.0)
        coverage = cv2.GaussianBlur(cv2.resize(weight, grid, interpolation=cv2.INTER_AREA), (0, 0),
                                    cfg["bg_gain_sigma"])
        num = cv2.GaussianBlur(live, (0, 0), cfg["bg_gain_sigma"])
        den = cv2.GaussianBlur(plate, (0, 0), cfg["bg_gain_sigma"])
        # Cells with no data nearby fall back to the global gain instead of 1.0.
        total_plate = plate.reshape(-1, 3).sum(axis=0)
        global_gain = np.where(total_plate > 1e-3, live.reshape(-1, 3).sum(axis=0) / np.maximum(total_plate, 1e-3),
                               1.0).astype(np.float32)
        has_data = (coverage > cfg["bg_gain_min_weight"])[..., None] & (den > 1e-3)
        gain = np.where(has_data, num / np.maximum(den, 1e-3), global_gain)
        gain = np.clip(gain, cfg["bg_gain_min"], cfg["bg_gain_max"]).astype(np.float32)
        self.gain_small = gain
        full = cv2.resize(gain, self.size, interpolation=cv2.INTER_LINEAR)
        return cv2.multiply(self.plate, full, dtype=cv2.CV_8U)

    def _update_noise(self, now: float, sub: np.ndarray) -> None:
        """Re-estimate the sensor noise sigma per channel from two consecutive frames.

        Frame to frame differences on safe pixels hold only sensor noise (sigma times sqrt 2).
        Comparing against the plate would also count the jitter of the running median.
        """
        prev, self._prev_sub = self._prev_sub, sub
        if prev is None or now < self._next_noise:
            return
        pick = self.safe[::2, ::2]
        if np.count_nonzero(pick) < 1000:
            return
        self._next_noise = now + self.cfg["bg_noise_update_s"]
        diff = sub[pick].astype(np.float32) - prev[pick]
        # Dither so the median of 8 bit differences does not snap to whole levels.
        diff += self._dither.uniform(-0.5, 0.5, diff.shape).astype(np.float32)
        center = np.median(diff, axis=0)
        mad = np.median(np.abs(diff - center), axis=0)
        self.sigma = (1.4826 / math.sqrt(2.0) * mad).astype(np.float32)

    def plate_view(self, adj: np.ndarray, zone: np.ndarray) -> np.ndarray:
        """adj plus synthesized sensor noise inside the zone (float 0..1 or bool)."""
        cfg = self.cfg
        if not cfg["bg_noise_on"] or float(self.sigma.max()) <= 0.0:
            return adj
        x0, y0, w, h = cv2.boundingRect((zone > 0).view(np.uint8))
        if w == 0 or h == 0:
            return adj
        tile_px = cfg["bg_noise_tile_px"]
        tile = self._unit_tiles[int(self._rng.integers(cfg["bg_noise_tiles"]))]
        oy, ox = (int(v) for v in self._rng.integers(tile_px, size=2))
        # Pixel (y, x) gets tile[(y + oy) % tile_px, (x + ox) % tile_px]: roll once, then repeat.
        rolled = np.roll(tile, (-((y0 + oy) % tile_px), -((x0 + ox) % tile_px)), axis=(0, 1))
        noise = np.tile(rolled, (-(-h // tile_px), -(-w // tile_px), 1))[:h, :w]
        weight = zone[y0:y0 + h, x0:x0 + w].astype(np.float32)
        noise = cv2.multiply(noise, cv2.merge([weight * s for s in self.sigma]))
        out = adj.copy()
        # Saturating add: values below 0 clip to 0 (convertScaleAbs would mirror them upward).
        out[y0:y0 + h, x0:x0 + w] = cv2.add(adj[y0:y0 + h, x0:x0 + w], noise, dtype=cv2.CV_8U)
        return out

    def fill_unknown(self, adj: np.ndarray, zone: np.ndarray) -> np.ndarray:
        """Experimental: estimate unknown pixels inside the zone with FSR inpainting (never TELEA)."""
        hole = zone & ~self.known()
        if not hole.any():
            return adj
        ys, xs = np.nonzero(hole)
        pad = self.cfg["inpaint_pad_px"]
        h, w = hole.shape
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
        crop = np.ascontiguousarray(adj[y0:y1, x0:x1])
        hole_crop = hole[y0:y1, x0:x1]
        # xphoto marks VALID pixels with non-zero, the opposite of cv2.inpaint.
        valid = np.where(hole_crop, np.uint8(0), np.uint8(255))
        filled = np.zeros_like(crop)
        cv2.xphoto.inpaint(crop, valid, filled, cv2.xphoto.INPAINT_FSR_FAST)
        out = adj.copy()
        out[y0:y1, x0:x1][hole_crop] = filled[hole_crop]
        return out
