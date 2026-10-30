"""MatteEngine: soft person matte at display resolution and the removal zone around it."""

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class MatteEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.matte = None  # type: Optional[np.ndarray]
        self.person = None  # type: Optional[np.ndarray]
        self._dist = None  # type: Optional[np.ndarray]

    def reset(self) -> None:
        self.matte = None
        self.person = None
        self._dist = None

    def update(self, frame: np.ndarray, conf_lo: np.ndarray) -> np.ndarray:
        """Refine the (already time smoothed) low resolution confidence into a display size matte."""
        cfg = self.cfg
        h, w = frame.shape[:2]
        coarse = cv2.resize(conf_lo, (w, h), interpolation=cv2.INTER_LINEAR)
        guide = frame.astype(np.float32)
        guide *= 1.0 / 255.0
        matte = cv2.ximgproc.guidedFilter(guide, coarse, cfg["matte_radius"], cfg["matte_eps"],
                                          dDepth=-1, scale=cfg["matte_scale"])
        np.clip(matte, 0.0, 1.0, out=matte)
        matte[matte < cfg["matte_floor"]] = 0.0
        person = matte > cfg["matte_on"]
        # Hysteresis: a pixel that was person stays person down to matte_keep.
        if self.person is not None and self.person.shape == person.shape:
            person |= (matte > cfg["matte_keep"]) & self.person
        self.matte, self.person = matte, person
        self._dist = None
        return matte

    def _distance(self) -> Tuple[Optional[Tuple[slice, slice]], np.ndarray]:
        """(box, distance) where distance is in pixels to the nearest pixel with any matte left.

        Growing the whole matte support instead of only the person merges the faint matte
        islands the guided filter leaves near textured edges into one zone without gaps.
        The box holds the support plus everything the dilation, shadow ring and feather can reach,
        so the zone work skips the rest of the frame. Every support pixel is inside it, so distances
        inside the box equal full frame distances. box is None when no matte is left at all.
        Computed once per matte update, since core and zone both need it.
        """
        if self._dist is None:
            cfg = self.cfg
            h, w = self.matte.shape
            x, y, bw, bh = cv2.boundingRect((self.matte > 0.0).view(np.uint8))
            if bw == 0 or bh == 0:
                self._dist = (None, np.zeros((0, 0), dtype=np.float32))
            else:
                pad = (cfg["zone_dilate_px"] + cfg["shadow_ring_px"]
                       + int(math.ceil(4.0 * cfg["zone_feather_px"])) + 2)
                # Even edges, so the half resolution shadow grid lines up with display pixels
                # and OpenCV can take its fast exact 2x path.
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
                box = (slice(y0 - y0 % 2, min(h, y1 + y1 % 2)), slice(x0 - x0 % 2, min(w, x1 + x1 % 2)))
                outside = np.where(self.matte[box] > 0.0, np.uint8(0), np.uint8(255))
                self._dist = (box, cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_5))
        return self._dist

    def core(self) -> np.ndarray:
        """Matte support dilated by zone_dilate_px (bool, frame size)."""
        box, dist = self._distance()
        core = np.zeros(self.matte.shape, dtype=bool)
        if box is not None:
            core[box] = dist <= self.cfg["zone_dilate_px"]
        return core

    def zone(self, frame: np.ndarray, plate_adj: np.ndarray, known: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Feathered removal zone Zf (float32 0..1) and the shadow part of it (bool)."""
        cfg = self.cfg
        zf = np.zeros(self.matte.shape, dtype=np.float32)
        shadow = np.zeros(self.matte.shape, dtype=bool)
        box, dist = self._distance()
        if box is None:
            return zf, shadow
        core = dist <= cfg["zone_dilate_px"]
        if cfg["shadow_on"]:
            shadow[box] = self._shadow(frame[box], plate_adj[box], known[box], core, dist)
        hard = (core | shadow[box]).astype(np.float32)
        # Feather outward only, so every pixel inside the zone is fully replaced.
        # The box keeps more than the Gaussian reach free around the zone, so cropping changes nothing.
        zf[box] = np.maximum(cv2.GaussianBlur(hard, (0, 0), cfg["zone_feather_px"]), hard)
        return zf, shadow

    def _shadow(self, frame: np.ndarray, plate_adj: np.ndarray, known: np.ndarray, core: np.ndarray,
                dist: np.ndarray) -> np.ndarray:
        """Pixels near the person that are darker than the room but keep its color (all arrays of one box).

        Shadows are smooth, so the color test runs on 2x2 blocks; the exact ring and known room are applied
        at full resolution afterwards. A block counts as ring or core when any of its pixels is, so no ring
        pixel next to the core is left out, and as known only when all of its pixels are.
        """
        cfg = self.cfg
        ring = (dist <= cfg["zone_dilate_px"] + cfg["shadow_ring_px"]) & ~core
        if not ring.any():
            return np.zeros_like(core)
        h, w = core.shape
        half = ((w + 1) // 2, (h + 1) // 2)

        def blocks(mask: np.ndarray) -> np.ndarray:
            return cv2.resize(mask.view(np.uint8) * np.uint8(255), half, interpolation=cv2.INTER_AREA)

        live = cv2.resize(frame, half, interpolation=cv2.INTER_AREA).astype(np.float32)
        plate = cv2.resize(plate_adj, half, interpolation=cv2.INTER_AREA).astype(np.float32)
        luma_live = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
        luma_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        chroma_live = live / (live.sum(axis=2, keepdims=True) + 1.0)
        chroma_plate = plate / (plate.sum(axis=2, keepdims=True) + 1.0)
        same_color = np.abs(chroma_live - chroma_plate).max(axis=2) < cfg["shadow_chroma_max"]
        darker = luma_live < cfg["shadow_ratio_max"] * luma_plate
        core_b = blocks(core) > 0
        candidate = ((blocks(ring) > 0) & (blocks(known) == 255) & darker & same_color).view(np.uint8)
        # shadow_close_px is in display pixels; a block is two of them.
        k = max(1, cfg["shadow_close_px"] // 2) | 1
        closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        # Keep only shadow regions that touch the person zone.
        _, labels = cv2.connectedComponents(closed | core_b.view(np.uint8), connectivity=8)
        touching = np.unique(labels[core_b])
        keep = (np.isin(labels, touching[touching > 0]) & (closed > 0)).view(np.uint8)
        full = cv2.resize(keep, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return full & ring & known
