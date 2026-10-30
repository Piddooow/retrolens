"""MatteEngine: soft person matte at display resolution and the removal zone around it."""

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class MatteEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.matte = None  # type: Optional[np.ndarray]
        self.person = None  # type: Optional[np.ndarray]

    def reset(self) -> None:
        self.matte = None
        self.person = None

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
        return matte

    def _distance(self) -> np.ndarray:
        """Distance in pixels to the nearest pixel with any matte left (this includes the person).

        Growing the whole matte support instead of only the person merges the faint matte
        islands the guided filter leaves near textured edges into one zone without gaps.
        """
        outside = np.where(self.matte > 0.0, np.uint8(0), np.uint8(255))
        return cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_5)

    def core(self, dist: Optional[np.ndarray] = None) -> np.ndarray:
        """Matte support dilated by zone_dilate_px (bool)."""
        if dist is None:
            dist = self._distance()
        return dist <= self.cfg["zone_dilate_px"]

    def zone(self, frame: np.ndarray, plate_adj: np.ndarray, known: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Feathered removal zone Zf (float32 0..1) and the shadow part of it (bool)."""
        cfg = self.cfg
        dist = self._distance()
        core = self.core(dist)
        if cfg["shadow_on"]:
            shadow = self._shadow(frame, plate_adj, known, core, dist)
        else:
            shadow = np.zeros_like(core)
        hard = (core | shadow).astype(np.float32)
        # Feather outward only, so every pixel inside the zone is fully replaced.
        zf = np.maximum(cv2.GaussianBlur(hard, (0, 0), cfg["zone_feather_px"]), hard)
        return zf, shadow

    def _shadow(self, frame: np.ndarray, plate_adj: np.ndarray, known: np.ndarray, core: np.ndarray,
                dist: np.ndarray) -> np.ndarray:
        """Pixels near the person that are darker than the room but keep its color."""
        cfg = self.cfg
        shadow = np.zeros_like(core)
        reach = dist <= cfg["zone_dilate_px"] + cfg["shadow_ring_px"]
        x, y, w, h = cv2.boundingRect(reach.view(np.uint8))
        if w == 0 or h == 0:
            return shadow
        box = (slice(y, y + h), slice(x, x + w))
        live = frame[box].astype(np.float32)
        plate = plate_adj[box].astype(np.float32)
        luma_live = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
        luma_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        chroma_live = live / (live.sum(axis=2, keepdims=True) + 1.0)
        chroma_plate = plate / (plate.sum(axis=2, keepdims=True) + 1.0)
        same_color = np.abs(chroma_live - chroma_plate).max(axis=2) < cfg["shadow_chroma_max"]
        darker = luma_live < cfg["shadow_ratio_max"] * luma_plate
        ring = reach[box] & ~core[box]
        candidate = (ring & darker & same_color & known[box]).view(np.uint8)
        k = cfg["shadow_close_px"]
        closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        # Keep only shadow regions that touch the person zone.
        _, labels = cv2.connectedComponents((closed | core[box].view(np.uint8)), connectivity=8)
        touching = np.unique(labels[core[box]])
        keep = np.isin(labels, touching[touching > 0])
        shadow[box] = keep & (closed > 0) & ring & known[box]
        return shadow
