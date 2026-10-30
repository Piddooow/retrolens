"""DissolveRenderer: the frozen person turns to dust at display resolution."""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# DUST style: the ash color a particle fades to is its luminance times DUST_LUMA plus its color times DUST_TINT.
DUST_LUMA = 0.55
DUST_TINT = 0.32
LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)
# Bilinear splat corners, as (dx, dy) offsets from the lower half resolution pixel.
CORNER_DX = np.array([0, 1, 0, 1], dtype=np.int32)
CORNER_DY = np.array([0, 0, 1, 1], dtype=np.int32)
CHANNELS = np.arange(4, dtype=np.int32)

Specks = Tuple[np.ndarray, np.ndarray, np.ndarray]


def smoothstep(x: Any) -> Any:
    """Ease 0..1 with zero slope at both ends; works on scalars and arrays."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def normalisasi(nilai: np.ndarray) -> np.ndarray:
    rendah, tinggi = float(nilai.min()), float(nilai.max())
    if tinggi - rendah < 1e-9:
        return np.zeros(nilai.shape, dtype=np.float32)
    return ((nilai - rendah) / (tinggi - rendah)).astype(np.float32)


def noise_fraktal(tinggi: int, lebar: int, oktaf: Tuple[int, ...], amp_kali: float,
                  rng: np.random.Generator) -> np.ndarray:
    total = np.zeros((tinggi, lebar), dtype=np.float32)
    amp = 1.0
    for sel in oktaf:
        kasar = rng.random((tinggi // sel + 2, lebar // sel + 2), dtype=np.float32)
        total += amp * cv2.resize(kasar, (lebar, tinggi), interpolation=cv2.INTER_CUBIC)
        amp *= amp_kali
    return normalisasi(total)


class DissolveRenderer:
    """Body and dust layers drawn over a base frame, limited to a region of interest."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.size = (0, 0)
        self.box = None  # type: Optional[Tuple[int, int, int, int]]
        self.roi = (0, 0, 0, 0)
        self.count = 0
        # Quality metrics of the last frame: visible mass relative to the start, largest particle move
        # since the previous frame (display px), and particles drawn.
        self.mass = 0.0
        self.step_max = 0.0
        self.visible = 0
        self.last_alpha = None  # type: Optional[np.ndarray]
        self._prev_xy = np.zeros((0, 2), dtype=np.float32)

    # Preparation (once, under the white flash)

    def siapkan(self, snapshot: np.ndarray, matte: np.ndarray, zone: np.ndarray, cell: int) -> None:
        """snapshot and matte of the frozen frame, zone = its removal zone, cell = particle cell size in px."""
        cfg = self.cfg
        h, w = matte.shape
        self.size = (w, h)
        self.cell = cell
        self.end = cfg["debu_mulai_detik"] + cfg["debu_sapuan_detik"] + cfg["debu_umur_detik"]
        self.matte_sum = float(matte.sum())
        self.count = 0
        self.mass = 0.0
        self.step_max = 0.0
        self.visible = 0
        self.last_alpha = None
        self.roi = self._roi(zone)
        bx, by, bw, bh = cv2.boundingRect((matte > 0).view(np.uint8))
        if bw == 0 or bh == 0:
            self.box = None
            self._prev_xy = np.zeros((0, 2), dtype=np.float32)
            return
        self.box = (bx, by, bw, bh)
        box = (slice(by, by + bh), slice(bx, bx + bw))
        rng = np.random.default_rng(cfg["debu_seed"])

        # Release map: head first, broken up by coarse and fine noise.
        ynorm = np.clip((np.arange(by, by + bh, dtype=np.float32) - by) / max(1, bh - 1), 0.0, 1.0)[:, None]
        coarse = noise_fraktal(max(1, h // 4), max(1, w // 4), tuple(cfg["debu_oktaf"]),
                               cfg["debu_amplitudo_oktaf"], rng)
        coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)[box]
        fine = rng.random((max(1, h // 2), max(1, w // 2)), dtype=np.float32)
        fine = cv2.resize(fine, (w, h), interpolation=cv2.INTER_NEAREST)[box]
        order = (cfg["debu_bobot_vertikal"] * ynorm + cfg["debu_bobot_noise"] * coarse
                 + cfg["fine_noise_weight"] * fine).astype(np.float32)
        self.matte_box = matte[box].astype(np.float32)
        inside = self.matte_box > 0
        lo, hi = float(order[inside].min()), float(order[inside].max())
        order = np.clip((order - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        self.T = (cfg["debu_mulai_detik"] + order * cfg["debu_sapuan_detik"]).astype(np.float32)
        self.snap_box = np.ascontiguousarray(snapshot[box])

        self._particles(rng)

    def _particles(self, rng: np.random.Generator) -> None:
        """One particle per cell with mean matte above matte_floor."""
        cfg = self.cfg
        c = self.cell
        bx, by, bw, bh = self.box
        gx0, gy0 = bx // c, by // c
        nx, ny = -(-(bx + bw) // c) - gx0, -(-(by + bh) // c) - gy0
        oy, ox = by - gy0 * c, bx - gx0 * c
        # Pad the box to whole cells; the padding has zero matte and adds nothing.
        m = np.zeros((ny * c, nx * c), dtype=np.float32)
        col = np.zeros((ny * c, nx * c, 3), dtype=np.float32)
        rel = np.zeros((ny * c, nx * c), dtype=np.float32)
        m[oy:oy + bh, ox:ox + bw] = self.matte_box
        col[oy:oy + bh, ox:ox + bw] = self.snap_box * self.matte_box[..., None]
        rel[oy:oy + bh, ox:ox + bw] = self.T * self.matte_box

        def cells(a: np.ndarray) -> np.ndarray:
            return a.reshape((ny, c, nx, c) + a.shape[2:]).sum(axis=(1, 3))

        msum = cells(m)
        iy, ix = np.nonzero(msum > cfg["matte_floor"] * c * c)
        n = len(iy)
        self.count = n
        weight = msum[iy, ix]
        self.p_mass = (weight / (c * c)).astype(np.float32)
        # Matte weighted means, so a particle carries exactly the color and timing of the body it replaces.
        self.p_color = (cells(col)[iy, ix] / weight[:, None]).astype(np.float32)
        self.p_release = (cells(rel)[iy, ix] / weight).astype(np.float32)
        self.p_x0 = ((gx0 + ix + 0.5) * c).astype(np.float32)
        self.p_y0 = ((gy0 + iy + 0.5) * c).astype(np.float32)
        luma = (self.p_color * LUMA_BGR).sum(axis=1)
        self.p_ash = (DUST_LUMA * luma[:, None] + DUST_TINT * self.p_color).astype(np.float32)
        self.p_jitter = rng.uniform(cfg["debu_jitter_min"], cfg["debu_jitter_maks"], (n, 2)).astype(np.float32)
        self.p_phase = rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
        self._prev_xy = np.full((n, 2), np.nan, dtype=np.float32)

    def _roi(self, zone: np.ndarray) -> Tuple[int, int, int, int]:
        """Zone bounding box grown by the farthest any particle and its soft splat can reach."""
        cfg = self.cfg
        w, h = self.size
        x, y, bw, bh = cv2.boundingRect((zone > 0).view(np.uint8))
        if bw == 0 or bh == 0:
            return (0, 0, 0, 0)
        life, jit = cfg["debu_umur_detik"], cfg["debu_jitter_maks"]
        tx = (cfg["debu_angin_kiri"] * life + 0.5 * cfg["debu_percepatan_kiri"] * life * life) * w * jit
        ty = (cfg["debu_angin_naik"] * life + 0.5 * cfg["debu_percepatan_naik"] * life * life) * h * jit
        sway = 2.0 * cfg["debu_goyang_px"]
        # Bilinear splat (3 px), Gaussian kernel (2 px per half resolution pixel) and upsampling (2 px).
        pad = 3 + 2 * (int(math.ceil(4.0 * cfg["dust_blur"])) + 1) + 2
        left, right = (0.0, tx) if cfg["debu_balik_horizontal"] else (tx, 0.0)
        x0 = int(math.floor(x - left - pad))
        x1 = int(math.ceil(x + bw + right + pad))
        y0 = int(math.floor(y - ty - sway - pad))
        y1 = int(math.ceil(y + bh + sway + pad))
        # Even edges keep the half resolution splat grid aligned with display pixels.
        return (max(0, x0 - x0 % 2), max(0, y0 - y0 % 2), min(w, x1 + x1 % 2), min(h, y1 + y1 % 2))

    # Per frame

    def render(self, out: np.ndarray, t: float, extra: Optional[Specks] = None) -> None:
        """Draw body and dust at t seconds since the snap into out (display frame, uint8, in place).

        extra = more specks to splat in the same pass (ash), as (x, y, weights).
        """
        self.visible = 0
        self.step_max = 0.0
        body_sum = 0.0
        particle_sum = 0.0
        splats = []  # type: List[Specks]
        if self.box is not None and t < self.end:
            body_sum = self._body(out, t)
            moving = self._moving(t)
            if moving is not None:
                particle_sum = moving[3]
                splats.append(moving[:3])
        self.mass = (body_sum + particle_sum * self.cell * self.cell) / max(self.matte_sum, 1e-6)
        if extra is not None:
            splats.append(extra)
        if not splats:
            self.last_alpha = None
            return
        w, h = self.size
        # Ash covers the whole frame; dust alone stays inside the ROI.
        region = (0, 0, w, h) if extra is not None else self.roi
        self._splat(out, splats, region)

    def _body(self, out: np.ndarray, t: float) -> float:
        bx, by, bw, bh = self.box
        s = smoothstep((self.T - np.float32(t)) * np.float32(1.0 / self.cfg["edge_soft_s"]) + np.float32(0.5))
        alpha = self.matte_box * s
        total = float(alpha.sum())
        if total > 0.0:
            region = out[by:by + bh, bx:bx + bw]
            region[:] = cv2.blendLinear(self.snap_box, region, alpha, 1.0 - alpha)
        return total

    def _moving(self, t: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """(x, y, weights, alpha sum) of the visible particles, or None."""
        cfg = self.cfg
        w, h = self.size
        edge_inv = np.float32(1.0 / cfg["edge_soft_s"])
        tt = np.float32(t)
        a = np.maximum(tt - self.p_release, np.float32(0.0))
        u = a * np.float32(1.0 / cfg["debu_umur_detik"])
        # Birth is the exact complement of the body fade, so mass is handed over without a jump.
        birth = 1.0 - smoothstep((self.p_release - tt) * edge_inv + np.float32(0.5))
        alpha = self.p_mass * birth * (1.0 - smoothstep(u))
        on = np.nonzero(alpha > 0)[0]
        xy = np.full((self.count, 2), np.nan, dtype=np.float32)
        if len(on) == 0:
            self._prev_xy = xy
            return None
        a = a[on]
        jit = self.p_jitter[on]
        phase = self.p_phase[on]
        arah_x = 1.0 if cfg["debu_balik_horizontal"] else -1.0
        dx = arah_x * (cfg["debu_angin_kiri"] * a + 0.5 * cfg["debu_percepatan_kiri"] * a * a) * w * jit[:, 0]
        dy = -(cfg["debu_angin_naik"] * a + 0.5 * cfg["debu_percepatan_naik"] * a * a) * h * jit[:, 1]
        # The sway starts at zero, so a particle does not jump when it is released.
        dy += cfg["debu_goyang_px"] * (np.sin(2.0 * math.pi * cfg["debu_goyang_hz"] * a + phase) - np.sin(phase))
        x = (self.p_x0[on] + dx).astype(np.float32)
        y = (self.p_y0[on] + dy).astype(np.float32)

        xy[on, 0] = x
        xy[on, 1] = y
        both = ~np.isnan(xy[:, 0]) & ~np.isnan(self._prev_xy[:, 0])
        if both.any():
            self.step_max = float(np.hypot(*(xy[both] - self._prev_xy[both]).T).max())
        self._prev_xy = xy
        self.visible = len(on)

        k = np.minimum(u[on], 1.0)[:, None]
        color = self.p_color[on] * (1.0 - k) + self.p_ash[on] * k
        al = alpha[on]
        weights = np.empty((len(on), 4), dtype=np.float32)
        weights[:, 0] = al
        weights[:, 1:] = color * al[:, None]
        # A cell covers (cell / 2) squared half resolution pixels, so fresh dense dust is as opaque as the body.
        weights *= (self.cell * 0.5) ** 2
        return x, y, weights, float(alpha.sum())

    def _splat(self, out: np.ndarray, splats: List[Specks], region: Tuple[int, int, int, int]) -> None:
        cfg = self.cfg
        w, h = self.size
        w2, h2 = (w + 1) // 2, (h + 1) // 2
        x = np.concatenate([s[0] for s in splats])
        y = np.concatenate([s[1] for s in splats])
        weights = np.concatenate([s[2] for s in splats])
        # Half resolution pixel j has its center at display coordinate 2j + 1.
        qx = x * np.float32(0.5) - np.float32(0.5)
        qy = y * np.float32(0.5) - np.float32(0.5)
        jx = np.floor(qx)
        jy = np.floor(qy)
        fx = (qx - jx)[:, None]
        fy = (qy - jy)[:, None]
        cx = jx.astype(np.int32)[:, None] + CORNER_DX
        cy = jy.astype(np.int32)[:, None] + CORNER_DY
        corner = np.where(CORNER_DX == 1, fx, 1.0 - fx) * np.where(CORNER_DY == 1, fy, 1.0 - fy)
        # Corners off the frame get zero weight on a clamped index, which avoids slow boolean indexing.
        corner *= (cx >= 0) & (cx < w2) & (cy >= 0) & (cy < h2)
        pixel = np.clip(cy, 0, h2 - 1) * w2 + np.clip(cx, 0, w2 - 1)
        index = (pixel * 4)[..., None] + CHANNELS
        value = corner[..., None] * weights[:, None, :]
        buf = np.bincount(index.ravel(), value.ravel(), minlength=h2 * w2 * 4)
        buf = cv2.GaussianBlur(buf.astype(np.float32).reshape(h2, w2, 4), (0, 0), cfg["dust_blur"])
        buf *= np.float32(cfg["dust_gain"])
        self.last_alpha = buf[..., 0]

        x0, y0, x1, y1 = region
        if x1 <= x0 or y1 <= y0:
            return
        up = cv2.resize(buf, (2 * w2, 2 * h2), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        alpha, b, g, r = cv2.split(up)
        # Dividing all four channels by alpha where it passes 1 is the same as covering min(alpha, 1)
        # with the color premultiplied / alpha, which one blendLinear does at display resolution.
        cover = cv2.min(alpha, 1.0)
        den = cv2.max(alpha, 1e-12)
        color = cv2.divide(cv2.merge([b, g, r]), cv2.merge([den, den, den]))
        dst = out[y0:y1, x0:x1]
        dst[:] = cv2.blendLinear(dst, cv2.convertScaleAbs(color), 1.0 - cover, cover)
