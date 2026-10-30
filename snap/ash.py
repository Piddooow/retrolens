"""Small ash specks drifting with the wind after the flash, drawn as soft splats by the dissolve."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from snap.config import hex_ke_bgr
from snap.dissolve import smoothstep


class AshAmbient:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.warna = hex_ke_bgr(cfg["abu_warna"])
        self.x0 = np.zeros(0, dtype=np.float32)
        self.size = (1, 1)

    def siapkan(self, size: Tuple[int, int], rng: np.random.Generator) -> None:
        """size = (width, height) of the display frame."""
        n = self.cfg["abu_jumlah"]
        self.size = size
        self.x0 = rng.uniform(0, size[0], n).astype(np.float32)
        self.y0 = rng.uniform(0, size[1], n).astype(np.float32)
        # Splat weight is the speck area in half resolution pixels.
        self.area = (rng.integers(1, self.cfg["abu_ukuran_maks"] + 1, n) ** 2).astype(np.float32)

    def aktif(self, t: float) -> bool:
        cfg = self.cfg
        return cfg["abu_ambient"] and len(self.x0) > 0 and cfg["debu_mulai_detik"] <= t < cfg["abu_selesai_detik"]

    def specks(self, t: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(x, y, weights) of the specks at t seconds since the snap, or None when none are visible.

        weights has the columns alpha, B, G, R times alpha, each times the speck area.
        """
        if not self.aktif(t):
            return None
        cfg = self.cfg
        lebar, tinggi = self.size
        mulai, akhir = cfg["debu_mulai_detik"], cfg["abu_selesai_detik"]
        a = t - mulai
        alpha = cfg["abu_alpha"] * (1.0 - float(smoothstep(a / (akhir - mulai))))
        arah_x = 1.0 if cfg["debu_balik_horizontal"] else -1.0
        rel = cfg["abu_kecepatan_relatif"]
        x = np.mod(self.x0 + arah_x * rel * cfg["debu_angin_kiri"] * lebar * a, lebar)
        y = np.mod(self.y0 - rel * cfg["debu_angin_naik"] * tinggi * a, tinggi)
        weights = np.empty((len(x), 4), dtype=np.float32)
        weights[:, 0] = alpha
        weights[:, 1:] = alpha * self.warna
        weights *= self.area[:, None]
        return x.astype(np.float32), y.astype(np.float32), weights
