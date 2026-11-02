"""PersonSegmenter: MediaPipe selfie segmenter wrapper; high mask values mean person."""

from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class PersonSegmenter:
    """Pembungkus ImageSegmenter selfie. Nilai mask tinggi berarti orang."""

    def __init__(self, path_model: str, cfg: Dict[str, Any]):
        vision = mp.tasks.vision
        opsi = vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=path_model),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.cfg = cfg
        self.mask = None  # type: Optional[np.ndarray]
        # Set by reset (main thread), applied by perbarui (the inference thread), so only one thread
        # ever touches the smoothed mask.
        self._reset = False
        self._seg = vision.ImageSegmenter.create_from_options(opsi)

    def confidence(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        """Mask float32 (tinggi, lebar) seukuran input, 0 background sampai 1 orang."""
        gambar = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hasil = self._seg.segment_for_video(gambar, ts_ms)
        # category_mask model ini terbalik (0 = orang), jadi hanya confidence mask yang dipakai.
        return np.squeeze(hasil.confidence_masks[0].numpy_view()).astype(np.float32, copy=True)

    def perbarui(self, bgr: np.ndarray, ts_ms: int) -> np.ndarray:
        """Mask yang dihaluskan antar frame untuk frame BGR resolusi proses."""
        baru = self.confidence(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts_ms)
        if self._reset:
            self._reset = False
            self.mask = None
        if self.mask is None or self.mask.shape != baru.shape:
            self.mask = baru
        else:
            e = self.cfg["mask_ema_baru"]
            self.mask = cv2.addWeighted(baru, e, self.mask, 1.0 - e, 0.0)
        return self.mask

    def reset(self) -> None:
        """Start the time smoothing over with the next frame."""
        self._reset = True

    @staticmethod
    def mask_snap(mask: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Mask biner bersih untuk efek dan fraksi luas orang terhadap frame."""
        kernel = np.ones((cfg["mask_kernel"], cfg["mask_kernel"]), dtype=np.uint8)
        biner = (mask > cfg["mask_ambang"]).astype(np.uint8)
        biner = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel)
        jumlah, label, stats, _ = cv2.connectedComponentsWithStats(biner, connectivity=8)
        simpan = np.zeros(jumlah, dtype=np.uint8)
        simpan[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg["mask_luas_min_komponen"] * biner.size
        biner = cv2.erode(simpan[label], kernel, iterations=1).astype(bool)
        return biner, float(biner.mean())

    def close(self) -> None:
        self._seg.close()
