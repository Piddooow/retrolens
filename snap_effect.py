"""Mode SNAP untuk Retrolens: orang di depan kamera hancur jadi debu setelah jentikan jari."""

import re
from typing import Any, Dict, List

import mediapipe as mp
import numpy as np

CONFIG = {
    # kamera dan resolusi
    "kamera_indeks": 0,
    "kamera_lebar": 1280,
    "kamera_tinggi": 720,
    "kamera_gagal_maks": 30,
    "proses_lebar": 640,
    "proses_tinggi": 360,

    # gesture (AWAL, setel lewat overlay debug)
    "cubit_masuk": 0.35,
    "cubit_tahan_detik": 0.20,
    "cubit_lepas": 0.70,
    "lepas_maks_detik": 0.12,
    "tengah_turun_min": 0.15,
    "siap_maks_detik": 1.5,
    "toleransi_hilang_detik": 0.15,

    # suara (AWAL, utamakan hasil kalibrasi)
    "audio_aktif": True,
    "audio_blocksize": 512,
    "audio_highpass_hz": 1500,
    "audio_onset_kali": 6.0,
    "audio_jendela_noise_detik": 2.0,
    "audio_dan_jendela_detik": 0.25,
    "audio_ambang_awal_kali_noise": 10.0,
    "audio_ring_blok": 64,
    "audio_diam_detik": 2.0,
    "audio_putus_detik": 1.0,
    "audio_kalibrasi_hening_detik": 2.0,
    "audio_kalibrasi_jentik_detik": 6.0,
    "audio_kalibrasi_jumlah": 3,
    "audio_kalibrasi_jarak_detik": 0.3,
    "mode_trigger": "GESTUR",
    "cooldown_detik": 1.0,

    # background
    "plate_bobot_baru": 0.1,
    "plate_ambang_bg": 0.2,
    "plate_hitung_mundur": 3,
    "plate_jumlah_frame": 30,
    "plate_maks_orang": 0.005,
    "plate_ambang_orang": 0.5,
    "plate_inpaint_radius": 5,
    "eksposur_min": 0.8,
    "eksposur_maks": 1.25,
    "latar_live": False,

    # segmentasi
    "mask_ema_baru": 0.6,
    "mask_ambang": 0.5,
    "mask_kernel": 3,
    "mask_luas_min_komponen": 0.005,
    "mask_luas_min_orang": 0.01,

    # flash (terukur dari reference)
    "flash_naik_detik": 0.33,
    "flash_naik_tabel": ((0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                         (0.0, 0.14, 0.33, 0.57, 0.85, 1.0)),
    "flash_tahan_detik": 1.0,
    "flash_turun_detik": 1.8,
    "flash_turun_tabel": ((0.0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1.0),
                          (1.0, 0.966, 0.912, 0.828, 0.71, 0.559, 0.412, 0.265, 0.118, 0.0)),
    "flash_tunda_ujung_detik": 0.07,
    "flash_warna_dekat": "#647AB7",
    "flash_warna_jauh": "#857A94",
    "flash_asal_cadangan": (0.10, 0.90),

    # debu (terukur dan teruji di prototipe)
    "debu_mulai_detik": 1.33,
    "debu_sapuan_detik": 1.8,
    "debu_umur_detik": 1.2,
    "debu_ukuran_butir": 2,
    "debu_ukuran_butir_turun": 3,
    "debu_bobot_vertikal": 0.6,
    "debu_bobot_noise": 0.4,
    "debu_oktaf": (4, 8, 16, 32, 64),
    "debu_amplitudo_oktaf": 0.55,
    "debu_angin_kiri": 0.08,
    "debu_angin_naik": 0.03,
    "debu_percepatan_kiri": 0.06,
    "debu_percepatan_naik": 0.04,
    "debu_jitter_min": 0.4,
    "debu_jitter_maks": 1.6,
    "debu_goyang_px": 2.0,
    "debu_goyang_hz": 1.5,
    "debu_seed": 7,

    # abu melayang
    "abu_ambient": True,
    "abu_jumlah": 150,
    "abu_warna": "#B0B0B0",
    "abu_alpha": 0.6,
    "abu_ukuran_maks": 2,
    "abu_kecepatan_relatif": 0.4,
    "abu_selesai_detik": 5.0,

    # linimasa
    "kosong_detik": 2.0,
    "kembali_detik": 0.5,
    "pesan_detik": 1.5,

    # performa
    "fps_target": 24,
    "fps_batas_turun": 20,
    "fps_jendela_frame": 30,
    "fps_turun_detik": 2.0,
}

MODE_TRIGGER = ("GESTUR", "SUARA", "ATAU", "DAN")

_KUNCI_BOOL = ("audio_aktif", "latar_live", "abu_ambient")
_KUNCI_INT = ("kamera_indeks", "kamera_lebar", "kamera_tinggi", "kamera_gagal_maks", "proses_lebar",
              "proses_tinggi", "audio_blocksize", "audio_ring_blok", "audio_kalibrasi_jumlah",
              "plate_hitung_mundur", "plate_jumlah_frame", "plate_inpaint_radius", "mask_kernel",
              "debu_ukuran_butir", "debu_ukuran_butir_turun", "debu_seed", "abu_jumlah",
              "abu_ukuran_maks", "fps_target", "fps_batas_turun", "fps_jendela_frame")
_KUNCI_PECAHAN = ("plate_bobot_baru", "plate_ambang_bg", "plate_maks_orang", "plate_ambang_orang",
                  "mask_ema_baru", "mask_ambang", "mask_luas_min_komponen", "mask_luas_min_orang",
                  "debu_bobot_vertikal", "debu_bobot_noise", "debu_amplitudo_oktaf", "abu_alpha")
_KUNCI_WARNA = ("flash_warna_dekat", "flash_warna_jauh", "abu_warna")
_KUNCI_TABEL = ("flash_naik_tabel", "flash_turun_tabel")


def _angka(nilai: Any) -> bool:
    return isinstance(nilai, (int, float)) and not isinstance(nilai, bool)


def validasi_config(cfg: Dict[str, Any]) -> List[str]:
    """Kembalikan daftar kesalahan CONFIG; kosong berarti aman dipakai."""
    salah = []  # type: List[str]
    hilang = [k for k in CONFIG if k not in cfg]
    if hilang:
        return ["kunci hilang: " + ", ".join(hilang)]

    for k in _KUNCI_BOOL:
        if not isinstance(cfg[k], bool):
            salah.append("%s harus True atau False" % k)
    for k in _KUNCI_INT:
        if not isinstance(cfg[k], int) or isinstance(cfg[k], bool):
            salah.append("%s harus bilangan bulat" % k)
    for k, v in cfg.items():
        if k in _KUNCI_BOOL or k in _KUNCI_WARNA or k in _KUNCI_TABEL or k == "mode_trigger":
            continue
        if k in ("debu_oktaf", "flash_asal_cadangan"):
            continue
        if not _angka(v):
            salah.append("%s harus angka" % k)
        elif k == "kamera_indeks" or k == "debu_seed":
            if v < 0:
                salah.append("%s tidak boleh negatif" % k)
        elif v <= 0:
            salah.append("%s harus lebih dari 0 (sekarang %r)" % (k, v))
    for k in _KUNCI_PECAHAN:
        if _angka(cfg[k]) and not 0 < cfg[k] <= 1:
            salah.append("%s harus di antara 0 dan 1" % k)
    for k in _KUNCI_WARNA:
        if not isinstance(cfg[k], str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", cfg[k]):
            salah.append("%s harus warna hex seperti #A1B2C3" % k)
    for k in _KUNCI_TABEL:
        tabel = cfg[k]
        try:
            xs, ys = tabel
            xs = [float(x) for x in xs]
            ys = [float(y) for y in ys]
        except (TypeError, ValueError):
            salah.append("%s harus berisi dua deret angka" % k)
            continue
        if len(xs) != len(ys) or len(xs) < 2:
            salah.append("%s: panjang deret x dan y harus sama dan minimal 2" % k)
        elif xs[0] != 0.0 or xs[-1] != 1.0 or any(b <= a for a, b in zip(xs, xs[1:])):
            salah.append("%s: deret x harus naik dari 0 sampai 1" % k)
        elif any(not 0.0 <= y <= 1.0 for y in ys):
            salah.append("%s: deret y harus di antara 0 dan 1" % k)

    if cfg["mode_trigger"] not in MODE_TRIGGER:
        salah.append("mode_trigger harus salah satu dari " + ", ".join(MODE_TRIGGER))
    oktaf = cfg["debu_oktaf"]
    if (not isinstance(oktaf, (tuple, list)) or not oktaf
            or any(not isinstance(o, int) or isinstance(o, bool) or o < 1 for o in oktaf)):
        salah.append("debu_oktaf harus deret bilangan bulat positif")
    asal = cfg["flash_asal_cadangan"]
    if (not isinstance(asal, (tuple, list)) or len(asal) != 2
            or any(not _angka(a) or not 0 <= a <= 1 for a in asal)):
        salah.append("flash_asal_cadangan harus (x, y) dengan nilai 0 sampai 1")
    # Pemeriksaan silang hanya aman jika semua tipe sudah benar.
    if salah:
        return salah

    if cfg["cubit_masuk"] >= cfg["cubit_lepas"]:
        salah.append("cubit_masuk harus lebih kecil dari cubit_lepas")
    if cfg["debu_jitter_min"] > cfg["debu_jitter_maks"]:
        salah.append("debu_jitter_min tidak boleh lebih besar dari debu_jitter_maks")
    if not cfg["eksposur_min"] <= 1.0 <= cfg["eksposur_maks"]:
        salah.append("eksposur_min harus <= 1 dan eksposur_maks harus >= 1")
    if cfg["fps_batas_turun"] > cfg["fps_target"]:
        salah.append("fps_batas_turun tidak boleh lebih besar dari fps_target")
    for k in ("debu_ukuran_butir", "debu_ukuran_butir_turun"):
        if cfg[k] > cfg["proses_lebar"] // 8 or cfg[k] > cfg["proses_tinggi"] // 8:
            salah.append("%s terlalu besar untuk resolusi proses" % k)
    if cfg["mask_kernel"] % 2 == 0:
        salah.append("mask_kernel harus ganjil")
    return salah


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
        self._seg = vision.ImageSegmenter.create_from_options(opsi)

    def confidence(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        """Mask float32 (tinggi, lebar) seukuran input, 0 background sampai 1 orang."""
        gambar = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hasil = self._seg.segment_for_video(gambar, ts_ms)
        # category_mask model ini terbalik (0 = orang), jadi hanya confidence mask yang dipakai.
        return np.squeeze(hasil.confidence_masks[0].numpy_view()).astype(np.float32, copy=True)

    def close(self) -> None:
        self._seg.close()
