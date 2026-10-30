"""CONFIG for the SNAP effect and its validator. Every tunable number lives here."""

import math
import re
from typing import Any, Dict, List

import numpy as np

CONFIG = {
    # kamera dan resolusi
    "kamera_indeks": 0,
    "kamera_lebar": 1280,
    "kamera_tinggi": 720,
    "kamera_gagal_maks": 30,
    "proses_lebar": 640,
    "proses_tinggi": 360,

    # gesture (disetel dari rekaman jentikan asli: rasio_cubit saat lepas 0.55 sampai 0.69,
    # rasio_tengah turun 40 sampai 50% dari nilai tertinggi selama SIAP)
    "cubit_masuk": 0.35,
    "cubit_tahan_detik": 0.20,
    "cubit_lepas": 0.45,
    "lepas_maks_detik": 0.12,
    "tengah_turun_min": 0.30,
    "tangan_cocok_maks": 0.25,
    "siap_maks_detik": 1.5,
    "toleransi_hilang_detik": 0.15,

    # hold gestures (MediaPipe GestureRecognizer, model file next to main.py)
    "gesture_model": "gesture_recognizer.task",
    "hold_min_score": 0.7,
    "hold_agree_frac": 0.8,
    "hold_gap_max_s": 0.15,
    "hold_release_s": 0.3,
    "gesture_cooldown_s": 0.6,
    "hold_thumb_up_s": 1.0,
    "hold_thumb_down_s": 1.0,
    # The snap release looks like a fist for under 0.2 s, so a fist only counts after this long.
    "hold_fist_min_s": 0.25,

    # hold progress ring
    "ring_radius_px": 28,
    "ring_pulse_px": 34,
    "ring_stroke_px": 3,
    "ring_alpha": 0.85,
    "ring_pulse_s": 0.15,
    "hud_fade_s": 0.2,

    # suara (AWAL, utamakan hasil kalibrasi)
    "audio_aktif": True,
    "audio_blocksize": 512,
    "audio_highpass_hz": 1500,
    "audio_onset_kali": 6.0,
    "audio_jendela_noise_detik": 2.0,
    "audio_dan_jendela_detik": 0.25,
    "audio_ambang_awal_kali_noise": 20.0,
    "audio_ring_blok": 64,
    "audio_diam_detik": 2.0,
    "audio_putus_detik": 1.0,
    "audio_kalibrasi_hening_detik": 2.0,
    "audio_kalibrasi_jentik_detik": 6.0,
    "audio_kalibrasi_jumlah": 3,
    "audio_kalibrasi_jarak_detik": 0.3,
    "mode_trigger": "GESTUR",
    "cooldown_detik": 1.0,

    # background (manual capture countdown and frame count)
    "plate_hitung_mundur": 3,
    "plate_jumlah_frame": 30,

    # background engine v2 (display resolution)
    "bg_safe_conf": 0.05,
    "bg_safe_margin_px": 16,
    "bg_safe_frames": 3,
    "bg_median_step": 2,
    "bg_known_samples": 15,
    "bg_auto_capture_s": 1.0,
    "bg_auto_capture_frames": 20,
    "bg_still_diff": 3.0,
    "bg_absent_frac": 0.001,
    # Measured on the user's footage: 640x360 keeps phase correlation linear (320x180 does not),
    # still-camera jitter stays under 0.38 px, so 0.75 px here (1.5 px at 720p) catches a 3 px move.
    "bg_moved_grid": (640, 360),
    "bg_shift_max_px": 0.75,
    "bg_phase_min_response": 0.15,
    "bg_moved_check_s": 0.5,
    "bg_moved_confirm": 2,
    "bg_moved_min_known": 0.2,
    "bg_moved_mask_blur": 2.0,
    "bg_gain_grid": (160, 90),
    "bg_gain_sigma": 6.0,
    "bg_gain_min": 0.7,
    "bg_gain_max": 1.4,
    "bg_gain_min_weight": 0.001,
    "bg_noise_on": True,
    "bg_noise_tiles": 8,
    "bg_noise_tile_px": 256,
    "bg_noise_update_s": 1.0,
    "bg_coverage_min": 0.995,
    "allow_inpaint_fallback": False,
    "inpaint_pad_px": 32,
    "coverage_tint_alpha": 0.25,

    # matte and removal zone (display resolution)
    "matte_radius": 8,
    "matte_eps": 0.001,
    "matte_scale": 0.5,
    "matte_floor": 0.02,
    "matte_on": 0.5,
    "matte_keep": 0.3,
    "zone_dilate_px": 10,
    "zone_feather_px": 4.0,
    # Starting values; tune on uji_lokal/v2/sit_still.mp4 once it is recorded.
    "shadow_on": True,
    "shadow_ring_px": 80,
    "shadow_ratio_max": 0.85,
    "shadow_chroma_max": 0.04,
    "shadow_close_px": 7,
    "debug_alpha": 0.45,
    "coverage_tint_fade_s": 0.6,

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
    "flash_ambang_putih": 0.5,

    # debu (terukur dan teruji di prototipe)
    "debu_mulai_detik": 1.33,
    "debu_sapuan_detik": 1.8,
    "debu_umur_detik": 1.2,
    # One dust particle per cell of this many display pixels; the larger size is used when fps stays low.
    "particle_cell_px": 4,
    "debu_ukuran_butir_turun": 6,
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
    "debu_goyang_px": 4.0,
    "debu_goyang_hz": 1.5,
    "debu_seed": 7,
    "debu_balik_horizontal": False,

    # dissolve v2 (display resolution)
    "edge_soft_s": 0.08,
    "fine_noise_weight": 0.06,
    "dust_blur": 0.6,
    "dust_gain": 1.6,
    "max_particle_step_px": 12,

    # abu melayang
    "abu_ambient": True,
    "abu_jumlah": 150,
    "abu_warna": "#B0B0B0",
    "abu_alpha": 0.6,
    # Speck size in half resolution pixels.
    "abu_ukuran_maks": 2,
    "abu_kecepatan_relatif": 0.4,
    # Must end with the dust, so the zone shows only the room afterwards.
    "abu_selesai_detik": 4.33,

    # linimasa
    "kosong_detik": 2.0,
    "kembali_detik": 0.5,
    "pesan_detik": 1.5,

    # performa
    "fps_target": 24,
    "fps_batas_turun": 20,
    "fps_jendela_frame": 30,
    "fps_turun_detik": 2.0,
    "profiler_window": 60,

    # lembar uji (--uji)
    "uji_jumlah_frame": 9,
    "uji_selang_detik": 0.4,
}

MODE_TRIGGER = ("GESTUR", "SUARA", "ATAU", "DAN")
# Frame rate at which motion limits such as max_particle_step_px are defined.
REFERENCE_FPS = 30

_KUNCI_BOOL = ("audio_aktif", "abu_ambient", "debu_balik_horizontal", "bg_noise_on",
               "allow_inpaint_fallback", "shadow_on")
_KUNCI_INT = ("kamera_indeks", "kamera_lebar", "kamera_tinggi", "kamera_gagal_maks", "proses_lebar",
              "proses_tinggi", "audio_blocksize", "audio_ring_blok", "audio_kalibrasi_jumlah",
              "plate_hitung_mundur", "plate_jumlah_frame", "mask_kernel",
              "particle_cell_px", "debu_ukuran_butir_turun", "debu_seed", "abu_jumlah",
              "abu_ukuran_maks", "fps_target", "fps_batas_turun", "fps_jendela_frame", "uji_jumlah_frame",
              "profiler_window", "bg_safe_margin_px", "bg_safe_frames", "bg_median_step",
              "bg_known_samples", "bg_auto_capture_frames", "bg_moved_confirm", "bg_noise_tiles",
              "bg_noise_tile_px", "zone_dilate_px", "inpaint_pad_px", "matte_radius", "shadow_ring_px",
              "shadow_close_px", "max_particle_step_px", "ring_radius_px", "ring_pulse_px", "ring_stroke_px")
_KUNCI_PECAHAN = ("bg_safe_conf", "bg_absent_frac", "bg_phase_min_response", "bg_moved_min_known",
                  "bg_coverage_min", "coverage_tint_alpha", "bg_gain_min_weight", "matte_eps", "matte_scale",
                  "matte_floor", "matte_on", "matte_keep", "shadow_ratio_max", "shadow_chroma_max",
                  "debug_alpha", "mask_ema_baru", "mask_ambang", "mask_luas_min_komponen", "mask_luas_min_orang",
                  "debu_bobot_vertikal", "debu_bobot_noise", "debu_amplitudo_oktaf", "abu_alpha",
                  "flash_ambang_putih", "tengah_turun_min", "tangan_cocok_maks", "fine_noise_weight",
                  "hold_min_score", "hold_agree_frac", "ring_alpha")
_KUNCI_WARNA = ("flash_warna_dekat", "flash_warna_jauh", "abu_warna")
_KUNCI_TABEL = ("flash_naik_tabel", "flash_turun_tabel")
_KUNCI_GRID = ("bg_moved_grid", "bg_gain_grid")


def hex_ke_bgr(warna: str) -> np.ndarray:
    return np.array([int(warna[5:7], 16), int(warna[3:5], 16), int(warna[1:3], 16)], dtype=np.float32)


def particle_step_px(cfg: Dict[str, Any]) -> float:
    """Upper bound of one particle's move per frame at REFERENCE_FPS on the camera frame size."""
    life, jit = cfg["debu_umur_detik"], cfg["debu_jitter_maks"]
    vx = (cfg["debu_angin_kiri"] + cfg["debu_percepatan_kiri"] * life) * cfg["kamera_lebar"] * jit
    vy = ((cfg["debu_angin_naik"] + cfg["debu_percepatan_naik"] * life) * cfg["kamera_tinggi"] * jit
          + 2.0 * math.pi * cfg["debu_goyang_hz"] * cfg["debu_goyang_px"])
    return math.hypot(vx, vy) / REFERENCE_FPS


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
        if k in _KUNCI_BOOL or k in _KUNCI_WARNA or k in _KUNCI_TABEL or k in ("mode_trigger", "gesture_model"):
            continue
        if k in ("debu_oktaf", "flash_asal_cadangan") or k in _KUNCI_GRID:
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

    if not isinstance(cfg["gesture_model"], str) or not cfg["gesture_model"].endswith(".task"):
        salah.append("gesture_model must be a .task file name")
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
    for k in _KUNCI_GRID:
        grid = cfg[k]
        if (not isinstance(grid, (tuple, list)) or len(grid) != 2
                or any(not isinstance(g, int) or isinstance(g, bool) or g < 2 for g in grid)):
            salah.append("%s must be (width, height) with whole numbers of at least 2" % k)
    # Pemeriksaan silang hanya aman jika semua tipe sudah benar.
    if salah:
        return salah

    if cfg["cubit_masuk"] >= cfg["cubit_lepas"]:
        salah.append("cubit_masuk harus lebih kecil dari cubit_lepas")
    if cfg["debu_jitter_min"] > cfg["debu_jitter_maks"]:
        salah.append("debu_jitter_min tidak boleh lebih besar dari debu_jitter_maks")
    if not cfg["bg_gain_min"] < 1.0 < cfg["bg_gain_max"]:
        salah.append("bg_gain_min must be below 1 and bg_gain_max above 1")
    # Counters are uint8, so sample thresholds must fit.
    if cfg["bg_known_samples"] > 255 or cfg["bg_safe_frames"] > 254:
        salah.append("bg_known_samples must be at most 255 and bg_safe_frames at most 254")
    if cfg["bg_median_step"] > 255:
        salah.append("bg_median_step must be at most 255")
    if cfg["fps_batas_turun"] > cfg["fps_target"]:
        salah.append("fps_batas_turun tidak boleh lebih besar dari fps_target")
    for k in ("particle_cell_px", "debu_ukuran_butir_turun"):
        if not 2 <= cfg[k] <= 16:
            salah.append("%s must be from 2 to 16 display pixels" % k)
    dust_end = cfg["debu_mulai_detik"] + cfg["debu_sapuan_detik"] + cfg["debu_umur_detik"]
    if cfg["abu_selesai_detik"] > dust_end + 1e-9:
        salah.append("abu_selesai_detik must not be later than the end of the dust (%.2f s)" % dust_end)
    if particle_step_px(cfg) > cfg["max_particle_step_px"]:
        salah.append("wind settings move dust %.1f px per frame at %d fps (limit max_particle_step_px %d)" % (
            particle_step_px(cfg), REFERENCE_FPS, cfg["max_particle_step_px"]))
    if cfg["mask_kernel"] % 2 == 0:
        salah.append("mask_kernel harus ganjil")
    if cfg["ring_pulse_px"] <= cfg["ring_radius_px"]:
        salah.append("ring_pulse_px must be larger than ring_radius_px")
    if cfg["shadow_close_px"] % 2 == 0:
        salah.append("shadow_close_px must be odd")
    if not cfg["matte_floor"] < cfg["matte_keep"] < cfg["matte_on"]:
        salah.append("matte values must satisfy matte_floor < matte_keep < matte_on")
    # Pergantian ke frame beku harus terjadi saat seluruh layar sudah putih penuh.
    if cfg["flash_tunda_ujung_detik"] >= cfg["flash_tahan_detik"]:
        salah.append("flash_tunda_ujung_detik harus lebih kecil dari flash_tahan_detik")
    if cfg["debu_mulai_detik"] < cfg["flash_naik_detik"] + cfg["flash_tunda_ujung_detik"]:
        salah.append("debu_mulai_detik tidak boleh sebelum layar putih penuh")
    return salah
