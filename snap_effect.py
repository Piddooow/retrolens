"""Mode SNAP untuk Retrolens: orang di depan kamera hancur jadi debu setelah jentikan jari."""

import argparse
import collections
import json
import math
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    import sounddevice as sd
except Exception:  # PortAudio tidak terpasang: mode suara dimatikan, sisanya tetap jalan.
    sd = None

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

    # background
    "plate_bobot_baru": 0.1,
    "plate_ambang_bg": 0.2,
    "plate_hitung_mundur": 3,
    "plate_jumlah_frame": 30,
    "plate_maks_orang": 0.005,
    "plate_ambang_orang": 0.5,
    "plate_inpaint_radius": 5,
    "plate_tepi_px": 4,
    "plate_beda_maks": 16.0,
    "plate_cocok_min": 0.05,
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
    "flash_ambang_putih": 0.5,

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
    "debu_balik_horizontal": False,

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

    # lembar uji (--uji)
    "uji_jumlah_frame": 9,
    "uji_selang_detik": 0.4,
}

MODE_TRIGGER = ("GESTUR", "SUARA", "ATAU", "DAN")

_KUNCI_BOOL = ("audio_aktif", "latar_live", "abu_ambient", "debu_balik_horizontal")
_KUNCI_INT = ("kamera_indeks", "kamera_lebar", "kamera_tinggi", "kamera_gagal_maks", "proses_lebar",
              "proses_tinggi", "audio_blocksize", "audio_ring_blok", "audio_kalibrasi_jumlah",
              "plate_hitung_mundur", "plate_jumlah_frame", "plate_inpaint_radius", "plate_tepi_px", "mask_kernel",
              "debu_ukuran_butir", "debu_ukuran_butir_turun", "debu_seed", "abu_jumlah",
              "abu_ukuran_maks", "fps_target", "fps_batas_turun", "fps_jendela_frame", "uji_jumlah_frame")
_KUNCI_PECAHAN = ("plate_bobot_baru", "plate_ambang_bg", "plate_maks_orang", "plate_ambang_orang", "plate_cocok_min",
                  "mask_ema_baru", "mask_ambang", "mask_luas_min_komponen", "mask_luas_min_orang",
                  "debu_bobot_vertikal", "debu_bobot_noise", "debu_amplitudo_oktaf", "abu_alpha",
                  "flash_ambang_putih", "tengah_turun_min", "tangan_cocok_maks")
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
    # Pergantian ke frame beku harus terjadi saat seluruh layar sudah putih penuh.
    if cfg["flash_tunda_ujung_detik"] >= cfg["flash_tahan_detik"]:
        salah.append("flash_tunda_ujung_detik harus lebih kecil dari flash_tahan_detik")
    if cfg["debu_mulai_detik"] < cfg["flash_naik_detik"] + cfg["flash_tunda_ujung_detik"]:
        salah.append("debu_mulai_detik tidak boleh sebelum layar putih penuh")
    return salah



def hex_ke_bgr(warna: str) -> np.ndarray:
    return np.array([int(warna[5:7], 16), int(warna[3:5], 16), int(warna[1:3], 16)], dtype=np.float32)


def normalisasi(nilai: np.ndarray) -> np.ndarray:
    rendah, tinggi = float(nilai.min()), float(nilai.max())
    if tinggi - rendah < 1e-9:
        return np.zeros(nilai.shape, dtype=np.float32)
    return ((nilai - rendah) / (tinggi - rendah)).astype(np.float32)


def kurva_flash(t: Any, cfg: Dict[str, Any]) -> np.ndarray:
    """Kekuatan flash 0..1 pada t detik sejak jentikan; t boleh skalar atau array."""
    t = np.asarray(t, dtype=np.float32)
    naik, tahan, turun = cfg["flash_naik_detik"], cfg["flash_tahan_detik"], cfg["flash_turun_detik"]
    xn, yn = cfg["flash_naik_tabel"]
    xt, yt = cfg["flash_turun_tabel"]
    # np.interp menahan nilai di ujung tabel, jadi cukup dipotong per fase.
    s_naik = np.interp(t / naik, xn, yn)
    s_turun = np.interp((t - naik - tahan) / turun, xt, yt)
    hasil = np.where(t < naik, s_naik, np.where(t < naik + tahan, 1.0, s_turun))
    hasil = np.where((t < 0) | (t >= naik + tahan + turun), 0.0, hasil)
    return hasil.astype(np.float32)


def noise_fraktal(tinggi: int, lebar: int, oktaf: Tuple[int, ...], amp_kali: float,
                  rng: np.random.Generator) -> np.ndarray:
    total = np.zeros((tinggi, lebar), dtype=np.float32)
    amp = 1.0
    for sel in oktaf:
        kasar = rng.random((tinggi // sel + 2, lebar // sel + 2), dtype=np.float32)
        total += amp * cv2.resize(kasar, (lebar, tinggi), interpolation=cv2.INTER_CUBIC)
        amp *= amp_kali
    return normalisasi(total)


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
        if self.mask is None or self.mask.shape != baru.shape:
            self.mask = baru
        else:
            e = self.cfg["mask_ema_baru"]
            self.mask = cv2.addWeighted(baru, e, self.mask, 1.0 - e, 0.0)
        return self.mask

    def reset(self) -> None:
        self.mask = None

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


class BackgroundModel:
    """Plate berjalan otomatis dan clean plate manual, keduanya di resolusi proses."""

    OK = "ok"
    ADA_ORANG = "ada_orang"
    TIDAK_COCOK = "tidak_cocok"

    def __init__(self, cfg: Dict[str, Any], lebar: int, tinggi: int):
        self.cfg = cfg
        self.lebar, self.tinggi = lebar, tinggi
        self.plate = None  # type: Optional[np.ndarray]
        self.manual = None  # type: Optional[np.ndarray]
        self.terlihat = np.zeros((tinggi, lebar), dtype=bool)
        self._jumlah = None  # type: Optional[np.ndarray]
        self._n = 0
        self._ada_orang = False

    @property
    def status(self) -> str:
        if self.manual is not None:
            return "BG TERSIMPAN"
        return "BG OTOMATIS %d%%" % int(round(100.0 * float(self.terlihat.mean())))

    def _sesuaikan(self, frame: np.ndarray, plate: np.ndarray, area: np.ndarray) -> Tuple[np.ndarray, float]:
        """Rasio eksposur per kanal plate terhadap frame, dan beda rata-rata setelah dikoreksi."""
        if not area.any():
            return np.ones(3, dtype=np.float32), 0.0
        f = frame[area].astype(np.float32)
        p = plate[area].astype(np.float32)
        rasio = np.clip(f.mean(axis=0) / np.maximum(p.mean(axis=0), 1.0),
                        self.cfg["eksposur_min"], self.cfg["eksposur_maks"]).astype(np.float32)
        return rasio, float(np.abs(f - p * rasio).mean())

    def perbarui(self, frame: np.ndarray, conf: np.ndarray) -> None:
        f = frame.astype(np.float32)
        bg = conf < self.cfg["plate_ambang_bg"]
        if self.plate is None:
            self.plate = f.copy()
        # Piksel yang baru pertama kali terlihat langsung diambil utuh, sisanya dirata-rata pelan.
        bobot = np.where(self.terlihat, np.float32(self.cfg["plate_bobot_baru"]), np.float32(1.0))
        bobot[~bg] = 0.0
        self.plate += (f - self.plate) * bobot[..., None]
        self.terlihat |= bg

    def set_manual(self, frame: np.ndarray) -> None:
        self.manual = frame.astype(np.float32)

    def mulai_manual(self) -> None:
        self._jumlah = np.zeros((self.tinggi, self.lebar, 3), dtype=np.float64)
        self._n = 0
        self._ada_orang = False

    def tambah_manual(self, frame: np.ndarray, conf: np.ndarray) -> Optional[str]:
        """None selama masih merekam, lalu OK, ADA_ORANG, atau TIDAK_COCOK."""
        if self._jumlah is None:
            return self.ADA_ORANG
        if float((conf > self.cfg["plate_ambang_orang"]).mean()) > self.cfg["plate_maks_orang"]:
            self._ada_orang = True
        self._jumlah += frame
        self._n += 1
        if self._n < self.cfg["plate_jumlah_frame"]:
            return None
        rata = (self._jumlah / self._n).astype(np.float32)
        self._jumlah = None
        if self._ada_orang:
            return self.ADA_ORANG
        # Kamera tertutup atau bergeser terlihat dari area yang sudah dikenal plate otomatis.
        if self.plate is not None and self.terlihat.mean() >= self.cfg["plate_cocok_min"]:
            _, beda = self._sesuaikan(self.plate, rata, self.terlihat)
            if beda > self.cfg["plate_beda_maks"]:
                return self.TIDAK_COCOK
        self.set_manual(rata)
        return self.OK

    def plate_untuk_snap(self, frame: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """Plate uint8 terkoreksi eksposur, apakah lengkap tanpa inpaint, dan apakah plate manual dibuang."""
        bg_live = conf < self.cfg["plate_ambang_bg"]
        dibuang = False
        if self.manual is not None:
            rasio, beda = self._sesuaikan(frame, self.manual, bg_live)
            if beda <= self.cfg["plate_beda_maks"]:
                return cv2.convertScaleAbs(self.manual * rasio), True, False
            # Kamera bergeser sejak plate direkam; plate manual tidak bisa dipakai lagi.
            self.manual = None
            dibuang = True

        if self.plate is not None:
            dasar = self.plate
            terlihat = self.terlihat
        else:
            dasar = frame.astype(np.float32)
            terlihat = bg_live
        rasio, _ = self._sesuaikan(frame, dasar, terlihat & bg_live)
        hasil = cv2.convertScaleAbs(dasar * rasio)
        kurang = (~terlihat).astype(np.uint8)
        if not kurang.any():
            return hasil, True, dibuang
        # Tepi diperlebar supaya warna rambut dan baju di batas siluet tidak ikut dipakai inpaint.
        tepi = 2 * self.cfg["plate_tepi_px"] + 1
        kurang = cv2.dilate(kurang, np.ones((tepi, tepi), dtype=np.uint8))
        hasil = cv2.inpaint(hasil, kurang * 255, self.cfg["plate_inpaint_radius"], cv2.INPAINT_TELEA)
        return hasil, False, dibuang


class FlashOverlay:
    """Flash radial dari titik jentikan, kurva dari pengukuran reference snap."""

    def __init__(self, cfg: Dict[str, Any], lebar: int, tinggi: int):
        self.cfg = cfg
        self.lebar, self.tinggi = lebar, tinggi
        self.dekat = hex_ke_bgr(cfg["flash_warna_dekat"])
        self.jauh = hex_ke_bgr(cfg["flash_warna_jauh"])
        naik = cfg["flash_naik_detik"]
        self.mulai_putih = naik + cfg["flash_tunda_ujung_detik"]
        self.akhir_putih = naik + cfg["flash_tahan_detik"]
        self.akhir = self.akhir_putih + cfg["flash_turun_detik"] + cfg["flash_tunda_ujung_detik"]
        self.jarak = np.zeros((tinggi, lebar), dtype=np.float32)
        self.warna_dasar = np.zeros((tinggi, lebar, 3), dtype=np.float32)

    def mulai(self, asal: Tuple[float, float]) -> None:
        """asal dalam koordinat ternormalisasi 0..1."""
        ys, xs = np.mgrid[0:self.tinggi, 0:self.lebar].astype(np.float32) + 0.5
        d = np.hypot(xs - asal[0] * self.lebar, ys - asal[1] * self.tinggi)
        self.jarak = (d / d.max()).astype(np.float32)
        j = self.jarak[..., None]
        self.warna_dasar = (self.dekat * (1.0 - j) + self.jauh * j).astype(np.float32)

    def kekuatan(self, t: float) -> np.ndarray:
        return kurva_flash(t - self.jarak * self.cfg["flash_tunda_ujung_detik"], self.cfg)

    def putih_penuh(self, t: float) -> bool:
        return self.mulai_putih <= t < self.akhir_putih

    def komposit(self, frame: np.ndarray, t: float) -> np.ndarray:
        if t <= 0 or t >= self.akhir:
            return frame
        if self.putih_penuh(t):
            return np.full_like(frame, 255)
        s = self.kekuatan(t)
        if float(s.max()) <= 0.0:
            return frame
        ambang = self.cfg["flash_ambang_putih"]
        ke_putih = np.clip((s - ambang) / (1.0 - ambang), 0.0, 1.0)[..., None]
        lapisan_warna = (self.warna_dasar * (1.0 - ke_putih) + 255.0 * ke_putih) * s[..., None]
        sisa = 1.0 - s
        tinggi, lebar = frame.shape[:2]
        # Lapisan flash halus, jadi dihitung kecil lalu diperbesar supaya frame tetap tajam.
        if (lebar, tinggi) != (self.lebar, self.tinggi):
            lapisan_warna = cv2.resize(lapisan_warna, (lebar, tinggi), interpolation=cv2.INTER_LINEAR)
            sisa = cv2.resize(sisa, (lebar, tinggi), interpolation=cv2.INTER_LINEAR)
        return cv2.convertScaleAbs(frame.astype(np.float32) * sisa[..., None] + lapisan_warna)


class Disintegration:
    """Butiran debu dari siluet beku, digambar langsung ke posisi barunya tiap frame."""

    def __init__(self, cfg: Dict[str, Any], lebar: int, tinggi: int):
        self.cfg = cfg
        self.lebar, self.tinggi = lebar, tinggi
        self.jumlah = 0
        self.ukuran = cfg["debu_ukuran_butir"]
        self._offset = np.zeros(1, dtype=np.int64)
        self._idx0 = np.zeros(0, dtype=np.int64)
        self.lepas = np.zeros(0, dtype=np.float32)

    def siapkan(self, beku: np.ndarray, mask: np.ndarray, ukuran: int) -> None:
        cfg = self.cfg
        s = ukuran
        gl, gt = self.lebar // s, self.tinggi // s
        warna_grid = cv2.resize(beku[:gt * s, :gl * s], (gl, gt), interpolation=cv2.INTER_AREA)
        mask_grid = cv2.resize(mask[:gt * s, :gl * s].astype(np.float32), (gl, gt), interpolation=cv2.INTER_AREA)
        iy, ix = np.nonzero(mask_grid > cfg["mask_ambang"])
        rng = np.random.default_rng(cfg["debu_seed"])

        self.ukuran = s
        self.jumlah = len(iy)
        self._offset = np.array([dy * self.lebar + dx for dy in range(s) for dx in range(s)], dtype=np.int64)
        self.x0 = (ix * s).astype(np.float32)
        self.y0 = (iy * s).astype(np.float32)
        self._idx0 = (iy * s).astype(np.int64) * self.lebar + (ix * s).astype(np.int64)
        self.warna = warna_grid[iy, ix].astype(np.float32)
        self._beku = beku.reshape(-1, 3).astype(np.float32)
        if self.jumlah == 0:
            self.lepas = np.zeros(0, dtype=np.float32)
            return

        noise = noise_fraktal(gt, gl, tuple(cfg["debu_oktaf"]), cfg["debu_amplitudo_oktaf"], rng)[iy, ix]
        ynorm = (iy - iy.min()) / float(max(1, iy.max() - iy.min()))
        urutan = normalisasi(cfg["debu_bobot_vertikal"] * ynorm + cfg["debu_bobot_noise"] * normalisasi(noise))
        self.lepas = (urutan * cfg["debu_sapuan_detik"]).astype(np.float32)
        self.jitter = rng.uniform(cfg["debu_jitter_min"], cfg["debu_jitter_maks"],
                                  (self.jumlah, 2)).astype(np.float32)
        self.fase = rng.uniform(0.0, 2.0 * math.pi, self.jumlah).astype(np.float32)

    def posisi(self, pilih: np.ndarray, umur_butir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posisi kiri atas butir terpilih setelah terbang selama umur_butir detik."""
        cfg = self.cfg
        at = umur_butir
        jit = self.jitter[pilih]
        arah_x = 1.0 if cfg["debu_balik_horizontal"] else -1.0
        dx = arah_x * (cfg["debu_angin_kiri"] * at + 0.5 * cfg["debu_percepatan_kiri"] * at * at) * self.lebar * jit[:, 0]
        dy = -(cfg["debu_angin_naik"] * at + 0.5 * cfg["debu_percepatan_naik"] * at * at) * self.tinggi * jit[:, 1]
        fase = self.fase[pilih]
        # Goyangan dimulai dari nol supaya butir tidak melompat saat lepas.
        dy += cfg["debu_goyang_px"] * (np.sin(2.0 * math.pi * cfg["debu_goyang_hz"] * at + fase) - np.sin(fase))
        return self.x0[pilih] + dx, self.y0[pilih] + dy

    def selesai(self, t_debu: float) -> bool:
        return self.jumlah == 0 or t_debu >= float(self.lepas.max()) + self.cfg["debu_umur_detik"]

    def render(self, kanvas: np.ndarray, t_debu: float) -> int:
        """Gambar butir ke kanvas float32 (tinggi*lebar, 3); kembalikan jumlah butir yang tampil."""
        if self.jumlah == 0:
            return 0
        cfg = self.cfg
        umur = cfg["debu_umur_detik"]
        a = np.float32(t_debu) - self.lepas

        # Butir yang belum lepas memakai piksel asli frame beku, jadi badan tetap tajam.
        belum = a <= 0
        if belum.any():
            dasar = self._idx0[belum]
            for off in self._offset:
                idx = dasar + off
                kanvas[idx] = self._beku[idx]

        terbang = np.nonzero((a > 0) & (a < umur))[0]
        if len(terbang) == 0:
            return int(belum.sum())
        at = a[terbang]
        xf, yf = self.posisi(terbang, at)

        s = self.ukuran
        x = np.rint(xf).astype(np.int64)
        y = np.rint(yf).astype(np.int64)
        di_layar = (x >= 0) & (x <= self.lebar - s) & (y >= 0) & (y <= self.tinggi - s)
        x, y = x[di_layar], y[di_layar]
        alpha = (1.0 - at[di_layar] / umur)[:, None]
        warna_a = self.warna[terbang[di_layar]] * alpha
        sisa = 1.0 - alpha
        dasar = y * self.lebar + x
        for off in self._offset:
            idx = dasar + off
            kanvas[idx] = kanvas[idx] * sisa + warna_a
        return int(belum.sum()) + len(x)


class AshAmbient:
    """Bintik abu kecil yang melayang searah angin setelah flash."""

    def __init__(self, cfg: Dict[str, Any], lebar: int, tinggi: int):
        self.cfg = cfg
        self.lebar, self.tinggi = lebar, tinggi
        self.warna = hex_ke_bgr(cfg["abu_warna"])
        self.x0 = np.zeros(0, dtype=np.float32)

    def siapkan(self, rng: np.random.Generator) -> None:
        n = self.cfg["abu_jumlah"]
        self.x0 = rng.uniform(0, self.lebar, n).astype(np.float32)
        self.y0 = rng.uniform(0, self.tinggi, n).astype(np.float32)
        self.ukuran = rng.integers(1, self.cfg["abu_ukuran_maks"] + 1, n)

    def render(self, kanvas: np.ndarray, t: float) -> None:
        """t dalam detik sejak jentikan; kanvas float32 (tinggi*lebar, 3)."""
        cfg = self.cfg
        mulai, akhir = cfg["debu_mulai_detik"], cfg["abu_selesai_detik"]
        if not cfg["abu_ambient"] or len(self.x0) == 0 or t < mulai or t >= akhir:
            return
        a = t - mulai
        alpha = cfg["abu_alpha"] * (1.0 - a / (akhir - mulai))
        arah_x = 1.0 if cfg["debu_balik_horizontal"] else -1.0
        rel = cfg["abu_kecepatan_relatif"]
        x = np.mod(self.x0 + arah_x * rel * cfg["debu_angin_kiri"] * self.lebar * a, self.lebar)
        y = np.mod(self.y0 - rel * cfg["debu_angin_naik"] * self.tinggi * a, self.tinggi)
        for s in np.unique(self.ukuran):
            pilih = self.ukuran == s
            xi = np.minimum(x[pilih].astype(np.int64), self.lebar - s)
            yi = np.minimum(y[pilih].astype(np.int64), self.tinggi - s)
            dasar = yi * self.lebar + xi
            for dy in range(s):
                for dx in range(s):
                    idx = dasar + dy * self.lebar + dx
                    kanvas[idx] = kanvas[idx] * (1.0 - alpha) + self.warna * alpha


class SnapController:
    """State machine SNAP. Semua waktu (now) berupa detik monotonic yang diberikan pemanggil."""

    LIVE = "LIVE"
    HITUNG_MUNDUR = "HITUNG_MUNDUR"
    SNAP = "SNAP"
    KOSONG = "KOSONG"
    KEMBALI = "KEMBALI"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        lebar, tinggi = cfg["proses_lebar"], cfg["proses_tinggi"]
        self.lebar, self.tinggi = lebar, tinggi
        self.bg = BackgroundModel(cfg, lebar, tinggi)
        self.flash = FlashOverlay(cfg, lebar, tinggi)
        self.debu = Disintegration(cfg, lebar, tinggi)
        self.abu = AshAmbient(cfg, lebar, tinggi)
        self.ukuran_butir = cfg["debu_ukuran_butir"]

        self.t_ganti = self.flash.mulai_putih
        self.debu_akhir = cfg["debu_mulai_detik"] + cfg["debu_sapuan_detik"] + cfg["debu_umur_detik"]
        self.kosong_akhir = self.debu_akhir + cfg["kosong_detik"]
        self.kembali_akhir = self.kosong_akhir + cfg["kembali_detik"]

        self.state = self.LIVE
        self.t0 = 0.0
        self.cooldown_sampai = -math.inf
        self.t_rekam = 0.0
        self.butir_tampil = 0
        self._pesan = ""
        self._pesan_sampai = -math.inf
        self._frame = None  # type: Optional[np.ndarray]
        self._mask = None  # type: Optional[np.ndarray]
        self._kanvas = np.zeros((tinggi, lebar, 3), dtype=np.float32)
        self._kanvas_datar = self._kanvas.reshape(-1, 3)
        self._kosongkan_snapshot()

    def _kosongkan_snapshot(self) -> None:
        self._beku = None  # type: Optional[np.ndarray]
        self._biner = None  # type: Optional[np.ndarray]
        self._conf_beku = None  # type: Optional[np.ndarray]
        self._plate_f = None  # type: Optional[np.ndarray]
        self._bg_lengkap = True
        self._bg_dibuang = False
        self._siap = False
        self._cache = None  # type: Optional[np.ndarray]

    def set_pesan(self, teks: str, now: float, durasi: Optional[float] = None) -> None:
        self._pesan = teks
        self._pesan_sampai = now + (self.cfg["pesan_detik"] if durasi is None else durasi)

    def pesan(self, now: float) -> str:
        return self._pesan if now < self._pesan_sampai else ""

    def waktu_efek(self, now: float) -> float:
        return now - self.t0 if self.state in (self.SNAP, self.KOSONG, self.KEMBALI) else 0.0

    def butuh_segmentasi(self) -> bool:
        return self.state in (self.LIVE, self.HITUNG_MUNDUR)

    def siap_dipicu(self, now: float) -> bool:
        return self.state == self.LIVE and now >= self.cooldown_sampai

    def sisa_hitung_mundur(self, now: float) -> int:
        return max(0, int(math.ceil(self.t_rekam - now)))

    def amati(self, now: float, frame_proses: np.ndarray, mask: np.ndarray) -> None:
        """Masukkan frame resolusi proses dan mask orang terbaru (hanya saat LIVE atau HITUNG_MUNDUR)."""
        self._frame = frame_proses
        self._mask = mask
        if self.state == self.LIVE:
            self.bg.perbarui(frame_proses, mask)
        elif self.state == self.HITUNG_MUNDUR and now >= self.t_rekam:
            hasil = self.bg.tambah_manual(frame_proses, mask)
            if hasil is None:
                return
            self.set_pesan({
                BackgroundModel.OK: "Background tersimpan",
                BackgroundModel.ADA_ORANG: "Masih ada orang di frame, tekan B lagi",
                BackgroundModel.TIDAK_COCOK: "Background tidak cocok. Keluar dari frame tanpa menutup kamera, tekan B lagi",
            }[hasil], now, self.cfg["pesan_detik"] * 2)
            self._masuk_live(now)

    def mulai_rekam_plate(self, now: float) -> None:
        if self.state != self.LIVE:
            return
        self.state = self.HITUNG_MUNDUR
        self.t_rekam = now + self.cfg["plate_hitung_mundur"]
        self.bg.mulai_manual()

    def picu(self, now: float, asal: Optional[Tuple[float, float]] = None) -> bool:
        """Mulai efek dari frame dan mask terakhir. asal = titik jentik ternormalisasi 0..1."""
        if not self.siap_dipicu(now):
            return False
        if self._frame is None or self._mask is None:
            self.set_pesan("Tidak ada orang terdeteksi", now)
            return False
        biner, luas = PersonSegmenter.mask_snap(self._mask, self.cfg)
        if luas < self.cfg["mask_luas_min_orang"]:
            self.set_pesan("Tidak ada orang terdeteksi", now)
            return False
        self._kosongkan_snapshot()
        # Snapshot dibekukan di sini; mask live tidak dipakai lagi selama efek.
        self._beku = self._frame.copy()
        self._biner = biner
        self._conf_beku = self._mask.copy()
        self.flash.mulai(asal if asal is not None else tuple(self.cfg["flash_asal_cadangan"]))
        self.state = self.SNAP
        self.t0 = now
        return True

    def reset(self, now: float) -> None:
        self._masuk_live(now)

    def _masuk_live(self, now: float) -> None:
        self.state = self.LIVE
        self.cooldown_sampai = now + self.cfg["cooldown_detik"]
        self.butir_tampil = 0
        self._kosongkan_snapshot()

    def _siapkan(self) -> None:
        """Dipanggil sekali saat layar putih penuh, jadi jeda persiapan tidak terlihat."""
        plate, self._bg_lengkap, self._bg_dibuang = self.bg.plate_untuk_snap(self._beku, self._conf_beku)
        self._plate_f = plate.astype(np.float32)
        self.debu.siapkan(self._beku, self._biner, self.ukuran_butir)
        self.abu.siapkan(np.random.default_rng(self.cfg["debu_seed"] + 1))
        self._siap = True

    def _komposisi(self, t: float, frame_tampil: np.ndarray) -> np.ndarray:
        tinggi, lebar = frame_tampil.shape[:2]
        statis = t >= self.debu_akhir and t >= self.cfg["abu_selesai_detik"] and not self.cfg["latar_live"]
        if statis and self._cache is not None and self._cache.shape == frame_tampil.shape:
            return self._cache
        if self.cfg["latar_live"]:
            kecil = cv2.resize(frame_tampil, (self.lebar, self.tinggi), interpolation=cv2.INTER_AREA)
            np.copyto(self._kanvas, kecil, casting="unsafe")
        else:
            np.copyto(self._kanvas, self._plate_f)
        t_debu = t - self.cfg["debu_mulai_detik"]
        self.butir_tampil = self.debu.render(self._kanvas_datar, t_debu) if t < self.debu_akhir else 0
        self.abu.render(self._kanvas_datar, t)
        hasil = cv2.resize(cv2.convertScaleAbs(self._kanvas), (lebar, tinggi), interpolation=cv2.INTER_LINEAR)
        if statis:
            self._cache = hasil
        return hasil

    def render(self, now: float, frame_tampil: np.ndarray) -> np.ndarray:
        """Frame akhir (uint8, resolusi tampilan) tanpa teks HUD."""
        if self.state in (self.LIVE, self.HITUNG_MUNDUR):
            return frame_tampil
        t = now - self.t0
        if t >= self.kembali_akhir:
            self._masuk_live(now)
            return frame_tampil
        if t >= self.kosong_akhir:
            self.state = self.KEMBALI
        elif t >= self.debu_akhir and self.state == self.SNAP:
            self.state = self.KOSONG
            if self._bg_dibuang:
                self.set_pesan("Kamera bergeser sejak background direkam, tekan B untuk rekam ulang", now,
                               self.cfg["kosong_detik"])
            elif not self._bg_lengkap:
                self.set_pesan("Background belum lengkap, tekan B untuk hasil terbaik", now,
                               self.cfg["kosong_detik"])

        if t < self.t_ganti:
            hasil = frame_tampil
        else:
            if not self._siap:
                self._siapkan()
            if self.flash.putih_penuh(t):
                return np.full_like(frame_tampil, 255)
            hasil = self._komposisi(t, frame_tampil)
        if self.state == self.KEMBALI:
            k = (t - self.kosong_akhir) / self.cfg["kembali_detik"]
            hasil = cv2.addWeighted(hasil, 1.0 - k, frame_tampil, k, 0.0)
        return self.flash.komposit(hasil, t)


class SnapGestureDetector:
    """Deteksi jentikan dari landmark: cubit jempol ke jari tengah, lalu lepas cepat sambil jari tengah turun."""

    DIAM = "DIAM"
    SIAP = "SIAP"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.tangan = []  # type: List[Dict[str, Any]]

    def reset(self) -> None:
        self.tangan = []

    @staticmethod
    def rasio(lm: np.ndarray) -> Tuple[float, float]:
        """(rasio_cubit, rasio_tengah), keduanya dibagi jarak pergelangan ke pangkal jari tengah."""
        skala = max(float(np.linalg.norm(lm[0] - lm[9])), 1e-6)
        return float(np.linalg.norm(lm[4] - lm[12])) / skala, float(np.linalg.norm(lm[12] - lm[0])) / skala

    def _jejak_baru(self, lm: np.ndarray) -> Dict[str, Any]:
        return {"state": self.DIAM, "lm": lm, "c": 0.0, "m": 0.0, "t_cubit": None, "t_siap": 0.0,
                "m_ref": 0.0, "titik": (0.0, 0.0), "t_buka": None, "t_hilang": None, "perlu_buka": False,
                "t_jentik": -math.inf}

    def _cocokkan(self, daftar: List[np.ndarray], lebar: int) -> Dict[int, int]:
        """Pasangkan tangan baru ke jejak lama lewat pergelangan terdekat; urutan dari MediaPipe bisa tertukar."""
        pasangan = sorted((float(np.linalg.norm(jejak["lm"][0] - lm[0])), i, j)
                          for i, jejak in enumerate(self.tangan) for j, lm in enumerate(daftar))
        hasil = {}  # type: Dict[int, int]
        dipakai = set()
        for jarak, i, j in pasangan:
            if j in hasil or i in dipakai or jarak > self.cfg["tangan_cocok_maks"] * lebar:
                continue
            hasil[j] = i
            dipakai.add(i)
        return hasil

    def perbarui(self, now: float, daftar: List[np.ndarray], lebar: int,
                 tinggi: int) -> List[Tuple[float, Tuple[float, float]]]:
        """daftar berisi landmark (21, 2) piksel per tangan. Kembalikan event (waktu, titik jentik 0..1)."""
        cfg = self.cfg
        event = []  # type: List[Tuple[float, Tuple[float, float]]]
        cocok = self._cocokkan(daftar, lebar)
        lama = self.tangan
        baru = []  # type: List[Dict[str, Any]]

        for i, jejak in enumerate(lama):
            if i in cocok.values():
                continue
            if jejak["t_hilang"] is None:
                jejak["t_hilang"] = now
            if now - jejak["t_hilang"] <= cfg["toleransi_hilang_detik"]:
                baru.append(jejak)

        for j, lm in enumerate(daftar):
            jejak = lama[cocok[j]] if j in cocok else self._jejak_baru(lm)
            c, m = self.rasio(lm)
            jejak["lm"], jejak["c"], jejak["m"] = lm, c, m
            titik = (float(lm[4][0] + lm[12][0]) / 2.0 / lebar, float(lm[4][1] + lm[12][1]) / 2.0 / tinggi)
            baru.append(jejak)

            if jejak["t_hilang"] is not None:
                hilang = now - jejak["t_hilang"]
                jejak["t_hilang"] = None
                # Lepasan jentik sering kabur sehingga tangan hilang sebentar dari deteksi.
                if jejak["state"] == self.SIAP and hilang <= cfg["toleransi_hilang_detik"] and c > cfg["cubit_lepas"]:
                    event.append((now, jejak["titik"]))
                    self._ke_diam(jejak, now, jentik=True)
                    continue

            if jejak["state"] == self.DIAM:
                if c >= cfg["cubit_masuk"]:
                    jejak["t_cubit"] = None
                    jejak["perlu_buka"] = False
                elif not jejak["perlu_buka"]:
                    if jejak["t_cubit"] is None:
                        jejak["t_cubit"] = now
                    if now - jejak["t_cubit"] >= cfg["cubit_tahan_detik"]:
                        jejak.update(state=self.SIAP, t_siap=now, m_ref=m, titik=titik, t_buka=None)
            elif c < cfg["cubit_masuk"]:
                jejak.update(t_buka=None, m_ref=max(jejak["m_ref"], m), titik=titik)
                if now - jejak["t_siap"] > cfg["siap_maks_detik"]:
                    # Harus membuka tangan dulu sebelum bisa SIAP lagi.
                    self._ke_diam(jejak, now)
                    jejak["perlu_buka"] = True
            else:
                if jejak["t_buka"] is None:
                    jejak["t_buka"] = now
                if now - jejak["t_buka"] > cfg["lepas_maks_detik"]:
                    self._ke_diam(jejak, now)
                elif c > cfg["cubit_lepas"] and m <= jejak["m_ref"] * (1.0 - cfg["tengah_turun_min"]):
                    event.append((now, jejak["titik"]))
                    self._ke_diam(jejak, now, jentik=True)

        self.tangan = baru
        return event

    def _ke_diam(self, jejak: Dict[str, Any], now: float, jentik: bool = False) -> None:
        jejak.update(state=self.DIAM, t_cubit=None, t_buka=None)
        if jentik:
            jejak["t_jentik"] = now


class OnsetAudio:
    """Deteksi onset per blok audio; terpisah dari thread supaya bisa diuji dengan sinyal sintetis."""

    def __init__(self, cfg: Dict[str, Any], samplerate: int):
        n = cfg["audio_blocksize"]
        self.cfg = cfg
        self.n = n
        self.jendela = np.hanning(n).astype(np.float32)
        self.pita = np.fft.rfftfreq(n, 1.0 / samplerate) >= cfg["audio_highpass_hz"]
        panjang = max(1, int(cfg["audio_jendela_noise_detik"] * samplerate / n))
        self.riwayat = collections.deque(maxlen=panjang)  # type: collections.deque
        self.sebelum = 0.0
        self.noise = 0.0
        self.ambang_kalibrasi = None  # type: Optional[float]

    def energi(self, blok: np.ndarray) -> float:
        if len(blok) != self.n:
            blok = np.resize(blok, self.n) if len(blok) else np.zeros(self.n, dtype=np.float32)
        return float(np.mean(np.abs(np.fft.rfft(blok * self.jendela)[self.pita]) ** 2))

    def ambang(self) -> float:
        if self.ambang_kalibrasi is not None:
            return self.ambang_kalibrasi
        return self.noise * self.cfg["audio_ambang_awal_kali_noise"]

    def proses(self, blok: np.ndarray) -> Tuple[float, bool]:
        e = self.energi(blok)
        self.noise = float(np.median(self.riwayat)) if self.riwayat else e
        # Blok awal stream sering nol, jadi tunggu riwayat noise penuh sebelum menandai onset.
        siap = len(self.riwayat) == self.riwayat.maxlen
        onset = siap and e > self.ambang() and e > self.cfg["audio_onset_kali"] * self.sebelum
        self.riwayat.append(e)
        self.sebelum = e
        return e, onset


class AudioSnapDetector:
    """Bunyi jentikan dari mikrofon. Callback hanya menyalin ke ring buffer; analisis di thread pekerja."""

    def __init__(self, cfg: Dict[str, Any], path_kalibrasi: str):
        self.cfg = cfg
        self.path_kalibrasi = path_kalibrasi
        self.aktif = False
        self.error = ""
        self.diam = False
        self.level = 0.0
        self.ambang = 0.0
        self.onset = None  # type: Optional[OnsetAudio]
        self.ambang_tersimpan = self._muat_kalibrasi()
        self._event = queue.Queue()  # type: queue.Queue
        self._pesan = queue.Queue()  # type: queue.Queue
        self._antrean = queue.Queue()  # type: queue.Queue
        self._stop = threading.Event()
        self._kunci = threading.Lock()
        self._kal = None  # type: Optional[Dict[str, Any]]
        self._stream = None
        self._thread = None  # type: Optional[threading.Thread]
        self._ring = np.zeros((cfg["audio_ring_blok"], cfg["audio_blocksize"]), dtype=np.float32)
        self._waktu = np.zeros(cfg["audio_ring_blok"], dtype=np.float64)
        self._tulis = 0

    def _muat_kalibrasi(self) -> Optional[float]:
        try:
            with open(self.path_kalibrasi) as f:
                ambang = float(json.load(f)["ambang"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return ambang if ambang > 0 else None

    def mulai(self) -> bool:
        if sd is None:
            self.error = "Modul sounddevice tidak tersedia"
            return False
        try:
            info = sd.query_devices(kind="input")
            samplerate = int(info["default_samplerate"])
            self.onset = OnsetAudio(self.cfg, samplerate)
            self.onset.ambang_kalibrasi = self.ambang_tersimpan
            self._stream = sd.InputStream(samplerate=samplerate, blocksize=self.cfg["audio_blocksize"],
                                          channels=1, dtype="float32", callback=self._callback)
            self._stream.start()
        except Exception as e:
            self.error = "Mikrofon tidak bisa dibuka (%s)" % e
            self._tutup_stream()
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._kerja, name="audio-snap", daemon=True)
        self._thread.start()
        self.aktif = True
        return True

    def _callback(self, indata: np.ndarray, frames: int, waktu: Any, status: Any) -> None:
        # Berjalan di thread audio real-time: tanpa alokasi array baru, hanya salin ke slot ring buffer.
        i = self._tulis % len(self._ring)
        self._ring[i, :frames] = indata[:frames, 0]
        self._waktu[i] = time.monotonic()
        self._tulis += 1
        self._antrean.put_nowait(i)

    def _kerja(self) -> None:
        cfg = self.cfg
        terakhir = time.monotonic()
        t_bersuara = terakhir
        try:
            while not self._stop.is_set():
                try:
                    i = self._antrean.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - terakhir > cfg["audio_putus_detik"]:
                        self.error = "Mikrofon terputus"
                        self.aktif = False
                        return
                    continue
                blok = self._ring[i].copy()
                t = float(self._waktu[i])
                terakhir = time.monotonic()
                # Izin mikrofon yang ditolak di macOS tidak memunculkan error, hanya sampel nol.
                if np.any(blok):
                    t_bersuara = t
                    self.diam = False
                elif t - t_bersuara > cfg["audio_diam_detik"]:
                    self.diam = True
                e, onset = self.onset.proses(blok)
                self.level = e
                self.ambang = self.onset.ambang()
                with self._kunci:
                    kal = self._kal
                if kal is not None:
                    self._langkah_kalibrasi(kal, t, e)
                elif onset:
                    self._event.put(t)
        except Exception as e:
            self.error = "Audio berhenti (%s)" % e
            self.aktif = False

    def mulai_kalibrasi(self, now: float) -> bool:
        if not self.aktif:
            return False
        with self._kunci:
            self._kal = {"mulai": now, "hening": [], "jentik": []}
        return True

    def sedang_kalibrasi(self) -> bool:
        with self._kunci:
            return self._kal is not None

    def teks_kalibrasi(self, now: float) -> str:
        with self._kunci:
            kal = self._kal
        if kal is None:
            return ""
        dt = now - kal["mulai"]
        hening = self.cfg["audio_kalibrasi_hening_detik"]
        if dt < hening:
            return "Kalibrasi suara: diam sebentar... %d" % int(math.ceil(hening - dt))
        sisa = hening + self.cfg["audio_kalibrasi_jentik_detik"] - dt
        return "Kalibrasi suara: jentikkan jari %d kali... %d" % (
            self.cfg["audio_kalibrasi_jumlah"], max(1, int(math.ceil(sisa))))

    def _langkah_kalibrasi(self, kal: Dict[str, Any], t: float, e: float) -> None:
        dt = t - kal["mulai"]
        hening = self.cfg["audio_kalibrasi_hening_detik"]
        if dt < 0:
            return
        if dt < hening:
            kal["hening"].append(e)
        elif dt < hening + self.cfg["audio_kalibrasi_jentik_detik"]:
            kal["jentik"].append((t, e))
        else:
            with self._kunci:
                self._kal = None
            self._pesan.put(self._selesaikan_kalibrasi(kal))

    def _selesaikan_kalibrasi(self, kal: Dict[str, Any]) -> str:
        cfg = self.cfg
        if not kal["hening"] or not kal["jentik"]:
            return "Kalibrasi gagal, tekan K lagi"
        # Puncak saat hening, bukan median: median terlalu rendah sehingga suara bicara ikut memicu.
        noise = max(kal["hening"])
        puncak = []  # type: List[Tuple[float, float]]
        for t, e in sorted(kal["jentik"], key=lambda p: -p[1]):
            if e <= noise:
                break
            if all(abs(t - tp) >= cfg["audio_kalibrasi_jarak_detik"] for tp, _ in puncak):
                puncak.append((t, e))
            if len(puncak) == cfg["audio_kalibrasi_jumlah"]:
                break
        if len(puncak) < cfg["audio_kalibrasi_jumlah"]:
            return "Kalibrasi gagal: hanya %d jentikan terdengar, tekan K lagi" % len(puncak)
        ambang = math.sqrt(max(noise, 1e-12) * min(e for _, e in puncak))
        self.onset.ambang_kalibrasi = ambang
        self.ambang_tersimpan = ambang
        try:
            with open(self.path_kalibrasi, "w") as f:
                json.dump({"ambang": ambang, "noise": noise, "jentik_terlemah": min(e for _, e in puncak),
                           "dibuat": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        except OSError as e:
            return "Kalibrasi dipakai tapi gagal disimpan (%s)" % e
        return "Kalibrasi suara tersimpan"

    def ambil_event(self) -> List[float]:
        return self._kuras(self._event)

    def ambil_pesan(self) -> List[str]:
        return self._kuras(self._pesan)

    @staticmethod
    def _kuras(q: queue.Queue) -> List[Any]:
        isi = []
        while True:
            try:
                isi.append(q.get_nowait())
            except queue.Empty:
                return isi

    def _tutup_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as e:
            print("Gagal menutup stream audio:", e)
        self._stream = None

    def hentikan(self) -> None:
        self._stop.set()
        self._tutup_stream()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.aktif = False


class PemicuGabungan:
    """Gabungkan event gesture dan suara sesuai mode trigger."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._gestur = []  # type: List[Tuple[float, Tuple[float, float]]]
        self._suara = []  # type: List[float]

    def reset(self) -> None:
        self._gestur = []
        self._suara = []

    def periksa(self, now: float, mode: str, gestur: List[Tuple[float, Tuple[float, float]]],
                suara: List[float]) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """(dipicu, titik asal flash). Titik asal None berarti pakai posisi cadangan."""
        if mode == "GESTUR":
            return (True, gestur[-1][1]) if gestur else (False, None)
        if mode == "SUARA":
            return (True, None) if suara else (False, None)
        if mode == "ATAU":
            if gestur:
                return True, gestur[-1][1]
            return (True, None) if suara else (False, None)
        # DAN: suara biasanya tiba lebih dulu daripada gambar kamera, jadi keduanya ditahan sebentar.
        jendela = self.cfg["audio_dan_jendela_detik"]
        self._gestur = [g for g in self._gestur + gestur if now - g[0] <= jendela]
        self._suara = [t for t in self._suara + suara if now - t <= jendela]
        for t_g, asal in self._gestur:
            if any(abs(t_g - t_s) <= jendela for t_s in self._suara):
                self.reset()
                return True, asal
        return False, None


def buat_lembar_uji(path_gambar: str, path_bg: Optional[str], path_keluar: Optional[str],
                    cfg: Dict[str, Any]) -> int:
    gambar = cv2.imread(path_gambar)
    if gambar is None:
        print("Gambar tidak bisa dibaca: %s" % path_gambar)
        return 1
    path_seg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfie_segmenter.tflite")
    if not os.path.isfile(path_seg):
        print("File model 'selfie_segmenter.tflite' tidak ditemukan. Lokasi yang diharapkan: %s" % path_seg)
        return 1

    ukuran_proses = (cfg["proses_lebar"], cfg["proses_tinggi"])
    ctrl = SnapController(cfg)
    seg = PersonSegmenter(path_seg, cfg)
    try:
        kecil = cv2.resize(gambar, ukuran_proses, interpolation=cv2.INTER_AREA)
        mask = seg.perbarui(kecil, 0)
    finally:
        seg.close()
    if path_bg:
        bg = cv2.imread(path_bg)
        if bg is None:
            print("Gambar background tidak bisa dibaca: %s" % path_bg)
            return 1
        ctrl.bg.set_manual(cv2.resize(bg, ukuran_proses, interpolation=cv2.INTER_AREA))
    ctrl.amati(0.0, kecil, mask)
    if not ctrl.picu(0.0):
        print(ctrl.pesan(0.0))
        return 1

    tinggi, lebar = gambar.shape[:2]
    kolom = int(math.ceil(math.sqrt(cfg["uji_jumlah_frame"])))
    baris = int(math.ceil(cfg["uji_jumlah_frame"] / float(kolom)))
    ubin_l, ubin_t = lebar // kolom, tinggi // kolom
    lembar = np.zeros((ubin_t * baris, ubin_l * kolom, 3), dtype=np.uint8)
    for i in range(cfg["uji_jumlah_frame"]):
        t = i * cfg["uji_selang_detik"]
        ubin = cv2.resize(ctrl.render(t, gambar), (ubin_l, ubin_t), interpolation=cv2.INTER_AREA)
        teks = "t = %.1f s  %s" % (t, ctrl.state)
        cv2.putText(ubin, teks, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ubin, teks, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        b, k = divmod(i, kolom)
        lembar[b * ubin_t:(b + 1) * ubin_t, k * ubin_l:(k + 1) * ubin_l] = ubin

    if not path_keluar:
        dasar, _ = os.path.splitext(path_gambar)
        path_keluar = dasar + "_lembar.png"
    if not cv2.imwrite(path_keluar, lembar):
        print("Gagal menyimpan: %s" % path_keluar)
        return 1
    print("Lembar contoh disimpan ke %s (%d butir debu)" % (path_keluar, ctrl.debu.jumlah))
    return 0


def main_cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Uji efek SNAP tanpa webcam.")
    parser.add_argument("--uji", required=True, metavar="GAMBAR", help="foto orang yang akan dihancurkan")
    parser.add_argument("--bg", metavar="GAMBAR", help="foto background kosong dari sudut kamera yang sama")
    parser.add_argument("--keluar", metavar="PATH", help="lokasi file lembar contoh")
    args = parser.parse_args(argv)
    salah = validasi_config(CONFIG)
    if salah:
        print("CONFIG tidak valid:\n  - " + "\n  - ".join(salah))
        return 1
    return buat_lembar_uji(args.uji, args.bg, args.keluar, CONFIG)


if __name__ == "__main__":
    sys.exit(main_cli())
