"""Snap trigger by sound: onset detection, the microphone thread with calibration, and the combined trigger."""

import collections
import json
import math
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except Exception:  # PortAudio tidak terpasang: mode suara dimatikan, sisanya tetap jalan.
    sd = None


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
            print("Sound input unavailable: the sounddevice module is not installed")
            self.error = "Sound is not available on this computer"
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
            print("Microphone error: %s" % e)
            self.error = "Cannot open the microphone"
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
                        self.error = "The microphone was disconnected"
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
            print("Audio stopped: %s" % e)
            self.error = "Sound stopped working"
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
            return "Sound check: stay quiet... %d" % int(math.ceil(hening - dt))
        sisa = hening + self.cfg["audio_kalibrasi_jentik_detik"] - dt
        return "Sound check: snap %d times... %d" % (
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
            return "Sound check failed. Press K to try again"
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
            return "Sound check failed: I heard %d of %d snaps. Press K to try again" % (
                len(puncak), cfg["audio_kalibrasi_jumlah"])
        ambang = math.sqrt(max(noise, 1e-12) * min(e for _, e in puncak))
        self.onset.ambang_kalibrasi = ambang
        self.ambang_tersimpan = ambang
        try:
            with open(self.path_kalibrasi, "w") as f:
                json.dump({"ambang": ambang, "noise": noise, "jentik_terlemah": min(e for _, e in puncak),
                           "dibuat": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        except OSError as e:
            print("Cannot save %s: %s" % (self.path_kalibrasi, e))
            return "Sound check works, but it could not be saved"
        return "Sound check saved"

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
