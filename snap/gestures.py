"""Hand gestures: the snap (thumb and middle finger), the metal hand, and held GestureRecognizer labels."""

import collections
import math
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

# One hand for HoldGestureDetector.update: landmarks (21, 2) in pixels, top label, its score, and
# whether the snap detector is in SIAP on this hand.
Hand = Tuple[np.ndarray, str, float, bool]
# Label the app gives the metal hand; the GestureRecognizer has no such category.
METAL = "Metal"
# A finger counts as stretched when its tip is this much farther from the wrist than its middle joint.
METAL_STRETCH = 1.15


def tangan_metal(lm: np.ndarray) -> bool:
    """The metal hand: index and pinky stretched, middle and ring folded (with the thumb in, or out as in the
    "I love you" sign). Distances from the wrist, so any rotation and distance from the camera works."""
    pergelangan = lm[0]

    def jarak(i: int) -> float:
        return float(np.hypot(*(lm[i] - pergelangan)))

    def lurus(ujung: int, sendi: int) -> bool:
        return jarak(ujung) > METAL_STRETCH * jarak(sendi)

    def tekuk(ujung: int, sendi: int) -> bool:
        return jarak(ujung) < jarak(sendi)

    return lurus(8, 6) and lurus(20, 18) and tekuk(12, 10) and tekuk(16, 14)


def pasangkan(lama: List[np.ndarray], baru: List[np.ndarray], maks_px: float) -> Dict[int, int]:
    """Pasangkan tangan baru ke jejak lama lewat pergelangan terdekat; urutan dari MediaPipe bisa tertukar.

    Kembalikan {indeks baru: indeks lama}.
    """
    pasangan = sorted((float(np.linalg.norm(a[0] - b[0])), i, j)
                      for i, a in enumerate(lama) for j, b in enumerate(baru))
    hasil = {}  # type: Dict[int, int]
    dipakai = set()
    for jarak, i, j in pasangan:
        if j in hasil or i in dipakai or jarak > maks_px:
            continue
        hasil[j] = i
        dipakai.add(i)
    return hasil


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
        return pasangkan([jejak["lm"] for jejak in self.tangan], daftar, self.cfg["tangan_cocok_maks"] * lebar)

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


class HoldGestureDetector:
    """Held GestureRecognizer labels, tracked per hand.

    A one shot hold (label -> seconds) fires once when one hand shows the label for that long, with at
    least hold_agree_frac of the frames in that window agreeing and no gap longer than hold_gap_max_s.
    After firing, the label must be gone for hold_release_s before it can fire again, and nothing fires
    during the global cooldown. A continuous hold (label -> minimum seconds) is reported in held while
    the label stays up. A hand whose snap detector is in SIAP adds no labels.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.tangan = []  # type: List[Dict[str, Any]]
        self.held = set()  # type: Set[str]
        self._now = 0.0
        self._holds = {}  # type: Dict[str, float]
        self._id_berikut = 0
        self._cooldown_sampai = -math.inf
        # Labels that fired: the time they were last seen gone, or None while still visible.
        self._diblok = {}  # type: Dict[str, Optional[float]]

    def reset(self) -> None:
        """Forget the tracked hands; release and cooldown keep running on time."""
        self.tangan = []
        self.held = set()

    def cooldown(self, now: float) -> None:
        """Start the global cooldown, for example after a snap."""
        self._cooldown_sampai = max(self._cooldown_sampai, now + self.cfg["gesture_cooldown_s"])

    def _lepas(self, now: float, terlihat: Set[str]) -> None:
        for label, sejak in list(self._diblok.items()):
            if label in terlihat:
                self._diblok[label] = None
            elif sejak is None:
                self._diblok[label] = now
            elif now - sejak >= self.cfg["hold_release_s"]:
                del self._diblok[label]

    def update(self, now: float, hands: List[Hand], holds: Dict[str, float], continuous: Dict[str, float],
               lebar: int) -> List[Tuple[str, Tuple[float, float]]]:
        """Feed one frame. Returns the one shot holds that fired as (label, center of the hand in px)."""
        cfg = self.cfg
        self._now = now
        self._holds = holds
        min_score, gap = cfg["hold_min_score"], cfg["hold_gap_max_s"]
        self._lepas(now, {label for _, label, score, _ in hands if score >= min_score})

        cocok = pasangkan([t["lm"] for t in self.tangan], [h[0] for h in hands], cfg["tangan_cocok_maks"] * lebar)
        baru = []  # type: List[Dict[str, Any]]
        for i, jejak in enumerate(self.tangan):
            if i in cocok.values():
                continue
            if jejak["t_hilang"] is None:
                jejak["t_hilang"] = now
            if now - jejak["t_hilang"] <= cfg["toleransi_hilang_detik"]:
                baru.append(jejak)

        fired = []  # type: List[Tuple[str, Tuple[float, float]]]
        for j, (lm, label, score, siap) in enumerate(hands):
            if j in cocok:
                jejak = self.tangan[cocok[j]]
            else:
                jejak = {"id": self._id_berikut, "label": None, "t0": 0.0, "last": 0.0,
                         "frames": collections.deque()}  # type: Dict[str, Any]
                self._id_berikut += 1
            jejak["lm"], jejak["t_hilang"] = lm, None
            baru.append(jejak)
            dicari = label in holds or label in continuous
            kandidat = label if dicari and score >= min_score and not siap and label not in self._diblok else None
            if jejak["label"] is not None and now - jejak["last"] > gap:
                jejak["label"] = None
            if jejak["label"] is None:
                if kandidat is None:
                    continue
                jejak.update(label=kandidat, t0=now, last=now, frames=collections.deque())
            setuju = kandidat == jejak["label"]
            if setuju:
                jejak["last"] = now
            frames = jejak["frames"]  # type: Deque[Tuple[float, bool]]
            frames.append((now, setuju))
            target = jejak["label"]
            if target not in holds:
                continue
            durasi = holds[target]
            while frames and frames[0][0] < now - durasi - 1e-6:
                frames.popleft()
            if now - jejak["t0"] < durasi - 1e-6 or now < self._cooldown_sampai:
                continue
            if sum(1 for _, s in frames if s) >= cfg["hold_agree_frac"] * len(frames):
                fired.append((target, (float(lm[9][0]), float(lm[9][1]))))
                self._diblok[target] = None
                self.cooldown(now)
                jejak["label"] = None

        self.tangan = baru
        self.held = {t["label"] for t in baru
                     if t["label"] in continuous and now - t["last"] <= gap
                     and t["last"] - t["t0"] >= continuous[t["label"]] - 1e-6}
        return fired

    def progress(self) -> List[Tuple[int, Tuple[float, float], float]]:
        """(hand id, center in px, 0..1) of every one shot hold in progress on a visible hand."""
        hasil = []
        for t in self.tangan:
            if t["t_hilang"] is None and t["label"] in self._holds:
                bagian = min(1.0, (self._now - t["t0"]) / self._holds[t["label"]])
                hasil.append((t["id"], (float(t["lm"][9][0]), float(t["lm"][9][1])), bagian))
        return hasil
