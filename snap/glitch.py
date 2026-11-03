"""GlitchRenderer: one step of the Spider-Verse style glitch (snap/glitch_score.py) drawn on a camera frame.

A step is held for a whole glitch tick (several camera frames). Its geometry (cut map, recolor cells, copy
offsets, bursts) is built once on the step's first frame from a seeded generator, relative to the person's
bounding box; every frame then only recolors and stacks the live body with OpenCV, so the picture under
the glitch stays live. Per frame layers are premultiplied BGRA uint8 (saturating OpenCV math); the
geometry is drawn in float32 and stored ready to paste.
"""

import math
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from snap.config import hex_ke_bgr
from snap.glitch_score import GHOST_ALPHA, PALETTE, SEE_THROUGH, Step

# Offsets per level as a share of the body height: cut slides and offset copies.
CUT_SHIFT = (0.0, 0.010, 0.025, 0.050)
COPY_SHIFT = (0.0, 0.025, 0.060, 0.110)
# Cuts per level (low, high) and the hairline gap each cut leaves.
CUT_PIECES = ((0, 0), (1, 2), (3, 5), (6, 9))
GAP_SHARE = 0.010
# Tritone: the darkest and brightest shares of the body take the shadow and highlight tones, the rest the
# flat color, so the body reads flat like the reference while features stay readable.
TRITONE_DARK_SHARE = 0.12
TRITONE_LIGHT_SHARE = 0.08
TRITONE_SHADOW = 0.35
TRITONE_LIGHT = 0.55
# Every PERCENTILE_STRIDE-th body pixel is enough to place the tritone thresholds.
PERCENTILE_STRIDE = 5
# Room around the body box, as a share of the body height: for the body group (copies and cut slides)
# and for the bursts and blobs.
GROUP_PAD = 0.25
EFFECT_PAD = 0.85
# Below this share kept, what is left of the body is one horizontal strip (#332 to #340 of reference 2).
STRIP_KEEP = 0.2
# Random share added to a cell's height (0 top, 1 bottom) when picking which cells stay shown.
KEEP_JITTER = 0.15
# Full frame backgrounds keep this many variants each; a step picks one by its seed.
BACKGROUND_VARIANTS = 4
MATTE_CUTOFF = 0.5
# The matte range that becomes the body's edge; the reference edges are crisp, and a soft matte edge
# would drag a halo of the room along with every move.
EDGE_LOW = 0.3
EDGE_HIGH = 0.7

Kotak = Tuple[int, int, int, int]


def nada(nama: str) -> np.ndarray:
    """Tritone lookup table (1, 256, 3) uint8: index 0 shadow, 1 flat color, 2 highlight."""
    c = hex_ke_bgr(PALETTE[nama])
    if nama == "BLACK":
        gelap, terang = c, np.full(3, 90.0, dtype=np.float32)
    else:
        gelap, terang = c * TRITONE_SHADOW, c + (255.0 - c) * TRITONE_LIGHT
    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    lut[0, 0], lut[0, 1], lut[0, 2:] = np.round(gelap), np.round(c), np.round(terang)
    return lut


def geser(gambar: np.ndarray, dx: int, dy: int, isi: float = 0) -> np.ndarray:
    """Integer shift (no resampling, so flat colors stay exact); uncovered pixels get isi."""
    h, w = gambar.shape[:2]
    hasil = np.full_like(gambar, isi)
    if abs(dx) >= w or abs(dy) >= h:
        return hasil
    hasil[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] = \
        gambar[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return hasil


def over(atas: np.ndarray, bawah: np.ndarray) -> np.ndarray:
    """Premultiplied BGRA 'over' (float32 or uint8)."""
    if atas.dtype == np.uint8:
        sisa = cv2.bitwise_not(cv2.extractChannel(atas, 3))
        return cv2.add(atas, cv2.multiply(bawah, cv2.merge([sisa, sisa, sisa, sisa]), scale=1.0 / 255.0))
    sisa = 1.0 - atas[..., 3]
    return cv2.add(atas, cv2.multiply(bawah, cv2.merge([sisa, sisa, sisa, sisa])))


def rata(nama: str, alpha: np.ndarray) -> np.ndarray:
    """Premultiplied flat palette color layer, uint8 for a uint8 alpha, else float32 (0..1)."""
    c = hex_ke_bgr(PALETTE[nama])
    if alpha.dtype == np.uint8:
        return cv2.multiply(cv2.merge([alpha, alpha, alpha, alpha]), (float(c[0]), float(c[1]), float(c[2]), 255.0),
                            scale=1.0 / 255.0)
    c = c / 255.0
    return cv2.merge([alpha * float(c[0]), alpha * float(c[1]), alpha * float(c[2]), alpha])


def premultiply(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """uint8 premultiplied BGRA from uint8 BGR and alpha."""
    b, g, r = cv2.split(bgr)
    return cv2.multiply(cv2.merge([b, g, r, alpha]), cv2.merge([alpha, alpha, alpha, np.full_like(alpha, 255)]),
                        scale=1.0 / 255.0)


class Tempelan:
    """A premultiplied layer ready to paste many times: its BGR part and 255 minus its alpha, both uint8."""

    def __init__(self, lapis: np.ndarray):
        if lapis.dtype != np.uint8:
            lapis = np.clip(lapis * 255.0 + 0.5, 0, 255).astype(np.uint8)
        sisa = cv2.bitwise_not(cv2.extractChannel(lapis, 3))
        self.bgr = cv2.cvtColor(lapis, cv2.COLOR_BGRA2BGR)
        self.sisa = cv2.merge([sisa, sisa, sisa])


def tempel(gambar: np.ndarray, kotak: Kotak, lapis) -> None:
    """Composite a premultiplied layer (uint8 BGRA or a Tempelan) onto the image region (x0, y0, x1, y1)."""
    if lapis is None:
        return
    if not isinstance(lapis, Tempelan):
        lapis = Tempelan(lapis)
    x0, y0, x1, y1 = kotak
    bagian = gambar[y0:y1, x0:x1]
    gambar[y0:y1, x0:x1] = cv2.add(cv2.multiply(bagian, lapis.sisa, scale=1.0 / 255.0), lapis.bgr)


def indeks_nada(luma: np.ndarray, pilih: np.ndarray) -> np.ndarray:
    """Tritone index (uint8 0, 1, 2) for luma, with thresholds from the selected pixels."""
    nilai = luma[pilih][::PERCENTILE_STRIDE]
    if nilai.size == 0:
        return np.ones(luma.shape, dtype=np.uint8)
    lo, hi = np.percentile(nilai, (100.0 * TRITONE_DARK_SHARE, 100.0 * (1.0 - TRITONE_LIGHT_SHARE)))
    # Only pixels below the dark threshold take the shadow, so an evenly lit area keeps the flat color.
    return (luma >= lo).view(np.uint8) + (luma > hi).view(np.uint8)


def warnai(indeks: np.ndarray, nama: str) -> np.ndarray:
    """BGR uint8 image of a palette color's tritone for a tone index map."""
    return cv2.LUT(cv2.merge([indeks, indeks, indeks]), nada(nama))


def bintang(rng: np.random.Generator, pusat: Tuple[float, float], r_luar: float, r_dalam: float,
            ujung: int) -> np.ndarray:
    """Irregular star polygon (int32 points) like the reference bursts."""
    sudut0 = rng.uniform(0.0, 2.0 * math.pi)
    titik = []
    for i in range(2 * ujung):
        a = sudut0 + math.pi * i / ujung + rng.uniform(-0.18, 0.18)
        r = (r_luar * rng.uniform(0.65, 1.15)) if i % 2 == 0 else (r_dalam * rng.uniform(0.7, 1.2))
        titik.append((pusat[0] + r * math.cos(a), pusat[1] + r * math.sin(a)))
    return np.round(np.array(titik)).astype(np.int32)


def isi_poligon(shape: Tuple[int, int], poli: np.ndarray) -> np.ndarray:
    peta = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(peta, [poli], 255, cv2.LINE_AA)
    return peta.astype(np.float32) * (1.0 / 255.0)


def arah(rng: np.random.Generator, jarak: float) -> Tuple[int, int]:
    """Mostly sideways offset, as in the reference copies."""
    if jarak <= 0.0:
        return 0, 0
    sudut = rng.normal(0.0, 0.45) + (math.pi if rng.random() < 0.5 else 0.0)
    return int(round(jarak * math.cos(sudut))), int(round(jarak * math.sin(sudut)))


def perluas(kotak: Kotak, pad: int, lebar: int, tinggi: int) -> Kotak:
    bx, by, bw, bh = kotak
    return max(0, bx - pad), max(0, by - pad), min(lebar, bx + bw + pad), min(tinggi, by + bh + pad)


def perlu_ruang(step: Step) -> bool:
    """True when the step moves the body away from where the person is, so the room must show there."""
    if step.get("bg") in ("REGIONS", "HALFTONE"):
        return False
    return "cut" in step or any(n == "BODY" and lv > 0 for n, lv in step.get("copies", ()))


class Rencana:
    """The geometry of one step, built on its first frame and reused while the step is held."""

    def __init__(self, kunci: Tuple[int, int, int], step: Step, g: Kotak, e: Kotak):
        # (seed, width, height) and the step itself: holding the step keeps a new step from ever matching.
        self.kunci = kunci
        self.step = step
        # Group region (body, copies, cuts) and effect region (bursts, blobs), as (x0, y0, x1, y1).
        self.g = g
        self.e = e
        self.kuning = False
        # Recolor cells: label map (0 = none, k = cell k) and the palette name of each cell.
        self.sel = None  # type: Optional[np.ndarray]
        self.sel_warna = []  # type: List[str]
        # Body parts still shown (uint8 0 or 255), None for all of it.
        self.sisa = None  # type: Optional[np.ndarray]
        self.buram = []  # type: List[Tuple[str, int, int]]
        self.badan = [(0, 0)]  # type: List[Tuple[int, int]]
        self.tembus = []  # type: List[Tuple[str, int, int]]
        # Mirror: (palette name or BODY, map x, map y).
        self.cermin = None  # type: Optional[Tuple[str, np.ndarray, np.ndarray]]
        self.potong = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]
        self.belakang = None  # type: Optional[Tempelan]
        self.depan = None  # type: Optional[Tempelan]
        self.garis = None  # type: Optional[Tempelan]


class GlitchRenderer:
    """Draws glitch steps; full frame backgrounds are built once per variant and frame size."""

    def __init__(self):
        self._latar: Dict[Tuple[str, int, int, int], Any] = {}
        # Guards _latar: the backgrounds are built on a helper thread while frames are drawn.
        self._kunci = threading.Lock()
        self._dipanaskan: Set[Tuple[int, int]] = set()
        self._rencana = None  # type: Optional[Rencana]

    # Full frame backgrounds

    def latar(self, jenis: str, varian: int, lebar: int, tinggi: int) -> Any:
        """REGIONS and HALFTONE: BGR uint8 frame. SHARDS: an overlay (Tempelan)."""
        kunci = (jenis, varian % BACKGROUND_VARIANTS, lebar, tinggi)
        with self._kunci:
            if kunci not in self._latar:
                rng = np.random.default_rng(1000 + 17 * kunci[1] + ("REGIONS", "SHARDS", "HALFTONE").index(jenis))
                buat = {"REGIONS": _wilayah, "SHARDS": lambda g, w, h: Tempelan(_serpihan(g, w, h)),
                        "HALFTONE": _halftone}[jenis]
                self._latar[kunci] = buat(rng, lebar, tinggi)
            return self._latar[kunci]

    def siapkan_latar(self, lebar: int, tinggi: int) -> None:
        """Build every background variant now, so no step pays for it later."""
        for jenis in ("REGIONS", "SHARDS", "HALFTONE"):
            for varian in range(BACKGROUND_VARIANTS):
                self.latar(jenis, varian, lebar, tinggi)

    def panaskan(self, lebar: int, tinggi: int) -> None:
        """Build the backgrounds for this frame size on a helper thread, once; cheap to call every frame.

        The score's first full frame background comes about a second in, by then they are ready.
        """
        if (lebar, tinggi) in self._dipanaskan:
            return
        self._dipanaskan.add((lebar, tinggi))
        threading.Thread(target=self.siapkan_latar, args=(lebar, tinggi), daemon=True).start()

    # One frame

    def render(self, frame: np.ndarray, alpha: np.ndarray, bersih: Optional[np.ndarray], step: Step,
               seed: int) -> np.ndarray:
        """frame BGR uint8, alpha = soft person matte (float32 0..1, frame size), bersih = the frame without
        the person (None when the room is not known: the body then stays under what moves).

        seed identifies the step: frames with the same seed reuse its geometry. Returns frame itself on a
        calm step or when nobody is in view; otherwise a new image.
        """
        if not step:
            return frame
        tinggi, lebar = frame.shape[:2]
        rencana = self._rencana
        if rencana is None or rencana.kunci != (seed, lebar, tinggi) or rencana.step is not step:
            rencana = self._buat_rencana(alpha, step, seed)
            self._rencana = rencana
        if rencana is None:
            return frame

        jenis_latar = step.get("bg")
        if jenis_latar in ("REGIONS", "HALFTONE"):
            hasil = self.latar(jenis_latar, seed, lebar, tinggi).copy()
        else:
            hasil = (frame if bersih is None else bersih).copy()
            if jenis_latar == "SHARDS":
                tempel(hasil, (0, 0, lebar, tinggi), self.latar("SHARDS", seed, lebar, tinggi))
        tempel(hasil, rencana.e, rencana.belakang)
        tempel(hasil, rencana.g, self._kelompok(frame, alpha, rencana))
        tempel(hasil, rencana.e, rencana.depan)
        tempel(hasil, rencana.e, rencana.garis)
        return hasil

    def _kelompok(self, frame: np.ndarray, alpha: np.ndarray, r: Rencana) -> np.ndarray:
        """The body group of this frame: recolored body, copies in the reference's layering, then the cuts."""
        x0, y0, x1, y1 = r.g
        a = cv2.convertScaleAbs(cv2.max(alpha[y0:y1, x0:x1], EDGE_LOW), alpha=255.0 / (EDGE_HIGH - EDGE_LOW),
                                beta=-255.0 * EDGE_LOW / (EDGE_HIGH - EDGE_LOW))
        if r.sisa is not None:
            a = cv2.bitwise_and(a, r.sisa)
        body = a > 127
        potong_frame = frame[y0:y1, x0:x1]
        luma = cv2.GaussianBlur(cv2.cvtColor(potong_frame, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
        idx = indeks_nada(luma, body)

        isi = warnai(idx, "YELLOW") if r.kuning else potong_frame.copy()
        if r.sel is not None:
            self._isi_sel(isi, luma, body, r)
        badan = premultiply(isi, a)

        kelompok = np.zeros_like(badan)
        for nama, dx, dy in r.buram:
            # Opaque flat copies sit behind the body (#11 red, #12 and #80 blue).
            kelompok = over(kelompok, premultiply(warnai(geser(idx, dx, dy, 1), nama), geser(a, dx, dy)))
        if r.cermin is not None and r.cermin[0] == "BODY":
            kelompok = over(kelompok, cv2.remap(badan, r.cermin[1], r.cermin[2], cv2.INTER_NEAREST,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        for dx, dy in r.badan:
            kelompok = over(geser(badan, dx, dy), kelompok)
        if r.cermin is not None and r.cermin[0] != "BODY":
            balik = cv2.remap(a, r.cermin[1], r.cermin[2], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
            kelompok = over(rata(r.cermin[0], cv2.multiply(balik, GHOST_ALPHA)), kelompok)
        for nama, dx, dy in r.tembus:
            kelompok = over(rata(nama, cv2.multiply(geser(a, dx, dy), GHOST_ALPHA)), kelompok)
        if r.potong is not None:
            kelompok = cv2.remap(kelompok, r.potong[0], r.potong[1], cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return kelompok

    @staticmethod
    def _isi_sel(isi: np.ndarray, luma: np.ndarray, body: np.ndarray, r: Rencana) -> None:
        """Recolor the cells in place. Tones are picked within each cell, so a bright face in a dark outfit
        still reads flat; one lookup table covers every cell (code = 3 * cell + tone)."""
        sel = r.sel
        kecil_l, kecil_s, kecil_b = luma[::3, ::3], sel[::3, ::3], body[::3, ::3]
        batas_lo = np.zeros((1, 256), dtype=np.uint8)
        batas_hi = np.full((1, 256), 255, dtype=np.uint8)
        tabel = np.zeros((1, 256, 3), dtype=np.uint8)
        for k, nama in enumerate(r.sel_warna, start=1):
            nilai = kecil_l[(kecil_s == k) & kecil_b]
            if nilai.size:
                lo, hi = np.percentile(nilai, (100.0 * TRITONE_DARK_SHARE, 100.0 * (1.0 - TRITONE_LIGHT_SHARE)))
                batas_lo[0, k], batas_hi[0, k] = int(lo), int(hi)
            tabel[0, 3 * k:3 * k + 3] = nada(nama)[0, :3]
        nada_sel = cv2.add(cv2.compare(luma, cv2.LUT(sel, batas_lo), cv2.CMP_GE) // 255,
                           cv2.compare(luma, cv2.LUT(sel, batas_hi), cv2.CMP_GT) // 255)
        kode = cv2.add(cv2.multiply(sel, 3), nada_sel)
        topeng = ((sel > 0) & body).view(np.uint8)
        cv2.copyTo(cv2.LUT(cv2.merge([kode, kode, kode]), tabel), topeng, isi)

    # Geometry of a step

    def _buat_rencana(self, alpha: np.ndarray, step: Step, seed: int) -> Optional[Rencana]:
        tinggi, lebar = alpha.shape
        bx, by, bw, bh = cv2.boundingRect((alpha > MATTE_CUTOFF).view(np.uint8))
        if bw == 0 or bh == 0:
            return None
        u = float(bh)
        rng = np.random.default_rng(seed)
        g = perluas((bx, by, bw, bh), int(round(GROUP_PAD * u)), lebar, tinggi)
        e = perluas((bx, by, bw, bh), int(round(EFFECT_PAD * u)), lebar, tinggi)
        r = Rencana((seed, lebar, tinggi), step, g, e)
        shape_g = (g[3] - g[1], g[2] - g[0])
        shape_e = (e[3] - e[1], e[2] - e[0])
        kotak_g = (bx - g[0], by - g[1], bw, bh)
        kotak_e = (bx - e[0], by - e[1], bw, bh)

        belakang = None  # type: Optional[np.ndarray]
        for nama in step.get("blob", ()):
            gumpal = _gumpal(rng, shape_e, kotak_e, u, nama)
            belakang = gumpal if belakang is None else over(gumpal, belakang)
        if "burst" in step:
            isi, garis = _ledakan(rng, shape_e, kotak_e, u, step["burst"])
            r.garis = Tempelan(garis)
            # The pastel burst lies see-through over the body (#23); the others sit behind it.
            if step["burst"] == "PASTEL":
                r.depan = Tempelan(isi)
            elif step["burst"] != "OUTLINE":
                belakang = isi if belakang is None else over(isi, belakang)
        if belakang is not None:
            r.belakang = Tempelan(belakang)

        salinan = list(step.get("copies", ()))
        cermin = step.get("mirror")
        # See-through copies lie over the reference's yellow figure, so the body takes that yellow and the
        # overlaps get the reference's colors.
        r.kuning = any(n in SEE_THROUGH for n, _ in salinan) or cermin in SEE_THROUGH
        if step.get("keep", 1.0) < 1.0:
            r.sisa = _sel_sisa(rng, shape_g, kotak_g, alpha[g[1]:g[3], g[0]:g[2]] > MATTE_CUTOFF, step["keep"])
        if "fill" in step:
            r.sel, r.sel_warna = _sel_isi(rng, shape_g, kotak_g, alpha[g[1]:g[3], g[0]:g[2]] > MATTE_CUTOFF,
                                          step["fill"], step["cover"])
        badan = []
        for nama, lv in salinan:
            dx, dy = arah(rng, COPY_SHIFT[lv] * u)
            if nama == "BODY":
                badan.append((dx, dy))
            elif nama in SEE_THROUGH:
                r.tembus.append((nama, dx, dy))
            else:
                r.buram.append((nama, dx, dy))
        if badan:
            r.badan = badan
        if cermin is not None:
            gx = np.arange(shape_g[1], dtype=np.float32)[None, :].repeat(shape_g[0], 0)
            gy = np.arange(shape_g[0], dtype=np.float32)[:, None].repeat(shape_g[1], 1)
            dx = int(round(COPY_SHIFT[1] * u))
            r.cermin = (cermin, (2 * kotak_g[0] + bw - 1 + dx) - gx, gy)
        if "cut" in step:
            r.potong = _peta_potong(rng, shape_g, kotak_g, u, step["cut"])
        return r


# Step geometry

def _sel(rng: np.random.Generator, shape: Tuple[int, int], kotak: Kotak, garis: int) -> np.ndarray:
    """Label map (uint8) of convex cells made by random straight cuts through the body box."""
    bx, by, bw, bh = kotak
    jauh = 2.0 * (shape[0] + shape[1])
    label = np.zeros(shape, dtype=np.uint8)
    for i in range(garis):
        px, py = bx + rng.uniform(0.15, 0.85) * bw, by + rng.uniform(0.15, 0.85) * bh
        # Mostly horizontal or vertical cuts, a few diagonal ones.
        if rng.random() < 0.75:
            sudut = rng.choice([0.0, math.pi / 2]) + rng.normal(0.0, 0.12)
        else:
            sudut = rng.uniform(0.0, math.pi)
        ux, uy = math.cos(sudut), math.sin(sudut)
        nx, ny = -uy, ux
        # The half plane on the normal side, as a quad far larger than the region.
        sisi = np.round(np.array([(px - ux * jauh, py - uy * jauh), (px + ux * jauh, py + uy * jauh),
                                  (px + ux * jauh + nx * jauh, py + uy * jauh + ny * jauh),
                                  (px - ux * jauh + nx * jauh, py - uy * jauh + ny * jauh)])).astype(np.int32)
        bidang = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(bidang, [sisi], 1 << i)
        label = cv2.bitwise_or(label, bidang)
    return label


def _sel_isi(rng: np.random.Generator, shape: Tuple[int, int], kotak: Kotak, body: np.ndarray,
             warna_isi: Tuple[str, ...], cover: float) -> Tuple[np.ndarray, List[str]]:
    """Cells that get a flat color, until they cover the requested share of the body (on the first frame).

    Returns a uint8 label map (0 = no recolor, k = the k-th chosen cell) and the color of each chosen cell.
    """
    label = _sel(rng, shape, kotak, 4)
    total = max(1, int(body.sum()))
    peta = np.zeros(shape, dtype=np.uint8)
    nama = []  # type: List[str]
    luas = 0
    for k in rng.permutation(16):
        sel = label == k
        n = int((sel & body).sum())
        if n == 0:
            continue
        nama.append(warna_isi[len(nama) % len(warna_isi)])
        peta[sel] = len(nama)
        luas += n
        if luas >= cover * total:
            break
    return peta, nama


def _sel_sisa(rng: np.random.Generator, shape: Tuple[int, int], kotak: Kotak, body: np.ndarray,
              keep: float) -> np.ndarray:
    """Mask (uint8 0 or 255) of the body parts still shown, about keep of the body.

    Upper cells go last, like the head that lingers in the reference; below STRIP_KEEP what is left is one
    horizontal strip across the upper body.
    """
    bx, by, bw, bh = kotak
    hasil = np.zeros(shape, dtype=np.uint8)
    total = int(np.count_nonzero(body))
    if total == 0:
        return hasil
    target = keep * total
    if keep < STRIP_KEEP:
        baris = np.count_nonzero(body, axis=1)
        y0 = y1 = min(shape[0] - 1, by + int(rng.uniform(0.12, 0.35) * bh))
        luas = int(baris[y0])
        while luas < target and (y0 > 0 or y1 < shape[0] - 1):
            if y1 < shape[0] - 1:
                y1 += 1
                luas += int(baris[y1])
            if luas < target and y0 > 0:
                y0 -= 1
                luas += int(baris[y0])
        hasil[y0:y1 + 1] = 255
        return hasil
    label = _sel(rng, shape, kotak, 5)
    # A little jitter only: the head stays last like Rick's (#304 on) and comes first like Morty's.
    acak = rng.uniform(0.0, KEEP_JITTER, 32)
    ys = np.nonzero(body)[0].astype(np.float64)
    lab = label[body]
    jumlah = np.bincount(lab, minlength=32)
    rata_y = np.bincount(lab, weights=ys, minlength=32) / np.maximum(jumlah, 1)
    prioritas = (rata_y - by) / max(1, bh) + acak
    tampil = np.zeros(256, dtype=np.uint8)
    luas = 0
    for k in np.argsort(prioritas):
        if jumlah[k] == 0:
            continue
        tampil[k] = 255
        luas += int(jumlah[k])
        if luas >= target:
            break
    return cv2.LUT(label, tampil)


def _peta_potong(rng: np.random.Generator, shape: Tuple[int, int], kotak: Kotak, u: float,
                 level: int) -> Tuple[np.ndarray, np.ndarray]:
    """Remap of a sequence of straight cuts: the chunk on one side of each cut slides along it, and a thin gap
    stays on the cut line. Pixels that end up empty map to -1 (transparent)."""
    bx, by, bw, bh = kotak
    h, w = shape
    mx = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    my = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    celah = max(2, int(round(GAP_SHARE * u)))
    rendah, tinggi = CUT_PIECES[level]
    for _ in range(int(rng.integers(rendah, tinggi + 1))):
        pilih = rng.random()
        # Mostly horizontal cuts, then vertical, a few diagonal.
        sudut = 0.0 if pilih < 0.6 else (math.pi / 2 if pilih < 0.85 else rng.uniform(0.3, 1.2))
        sudut += rng.normal(0.0, 0.08)
        ux, uy = math.cos(sudut), math.sin(sudut)
        nx, ny = -uy, ux
        sepanjang = abs(ux) * bw + abs(uy) * bh
        px, py = bx + rng.uniform(0.1, 0.9) * bw, by + rng.uniform(0.1, 0.9) * bh
        awal, akhir = -rng.uniform(0.15, 0.7) * sepanjang, rng.uniform(0.15, 0.7) * sepanjang
        dalam = rng.uniform(0.06, 0.45) * u * (1 if rng.random() < 0.5 else -1)
        a0 = (px + ux * awal, py + uy * awal)
        a1 = (px + ux * akhir, py + uy * akhir)
        poli = np.round(np.array([a0, a1, (a1[0] + nx * dalam, a1[1] + ny * dalam),
                                  (a0[0] + nx * dalam, a0[1] + ny * dalam)])).astype(np.int32)
        jarak = CUT_SHIFT[level] * u * rng.uniform(0.6, 1.4) * (1 if rng.random() < 0.5 else -1)
        dx, dy = int(round(jarak * ux)), int(round(jarak * uy))
        # Work only in the window the chunk covers before and after sliding.
        x0 = max(0, int(poli[:, 0].min()) - abs(dx) - 1)
        x1 = min(w, int(poli[:, 0].max()) + abs(dx) + 2)
        y0 = max(0, int(poli[:, 1].min()) - abs(dy) - 1)
        y1 = min(h, int(poli[:, 1].max()) + abs(dy) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        keping = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(keping, [poli - np.array([x0, y0], dtype=np.int32)], 1)
        tujuan = geser(keping, dx, dy).astype(bool)
        asal = keping.astype(bool) & ~tujuan
        jx, jy = mx[y0:y1, x0:x1], my[y0:y1, x0:x1]
        gx, gy = geser(jx, dx, dy, -1.0), geser(jy, dx, dy, -1.0)
        jx[tujuan], jy[tujuan] = gx[tujuan], gy[tujuan]
        jx[asal], jy[asal] = -1.0, -1.0
        mulai = (int(round(a0[0] + ux * min(0.0, jarak))), int(round(a0[1] + uy * min(0.0, jarak))))
        selesai = (int(round(a1[0] + ux * max(0.0, jarak))), int(round(a1[1] + uy * max(0.0, jarak))))
        lx0 = max(0, min(mulai[0], selesai[0]) - celah)
        lx1 = min(w, max(mulai[0], selesai[0]) + celah + 1)
        ly0 = max(0, min(mulai[1], selesai[1]) - celah)
        ly1 = min(h, max(mulai[1], selesai[1]) + celah + 1)
        if lx1 > lx0 and ly1 > ly0:
            tepi = np.zeros((ly1 - ly0, lx1 - lx0), dtype=np.uint8)
            cv2.line(tepi, (mulai[0] - lx0, mulai[1] - ly0), (selesai[0] - lx0, selesai[1] - ly0), 1, celah)
            celah_ada = tepi.view(bool)
            mx[ly0:ly1, lx0:lx1][celah_ada] = -1.0
            my[ly0:ly1, lx0:lx1][celah_ada] = -1.0
    return mx, my


def _ledakan(rng: np.random.Generator, shape: Tuple[int, int], kotak: Kotak, u: float,
             jenis: str) -> Tuple[np.ndarray, np.ndarray]:
    """Burst fill and its outline, both premultiplied BGRA uint8."""
    bx, by, bw, bh = kotak
    pusat = (bx + bw * rng.uniform(0.4, 0.6), by + bh * rng.uniform(0.45, 0.65))
    isi = np.zeros(shape + (4,), dtype=np.uint8)
    warna_isi = {"MINT": (("MINT", 1.0),), "PASTEL": (("PASTEL_PINK", 0.6), ("PASTEL_CYAN", 0.6), ("MINT", 0.6)),
                 "RED": (("RED_BURST", 0.85),), "OUTLINE": ()}[jenis]
    for nama, opa in warna_isi:
        peta = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(peta, [bintang(rng, pusat, 0.65 * u, 0.30 * u, int(rng.integers(9, 14)))], 255, cv2.LINE_AA)
        isi = over(rata(nama, cv2.multiply(peta, opa)), isi)
    peta = np.zeros(shape, dtype=np.uint8)
    poli = bintang(rng, pusat, 0.75 * u, 0.30 * u, int(rng.integers(8, 12)))
    cv2.polylines(peta, [poli], True, 255, max(2, int(round(0.008 * u))), cv2.LINE_AA)
    return isi, rata("RED_LINE" if jenis == "RED" else "MINT_LINE", peta)


def _gumpal(rng: np.random.Generator, shape: Tuple[int, int], kotak: Kotak, u: float, nama: str) -> np.ndarray:
    """A big soft cornered shape behind the body (#68 to #70), premultiplied BGRA uint8."""
    bx, by, bw, bh = kotak
    pusat = (bx + bw * rng.uniform(0.2, 0.8), by + bh * rng.uniform(0.3, 0.7))
    poli = bintang(rng, pusat, 0.6 * u, 0.38 * u, int(rng.integers(5, 8)))
    # Blur then threshold rounds the corners into the reference's soft blob outline; done at a quarter of the
    # size, and the upscale gives the edge its antialiasing.
    kecil = np.zeros((max(1, shape[0] // 4), max(1, shape[1] // 4)), dtype=np.uint8)
    cv2.fillPoly(kecil, [np.round(poli / 4.0).astype(np.int32)], 255)
    kecil = cv2.threshold(cv2.GaussianBlur(kecil, (0, 0), max(1.0, 0.07 * u / 4.0)), 127, 255, cv2.THRESH_BINARY)[1]
    bentuk = cv2.resize(kecil, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return rata(nama, bentuk)


# Backgrounds (built once per variant)

def _wilayah(rng: np.random.Generator, lebar: int, tinggi: int) -> np.ndarray:
    """Large flat color regions with wavy borders and a thin cyan seam between them (#13, #82)."""
    lk, tk = lebar // 2, tinggi // 2
    n = 7
    titik = np.column_stack([rng.uniform(0, lk, n), rng.uniform(0, tk, n)]).astype(np.float32)
    yy, xx = np.mgrid[0:tk, 0:lk].astype(np.float32)
    # Domain warp gives the curved borders of the reference.
    wx = cv2.resize(rng.random((6, 10), dtype=np.float32), (lk, tk), interpolation=cv2.INTER_CUBIC) - 0.5
    wy = cv2.resize(rng.random((6, 10), dtype=np.float32), (lk, tk), interpolation=cv2.INTER_CUBIC) - 0.5
    xx += wx * 0.25 * lk
    yy += wy * 0.25 * tk
    jarak = (xx[..., None] - titik[:, 0]) ** 2 + (yy[..., None] - titik[:, 1]) ** 2
    label = np.argmin(jarak, axis=2).astype(np.uint8)
    daftar = ["REG_CYAN", "REG_BLUE", "REG_GREEN", "REG_RED", "REG_CYAN", "REG_LIME", "REG_BLUE"]
    lut = np.stack([hex_ke_bgr(PALETTE[n]) for n in daftar]).astype(np.uint8)
    label = cv2.resize(label, (lebar, tinggi), interpolation=cv2.INTER_NEAREST)
    gambar = lut[label]
    tepi = cv2.morphologyEx(label, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    gambar[tepi] = hex_ke_bgr(PALETTE["REG_CYAN"]).astype(np.uint8)
    return cv2.GaussianBlur(gambar, (3, 3), 0)


def _serpihan(rng: np.random.Generator, lebar: int, tinggi: int) -> np.ndarray:
    """Sweeping blue crescents and small cyan and green slivers over the frame (#12, #80)."""
    peta = {n: np.zeros((tinggi, lebar), dtype=np.uint8) for n in ("SHARD_BLUE", "SHARD_CYAN", "SHARD_GREEN")}
    for _ in range(3):
        # A crescent: one ellipse minus a slightly moved copy of itself, of which only an arc shows.
        pusat = (rng.uniform(0.1, 0.9) * lebar, rng.uniform(0.2, 1.0) * tinggi)
        sumbu = (rng.uniform(0.3, 0.6) * lebar, rng.uniform(0.25, 0.5) * tinggi)
        miring = rng.uniform(-25, 25)
        luar = np.zeros((tinggi, lebar), dtype=np.uint8)
        cv2.ellipse(luar, (int(pusat[0]), int(pusat[1])), (int(sumbu[0]), int(sumbu[1])), miring, 0, 360, 255, -1,
                    cv2.LINE_AA)
        geser_x, geser_y = rng.uniform(-0.12, 0.12) * lebar, rng.uniform(0.06, 0.18) * tinggi
        dalam = np.zeros_like(luar)
        cv2.ellipse(dalam, (int(pusat[0] + geser_x), int(pusat[1] + geser_y)),
                    (int(sumbu[0] * 1.02), int(sumbu[1] * 1.02)), miring, 0, 360, 255, -1, cv2.LINE_AA)
        sapu = np.zeros_like(luar)
        mulai = rng.uniform(0, 360)
        cv2.ellipse(sapu, (int(pusat[0]), int(pusat[1])), (int(sumbu[0] * 1.3), int(sumbu[1] * 1.3)), miring,
                    mulai, mulai + rng.uniform(70, 160), 255, -1)
        peta["SHARD_BLUE"] = cv2.max(peta["SHARD_BLUE"], cv2.min(cv2.subtract(luar, dalam), sapu))
    for _ in range(8):
        nama = "SHARD_CYAN" if rng.random() < 0.6 else "SHARD_GREEN"
        tepi_x = rng.choice([rng.uniform(0, 0.15), rng.uniform(0.85, 1.0), rng.uniform(0, 1)])
        cx, cy = tepi_x * lebar, rng.uniform(0, 1) * tinggi
        poli = np.round(np.array([(cx + rng.uniform(-0.08, 0.08) * lebar, cy + rng.uniform(-0.05, 0.05) * tinggi)
                                  for _ in range(3)])).astype(np.int32)
        cv2.fillPoly(peta[nama], [poli], 255, cv2.LINE_AA)
    hasil = np.zeros((tinggi, lebar, 4), dtype=np.float32)
    for nama, m in peta.items():
        hasil = over(rata(nama, m.astype(np.float32) * (1.0 / 255.0)), hasil)
    return hasil


def _halftone(rng: np.random.Generator, lebar: int, tinggi: int) -> np.ndarray:
    """Red dots on white with white burst shapes cleared out and blue shards at the edges (#15, #65)."""
    kertas = np.full((tinggi, lebar), 255, dtype=np.uint8)
    jarak = max(8, int(round(0.05 * tinggi)))
    gradien = cv2.resize(rng.random((3, 5), dtype=np.float32), (lebar, tinggi), interpolation=cv2.INTER_CUBIC)
    for y in range(jarak // 2, tinggi + jarak, jarak):
        geser_baris = (jarak // 2) if (y // jarak) % 2 else 0
        for x in range(geser_baris, lebar + jarak, jarak):
            g = float(gradien[min(y, tinggi - 1), min(x, lebar - 1)])
            cv2.circle(kertas, (x, y), int(round(jarak * (0.22 + 0.24 * g))), 0, -1, cv2.LINE_AA)
    kertas = cv2.GaussianBlur(kertas, (0, 0), 1.0)
    for _ in range(2):
        poli = bintang(rng, (rng.uniform(0.2, 0.8) * lebar, rng.uniform(0.2, 0.8) * tinggi),
                       0.45 * tinggi, 0.12 * tinggi, int(rng.integers(5, 8)))
        cv2.fillPoly(kertas, [poli], 255, cv2.LINE_AA)
    tinta = hex_ke_bgr(PALETTE["DOT"])
    k = kertas.astype(np.float32)[..., None] * (1.0 / 255.0)
    gambar = 255.0 * k + tinta * (1.0 - k)
    for _ in range(4):
        cx = rng.choice([rng.uniform(0, 0.2), rng.uniform(0.8, 1.0)]) * lebar
        cy = rng.uniform(0, 1) * tinggi
        poli = np.round(np.array([(cx + rng.uniform(-0.1, 0.1) * lebar, cy + rng.uniform(-0.1, 0.1) * tinggi)
                                  for _ in range(3)])).astype(np.int32)
        a = isi_poligon((tinggi, lebar), poli)[..., None]
        nama = "HALF_BLUE" if rng.random() < 0.6 else "SHARD_CYAN"
        gambar = hex_ke_bgr(PALETTE[nama]) * a + gambar * (1.0 - a)
    return np.clip(gambar + 0.5, 0, 255).astype(np.uint8)
