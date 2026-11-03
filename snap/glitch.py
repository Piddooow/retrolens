"""GlitchRenderer: one step of the Spider-Verse style glitch (snap/glitch_score.py) drawn on a camera frame.

The glitch stays on the person: the room is never painted over, it only shows where a piece of the body
moved away. A step is held for a whole glitch tick (several camera frames). Its geometry (cut map, recolor
cells, copy offsets, shown parts) is built once on the step's first frame from a seeded generator, relative
to the person's bounding box; every frame then only recolors and stacks the live body with OpenCV, so the
picture under the glitch stays live. Layers are premultiplied BGRA uint8 (saturating OpenCV math).
"""

import math
from typing import List, Optional, Tuple

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
# The tones are cel shading, not photo detail: the luminance is smoothed first and the tone map cleaned of
# specks (median of this size), so each tone reads as a few clean shapes like the reference's flat fills.
TONE_BLUR_SIGMA = 2.0
TONE_MEDIAN = 5
# Two passes: one median leaves a few specks where three tones meet, a second one removes them.
TONE_MEDIAN_PASSES = 2
# A pixel changes tone only when it is this many levels past a threshold: camera noise (about 2 levels)
# never flips a tone, so the flat shapes do not flicker from frame to frame.
TONE_NOISE = 2
# Room around the body box for copies and cut slides, as a share of the body height.
GROUP_PAD = 0.25
# Below this share kept, what is left of the body is one horizontal strip (#332 to #340 of reference 2).
STRIP_KEEP = 0.2
# Random share added to a cell's height (0 top, 1 bottom) when picking which cells stay shown.
KEEP_JITTER = 0.15
MATTE_CUTOFF = 0.5
# The body's edge is the matte cut at MATTE_CUTOFF with a 3x3 antialias, however soft the matte is: the
# reference edges are crisp, and a soft edge makes every copy glow and drags a halo of the room along.
EDGE_AA = 3
# Width of the edge ring that always takes the flat color.
EDGE_RING = 5
# Matte above this counts as the person when deciding where the room shows: it includes the soft fringe,
# so no outline of the person is left in a gap. GAP_REACH_PX limits that fringe to near a real gap.
FRINGE_MATTE = 0.1
GAP_REACH_PX = 9
EDGE_MARGIN = 0.05

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
    """Premultiplied BGRA uint8 'over'."""
    sisa = cv2.bitwise_not(cv2.extractChannel(atas, 3))
    return cv2.add(atas, cv2.multiply(bawah, cv2.merge([sisa, sisa, sisa, sisa]), scale=1.0 / 255.0))


def rata(nama: str, alpha: np.ndarray) -> np.ndarray:
    """Premultiplied flat palette color layer (uint8) for a uint8 alpha."""
    c = hex_ke_bgr(PALETTE[nama])
    return cv2.multiply(cv2.merge([alpha, alpha, alpha, alpha]), (float(c[0]), float(c[1]), float(c[2]), 255.0),
                        scale=1.0 / 255.0)


def premultiply(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """uint8 premultiplied BGRA from uint8 BGR and alpha."""
    b, g, r = cv2.split(bgr)
    return cv2.multiply(cv2.merge([b, g, r, alpha]), cv2.merge([alpha, alpha, alpha, np.full_like(alpha, 255)]),
                        scale=1.0 / 255.0)


def tempel(gambar: np.ndarray, kotak: Kotak, lapis: np.ndarray) -> None:
    """Composite a premultiplied BGRA uint8 layer onto the image region (x0, y0, x1, y1) in place."""
    x0, y0, x1, y1 = kotak
    sisa = cv2.bitwise_not(cv2.extractChannel(lapis, 3))
    bagian = gambar[y0:y1, x0:x1]
    gambar[y0:y1, x0:x1] = cv2.add(cv2.multiply(bagian, cv2.merge([sisa, sisa, sisa]), scale=1.0 / 255.0),
                                   cv2.cvtColor(lapis, cv2.COLOR_BGRA2BGR))


def indeks_nada(luma: np.ndarray, pilih: np.ndarray) -> np.ndarray:
    """Tritone index (uint8 0, 1, 2) for luma, with thresholds from the selected pixels (before cleaning)."""
    nilai = luma[pilih][::PERCENTILE_STRIDE]
    if nilai.size == 0:
        return np.ones(luma.shape, dtype=np.uint8)
    lo, hi = np.percentile(nilai, (100.0 * TRITONE_DARK_SHARE, 100.0 * (1.0 - TRITONE_LIGHT_SHARE)))
    # Only pixels clearly below the dark threshold take the shadow, so an evenly lit area keeps the flat color.
    return (luma >= lo - TONE_NOISE).view(np.uint8) + (luma > hi + TONE_NOISE).view(np.uint8)


def rapikan_nada(indeks: np.ndarray, tepi: np.ndarray) -> np.ndarray:
    """Clean tone map: the edge ring takes the flat color, then a median removes specks and the ring's seams."""
    indeks[tepi] = 1
    for _ in range(TONE_MEDIAN_PASSES):
        indeks = cv2.medianBlur(indeks, TONE_MEDIAN)
    return indeks


def luma_badan(bgr: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Smoothed luminance (uint8) of the body that only averages body pixels a pixel in from the edge (the
    edge pixels mix in the room), so the room never darkens or brightens the body into a false outline."""
    m = cv2.erode(body.view(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    abu = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    num = cv2.GaussianBlur(cv2.multiply(abu, m), (0, 0), TONE_BLUR_SIGMA)
    den = cv2.GaussianBlur(m, (0, 0), TONE_BLUR_SIGMA)
    return cv2.convertScaleAbs(cv2.divide(num, cv2.max(den, 1e-3)))


def warnai(indeks: np.ndarray, nama: str) -> np.ndarray:
    """BGR uint8 image of a palette color's tritone for a tone index map."""
    return cv2.LUT(cv2.merge([indeks, indeks, indeks]), nada(nama))


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
    return "cut" in step or any(n == "BODY" and lv > 0 for n, lv in step.get("copies", ()))


class Rencana:
    """The geometry of one step, built on its first frame and reused while the step is held."""

    def __init__(self, kunci: Tuple[int, int, int], step: Step, g: Kotak):
        # (seed, width, height) and the step itself: holding the step keeps a new step from ever matching.
        self.kunci = kunci
        self.step = step
        # Region of the body group (body, copies, cuts) as (x0, y0, x1, y1).
        self.g = g
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


class GlitchRenderer:
    """Draws glitch steps on the person; the geometry of the step being held is kept."""

    def __init__(self):
        self._rencana = None  # type: Optional[Rencana]

    # One frame

    def render(self, frame: np.ndarray, alpha: np.ndarray, bersih: Optional[np.ndarray], step: Step,
               seed: int) -> np.ndarray:
        """frame BGR uint8, alpha = soft person matte (float32 0..1, frame size), bersih = the frame without
        the person (None when the room is not known: the body then stays under what moves). For a step that
        hides parts of the body (keep), bersih is the base everywhere; otherwise it only shows where the
        person's own pixels are left uncovered, so the room never shows around a body that did not move.

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

        kelompok, opak = self._kelompok(frame, alpha, rencana)
        if bersih is None or "keep" in step:
            hasil = (frame if bersih is None else bersih).copy()
        else:
            hasil = frame.copy()
            x0, y0, x1, y1 = rencana.g
            # Only the opaque body hides the person; see-through copies over a gap still show the room. A gap is
            # where the person's own pixels lost the body; the margin above the cutoff keeps the body's own
            # antialiased edge from counting as one.
            a = alpha[y0:y1, x0:x1]
            celah = (a > MATTE_CUTOFF + EDGE_MARGIN) & (opak < 128)
            dekat = cv2.dilate(celah.view(np.uint8), np.ones((2 * GAP_REACH_PX + 1,) * 2, np.uint8)) > 0
            # Near a gap the room also goes under the person's fringe and under every pixel the body does not
            # fully cover (its antialiased edge), so nothing of the person shows through there.
            terbuka = (a > FRINGE_MATTE) & (opak < 255) & dekat
            cv2.copyTo(bersih[y0:y1, x0:x1], terbuka.view(np.uint8), hasil[y0:y1, x0:x1])
        tempel(hasil, rencana.g, kelompok)
        return hasil

    def _kelompok(self, frame: np.ndarray, alpha: np.ndarray, r: Rencana) -> Tuple[np.ndarray, np.ndarray]:
        """The body group of this frame (recolored body, copies in the reference's layering, then the cuts),
        and the alpha (uint8) of its opaque part alone."""
        x0, y0, x1, y1 = r.g
        a = cv2.GaussianBlur((alpha[y0:y1, x0:x1] > MATTE_CUTOFF).view(np.uint8) * np.uint8(255), (EDGE_AA, EDGE_AA),
                             0)
        if r.sisa is not None:
            a = cv2.bitwise_and(a, r.sisa)
        body = a > 127
        potong_frame = frame[y0:y1, x0:x1]
        luma = luma_badan(potong_frame, body)
        # The outermost pixels mix in the room behind; they keep the flat color, so the silhouette is filled
        # to its edge like the reference instead of getting a speckled outline.
        tepi = cv2.subtract(body.view(np.uint8), cv2.erode(body.view(np.uint8), np.ones((EDGE_RING,) * 2, np.uint8))) > 0
        idx = rapikan_nada(indeks_nada(luma, body), tepi)

        isi = warnai(idx, "YELLOW") if r.kuning else potong_frame.copy()
        if r.sel is not None:
            self._isi_sel(isi, luma, body, tepi, r)
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
        # See-through copies on a layer of their own, cut with the same pieces as the body.
        tembus = None  # type: Optional[np.ndarray]
        if r.cermin is not None and r.cermin[0] != "BODY":
            balik = cv2.remap(a, r.cermin[1], r.cermin[2], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
            tembus = rata(r.cermin[0], cv2.multiply(balik, GHOST_ALPHA))
        for nama, dx, dy in r.tembus:
            lapis = rata(nama, cv2.multiply(geser(a, dx, dy), GHOST_ALPHA))
            tembus = lapis if tembus is None else over(lapis, tembus)
        if r.potong is not None:
            kelompok = self._potong(kelompok, r)
            if tembus is not None:
                tembus = self._potong(tembus, r)
        opak = cv2.extractChannel(kelompok, 3)
        if tembus is not None:
            kelompok = over(tembus, kelompok)
        return kelompok, opak

    @staticmethod
    def _potong(lapis: np.ndarray, r: Rencana) -> np.ndarray:
        return cv2.remap(lapis, r.potong[0], r.potong[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)

    @staticmethod
    def _isi_sel(isi: np.ndarray, luma: np.ndarray, body: np.ndarray, tepi: np.ndarray, r: Rencana) -> None:
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
                batas_lo[0, k] = max(0, int(lo) - TONE_NOISE)
                batas_hi[0, k] = min(255, int(hi) + TONE_NOISE)
            tabel[0, 3 * k:3 * k + 3] = nada(nama)[0, :3]
        nada_sel = rapikan_nada(cv2.add(cv2.compare(luma, cv2.LUT(sel, batas_lo), cv2.CMP_GE) // 255,
                                        cv2.compare(luma, cv2.LUT(sel, batas_hi), cv2.CMP_GT) // 255), tepi)
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
        r = Rencana((seed, lebar, tinggi), step, g)
        shape_g = (g[3] - g[1], g[2] - g[0])
        kotak_g = (bx - g[0], by - g[1], bw, bh)

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
