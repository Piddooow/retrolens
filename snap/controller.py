"""SnapController v2: the SNAP state machine.

LIVE -> VANISH (flash and dissolve) -> INVISIBLE (until a reappear trigger) -> REAPPEAR (the dust
flies back) -> LIVE, plus PLATE_CAPTURE for learning the room. With the Glitch style, VANISH and REAPPEAR
play the glitch scores on the live body instead (no flash), and LIVE plays the glitch score while visible. Cancel returns to LIVE from any state
with a short eased fade. Effect timing runs on EffectClock, so fist and pointing up can freeze or
slow it outside the flash.
"""

import contextlib
import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from snap.ash import AshAmbient
from snap.background import CAPTURE_MESSAGES, BackgroundEngine
from snap.clock import EffectClock
from snap.config import STYLES
from snap.dissolve import DissolveRenderer, smoothstep
from snap.flash import FlashOverlay
from snap.glitch import GlitchRenderer, perlu_ruang
from snap.glitch_score import REAPPEAR_SCORE, SCORE, VANISH_SCORE
from snap.matte import MatteEngine
from snap.quality import QualityLadder
from snap.segmenter import PersonSegmenter

MSG_NEED_ROOM = "I still need to see the room behind you. Step out for 2 seconds, then snap again"
MSG_ESTIMATED = "Background estimated, result may not be perfect"
MSG_CANT_SEE = "I can't see behind you there"
MSG_NO_PERSON = "I can't see anyone. Step into the frame"
MSG_STYLE_LATER = "Pick an effect when this one has finished"
STYLE_NAMES = {"DUST": "Dust", "ASH": "Ash", "EMBER": "Ember", "GLITCH": "Glitch"}


class SnapController:
    """State machine SNAP. Semua waktu (now) berupa detik monotonic yang diberikan pemanggil."""

    LIVE = "LIVE"
    PLATE_CAPTURE = "PLATE_CAPTURE"
    VANISH = "VANISH"
    INVISIBLE = "INVISIBLE"
    REAPPEAR = "REAPPEAR"
    EFFECT_STATES = (VANISH, INVISIBLE, REAPPEAR)

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        lebar, tinggi = cfg["proses_lebar"], cfg["proses_tinggi"]
        self.lebar, self.tinggi = lebar, tinggi
        self.bg = BackgroundEngine(cfg)
        self.matte = MatteEngine(cfg)
        # Optional snap.profiler.Profiler; the app sets it to time the matte and background stages.
        self.prof = None  # type: Any
        self.bg_siap = False
        self.bg_persen = 0
        self.flash = FlashOverlay(cfg, lebar, tinggi)
        self.debu = DissolveRenderer(cfg)
        self.abu = AshAmbient(cfg)
        self.jam = EffectClock(cfg)
        self.ukuran_butir = cfg["particle_cell_px"]
        self.tangga = QualityLadder(cfg)
        self.gaya = cfg["style_default"]
        self.glitch = GlitchRenderer()
        # Glitch style while visible: start time of the score and the step shown last (-1 = none).
        self._glitch_t0 = None  # type: Optional[float]
        self.glitch_langkah = -1

        self.t_ganti = self.flash.mulai_putih
        self.debu_akhir = cfg["debu_mulai_detik"] + cfg["debu_sapuan_detik"] + cfg["debu_umur_detik"]
        self.rakit_akhir = cfg["reappear_sweep_s"] + cfg["reappear_life_s"]
        self.reappear_akhir = self.rakit_akhir + cfg["reappear_blend_s"]
        # Effect time when the current vanish and reappear end: the dust timing above, or the glitch scores'.
        self.vanish_akhir = self.debu_akhir
        self.akhir_kembali = self.reappear_akhir
        # The current vanish or reappear is a glitch one; each effect gets new glitch shapes.
        self._glitch_efek = False
        self._efek_ke = 0

        self.state = self.LIVE
        self.t0 = 0.0
        self.cooldown_sampai = -math.inf
        self.t_rekam = 0.0
        self.butir_tampil = 0
        self.waktu_efek_terakhir = 0.0
        self._target_waktu = 1.0
        self._pesan = ""
        self._pesan_sampai = -math.inf
        self._pudar = None  # type: Optional[Tuple[np.ndarray, float]]
        self._frame = None  # type: Optional[np.ndarray]
        self._frame_tampil = None  # type: Optional[np.ndarray]
        self._mask = None  # type: Optional[np.ndarray]
        self._tint = None  # type: Optional[np.ndarray]
        self._tint_t = -math.inf
        self._kosongkan_snapshot()

    def _kosongkan_snapshot(self) -> None:
        self._beku_tampil = None  # type: Optional[np.ndarray]
        self._matte_beku = None  # type: Optional[np.ndarray]
        self._zona = None  # type: Optional[np.ndarray]
        self._zf = None  # type: Optional[np.ndarray]
        # Pixels the experimental fallback estimated into the plate.
        self._terisi = None  # type: Optional[np.ndarray]
        self._perkiraan = False
        self._siap = False
        # Reveal map: 1 where never seen room shows the live frame, faded in over reveal_fade_s.
        self._buka = None  # type: Optional[np.ndarray]
        self._buka_maks = 0.0
        self._t_buka = 0.0
        # Base layer data (union zone, plate view) cached for one live frame and matte.
        self._dasar_kunci = None  # type: Optional[Tuple[np.ndarray, Optional[np.ndarray]]]
        self._dasar_hasil = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]
        # Base layer data used by the last rendered frame, for the quality tools.
        self._info_dasar = None  # type: Optional[Tuple[np.ndarray, np.ndarray]]
        self._keluaran = None  # type: Optional[np.ndarray]
        # Crossfade weight toward the live body at the end of a reappear (0..1).
        self._k_blend = 0.0

    # Messages: one at a time; the HUD shows the current one as a fading toast.

    def set_pesan(self, teks: str, now: float, durasi: Optional[float] = None) -> None:
        self._pesan = teks
        self._pesan_sampai = now + (self.cfg["pesan_detik"] if durasi is None else durasi)

    def pesan(self, now: float) -> str:
        return self._pesan if now < self._pesan_sampai else ""

    # Timing

    def waktu_efek(self, now: float) -> float:
        return self.jam.t if self.state in self.EFFECT_STATES else 0.0

    def siap_dipicu(self, now: float) -> bool:
        """A snap now vanishes (LIVE after the cooldown) or reappears (INVISIBLE)."""
        return (self.state == self.LIVE and now >= self.cooldown_sampai) or self.state == self.INVISIBLE

    def sisa_hitung_mundur(self, now: float) -> int:
        return max(0, int(math.ceil(self.t_rekam - now)))

    def atur_waktu(self, beku: bool, pelan: bool) -> None:
        """Requested time control for this frame: fist freezes, pointing up slows, nothing runs normally."""
        self._target_waktu = 0.0 if beku else (self.cfg["slow_rate"] if pelan else 1.0)

    def kontrol_waktu_aktif(self) -> bool:
        """Time control works during the dissolve and reappear, never during the flash (a glitch vanish has
        none)."""
        if self.state == self.REAPPEAR:
            return True
        return self.state == self.VANISH and (self._glitch_efek or self.jam.t >= self.flash.akhir)

    def _tahap(self, nama: str) -> Any:
        return self.prof.stage(nama) if self.prof is not None else contextlib.nullcontext()

    # Per frame input

    def amati(self, now: float, frame_tampil: np.ndarray, frame_proses: np.ndarray, mask: np.ndarray) -> None:
        """Latest display frame, its process size copy, and the person mask (every state)."""
        self._frame_tampil = frame_tampil
        self._frame = frame_proses
        self._mask = mask
        with self._tahap("matte"):
            self.matte.update(frame_tampil, mask)
        if self.state in self.EFFECT_STATES:
            if self.state == self.INVISIBLE:
                # Keep learning the room the user uncovers while walking around invisible.
                with self._tahap("background"):
                    for teks in self.bg.update(now, frame_tampil, mask, allow_auto=False):
                        self.set_pesan(teks, now, self.cfg["pesan_detik"] * 2)
            t = self.jam.t
            # The base layer also hides the live person; preparing it here keeps it out of the render stage.
            if (self.state != self.VANISH or self._glitch_efek
                    or (t >= self.t_ganti and not self.flash.putih_penuh(t))):
                self._dasar(frame_tampil, now)
            return
        with self._tahap("matte"):
            zona = self.matte.core()
        with self._tahap("background"):
            if self.state == self.LIVE:
                for teks in self.bg.update(now, frame_tampil, mask):
                    self.set_pesan(teks, now, self.cfg["pesan_detik"] * 2)
            elif self.state == self.PLATE_CAPTURE:
                if now < self.t_rekam:
                    self.bg.update(now, frame_tampil, mask, allow_auto=False)
                else:
                    hasil = self.bg.capture_frame(frame_tampil, mask)
                    if hasil is not None:
                        self.set_pesan(CAPTURE_MESSAGES[hasil], now, self.cfg["pesan_detik"] * 2)
                        self._masuk_live(now)
            # The shadow part never lowers coverage (it is only added where known), so the core is enough here.
            cakupan = self.bg.coverage(zona) if zona.any() else self.bg.known_fraction()
            self.bg_siap = cakupan >= self.cfg["bg_coverage_min"]
            self.bg_persen = int(math.floor(100.0 * cakupan))

    def zona_hapus(self) -> Tuple[np.ndarray, np.ndarray]:
        """Removal zone of the latest frame: feathered Zf (float 0..1) and its shadow part (bool)."""
        frame = self._frame_tampil
        return self.matte.zone(frame, self.bg.plate_adj(frame), self.bg.known())

    def peta_gain(self) -> np.ndarray:
        """Exposure and color gain map (grid size, 3 channels) for the latest frame."""
        self.bg.plate_adj(self._frame_tampil)
        return self.bg.gain_small

    def tint(self, now: float) -> Optional[Tuple[np.ndarray, float]]:
        """Soft blue overlay on the unknown background after a refused snap: (mask, opacity) or None."""
        if self._tint is None:
            return None
        hold = self.cfg["pesan_detik"]
        fade = self.cfg["coverage_tint_fade_s"]
        masuk = float(smoothstep((now - self._tint_t) / self.cfg["hud_fade_s"]))
        alpha = self.cfg["coverage_tint_alpha"] * masuk * (1.0 - float(smoothstep((now - self._tint_t - hold) / fade)))
        if now - self._tint_t >= hold + fade:
            self._tint = None
            return None
        return self._tint, alpha

    # Actions

    def mulai_rekam_plate(self, now: float) -> None:
        if self.state != self.LIVE:
            return
        self.state = self.PLATE_CAPTURE
        self.t_rekam = now + self.cfg["plate_hitung_mundur"]
        self.bg.cancel_capture()

    def pilih_gaya(self, gaya: str, now: float) -> bool:
        """Effect style (from LIVE or INVISIBLE); it applies from the next vanish or reappear, and Glitch also
        right away while visible."""
        if self.state not in (self.LIVE, self.INVISIBLE):
            self.set_pesan(MSG_STYLE_LATER, now)
            return False
        if gaya != self.gaya:
            self.gaya = gaya
            # The glitch score starts over, from its calm opening.
            self._glitch_t0 = None
        self.set_pesan(STYLE_NAMES[self.gaya], now, self.cfg["style_toast_s"])
        return True

    def ganti_gaya(self, now: float) -> bool:
        """The next effect style in STYLES order."""
        return self.pilih_gaya(STYLES[(STYLES.index(self.gaya) + 1) % len(STYLES)], now)

    def picu(self, now: float, asal: Optional[Tuple[float, float]] = None) -> bool:
        """Snap: vanish from LIVE, reappear from INVISIBLE. asal = snap point 0..1 for the flash."""
        if self.state == self.INVISIBLE:
            self._mulai_reappear(now)
            return True
        if not self.siap_dipicu(now):
            return False
        if self._frame_tampil is None or self._mask is None or self.matte.matte is None:
            self.set_pesan(MSG_NO_PERSON, now)
            return False
        _, luas = PersonSegmenter.mask_snap(self._mask, self.cfg)
        if luas < self.cfg["mask_luas_min_orang"]:
            self.set_pesan(MSG_NO_PERSON, now)
            return False
        zf, _ = self.zona_hapus()
        zona = zf > 0.0
        # Coverage gate: never vanish over background that was not seen.
        perkiraan = self.bg.coverage(zona) < self.cfg["bg_coverage_min"]
        if perkiraan and not self.cfg["allow_inpaint_fallback"]:
            self.set_pesan(MSG_NEED_ROOM, now, self.cfg["pesan_detik"] * 2)
            self._tint = zona & ~self.bg.known()
            self._tint_t = now
            return False
        self._kosongkan_snapshot()
        self._pudar = None
        self._terapkan_kualitas()
        # Snapshot dibekukan di sini; dissolve memakai frame dan matte ini sampai selesai.
        self._beku_tampil = self._frame_tampil.copy()
        self._matte_beku = self.matte.matte.copy()
        self._zona = zona
        self._zf = zf
        self._perkiraan = perkiraan
        self._mulai_buka(now)
        self._glitch_efek = self.gaya == "GLITCH"
        self._efek_ke += 1
        if self._glitch_efek:
            # The live body glitches out; nothing to prepare under a flash.
            self.vanish_akhir = len(VANISH_SCORE) / float(self.cfg["glitch_step_hz"])
            self._siap = True
        else:
            self.vanish_akhir = self.debu_akhir
            self.flash.mulai(asal if asal is not None else tuple(self.cfg["flash_asal_cadangan"]))
        self.state = self.VANISH
        self.t0 = now
        self.jam.start(now)
        return True

    def _terapkan_kualitas(self) -> None:
        """Use the quality ladder's level; called only when an effect starts, never in the middle of one."""
        atur = self.tangga.settings()
        self.matte.scale = atur["matte_scale"]
        self.matte.bayangan = atur["shadow"]
        self.debu.blur = atur["dust_blur"]
        self.ukuran_butir = atur["cell"]

    def _mulai_reappear(self, now: float) -> None:
        """Take the live frame and its matte as the body the dust assembles into (or the glitch brings in)."""
        cfg = self.cfg
        self._terapkan_kualitas()
        frame = self._frame_tampil
        if frame is None or self.matte.matte is None:
            # Nothing seen yet to assemble; simply come back.
            self._masuk_live(now)
            return
        zf, _ = self.zona_hapus()
        matte = self.matte.matte
        self._beku_tampil = frame.copy()
        self._matte_beku = matte.copy()
        self._zona = zf > 0.0
        self._zf = zf
        self._dasar_kunci = None
        self._glitch_efek = self.gaya == "GLITCH"
        self._efek_ke += 1
        if self._glitch_efek:
            self.akhir_kembali = len(REAPPEAR_SCORE) / float(cfg["glitch_step_hz"])
        else:
            self.akhir_kembali = self.reappear_akhir
            timing = (0.0, cfg["reappear_sweep_s"], cfg["reappear_life_s"])
            self.debu.siapkan(self._beku_tampil, self._matte_beku, zf, self.ukuran_butir, self.gaya, timing)
        self._siap = True
        self.state = self.REAPPEAR
        self.t0 = now
        self.jam.start(now)

    def reset(self, now: float, pudar: bool = True) -> None:
        """Cancel to LIVE; with pudar the last shown frame fades into the live camera over cancel_fade_s."""
        if pudar and self.state != self.LIVE and self._keluaran is not None:
            self._pudar = (self._keluaran, now)
        else:
            self._pudar = None
        self._masuk_live(now)

    def _masuk_live(self, now: float) -> None:
        self.state = self.LIVE
        self._glitch_t0 = None
        self._glitch_efek = False
        self.cooldown_sampai = now + self.cfg["cooldown_detik"]
        self.butir_tampil = 0
        self._target_waktu = 1.0
        self._kosongkan_snapshot()

    # Rendering

    def _siapkan(self) -> None:
        """Dipanggil sekali saat layar putih penuh, jadi jeda persiapan tidak terlihat."""
        if self._perkiraan:
            # Experimental: estimate the unseen room once and keep it in the plate for every frame.
            # The pixels stay unknown, so the room learning replaces them later.
            hole = self._zona & ~self.bg.known()
            self.bg.plate[:] = self.bg.fill_unknown(self.bg.plate, self._zona)
            self._terisi = hole
        self.debu.siapkan(self._beku_tampil, self._matte_beku, self._zf, self.ukuran_butir, self.gaya)
        tinggi, lebar = self._beku_tampil.shape[:2]
        self.abu.siapkan((lebar, tinggi), np.random.default_rng(self.cfg["debu_seed"] + 1))
        self._siap = True

    def _tak_terlihat(self) -> np.ndarray:
        """Room pixels with no sample at all; the plate has nothing to show there."""
        unseen = self.bg.count == 0
        if self._terisi is not None:
            unseen &= ~self._terisi
        return unseen

    def _mulai_buka(self, now: float) -> None:
        # Room that was never seen is shown live from the start (the coverage gate keeps it tiny).
        self._buka = self._tak_terlihat().astype(np.float32)
        self._buka_maks = float(self._buka.max())
        self._t_buka = now

    def _perbarui_buka(self, unseen: np.ndarray, now: float) -> None:
        if self._buka is None or self._buka.shape != unseen.shape:
            self._mulai_buka(now)
            return
        langkah = max(0.0, now - self._t_buka) / self.cfg["reveal_fade_s"]
        self._t_buka = now
        if self._buka_maks <= 0.0 and not unseen.any():
            return
        self._buka += np.clip(unseen.astype(np.float32) - self._buka, -langkah, langkah)
        self._buka_maks = float(self._buka.max())

    def _dasar(self, frame: np.ndarray, now: float) -> Tuple[np.ndarray, np.ndarray]:
        """Union removal zone (float 0..1) and plate view for one live frame."""
        kunci = self._dasar_kunci
        if kunci is not None and kunci[0] is frame and kunci[1] is self.matte.matte:
            return self._dasar_hasil
        known = self.bg.known()
        matte_ok = self.matte.matte is not None and self.matte.matte.shape == frame.shape[:2]
        with self._tahap("matte"):
            orang = self.matte.core() if matte_ok else None
        with self._tahap("background"):
            # Exposure is matched only on known room pixels away from both the snapshot and the live person.
            area = known & ~self._zona
            if orang is not None:
                area &= ~orang
            adj = self.bg.plate_adj(frame, area)
        zf = self._zf
        zf_live = None
        if matte_ok:
            with self._tahap("matte"):
                zf_live, _ = self.matte.zone(frame, adj, known)
                zf = np.maximum(zf, zf_live)
        unseen = self._tak_terlihat()
        self._perbarui_buka(unseen, now)
        if self._buka_maks > 0.0:
            # Never seen room keeps the live frame instead of a made up plate value, revealed softly.
            zf = zf * (1.0 - smoothstep(self._buka))
            if self.state == self.INVISIBLE and zf_live is not None:
                live = zf_live > 0.0
                total = int(np.count_nonzero(live))
                if total and np.count_nonzero(live & unseen) > (1.0 - self.cfg["bg_coverage_min"]) * total:
                    self.set_pesan(MSG_CANT_SEE, now)
        with self._tahap("background"):
            pv = self.bg.plate_view(adj, zf)
        self._dasar_kunci = (frame, self.matte.matte)
        self._dasar_hasil = (zf, pv)
        return zf, pv

    def _lapisan_dasar(self, frame_tampil: np.ndarray, now: float) -> Tuple[np.ndarray, Tuple[slice, slice]]:
        """Base layer: live frame outside the zone, plate view inside, feathered at the edge; and the zone box."""
        zf, pv = self._dasar(frame_tampil, now)
        self._info_dasar = (zf, pv)
        hasil = frame_tampil.copy()
        x, y, w, h = cv2.boundingRect((zf > 0.0).view(np.uint8))
        kotak = (slice(y, y + h), slice(x, x + w))
        if w > 0 and h > 0:
            bobot = np.ascontiguousarray(zf[kotak])
            hasil[kotak] = cv2.blendLinear(frame_tampil[kotak], pv[kotak], 1.0 - bobot, bobot)
        return hasil, kotak

    def _komposisi_glitch(self, t: float, frame_tampil: np.ndarray, now: float) -> np.ndarray:
        """Glitch vanish or reappear: the pieces of the live body that are still shown, over the base layer."""
        dasar, _ = self._lapisan_dasar(frame_tampil, now)
        self.butir_tampil = 0
        skor = VANISH_SCORE if self.state == self.VANISH else REAPPEAR_SCORE
        k = min(len(skor) - 1, int(math.floor(t * self.cfg["glitch_step_hz"] + 1e-6)))
        self.glitch_langkah = k
        matte = self.matte.matte
        if matte is None or matte.shape != frame_tampil.shape[:2]:
            return dasar
        seed = 1000000 + 1000 * self._efek_ke + (0 if self.state == self.VANISH else 500) + k
        hasil = self.glitch.render(frame_tampil, matte, dasar, skor[k], seed)
        # Nobody in view: the base layer is what shows.
        return dasar if hasil is frame_tampil else hasil

    def _komposisi(self, t: float, frame_tampil: np.ndarray, now: float) -> np.ndarray:
        hasil, kotak = self._lapisan_dasar(frame_tampil, now)
        h, w = kotak[0].stop - kotak[0].start, kotak[1].stop - kotak[1].start
        self.butir_tampil = 0
        self._k_blend = 0.0
        if self.state == self.VANISH and t < self.debu_akhir:
            self.debu.render(hasil, t, self.abu.specks(t))
            self.butir_tampil = self.debu.visible
        elif self.state == self.REAPPEAR:
            # The dissolve played backward: the last dust to leave is the first to come back.
            self.debu.render(hasil, self.rakit_akhir - t, None)
            self.butir_tampil = self.debu.visible
            k = float(smoothstep((t - self.rakit_akhir) / self.cfg["reappear_blend_s"]))
            if k > 0.0 and w > 0 and h > 0:
                # Crossfade the assembled snapshot to the live frame, only where the zone is: outside it
                # the composite already equals the live frame, so k = 1 ends exactly on the live frame.
                hasil[kotak] = cv2.addWeighted(hasil[kotak], 1.0 - k, frame_tampil[kotak], k, 0.0)
            self._k_blend = k
        return hasil

    def info_kualitas(self, lebar: int, tinggi: int) -> Dict[str, Any]:
        """Data for tools/measure_quality.py at display size; entries are None when not available.

        zone is the part of the removal zone that is fully replaced by the plate view.
        """
        info = {"zone": None, "plate_view": None, "mass": None, "step_px": None}  # type: Dict[str, Any]
        if self._info_dasar is not None:
            zf, pv = self._info_dasar
            penuh = (zf >= 1.0).view(np.uint8) * np.uint8(255)
            if penuh.shape != (tinggi, lebar):
                penuh = cv2.resize(penuh, (lebar, tinggi), interpolation=cv2.INTER_NEAREST)
                pv = cv2.resize(pv, (lebar, tinggi), interpolation=cv2.INTER_LINEAR)
            info["zone"] = penuh
            info["plate_view"] = pv
        if not self._siap or self._glitch_efek:
            # The glitch changes in steps by design; dust mass and particle steps do not apply to it.
            return info
        effect_t = self.waktu_efek_terakhir
        skala = lebar / float(self.debu.size[0])
        if self.state == self.REAPPEAR:
            # The live body fades in as the snapshot body fades out, so it counts as visible mass.
            k = self._k_blend
            info["mass"] = min(1.0, self.debu.mass * (1.0 - k) + k)
            info["step_px"] = self.debu.step_max * skala
        else:
            active = self.cfg["debu_mulai_detik"] <= effect_t < self.debu_akhir
            info["mass"] = self.debu.mass if active else (1.0 if effect_t < self.cfg["debu_mulai_detik"] else 0.0)
            info["step_px"] = self.debu.step_max * skala if active else 0.0
        return info

    def render(self, now: float, frame_tampil: np.ndarray) -> np.ndarray:
        """Frame akhir (uint8, resolusi tampilan) tanpa teks HUD."""
        if self.state in (self.LIVE, self.PLATE_CAPTURE):
            return self._render_live(now, frame_tampil)
        target = self._target_waktu if self.kontrol_waktu_aktif() else 1.0
        t = self.jam.update(now, target)
        self.waktu_efek_terakhir = t
        self._info_dasar = None
        self.glitch_langkah = -1
        # The tolerance keeps a summed clock that lands a hair under an end on a frame time from ending a frame
        # late (10 glitch steps at 12 Hz end exactly on frame 25 at 30 fps).
        if self.state == self.VANISH and t >= self.vanish_akhir - 1e-6:
            self.state = self.INVISIBLE
            if self._perkiraan:
                self.set_pesan(MSG_ESTIMATED, now, self.cfg["pesan_detik"] * 2)
        auto = self.cfg["auto_reappear_s"]
        if self.state == self.INVISIBLE and auto > 0.0 and t >= self.vanish_akhir + auto:
            self._mulai_reappear(now)
            t = self.jam.update(now, 1.0)
            self.waktu_efek_terakhir = t
        if self.state == self.REAPPEAR and t >= self.akhir_kembali - 1e-6:
            self._masuk_live(now)
            return self._render_live(now, frame_tampil)

        if self._glitch_efek and self.state in (self.VANISH, self.REAPPEAR):
            hasil = self._komposisi_glitch(t, frame_tampil, now)
        elif self.state == self.VANISH and t < self.t_ganti:
            hasil = frame_tampil
        else:
            if not self._siap:
                self._siapkan()
            if self.state == self.VANISH and self.flash.putih_penuh(t):
                hasil = np.full_like(frame_tampil, 255)
            else:
                hasil = self._komposisi(t, frame_tampil, now)
        if self.state == self.VANISH and not self._glitch_efek:
            hasil = self.flash.komposit(hasil, t)
        self._keluaran = hasil
        return hasil

    def _render_live(self, now: float, frame_tampil: np.ndarray) -> np.ndarray:
        if self._pudar is None:
            if self.state == self.LIVE and self.gaya == "GLITCH":
                return self._glitch_live(now, frame_tampil)
            self.glitch_langkah = -1
            return frame_tampil
        beku, mulai = self._pudar
        k = float(smoothstep((now - mulai) / self.cfg["cancel_fade_s"]))
        if k >= 1.0 or beku.shape != frame_tampil.shape:
            self._pudar = None
            return frame_tampil
        return cv2.addWeighted(beku, 1.0 - k, frame_tampil, k, 0.0)

    def _glitch_live(self, now: float, frame: np.ndarray) -> np.ndarray:
        """The reference score on the live person, one step per glitch tick; the camera picture stays live."""
        h, w = frame.shape[:2]
        if self._glitch_t0 is None:
            self._glitch_t0 = now
        n = max(0, int(math.floor((now - self._glitch_t0) * self.cfg["glitch_step_hz"] + 1e-6)))
        self.glitch_langkah = n
        step = SCORE[n % len(SCORE)]
        matte = self.matte.matte
        if not step or matte is None or matte.shape != (h, w):
            return frame
        bersih = self._tanpa_orang(frame) if perlu_ruang(step) else None
        # The step number is the seed: each pass through the score gets new shapes.
        return self.glitch.render(frame, matte, bersih, step, n)

    def _tanpa_orang(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """The live frame with the person and their shadow replaced by the learned room; None while the room
        behind them is not known well enough (the glitch then keeps the body under what moves)."""
        if not self.bg_siap or self.matte.matte is None:
            return None
        # Runs inside the app's render stage, so it is not timed as a stage of its own.
        known = self.bg.known()
        adj = self.bg.plate_adj(frame, known & ~self.matte.core())
        zf, _ = self.matte.zone(frame, adj, known)
        # Room that was never seen keeps the live frame instead of a made up plate value.
        zf = cv2.multiply(zf, known.astype(np.float32))
        pv = self.bg.plate_view(adj, zf)
        hasil = frame.copy()
        x, y, bw, bh = cv2.boundingRect((zf > 0.0).view(np.uint8))
        if bw > 0 and bh > 0:
            kotak = (slice(y, y + bh), slice(x, x + bw))
            bobot = np.ascontiguousarray(zf[kotak])
            hasil[kotak] = cv2.blendLinear(frame[kotak], pv[kotak], 1.0 - bobot, bobot)
        return hasil
