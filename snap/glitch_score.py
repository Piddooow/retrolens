"""The Glitch style's palette and scores, transcribed frame by frame from the references.

SCORE (while visible) comes from reference-effect-spider-verse-1.mp4: 94 frames at 13 fps, played in order,
one step per glitch tick. VANISH_SCORE and REAPPEAR_SCORE come from the finished shots in
reference-effect-spider-verse-2.mp4: Rick glitching out (frames 296 to 342 at 60 fps) and Morty glitching
in (frames 8 to 56), about 0.8 s each, so 10 steps at 12 Hz. Colors are k-means centers measured on the
reference frames (RGB hex, converted to BGR where used).

A step is a dict of operations, all optional (an empty dict is a calm step, the frame stays untouched):
    cut:    1 light, 2 medium, 3 strong. Polygon pieces of the body shift, leaving thin gaps.
    fill:   palette names; pieces of the body are recolored flat (tritone, so features stay readable).
    cover:  share of the body the fill pieces take, 0..1.
    copies: (palette name, offset level 0..3) per offset copy of the body. MAGENTA, CYAN and GREEN_GHOST
            are see-through at GHOST_ALPHA and lie over the body, which then takes the reference's yellow;
            BODY is the body itself; other names are flat opaque copies behind the body.
    mirror: palette name of a mirrored copy.
    burst:  MINT, PASTEL, RED or OUTLINE star burst behind the body (the outline is drawn over it).
    blob:   palette names of large soft shapes behind the body.
    bg:     REGIONS, SHARDS or HALFTONE, a full frame background for this step.
    keep:   share of the body still shown, 0..1; the rest shows the room (vanish and reappear only).
"""

from typing import Any, Dict, List, Tuple

PALETTE = {
    # The reference character's own color.
    "YELLOW": "#FCDA02",
    # See-through copies: values solved for 50% opacity from their measured color over white.
    "MAGENTA": "#F72969",
    "CYAN": "#33D1F7",
    "GREEN_GHOST": "#21F951",
    # Flat recolor pieces and opaque copies.
    "RED": "#FA020B",
    "SOFT_RED": "#FA3434",
    "GREEN": "#0CFD02",
    "BLUE": "#0330FA",
    "AQUA": "#02F8FB",
    "SKY": "#01B2FC",
    "ORANGE": "#FD8901",
    "BLACK": "#020202",
    "DEEP_TEAL": "#226967",
    "DEEP_RED": "#3C0503",
    "PURPLE": "#5B509E",
    # Opaque light green copy behind the body (#10).
    "LIME": "#ADFD33",
    # Full frame color regions (#13, #82).
    "REG_CYAN": "#31FDF4",
    "REG_BLUE": "#2D39FD",
    "REG_GREEN": "#2FFD48",
    "REG_RED": "#FB3346",
    "REG_LIME": "#DDFC36",
    # Background shards (#12, #80).
    "SHARD_BLUE": "#2E3DFB",
    "SHARD_CYAN": "#34F9F4",
    "SHARD_GREEN": "#31FC4A",
    # Halftone (#15, #65): dot ink on white, with blue shards.
    "DOT": "#EF4C79",
    "HALF_BLUE": "#4941E8",
    # Star bursts (#21, #23, #65, #67).
    "MINT": "#04FB9B",
    "MINT_LINE": "#306E51",
    "PASTEL_PINK": "#F98FA6",
    "PASTEL_CYAN": "#9EECE6",
    "RED_BURST": "#E35453",
    "RED_LINE": "#7A2B30",
    # Soft shapes behind the body (#68 to #70).
    "BLOB_CYAN": "#03CAFB",
    "BLOB_GREEN": "#02F379",
    # Reference 2 recolors: Rick glitching out (#296 to #330).
    "NAVY": "#03001C",
    "LIME_RICK": "#86D409",
    "MAGENTA_RICK": "#C4009D",
    "YELLOW_RICK": "#F5EA0C",
    "VIOLET": "#4A2DB7",
    "VIOLET_EDGE": "#1C06AB",
    # Reference 2 recolors: Morty glitching in (#8 to #54).
    "ORANGE_RED": "#E65609",
    "PLUM": "#450949",
    "YELLOW_WARM": "#F4C72A",
    "MAGENTA_MORTY": "#D20AB4",
    "CREAM": "#FBFB8C",
    "ORANGE_MORTY": "#ED8A08",
    "MINT_MORTY": "#54FA82",
    "BLUE_MORTY": "#2090DF",
    "PINK_MORTY": "#E31EE3",
}

# Opacity of the see-through copies; with it MAGENTA over YELLOW gives #F88136 (reference #FC7C29) and
# CYAN over YELLOW gives #96D57D (reference #A6D66A) and GREEN_GHOST over YELLOW gives #8DE92A (reference #8EEA1E).
GHOST_ALPHA = 0.5
SEE_THROUGH = ("MAGENTA", "CYAN", "GREEN_GHOST")
BURSTS = ("MINT", "PASTEL", "RED", "OUTLINE")
BACKGROUNDS = ("REGIONS", "SHARDS", "HALFTONE")
OPERATIONS = ("cut", "fill", "cover", "copies", "mirror", "burst", "blob", "bg", "keep")

Step = Dict[str, Any]

M, C, G, B = "MAGENTA", "CYAN", "GREEN_GHOST", "BODY"


def _cmy(level: int) -> Tuple[Tuple[str, int], ...]:
    """The three color copies of the reference at one offset level."""
    return ((B, level), (M, level), (C, level))


# One entry per reference frame, #0 to #93.
SCORE = [
    {}, {}, {}, {}, {}, {},
    {"cut": 1},                                                                     # 6
    {"cut": 1},                                                                     # 7
    {"cut": 3},                                                                     # 8
    {"cut": 3},                                                                     # 9
    {"cut": 1, "copies": (("LIME", 1),)},                                           # 10
    {"cut": 2, "copies": (("SOFT_RED", 3),)},                                       # 11
    {"bg": "SHARDS", "cut": 2, "copies": (("BLUE", 1),)},                           # 12
    {"bg": "REGIONS", "cut": 2, "fill": ("RED", "GREEN", "BLUE"), "cover": 0.25},   # 13
    {"bg": "SHARDS", "cut": 3, "copies": _cmy(3) + ((M, 2), (C, 1))},               # 14
    {"bg": "HALFTONE", "cut": 2, "copies": ((C, 1),), "fill": ("BLUE", "GREEN"), "cover": 0.1},  # 15
    {"cut": 1, "fill": ("RED", "PURPLE", "ORANGE"), "cover": 0.8, "copies": ((M, 1),)},  # 16
    {"cut": 2, "fill": ("GREEN", "BLUE", "RED"), "cover": 0.5},                     # 17
    {"cut": 2, "fill": ("BLUE", "RED", "GREEN"), "cover": 0.85},                    # 18
    {"cut": 3, "fill": ("GREEN", "BLUE", "RED"), "cover": 0.4},                     # 19
    {"cut": 2, "fill": ("GREEN",), "cover": 0.7},                                   # 20
    {"burst": "MINT", "cut": 2, "fill": ("RED", "BLUE"), "cover": 0.6},             # 21
    {"burst": "MINT", "cut": 3, "fill": ("AQUA", "GREEN", "YELLOW"), "cover": 0.6},  # 22
    {"burst": "PASTEL", "copies": _cmy(2)},                                         # 23
    {"burst": "OUTLINE", "copies": _cmy(3)},                                        # 24
    {"copies": _cmy(1)},                                                            # 25
    {"mirror": C},                                                                  # 26
    {"cut": 3},                                                                     # 27
    {"cut": 1},                                                                     # 28
    {"cut": 3},                                                                     # 29
    {"cut": 2},                                                                     # 30
    {"cut": 1, "copies": ((C, 1),)},                                                # 31
    {},                                                                             # 32
    {"cut": 1, "copies": _cmy(2)},                                                  # 33
    {"copies": ((M, 2),)},                                                          # 34
    {"copies": ((C, 1),), "mirror": "BODY"},                                        # 35
    {"copies": ((B, 2), (M, 1))},                                                   # 36
    {"copies": ((C, 1),)},                                                          # 37
    {"cut": 1},                                                                     # 38
    {"cut": 3},                                                                     # 39
    {"cut": 1, "copies": ((G, 1), (G, 2))},                                         # 40
    {},                                                                             # 41
    {"cut": 1},                                                                     # 42
    {"copies": _cmy(1)},                                                            # 43
    {"copies": _cmy(3)},                                                            # 44
    {"copies": ((C, 2),)},                                                          # 45
    {"copies": ((M, 1),)},                                                          # 46
    {"copies": ((M, 1), (C, 1))},                                                  # 47
    {}, {},                                                                         # 48, 49
    {"cut": 2},                                                                     # 50
    {"cut": 2, "copies": ((G, 1), (G, 2))},                                         # 51
    {"cut": 3},                                                                     # 52
    {"cut": 1},                                                                     # 53
    {"cut": 1, "copies": ((M, 1),)},                                                # 54
    {"copies": ((C, 1),)},                                                          # 55
    {"cut": 2, "copies": _cmy(3)},                                                  # 56
    {"cut": 2, "copies": _cmy(3)},                                                  # 57
    {"copies": _cmy(1)},                                                            # 58
    {"copies": ((M, 1),)},                                                          # 59
    {"copies": _cmy(3)},                                                            # 60
    {"copies": ((B, 2), (C, 2))},                                                   # 61
    {"cut": 2, "fill": ("BLACK", "AQUA", "GREEN"), "cover": 0.9},                   # 62
    {"cut": 1, "fill": ("AQUA",), "cover": 0.05},                                   # 63
    {"cut": 2, "copies": (("DEEP_TEAL", 2), ("DEEP_RED", 1), (M, 2), (C, 3))},      # 64
    {"bg": "HALFTONE", "burst": "RED", "copies": ((C, 1), (M, 1))},                 # 65
    {"bg": "HALFTONE", "burst": "RED", "copies": ((M, 1), (B, 1))},                 # 66
    {"burst": "RED", "copies": _cmy(2)},                                            # 67
    {"blob": ("BLOB_CYAN",), "burst": "OUTLINE", "copies": _cmy(2)},                # 68
    {"blob": ("BLOB_GREEN", "BLOB_CYAN"), "copies": _cmy(3)},                       # 69
    {"blob": ("BLOB_GREEN", "SOFT_RED"), "copies": _cmy(3)},                        # 70
    {"cut": 2, "copies": ((C, 3),), "fill": ("BLACK",), "cover": 0.05},             # 71
    {"cut": 3, "fill": ("BLACK",), "cover": 0.3},                                   # 72
    {"cut": 3, "fill": ("BLACK",), "cover": 0.4},                                   # 73
    {"cut": 3},                                                                     # 74
    {"cut": 3},                                                                     # 75
    {"cut": 3},                                                                     # 76
    {"cut": 2, "fill": ("BLUE", "RED", "GREEN", "BLACK"), "cover": 0.7},            # 77
    {"cut": 2, "fill": ("BLUE", "GREEN"), "cover": 0.5},                            # 78
    {"cut": 2, "fill": ("GREEN", "RED"), "cover": 0.25},                            # 79
    {"bg": "SHARDS", "cut": 3, "copies": (("BLUE", 2),)},                           # 80
    {"bg": "SHARDS", "cut": 3, "copies": (("BLUE", 3),)},                           # 81
    {"bg": "REGIONS", "cut": 2},                                                    # 82
    {"bg": "SHARDS", "cut": 3, "copies": _cmy(2) + (("BLUE", 1),)},                 # 83
    {"bg": "SHARDS", "cut": 3, "copies": _cmy(2) + (("BLUE", 2),)},                 # 84
    {"bg": "SHARDS", "cut": 2, "copies": ((C, 2),)},                                # 85
    {"cut": 2, "fill": ("GREEN", "SKY", "BLACK"), "cover": 0.6},                    # 86
    {"cut": 3, "fill": ("BLACK", "ORANGE", "RED"), "cover": 0.6},                   # 87
    {"cut": 2, "fill": ("ORANGE",), "cover": 0.6},                                  # 88
    {"cut": 2, "fill": ("GREEN", "AQUA"), "cover": 0.9},                            # 89
    {"cut": 1, "fill": ("AQUA",), "cover": 0.15},                                   # 90
    {}, {}, {},                                                                     # 91 to 93
]  # type: List[Step]


# Rick glitching out: recolored halves, the body drops away first, the head breaks up, one strip lingers.
VANISH_SCORE = [
    {"fill": ("NAVY",), "cover": 0.5, "copies": (("VIOLET_EDGE", 1),)},                              # 296
    {"fill": ("LIME_RICK", "NAVY"), "cover": 0.7, "copies": (("VIOLET_EDGE", 1),), "cut": 1},        # 300
    {"keep": 0.55, "fill": ("MAGENTA_RICK",), "cover": 0.5, "cut": 2},                               # 304
    {"keep": 0.5, "fill": ("MAGENTA_RICK", "YELLOW_RICK"), "cover": 0.5, "cut": 2},                  # 308
    {"keep": 0.5, "fill": ("YELLOW_RICK", "MAGENTA_RICK"), "cover": 0.8, "cut": 1},                  # 312
    {"keep": 0.3, "fill": ("YELLOW_RICK", "MAGENTA_RICK"), "cover": 0.7, "cut": 2},                  # 316
    {"keep": 0.25, "fill": ("YELLOW_RICK", "VIOLET"), "cover": 0.7, "cut": 3},                       # 320
    {"keep": 0.2, "fill": ("YELLOW_RICK", "VIOLET"), "cover": 0.6, "cut": 3},                        # 328
    {"keep": 0.08},                                                                                    # 332
    {"keep": 0.08},                                                                                    # 338
]  # type: List[Step]

# Morty glitching in: a recolored piece of the head first, then more pieces, recolors fading out.
REAPPEAR_SCORE = [
    {"keep": 0.1, "fill": ("ORANGE_RED", "PLUM"), "cover": 1.0},                                       # 8
    {"keep": 0.25, "fill": ("YELLOW_WARM", "MAGENTA_MORTY", "ORANGE_RED"), "cover": 1.0},             # 14
    {"keep": 0.3, "fill": ("YELLOW_WARM", "ORANGE_RED", "MAGENTA_MORTY", "PLUM"), "cover": 1.0},      # 22
    {"keep": 0.55, "fill": ("CREAM", "ORANGE_RED", "YELLOW_WARM"), "cover": 0.9, "cut": 1},           # 26
    {"keep": 0.65, "fill": ("ORANGE_MORTY", "YELLOW_RICK", "CREAM"), "cover": 0.8, "cut": 2},         # 30
    {"keep": 0.8, "fill": ("MAGENTA_MORTY", "YELLOW_RICK", "ORANGE_MORTY"), "cover": 0.7, "cut": 1},  # 34
    {"keep": 0.9, "fill": ("MINT_MORTY", "BLUE_MORTY", "MAGENTA_MORTY"), "cover": 0.6},               # 38
    {"keep": 0.95, "fill": ("MINT_MORTY", "BLUE_MORTY", "PINK_MORTY"), "cover": 0.5, "cut": 1},       # 42
    {"fill": ("YELLOW_RICK", "MINT_MORTY", "PINK_MORTY"), "cover": 0.45, "cut": 1},                   # 48
    {"fill": ("YELLOW_RICK", "PINK_MORTY"), "cover": 0.3},                                             # 52
]  # type: List[Step]


def calm(step: Step) -> bool:
    return not step


def validate_score(score: List[Step]) -> List[str]:
    """Problems in a score, empty when every step only uses known operations and colors."""
    errors = []
    for i, step in enumerate(score):
        for key, value in step.items():
            if key not in OPERATIONS:
                errors.append("step %d: unknown operation %s" % (i, key))
            elif key == "cut" and value not in (1, 2, 3):
                errors.append("step %d: cut must be 1, 2 or 3" % i)
            elif key == "fill" and not all(n in PALETTE for n in value):
                errors.append("step %d: unknown fill color" % i)
            elif key == "cover" and not 0.0 < value <= 1.0:
                errors.append("step %d: cover must be in (0, 1]" % i)
            elif key == "copies" and not all((n == "BODY" or n in PALETTE) and lv in (0, 1, 2, 3) for n, lv in value):
                errors.append("step %d: bad copy" % i)
            elif key == "mirror" and value != "BODY" and value not in PALETTE:
                errors.append("step %d: unknown mirror color" % i)
            elif key == "burst" and value not in BURSTS:
                errors.append("step %d: unknown burst" % i)
            elif key == "blob" and not all(n in PALETTE for n in value):
                errors.append("step %d: unknown blob color" % i)
            elif key == "bg" and value not in BACKGROUNDS:
                errors.append("step %d: unknown background" % i)
            elif key == "keep" and not 0.0 < value <= 1.0:
                errors.append("step %d: keep must be in (0, 1]" % i)
        if ("fill" in step) != ("cover" in step):
            errors.append("step %d: fill and cover go together" % i)
    return errors
