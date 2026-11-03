# Retrolens web: design direction

Direction proposed by the agent as one of three options and chosen by the owner (Piddooow): "Komik cetak Spider-Verse".

Design Read: a playful camera toy and its guided setup, for curious people at a laptop, in a comic print language taken from the glitch effect itself. Dial: ENERGY 3 / RHYTHM 2 / MOTION 2.

## Identity
The product's look is its own effect: flat print inks, colors that miss their registration, halftone dots. The pages around the camera borrow that print, so the guide and the effect feel like one thing.

## Palette (2 neutrals, 3 inks)
- Paper `#FFF6DA`: background of every page outside the camera. A warm print stock, not screen white.
- Ink `#141414`: all text and outlines.
- Yellow `#FCDA02`: the one accent. Primary buttons and the active step only. It is the reference figure's color.
- Magenta `#F72969` and Cyan `#33D1F7`: only in the misregistration motif (offset copies behind the title and the active step). Never text, never a fill behind text.
All three inks are measured from reference-effect-spider-verse-1.mp4 (snap/glitch_score.py PALETTE).

## Typography
- Anton for headlines: a condensed poster grotesque, the weight and squeeze of a comic cover title.
- Atkinson Hyperlegible for everything else: the guide is read at a glance while someone steps out of frame, so legibility wins.

## Motif
Misregistration: the title (and the active guide step) carries a magenta copy and a cyan copy offset by a few pixels, like ink plates that slipped. It jumps in steps, not smoothly, matching the glitch's 12 steps a second. A small field of halftone dots sits in one corner of a page, never behind text.

## Motion (dial 2)
- The title's plates slip once when a page appears (0.5 s, stepped), then rest.
- A button's plates slip while it is hovered or focused.
- Nothing loops forever. With prefers-reduced-motion, nothing slips.

## Shape
Hard print edges: 2 px ink borders, 4 px radius on buttons and panels. A 4 px offset ink shadow marks the one thing to press on a screen (elevation = "this is the action"), nothing else.

## Theme
Fixed: paper outside the camera, the camera picture full bleed inside it. Reason: the identity is print on paper, and the camera view shows the room as it is.
