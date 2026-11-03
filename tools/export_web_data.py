"""Write the settings and glitch scores the web app shares with the Python app, as JSON.

The web app (web/) reads web/src/data/*.json and never keeps its own copy of a timing, color or score, so
both apps always match. Run it after changing snap/config.py or snap/glitch_score.py:
    python tools/export_web_data.py
With --check it only reports whether the JSON files are up to date (exit code 1 when they are not).
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from snap.config import CONFIG, MODE_TRIGGER, REFERENCE_FPS, STYLES  # noqa: E402
from snap.glitch_score import GHOST_ALPHA, PALETTE, REAPPEAR_SCORE, SCORE, SEE_THROUGH, VANISH_SCORE  # noqa: E402

OUT = os.path.join(ROOT, "web", "src", "data")


def files() -> Dict[str, Any]:
    return {
        "config.json": {"config": CONFIG, "styles": list(STYLES), "modeTrigger": list(MODE_TRIGGER),
                        "referenceFps": REFERENCE_FPS},
        "glitch.json": {"palette": PALETTE, "ghostAlpha": GHOST_ALPHA, "seeThrough": list(SEE_THROUGH),
                        "score": SCORE, "vanishScore": VANISH_SCORE, "reappearScore": REAPPEAR_SCORE},
    }


def render(data: Any) -> str:
    # Tuples become lists; keys keep their order so the files diff cleanly.
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export shared settings and scores for the web app.")
    parser.add_argument("--check", action="store_true", help="only check that the JSON files are up to date")
    args = parser.parse_args()
    stale = []
    for name, data in files().items():
        path = os.path.join(OUT, name)
        text = render(data)
        current = open(path, encoding="utf-8").read() if os.path.isfile(path) else None
        if current == text:
            continue
        if args.check:
            stale.append(name)
            continue
        os.makedirs(OUT, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("Wrote %s" % os.path.relpath(path, ROOT))
    if stale:
        print("Out of date: %s. Run python tools/export_web_data.py" % ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
