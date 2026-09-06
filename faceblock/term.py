"""Terminal colour helpers.

Colour is disabled automatically when stdout is not a TTY, so piping output to a
file or a log stays free of escape codes, and when NO_COLOR is set - the
convention that lets users switch it off globally. FACEBLOCK_COLOR=always forces
it back on for cases like `less -R`.
"""

from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    if os.getenv("FACEBLOCK_COLOR", "").lower() == "always":
        return True
    if os.getenv("NO_COLOR") is not None or os.getenv("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


_ON = _enabled()


def _style(code: str):
    def apply(text: object) -> str:
        return f"\033[{code}m{text}\033[0m" if _ON else str(text)

    return apply


bold = _style("1")
dim = _style("2")
key = _style("96")  # bright cyan - the values that matter
head = _style("1;96")  # bold bright cyan - section headings
ok = _style("92")
warn = _style("93")
bad = _style("91")
link = _style("4;96")  # underlined, so terminals make it clickable-looking
