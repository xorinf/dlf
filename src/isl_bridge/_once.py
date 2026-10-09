"""Process-wide one-shot printer.

Tiny Ponytail-grade helper: `once.warn(msg)` prints to stderr exactly once per
process, idempotent across multiple call sites. We use this to deduplicate
warnings that come from inside frequently-called library code paths.
"""

from __future__ import annotations

import sys
from threading import Lock

_lock = Lock()
_seen: set[str] = set()


def warn(msg: str) -> None:
    with _lock:
        if msg in _seen:
            return
        _seen.add(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


def reset_for_tests() -> None:  # pragma: no cover
    with _lock:
        _seen.clear()
