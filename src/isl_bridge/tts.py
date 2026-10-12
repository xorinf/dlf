"""Phase 7 — Tiny text-to-speech wrapper.

No extra dependencies. On macOS we shell out to `say`. On other platforms we
fall back to a silent no-op so the rest of the pipeline still imports.

The TTS layer is intentionally dead-simple. We don't model voice selection,
language detection, or async queueing — those are upgrades only if a user
asks for them.

# ponytail: subprocess to `say`. If we ever need to swap to an online TTS
#            provider, this is the only call site to touch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def say(text: str, *, voice: str | None = None, out_path: Path | None = None) -> bool:
    """Speak `text`. Returns True on success, False on platform/driver failure.

    If `out_path` is given, write an audio file instead of speaking it
    (works on macOS via `say -o <file>.aiff`).
    """
    if not text:
        return False
    if sys.platform != "darwin" or shutil.which("say") is None:
        return False
    cmd = ["say"]
    if voice:
        cmd += ["-v", voice]
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-o", str(out_path), "--data-format=LEI16@22050"]
    cmd += [text]
    try:
        subprocess.run(cmd, check=True, timeout=10,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def main() -> int:
    p = __import__("argparse").ArgumentParser()
    p.add_argument("text")
    p.add_argument("--voice", default=None)
    p.add_argument("--out", type=Path, default=None,
                   help="If set, write an AIFF/LEI16 file instead of speaking.")
    args = p.parse_args()
    ok = say(args.text, voice=args.voice, out_path=args.out)
    print("OK" if ok else "UNAVAILABLE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())