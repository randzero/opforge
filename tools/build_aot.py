#!/usr/bin/env python3
"""AOT precompile entry point — compile every op at **build time** and persist it to disk.

    # build time (inside the image) — both dirs point at persistent locations
    TRITON_CACHE_DIR=/opt/opforge-triton \
    OPFORGE_CACHE_DIR=/opt/opforge-kernels \
    python3 tools/build_aot.py

    # at runtime, just mount the same two dirs — installed means warm start
    export TRITON_CACHE_DIR=/opt/opforge-triton
    export OPFORGE_CACHE_DIR=/opt/opforge-kernels

Division of labor between the two dirs: **Triton artifacts can only live in Triton's own
cache** (it has no export API); **CuTeDSL artifacts are managed by opforge itself** (the
object + manifest produced by `export_to_c`). So on a second run, the CuTeDSL line in the
report shows `loaded` (disk hit, no compilation).

Running it on a machine with no GPU prints ``skip:`` and returns non-zero — this is
deliberate: precompilation must happen on the **target architecture**, and artifacts
built for one architecture cannot be reused across architectures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _cache_dirs() -> list[tuple[str, str]]:
    triton_dir = os.environ.get("TRITON_CACHE_DIR")
    opforge_dir = os.environ.get("OPFORGE_CACHE_DIR")
    return [
        ("TRITON_CACHE_DIR", triton_dir or "(unset, Triton default ~/.triton/cache)"),
        ("OPFORGE_CACHE_DIR", opforge_dir or "(unset, opforge default <tmp>/<user>/opforge-cache)"),
    ]


def main() -> int:
    from opforge.aot import warm

    print("AOT precompile")
    for name, value in _cache_dirs():
        print(f"  {name} = {value}")
    print()

    report = warm()
    print(report.format())

    if not report.ok:
        print(
            "\nSome ops could not be precompiled. For a `skip:` entry, this usually means "
            "no GPU or an unavailable backend —\n"
            "precompilation must be done on the same architecture as deployment.",
            file=sys.stderr,
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
