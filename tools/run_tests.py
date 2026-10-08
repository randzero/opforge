#!/usr/bin/env python3
"""Minimal test runner — lets tests run in environments without pytest.

    PYTHONPATH=src python3 tools/run_tests.py [keyword]

The tests themselves are **pytest style** (plain ``test_*`` functions in modules), so if
pytest is installed, plain ``pytest`` works too. To skip a test, raise
``unittest.SkipTest`` (both runners honor it).
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "tests"


def _load(path: Path):
    name = f"_opforge_test_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _collect(keyword: str | None):
    cases = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module = _load(path)
        for name, fn in vars(module).items():
            if not (name.startswith("test_") and inspect.isfunction(fn)):
                continue
            if getattr(fn, "__module__", None) != module.__name__:
                continue
            if keyword and keyword not in f"{path.stem}::{name}":
                continue
            cases.append((path.name, name, fn))
    return cases


def main(argv: list[str]) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(TESTS_DIR))  # make `from support import ...` work
    keyword = argv[0] if argv else None
    cases = _collect(keyword)

    passed = failed = skipped = 0
    failures: list[str] = []

    for filename, name, fn in cases:
        try:
            fn()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"  SKIP {filename}::{name}  ({exc})")
        except Exception:
            failed += 1
            failures.append(f"{filename}::{name}\n{traceback.format_exc()}")
            print(f"  FAIL {filename}::{name}")
        else:
            passed += 1
            print(f"  ok   {filename}::{name}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped  (of {len(cases)})")
    for block in failures:
        print(f"\n--- failure ---\n{block}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
