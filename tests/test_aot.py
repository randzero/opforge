"""AOT precompilation: reports, prewarm compilation, and the core assertion that
nothing recompiles across processes."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from opforge.aot import WarmEntry, WarmReport, iter_kernels, warm
from support import requires_torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _snapshot(directory: str) -> set[str]:
    return {
        str(path.relative_to(directory))
        for path in Path(directory).rglob("*")
        if path.is_file()
    }


def test_iter_kernels_finds_the_package_operators():
    names = [handle.name for handle in iter_kernels()]
    assert "rms_norm" in names, names
    assert names == sorted(names), "collection order must be stable"


def test_warm_reports_every_kernel_with_a_status():
    report = warm()
    assert isinstance(report, WarmReport)
    # may exceed iter_kernels(): op modules that fail to import appear as `skip:ImportError` entries
    assert len(report.entries) >= len(iter_kernels())
    for entry in report.entries:
        assert isinstance(entry, WarmEntry)
        assert entry.status, entry
        assert entry.kernel


def test_python_backend_needs_no_prewarm():
    """An op that declares no DSL kernel uses the python backend and naturally needs no prewarm.

    A test-local op is used rather than a built-in one -- built-ins (such as `rms_norm`) now
    all carry DSL kernels, and `backend="auto"` would select triton whenever a GPU is present,
    tying the test to the machine environment.
    """
    from opforge import Contract, kernel

    def _plain_impl(x):
        return x

    def _plain_ref(x):
        return x

    handle = kernel(name="_plain", contract=Contract.FAST, reference=_plain_ref)(_plain_impl)
    report = warm(kernels=[handle])
    (entry,) = report.entries
    assert entry.backend == "python"
    assert entry.status == "no-prewarm"
    assert entry.ok and report.ok


def test_unavailable_backend_is_reported_not_raised():
    """Without a GPU / when the backend is unavailable this should be a ``skip:`` entry,
    not something that crashes the build-time job."""
    report = warm()
    assert isinstance(report.ok, bool)
    for entry in report.entries:
        assert entry.status, entry


def test_report_format_is_human_readable():
    text = warm(kernels=[h for h in iter_kernels() if h.name == "rms_norm"]).format()
    assert "kernels" in text and "ok=" in text
    assert "rms_norm" in text


def test_handle_build_is_idempotent():
    handles = [h for h in iter_kernels() if h.name == "rms_norm"]
    (handle,) = handles
    first = handle.build()
    second = handle.build()
    assert first is second, "the same artifact should be reused under the same contract"


def test_aot_build_is_cross_process_and_does_not_recompile():
    """The core AOT claim: compile at build time, **only load at runtime**.

    Two processes each run the build entry point, and the second must not add any compilation
    artifacts -- both cache directories are inspected:

    * `TRITON_CACHE_DIR`: Triton's own on-disk cache (the compilation artifacts land here);
    * `OPFORGE_CACHE_DIR`: opforge's artifact cache (CuTeDSL lands here via `export_to_c`).
    """
    torch = requires_torch()
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA device")

    from opforge.lang.triton_backend import triton_usable

    if not triton_usable():
        raise unittest.SkipTest("triton unavailable (see tools/gpu_env.sh)")

    triton_dir = tempfile.mkdtemp()
    opforge_dir = tempfile.mkdtemp()
    env = {
        **os.environ,
        "TRITON_CACHE_DIR": triton_dir,
        "OPFORGE_CACHE_DIR": opforge_dir,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH", "")])),
    }
    command = [sys.executable, str(ROOT / "tools" / "build_aot.py")]

    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
    assert first.returncode == 0, first.stderr[-800:]
    for directory in (triton_dir, opforge_dir):
        assert _snapshot(directory), f"the first build should write artifacts under {directory}"
    after_first = _snapshot(triton_dir) | _snapshot(opforge_dir)

    second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
    assert second.returncode == 0, second.stderr[-800:]
    after_second = _snapshot(triton_dir) | _snapshot(opforge_dir)

    new_files = sorted(after_second - after_first)
    assert not new_files, (
        f"the second process produced new compilation artifacts (a recompile happened): "
        f"{new_files[:5]}"
    )
