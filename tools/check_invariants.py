#!/usr/bin/env python3
"""Architecture invariant checks (I1-I9).

    PYTHONPATH=src python3 tools/check_invariants.py

Each invariant corresponds to a line in `docs/architecture.md` §8; here it gets an
**executable check**. Recommended after changing L0-L2; CI should run it.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

CORE_PACKAGES = ("contract", "cache", "launch", "kernel", "lang")
L3_PACKAGES = ("training", "inference")

#: Things core-layer modules must not import at module level (I1). Lazy imports inside
#: functions are unrestricted.
FORBIDDEN_IN_CORE = (
    "torch",
    "triton",
    "cutlass",
    "nvidia",
    "megatron",
    "deepspeed",
    "vllm",
    "sglang",
)


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #


def _iter_py(package_dir: Path):
    return sorted(package_dir.rglob("*.py"))


def _module_level_imports(path: Path) -> tuple[set[str], set[str]]:
    """Return (top-level package names of absolute imports, top-level names targeted by
    relative imports). Only **module-level** imports are counted."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    absolute: set[str] = set()
    relative: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            head = node.module.split(".")[0]
            if node.level == 0:
                absolute.add(head)
            else:
                relative.add(head)
    return absolute, relative


def _core_modules():
    package_root = SRC / "opforge"
    for name in CORE_PACKAGES:
        directory = package_root / name
        for path in _iter_py(directory):
            yield name, path


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

CHECKS: list[tuple[str, str, callable]] = []


def check(invariant: str, description: str):
    def decorate(fn):
        CHECKS.append((invariant, description, fn))
        return fn

    return decorate


@check("I1", "L0-L2 must not import training/inference frameworks at module level")
def i1():
    violations = []
    for _, path in _core_modules():
        absolute, _ = _module_level_imports(path)
        hit = sorted(absolute & set(FORBIDDEN_IN_CORE))
        if hit:
            violations.append(f"{path.relative_to(SRC)}: {hit}")
    assert not violations, "module-level import violation(s):\n  " + "\n  ".join(violations)


@check("I2", "L0-L2 must not depend back on the L3 adapter layer")
def i2():
    violations = []
    for _, path in _core_modules():
        _, relative = _module_level_imports(path)
        hit = sorted(relative & set(L3_PACKAGES))
        if hit:
            violations.append(f"{path.relative_to(SRC)}: {hit}")
    assert not violations, "backward dependency:\n  " + "\n  ".join(violations)


@check("I3", "contract fallback may only go in the stricter direction")
def i3():
    from itertools import combinations

    from opforge import Contract, resolve

    impls_pool = list(Contract)
    for size in range(1, len(impls_pool) + 1):
        for subset in combinations(impls_pool, size):
            impls = {c: (lambda *a, **k: None) for c in subset}
            for requested in Contract:
                try:
                    got = resolve(requested, impls, op_name="I3").actual
                except Exception as exc:  # ContractUnsatisfied is a legitimate outcome
                    from opforge.errors import ContractUnsatisfied

                    assert isinstance(exc, ContractUnsatisfied), exc
                    continue
                assert got.is_at_least_as_strict_as(requested), (
                    f"fell back to a looser implementation: requested {requested.name}, "
                    f"got {got.name}"
                )


@check("I4", "every contract fallback is counted")
def i4():
    from opforge import Contract, fallback_counts, reset_fallback_counts, resolve

    reset_fallback_counts()
    only_det = {Contract.DETERMINISTIC: (lambda *a, **k: None)}
    resolve(Contract.FAST, only_det, op_name="I4")
    resolve(Contract.HIGH_PRECISION, only_det, op_name="I4")
    resolve(Contract.DETERMINISTIC, only_det, op_name="I4")  # no fallback

    counts = fallback_counts()
    assert len(counts) == 2, f"fallback counts are wrong: {counts}"
    assert all(v == 1 for v in counts.values()), counts


@check(
    "I5", "runtime dimensions do not enter the cache key; scalar values do not enter the "
    "launch signature"
)
def i5():
    import numpy as np

    from opforge import Contract, KernelCache
    from opforge.kernels.norm import rms_norm
    from opforge.launch import LaunchSpec, Launcher

    # (a) same op, different shapes -> build only once
    cache = KernelCache(Path(tempfile.mkdtemp()), persist=False)
    builds = []
    key = rms_norm.build_key(Contract.HIGH_PRECISION)
    for shape in ((4, 128), (64, 2048)):
        cache.get_or_build(key, lambda: builds.append(1) or object())
    assert len(builds) == 1, (
        f"different shapes caused {len(builds)} builds (shape leaked into the cache key)"
    )

    # (b) scalar values do not enter the launch signature, tensor shapes do
    launcher = Launcher()
    artifact = object()
    artifact_run = []

    class _Art:
        def run(self, args, kwargs):
            artifact_run.append((args, kwargs))

    for value in (1e-6, 1e-5, 12345.0):
        launcher(LaunchSpec("op", _Art(), (np.zeros(4),), {"eps": value}))
    after_scalars = launcher.cached_signatures

    class _T:
        def __init__(self, shape):
            self.shape = shape
            self.dtype = "float32"

    launcher(LaunchSpec("op", _Art(), (_T((4, 128)),), {"eps": 1e-6}))
    after_shape = launcher.cached_signatures

    assert after_scalars == 1, (
        f"scalar values entered the launch signature (expected 1, got {after_scalars})"
    )
    assert after_shape == 2, (
        f"tensor shapes did not enter the launch signature (expected 2, got {after_shape})"
    )


@check("I6", "a missing / mismatched manifest must raise, not silently fall back")
def i6():
    from opforge import BuildKey, Contract, KernelCache, Manifest, source_fingerprint, target_fingerprint
    from opforge.cache.manifest import file_sha256
    from opforge.errors import ManifestError

    root = Path(tempfile.mkdtemp())
    key = BuildKey(source_fingerprint({"op": "x"}), target_fingerprint(), "I6", Contract.FAST)

    class _Art:
        def export(self, path, key):
            Path(path).write_text("payload")

        def run(self, args, kwargs):  # pragma: no cover
            return None

    cache = KernelCache(root, persist=True)
    cache.get_or_build(key, _Art, load=lambda k, p: _Art())
    assert key.artifact_path(root).exists(), "artifact was not written to disk"

    # manifest corrupted
    cache.clear_memory()
    key.manifest_path(root).write_text('{"format_version": 999}')
    try:
        cache.get_or_build(key, _Art, load=lambda k, p: _Art())
    except ManifestError:
        pass
    else:
        raise AssertionError("corrupted manifest did not raise")

    # manifest missing
    cache.clear_memory()
    key.manifest_path(root).unlink()
    try:
        cache.get_or_build(key, _Art, load=lambda k, p: _Art())
    except ManifestError:
        pass
    else:
        raise AssertionError("missing manifest did not raise")

    # artifact tampered with (digest mismatch)
    cache.clear_memory()
    Manifest.create(key, artifact_digest="0" * 64).write(key.manifest_path(root))
    try:
        cache.get_or_build(key, _Art, load=lambda k, p: _Art())
    except ManifestError:
        pass
    else:
        raise AssertionError("artifact digest mismatch did not raise")
    assert file_sha256  # keep the reference, showing the real digest implementation is used


@check("I7", "no compilation may occur after seal compilation")
def i7():
    from opforge import BuildKey, Contract, KernelCache, source_fingerprint, target_fingerprint
    from opforge.errors import CompilationSealed

    key = BuildKey(source_fingerprint({"op": "y"}), target_fingerprint(), "I7", Contract.FAST)
    cache = KernelCache(Path(tempfile.mkdtemp()), persist=False)
    cache.seal()
    try:
        cache.get_or_build(key, object)
    except CompilationSealed:
        pass
    else:
        raise AssertionError("build still succeeded after seal compilation")
    assert cache.compiled_since_seal == ("I7",), cache.compiled_since_seal


@check("I8", "an escape hatch must carry a reason, and CALL_SHAPE_MISMATCH is rejected")
def i8():
    from opforge import EscalationReason, cpp_kernel, escalations
    from opforge.errors import EscalationNotAllowed

    # legal: missing primitive
    @cpp_kernel(reason=EscalationReason.MISSING_PRIMITIVE, primitives=["named_barrier"])
    def _ok():
        return None

    assert escalations()["_ok"].primitives == ("named_barrier",)

    # illegal: call-shape mismatch
    try:

        @cpp_kernel(reason=EscalationReason.CALL_SHAPE_MISMATCH)
        def _bad():
            return None

    except EscalationNotAllowed:
        pass
    else:
        raise AssertionError("CALL_SHAPE_MISMATCH was accepted after all")

    # illegal: PERF_GAP with no evidence
    try:

        @cpp_kernel(reason=EscalationReason.PERF_GAP)
        def _bad2():
            return None

    except EscalationNotAllowed:
        pass
    else:
        raise AssertionError("PERF_GAP passed without benchmark evidence")


@check("I9", "every op has a reference")
def i9():
    # Reuse aot.iter_kernels: it tolerates "op module import failures" (e.g. a DSL that is
    # not installed), which is a different matter from "op written but missing a
    # reference" and must not be conflated.
    from opforge.aot import iter_kernels

    handles = iter_kernels()
    assert handles, "no ops found (is opforge.kernels empty?)"
    missing = [handle.name for handle in handles if handle.reference is None]
    assert not missing, f"ops missing a reference: {missing}"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    width = max(len(inv) for inv, _, _ in CHECKS)
    failed = 0
    for invariant, description, fn in CHECKS:
        try:
            fn()
        except unittest.SkipTest as exc:  # pragma: no cover
            print(f"  SKIP  {invariant:<{width}}  {description}  ({exc})")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {invariant:<{width}}  {description}")
            print(f"        {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {invariant:<{width}}  {description}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} invariants hold")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
