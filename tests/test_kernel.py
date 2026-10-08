"""L1 kernel layer: declaration validation, lazy reference resolution, call pipeline,
per-contract cache tiers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from opforge import (
    Contract,
    KernelCache,
    KernelHandle,
    RequiresAligned,
    Unavailable,
    fallback_counts,
    kernel,
    reset_fallback_counts,
)
from opforge.errors import KernelSpecError
from support import FakeTensor

# --------------------------------------------------------------------------- #
# Module-level probe ops: the reference is deliberately defined **after** the decorated
# function, to verify lazy resolution
# --------------------------------------------------------------------------- #


def _probe_det(x):
    return ("det", x)


def _probe_fast(x):
    return ("fast", x)


@kernel(
    name="_probe",
    contract=Contract.HIGH_PRECISION,
    variants={Contract.DETERMINISTIC: _probe_det, Contract.FAST: _probe_fast},
    tags=("test",),
)
def _probe(x):
    return ("hp", x)


def _probe_ref(x):
    return ("ref", x)


@kernel(name="_probe_det_only", contract=Contract.DETERMINISTIC)
def _probe_det_only(x):
    return "det"


def _probe_det_only_ref(x):
    return "ref"


@kernel(
    name="_probe_aligned",
    contract=Contract.FAST,
    preconditions=(RequiresAligned(16),),
)
def _probe_aligned(x):
    return "aligned"


def _probe_aligned_ref(x):
    return "ref"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _isolated(op) -> KernelHandle:
    """The same op declaration + a fresh, in-process cache instance (to keep cases from
    polluting each other)."""
    return KernelHandle(op.spec, cache=KernelCache(Path(tempfile.mkdtemp()), persist=False))


# --------------------------------------------------------------------------- #
# Declaration validation
# --------------------------------------------------------------------------- #


def test_decorator_rejects_non_contract():
    _raises(KernelSpecError, kernel, name="x", contract="fast")


def test_decorator_rejects_empty_name():
    _raises(KernelSpecError, kernel, name="", contract=Contract.FAST)


def test_decorator_rejects_duplicate_default_in_variants():
    _raises(
        KernelSpecError,
        kernel,
        name="x",
        contract=Contract.FAST,
        variants={Contract.FAST: _probe_fast},
    )


def test_decorator_rejects_bad_variant_key():
    _raises(
        KernelSpecError,
        kernel,
        name="x",
        contract=Contract.FAST,
        variants={"fast": _probe_fast},
    )


def test_decorator_rejects_non_positive_alignment():
    _raises(KernelSpecError, kernel, name="x", contract=Contract.FAST, min_alignment=0)


def test_spec_exposes_params_tags_and_contracts():
    assert [p.name for p in _probe.spec.params] == ["x"]
    assert _probe.spec.tags == ("test",)
    assert [c.name for c in _probe.spec.available_contracts] == [
        "DETERMINISTIC",
        "HIGH_PRECISION",
        "FAST",
    ]


# --------------------------------------------------------------------------- #
# Call pipeline
# --------------------------------------------------------------------------- #


def test_reference_is_resolved_lazily_by_naming_convention():
    assert _probe.reference is _probe_ref
    assert _probe_det_only.reference is _probe_det_only_ref


def test_contract_selects_the_matching_variant():
    handle = _isolated(_probe)
    assert handle(FakeTensor((2, 8)), contract=Contract.FAST)[0] == "fast"
    assert handle(FakeTensor((2, 8)), contract=Contract.DETERMINISTIC)[0] == "det"
    assert handle(FakeTensor((2, 8)))[0] == "hp"


def test_requesting_a_looser_contract_falls_back_and_is_counted():
    reset_fallback_counts()
    handle = _isolated(_probe_det_only)
    assert handle(FakeTensor((2, 8)), contract=Contract.FAST) == "det"
    counts = fallback_counts()
    assert len(counts) == 1, counts
    assert next(iter(counts))[0] == "_probe_det_only"


def test_requesting_a_stricter_contract_than_available_raises():
    from opforge.errors import ContractUnsatisfied

    handle = _isolated(_probe_det_only)  # only DETERMINISTIC
    # the reverse is impossible: requesting something stricter when already at the strictest;
    # a FAST-only op is used to verify
    only_fast = kernel(name="_only_fast", contract=Contract.FAST)(_probe_fast)
    isolated = KernelHandle(only_fast.spec, cache=KernelCache(Path(tempfile.mkdtemp()), persist=False))
    _raises(ContractUnsatisfied, isolated, FakeTensor((2, 8)), contract=Contract.DETERMINISTIC)
    assert handle is not None


def test_precondition_failure_returns_unavailable_not_exception():
    handle = _isolated(_probe_aligned)
    result = handle(FakeTensor((2, 8)))  # trailing dim 8 is not a multiple of 16
    assert isinstance(result, Unavailable)
    assert bool(result) is False
    assert "requires_aligned" in result.reason
    assert handle(FakeTensor((2, 16))) == "aligned"  # returns normally when the condition holds


def test_ref_bypasses_contracts_and_supports_fallback():
    tensor = FakeTensor((2, 8))
    assert _probe.ref(tensor) == ("ref", tensor)
    assert _probe_det_only.ref(tensor) == "ref"


def test_each_contract_gets_its_own_cache_entry():
    cache = KernelCache(Path(tempfile.mkdtemp()), persist=False)
    handle = KernelHandle(_probe.spec, cache=cache)
    for _ in range(3):
        handle(FakeTensor((2, 8)), contract=Contract.FAST)
    handle(FakeTensor((2, 8)), contract=Contract.DETERMINISTIC)
    handle(FakeTensor((2, 8)))
    assert cache.stats["misses"] == 3, cache.stats  # one for each of the three tiers
    assert cache.stats["hits"] == 2, cache.stats


def test_different_shapes_share_one_cache_entry():
    """Invariant I5: runtime shapes must not enter the cache key."""
    cache = KernelCache(Path(tempfile.mkdtemp()), persist=False)
    handle = KernelHandle(_probe.spec, cache=cache)
    for shape in ((2, 8), (64, 2048), (1, 1)):
        handle(FakeTensor(shape), contract=Contract.FAST)
    assert cache.stats["misses"] == 1, cache.stats
    assert cache.stats["hits"] == 2, cache.stats


def test_kernel_is_callable_and_introspectable():
    assert _probe.__name__ == "_probe"
    assert _probe.name == "_probe"
    assert _probe.contract is Contract.HIGH_PRECISION
    assert "KernelHandle" in repr(_probe)


def test_rejecting_non_contract_argument_type():
    handle = _isolated(_probe)
    try:
        handle(FakeTensor((2, 8)), contract="fast")
    except TypeError:
        return
    raise AssertionError("a wrong contract argument type should raise TypeError")
