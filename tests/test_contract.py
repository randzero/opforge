"""L2 contract layer: one-way fallback, fallback counts, preconditions."""

from __future__ import annotations

from itertools import combinations

from opforge import (
    CallContext,
    Contract,
    ExecContext,
    RequiresAligned,
    RequiresAllRanksLockstep,
    RequiresDtypeIn,
    RequiresMinLeadingDim,
    RequiresNoCapture,
    check_all,
    fallback_counts,
    reset_fallback_counts,
    resolve,
)
from opforge.errors import ContractUnsatisfied
from support import FakeTensor


def _noop(*args, **kwargs):
    return None


def test_resolve_never_falls_back_to_a_looser_contract():
    """Invariant I3: over every implementation combination, the actual result must be
    at least as strict as the one requested."""
    for size in range(1, len(Contract) + 1):
        for subset in combinations(list(Contract), size):
            impls = {c: _noop for c in subset}
            for requested in Contract:
                try:
                    actual = resolve(requested, impls, op_name="t").actual
                except ContractUnsatisfied:
                    continue
                assert actual.is_at_least_as_strict_as(requested), (requested, actual)


def test_resolve_prefers_the_cheapest_satisfying_implementation():
    impls = {Contract.DETERMINISTIC: _noop, Contract.FAST: _noop}
    assert resolve(Contract.FAST, impls, op_name="t").actual is Contract.FAST
    # requesting HIGH_PRECISION should not reach for the pricier DETERMINISTIC... but impls
    # has no HP, so it can only tighten
    assert resolve(Contract.HIGH_PRECISION, impls, op_name="t").actual is Contract.DETERMINISTIC


def test_resolve_raises_when_only_looser_implementations_exist():
    try:
        resolve(Contract.DETERMINISTIC, {Contract.FAST: _noop}, op_name="t")
    except ContractUnsatisfied:
        return
    raise AssertionError(
        "requesting DETERMINISTIC when only a FAST implementation exists "
        "must raise ContractUnsatisfied"
    )


def test_resolve_raises_on_empty_implementation_set():
    try:
        resolve(Contract.FAST, {}, op_name="t")
    except ContractUnsatisfied:
        return
    raise AssertionError("an empty implementation set must raise ContractUnsatisfied")


def test_fallbacks_are_counted_but_exact_hits_are_not():
    reset_fallback_counts()
    impls = {Contract.DETERMINISTIC: _noop, Contract.FAST: _noop}

    resolve(Contract.FAST, impls, op_name="k")
    assert fallback_counts() == {}, "an exact hit must not be counted"

    resolve(Contract.FAST, {Contract.DETERMINISTIC: _noop}, op_name="k")
    counts = fallback_counts()
    assert len(counts) == 1 and all(v == 1 for v in counts.values()), counts


def test_custom_fallback_hook_receives_resolution():
    seen = []
    resolve(
        Contract.FAST,
        {Contract.DETERMINISTIC: _noop},
        op_name="k",
        on_fallback=lambda name, resolved: seen.append((name, resolved.actual)),
    )
    assert seen == [("k", Contract.DETERMINISTIC)], seen


def test_preconditions_report_unavailable_instead_of_raising():
    preconditions = (
        RequiresAllRanksLockstep(),
        RequiresAligned(16),
        RequiresDtypeIn("float32", "bfloat16"),
    )

    ok = check_all(preconditions, CallContext(args=(FakeTensor((8, 4096)),)))
    assert ok.ok, ok.reason

    captured = check_all(
        preconditions,
        CallContext(exec=ExecContext(world_size=4, is_capturing=True), args=(FakeTensor((8, 4096)),)),
    )
    assert not captured.ok and "lockstep" in captured.reason, captured.reason

    misaligned = check_all(preconditions, CallContext(args=(FakeTensor((8, 1000)),)))
    assert not misaligned.ok and "multiple of 16" in misaligned.reason

    wrong_dtype = check_all(preconditions, CallContext(args=(FakeTensor((8, 4096), "int8"),)))
    assert not wrong_dtype.ok and "int8" in wrong_dtype.reason


def test_preconditions_pass_when_the_relevant_state_is_absent():
    """When a condition is not applicable it should pass (e.g. single rank, no capture,
    no dtype information)."""
    assert check_all((RequiresAllRanksLockstep(),), CallContext()).ok
    assert check_all((RequiresNoCapture(),), CallContext()).ok
    assert check_all((RequiresMinLeadingDim(64),), CallContext(args=(FakeTensor((8, 16)),))).ok is False
    assert check_all((RequiresDtypeIn("float32"),), CallContext(args=(object(),))).ok


def test_dtype_names_are_normalized_across_libraries():
    """``torch.float32`` and ``float32`` should be treated the same."""
    pre = (RequiresDtypeIn("float32"),)
    assert check_all(pre, CallContext(args=(FakeTensor((2,), "torch.float32"),))).ok


def test_contract_strictness_ordering():
    assert Contract.DETERMINISTIC.is_at_least_as_strict_as(Contract.FAST)
    assert not Contract.FAST.is_at_least_as_strict_as(Contract.DETERMINISTIC)
    assert Contract.HIGH_PRECISION.suffix == "hp"
    assert Contract.DETERMINISTIC.suffix == "det"
