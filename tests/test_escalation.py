"""L0 escape-hatch registry (invariant I8): only "missing primitive" and
"evidence-backed performance gap" are allowed."""

from __future__ import annotations

from opforge import EscalationReason, cpp_kernel, escalations
from opforge.errors import EscalationNotAllowed


def test_missing_primitive_is_allowed_and_recorded():
    @cpp_kernel(reason=EscalationReason.MISSING_PRIMITIVE, primitives=["named_barrier", "tma"])
    def probe_primitive():
        return None

    record = escalations()["probe_primitive"]
    assert record.reason is EscalationReason.MISSING_PRIMITIVE
    assert record.primitives == ("named_barrier", "tma")
    assert record.benchmark is None


def test_record_is_attached_to_the_function():
    @cpp_kernel(reason=EscalationReason.MISSING_PRIMITIVE, primitives=["tma"])
    def probe_attached():
        return None

    assert probe_attached.__opforge_escalation__.primitives == ("tma",)


def test_call_shape_mismatch_is_rejected_outright():
    """Class B problems must be solved within the DSL and must not drag the user into
    the nvcc toolchain."""
    try:

        @cpp_kernel(reason=EscalationReason.CALL_SHAPE_MISMATCH)
        def probe_shape():
            return None

    except EscalationNotAllowed as exc:
        assert "dtype bucketing" in str(exc)
        return
    raise AssertionError("CALL_SHAPE_MISMATCH was accepted after all")


def test_perf_gap_without_evidence_is_rejected():
    try:

        @cpp_kernel(reason=EscalationReason.PERF_GAP)
        def probe_perf():
            return None

    except EscalationNotAllowed:
        return
    raise AssertionError("PERF_GAP passed without accompanying benchmark evidence")


def test_perf_gap_with_evidence_is_allowed():
    @cpp_kernel(reason=EscalationReason.PERF_GAP, benchmark="bench/kernels/xyz.json")
    def probe_perf_ok():
        return None

    assert escalations()["probe_perf_ok"].benchmark == "bench/kernels/xyz.json"


def test_reason_must_be_an_enum_member():
    try:
        cpp_kernel(reason="missing_primitive")
    except TypeError:
        return
    raise AssertionError("a non-enum reason should raise TypeError")


def test_name_can_override_the_registry_key():
    @cpp_kernel(reason=EscalationReason.MISSING_PRIMITIVE, name="custom.name")
    def probe_named():
        return None

    assert "custom.name" in escalations()
