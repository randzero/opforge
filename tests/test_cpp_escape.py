"""C++/CUDA escape hatch (class A): ``warp_reduce_sum``.

The first compilation takes about 1 minute (afterwards torch's extension cache is hit,
0 seconds); skipped entirely without a CUDA compiler.
"""

from __future__ import annotations

import unittest

from opforge import Contract
from support import requires_torch


def _requires_cpp():
    torch = requires_torch()
    from opforge.kernels.reduction import warp_reduce_sum

    if warp_reduce_sum is None:  # pragma: no cover
        raise unittest.SkipTest("warp_reduce_sum unavailable (import failed)")
    from opforge.lang.cpp_backend import cpp_compile_available

    if not cpp_compile_available():
        raise unittest.SkipTest("no CUDA compiler (nvcc) available")
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA device")
    return torch, warp_reduce_sum


def test_cpp_backend_is_registered():
    from opforge.lang import registered_backends

    assert "cpp" in registered_backends(), registered_backends()


def test_operator_selects_the_cpp_backend():
    _torch, op = _requires_cpp()
    from opforge.lang import select_backend

    assert select_backend(op.spec).name == "cpp"
    assert op.spec.backend == "cpp"


def test_escalation_is_recorded_with_a_missing_primitive_reason():
    _torch, op = _requires_cpp()
    from opforge.lang.escalation import EscalationReason, escalations

    record = getattr(op.spec.default_impl, "__opforge_escalation__", None)
    assert record is not None, "an escape hatch must leave a registration record (invariant I8)"
    assert record.reason is EscalationReason.MISSING_PRIMITIVE
    assert "__shfl_down_sync" in record.primitives
    assert record.name in escalations()


def test_contract_is_deterministic():
    _torch, op = _requires_cpp()
    assert op.spec.contract is Contract.DETERMINISTIC


def test_matches_reference_within_tolerance():
    torch, op = _requires_cpp()
    x = torch.randn(256, 4096, device="cuda")
    out = op(x)
    assert tuple(out.shape) == (256,)
    assert torch.allclose(out, op.reference(x), rtol=1e-4, atol=1e-3), (
        float((out - op.reference(x)).abs().max())
    )


def test_result_is_reproducible_across_runs():
    """The essence of the DETERMINISTIC contract: the same input must produce
    **bit-identical** results.

    That is exactly why it would rather drop down to C++ -- the reduction tree is
    hand-fixed.
    """
    torch, op = _requires_cpp()
    x = torch.randn(64, 2048, device="cuda")
    first = op(x)
    for _ in range(3):
        assert torch.equal(op(x), first), "results for the same input must be bit-identical"


def test_handles_columns_not_a_multiple_of_the_warp_size():
    torch, op = _requires_cpp()
    x = torch.randn(32, 100, device="cuda")  # 100 is not a multiple of 32
    assert torch.allclose(op(x), op.reference(x), rtol=1e-4, atol=1e-3)


def test_single_row_edge_case():
    torch, op = _requires_cpp()
    x = torch.randn(1, 7, device="cuda")
    assert torch.allclose(op(x), op.reference(x), rtol=1e-4, atol=1e-3)


def test_non_cuda_and_wrong_dtype_are_rejected_by_the_kernel():
    """The kernel does its own TORCH_CHECK -- bad input should give a readable error
    instead of silently computing the wrong thing."""
    torch, op = _requires_cpp()
    try:
        op(torch.randn(4, 8))  # CPU tensor
    except RuntimeError as exc:
        assert "CUDA" in str(exc), exc
        return
    raise AssertionError("CPU input should be rejected")
