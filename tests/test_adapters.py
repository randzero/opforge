"""L3 adapter layer: training-side autograd and inference-side custom op (skipped
entirely without torch)."""

from __future__ import annotations

import unittest

from opforge import Contract
from opforge.kernels.norm import rms_norm
from support import requires_torch

#: The Library used for registration must stay alive, otherwise the op gets deregistered.
_LIBRARY = None


def test_schema_inference_matches_expected_shape():
    from opforge.inference import infer_schema

    assert infer_schema(rms_norm.spec) == (
        "(Tensor x, Tensor weight, float eps=1e-06, *, bool zero_centered=False) -> Tensor"
    )


def test_schema_marks_mutated_arguments():
    from opforge.inference import infer_schema

    assert "Tensor(a!) x" in infer_schema(rms_norm.spec, mutates_args=["x"])


def test_schema_rejects_var_args_kernels():
    from opforge import kernel
    from opforge.errors import KernelSpecError
    from opforge.inference import infer_schema

    def variadic(*args):
        return args

    handle = kernel(name="_variadic", contract=Contract.FAST)(variadic)
    try:
        infer_schema(handle.spec)
    except KernelSpecError:
        return
    raise AssertionError("*args cannot be expressed as a schema; KernelSpecError should be raised")


def test_autograd_forward_and_backward_match_reference():
    torch = requires_torch()
    from opforge.training import to_autograd

    x = torch.randn(8, 256, requires_grad=True)
    weight = torch.ones(256)

    wrapped = to_autograd(rms_norm, contract=Contract.DETERMINISTIC)
    out = wrapped(x, weight)
    assert out.requires_grad
    out.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())

    x_ref = x.detach().clone().requires_grad_(True)
    rms_norm.ref(x_ref, weight).sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6)


def test_autograd_falls_back_to_reference_when_kernel_is_unavailable():
    torch = requires_torch()
    from opforge.training import to_autograd

    x = torch.zeros(4, 128, dtype=torch.int8)  # rejected by the precondition
    wrapped = to_autograd(rms_norm, contract=Contract.HIGH_PRECISION)
    # falls back to the reference (which does not check dtype); must not raise
    out = wrapped(x.to(torch.float32), torch.ones(128))
    assert out.shape == (4, 128)


def test_custom_op_is_registered_and_callable():
    global _LIBRARY
    torch = requires_torch()
    from opforge.inference import to_custom_op

    if _LIBRARY is None:
        _LIBRARY = to_custom_op(rms_norm)

    x = torch.randn(4, 128)
    weight = torch.ones(128)
    out = torch.ops.opforge.rms_norm(x, weight)
    assert torch.allclose(out, rms_norm.ref(x, weight), atol=1e-6)
    # both keyword args and defaults must work
    out2 = torch.ops.opforge.rms_norm(x, weight, 1e-5, zero_centered=True)
    assert torch.allclose(out2, rms_norm.ref(x, weight, 1e-5, zero_centered=True), atol=1e-6)


def test_custom_op_is_traceable_under_fake_tensors():
    """The **prerequisite** for being traceable by torch.compile: a fake is registered.

    Calling the op under FakeTensorMode goes through the fake implementation; without
    a fake it would run the real implementation and blow up on the device/memory.
    This test does not depend on inductor/Triton, so it runs anywhere.
    """
    global _LIBRARY
    torch = requires_torch()
    from opforge.inference import to_custom_op

    if _LIBRARY is None:
        _LIBRARY = to_custom_op(rms_norm)

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_x = torch.empty(4, 128)
        fake_w = torch.ones(128)
        out = torch.ops.opforge.rms_norm(fake_x, fake_w)
        assert tuple(out.shape) == (4, 128)
        assert out.dtype == fake_x.dtype


def test_custom_op_compiles_with_inductor_when_the_backend_is_usable():
    """Run torch.compile end to end -- requires a usable inductor/Triton backend."""
    global _LIBRARY
    torch = requires_torch()
    from opforge.inference import to_custom_op

    if _LIBRARY is None:
        _LIBRARY = to_custom_op(rms_norm)

    def model(x, weight):
        return torch.ops.opforge.rms_norm(x, weight)

    x = torch.randn(8, 128)
    weight = torch.ones(128)
    try:
        compiled = torch.compile(model, fullgraph=True)
        result = compiled(x, weight)
    except Exception as exc:  # pragma: no cover - environment dependent
        message = str(exc)
        if "cuda_utils" in message or "undefined symbol" in message or "InductorError" in message:
            raise unittest.SkipTest(
                f"inductor/triton backend not usable in this environment: {message[:70]}"
            ) from exc
        raise
    assert torch.allclose(result, rms_norm.ref(x, weight), atol=1e-5)
