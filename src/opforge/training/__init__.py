"""L3 · training-side adaptation.

Wrap opforge operators as ``torch.autograd.Function``:

* **forward** goes through the kernel (at the requested contract level);
* **backward** is computed with **the reference's autograd** (a skeleton-stage
  approach); once a hand-written backward is ready it can be swapped in place, with no
  interface change;
* when a precondition is unsatisfied (returns ``Unavailable``) it **falls back to the
  reference** -- training can always run.

This layer **must not** be depended upon in reverse by L0-L2 (invariant I2): the
`opforge` core does not import this module.
"""

from __future__ import annotations

from typing import Any, Callable

from ..contract import Contract
from ..kernel import KernelHandle
from ..result import Unavailable


def _require_torch():  # noqa: ANN202 - the return value is the torch module
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "opforge.training requires torch; install `opforge[training]`"
        ) from exc
    return torch


def to_autograd(op: KernelHandle, *, contract: Contract | None = None) -> Callable:
    """Return a differentiable wrapper of ``op``.

    Args:
        op: an operator handle.
        contract: the contract level passed to the kernel; the operator's default
            contract when ``None``. On the training side you should usually pass
            ``Contract.DETERMINISTIC`` or ``HIGH_PRECISION`` explicitly.

    Returns:
        A callable object whose forward goes through the kernel / falls back to the
        reference, and whose backward is provided by the reference's autograd.

    Note:
        Requires the operator's implementation to be **differentiable** on torch
        tensors (e.g. dispatching to torch ops based on input type). If a pure kernel
        backend only returns numpy, backward is unavailable -- then a hand-written
        ``backward`` should be provided.
    """
    torch = _require_torch()
    requested = op.spec.contract if contract is None else contract

    def _forward(args: tuple, kwargs: dict) -> Any:
        out = op(*args, contract=requested, **kwargs)
        if isinstance(out, Unavailable):
            return op.ref(*args, **kwargs)
        return out

    class _KernelFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *flat):  # flat = (args, kwargs, *tensors)
            args, kwargs = flat[0], flat[1]
            tensors = flat[2:]
            ctx.args = args
            ctx.kwargs = kwargs
            ctx.tensor_slots = [
                i for i, a in enumerate(args) if isinstance(a, torch.Tensor)
            ]
            ctx.save_for_backward(*tensors)
            return _forward(args, kwargs)

        @staticmethod
        def backward(ctx, grad_out):  # noqa: ANN001
            args, kwargs = ctx.args, ctx.kwargs
            with torch.enable_grad():
                ref_args = list(args)
                leaves = []
                for slot, tensor in zip(ctx.tensor_slots, ctx.saved_tensors):
                    leaf = tensor.detach().requires_grad_(True)
                    ref_args[slot] = leaf
                    leaves.append(leaf)
                out = op.ref(*ref_args, **kwargs)
                grads = torch.autograd.grad(
                    out, leaves, grad_outputs=grad_out, allow_unused=True
                )
            return (None, None) + tuple(grads)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        return _KernelFunction.apply(tuple(args), dict(kwargs), *tensors)

    wrapped.__name__ = op.name  # type: ignore[attr-defined]
    wrapped.__doc__ = (
        f"Differentiable wrapper of `{op.name}` (contract={requested.name}); "
        f"backward uses the reference implementation."
    )
    return wrapped
