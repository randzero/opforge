"""``scale`` — a CuTeDSL backend integration example: ``y = alpha * x``.

A deliberately tiny operator: it makes clear "how a second kind of DSL plugs into the
`Backend` protocol" without getting bogged down in the details of attention / GEMM.

Three things worth noting:

1. **The host function is the operator implementation.** The `@cute.jit` `scale_host` is
   used directly as the decorated function of `@kernel` — the artifact produced by
   `cute.compile(host, *examples)` lives inside `Backend.build`;
2. **Example inputs are a required declaration**, not an optional optimization.
   `cute.compile` needs them to specialize the compilation, which is the same concept as
   Triton's "prewarm shapes";
3. **Compilation and execution have different environment requirements**: compilation
   only needs the toolchain, whereas execution additionally requires a matching driver
   version (`cudaErrorInsufficientDriver` means the driver is older than cuda-python).

**The signature must carry type annotations** (``x: cute.Tensor``,
``alpha: cutlass.Float32``): cross-process reuse goes through the official ``export_to_c``,
which generates C ABI structs from the parameter types. A bare ``torch.Tensor`` parameter
(no annotation) lets ``cute.compile`` pass, but at ``export`` time it reports
"Unsupported argument for c function argument generation". At runtime a ``torch.Tensor``
is still passed directly — the adaptation layer converts it to ``cute.Tensor`` via DLPack.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from opforge import Contract, kernel
from opforge.lang.cutedsl_backend import declare_cutedsl

BLOCK = 256


@cute.kernel
def _scale_kernel(gx, alpha, gy, n, block):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    index = bidx * block + tidx
    if index < n:
        gy[index] = gx[index] * alpha


@cute.jit
def _scale_host(x: cute.Tensor, alpha: cutlass.Float32, y: cute.Tensor):
    n = x.shape[0]
    _scale_kernel(x, alpha, y, n, BLOCK).launch(
        grid=[cute.ceil_div(n, BLOCK), 1, 1],
        block=[BLOCK, 1, 1],
    )


def _scale_ref(x, alpha, y):
    """Ground truth: simply a multiply (writes ``y`` in place, matching the kernel semantics)."""
    import torch

    torch.mul(x, alpha, out=y)
    return y


def _examples():
    """Sample inputs for compile-time specialization — also the AOT declaration of
    "which shapes to compile".

    They must be wrapped as ``cute.Tensor`` with ``from_dlpack`` (matching the host
    function's type annotations); ``alpha`` uses ``cutlass.Float32`` rather than a Python
    ``float``.
    """
    import torch
    from cutlass.cute.runtime import from_dlpack

    return (
        from_dlpack(torch.empty(1024, device="cuda", dtype=torch.float32)),
        cutlass.Float32(2.0),
        from_dlpack(torch.empty(1024, device="cuda", dtype=torch.float32)),
    )


declare_cutedsl(_scale_host, examples=_examples)

scale = kernel(
    name="scale",
    contract=Contract.HIGH_PRECISION,
    reference=_scale_ref,
    backend="cutedsl",
    tags=("elementwise", "gpu"),
)(_scale_host)
