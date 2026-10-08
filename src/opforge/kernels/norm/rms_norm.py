"""``rms_norm`` — a **real DSL operator**.

On CUDA it runs a real Triton kernel; on CPU it falls back to numpy/torch (so the whole
chain can run and be tested without a GPU). Both paths share the same reference and the
same contract.

The differences between the three contract tiers are **real**, not labels:

| Contract | Implementation | Difference |
|---|---|---|
| `DETERMINISTIC` | numpy/torch, **float64 accumulation** | most reproducible (ordered), slowest |
| `HIGH_PRECISION` (default) | Triton, **fp32 accumulation** | the production path |
| `FAST` | Triton, **input-dtype accumulation** | faster, less exact under fp16/bf16; allowed |

Backend selection goes through `backend="auto"`: when a usable Triton is present it picks
`triton` (prewarm-compiled at build time, see `tools/build_aot.py`), otherwise it picks the
`python` pass-through. **The same code runs in both environments.**
"""

from __future__ import annotations

import numpy as np

from opforge import Contract, RequiresDtypeIn, kernel
from opforge.lang.triton_backend import declare_triton

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORTED = True
except ImportError:  # pragma: no cover - decided by the environment
    triton = None
    tl = None
    _TRITON_IMPORTED = False

try:
    import torch as _torch
except ImportError:  # pragma: no cover - decided by the environment
    _torch = None

#: Upper bound on the width of a single tl.arange. For larger n_cols the kernel loops in blocks.
MAX_BLOCK = 4096


def _rms_norm_kernel_defs():
    """Keep the ``@triton.jit`` definition inside a function so the module still imports
    when triton is not installed."""
    if not _TRITON_IMPORTED:
        return None

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        n_cols,
        eps,
        ZERO_CENTERED: tl.constexpr,
        ACC_FP32: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        row_base = row * n_cols

        # First pass: sum of squares. BLOCK may be smaller than n_cols, so accumulate over a
        # loop of blocks (the mask handles the tail).
        #
        # ACC_FP32 decides whether the accumulator is fp32 or the **input dtype**. Note you
        # must not write `tl.zeros(..., dtype=acc.dtype)` — acc itself is fp32 there, so the
        # two branches would collapse into one and the contract would become an empty label.
        if ACC_FP32:
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
        else:
            acc = tl.zeros((BLOCK,), dtype=x_ptr.dtype.element_ty)
        for off in range(0, n_cols, BLOCK):
            idx = off + cols
            mask = idx < n_cols
            v = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0)
            acc += v * v

        variance = tl.sum(acc, axis=0).to(tl.float32) / n_cols
        inv_rms = 1.0 / tl.sqrt(variance + eps)

        # Second pass: normalize and write back
        for off in range(0, n_cols, BLOCK):
            idx = off + cols
            mask = idx < n_cols
            v = tl.load(x_ptr + row_base + idx, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            if ZERO_CENTERED:
                w = w + 1.0
            tl.store(
                out_ptr + row_base + idx,
                (v * inv_rms * w).to(out_ptr.dtype.element_ty),
                mask=mask,
            )

    return _rms_norm_kernel


_rms_norm_kernel = _rms_norm_kernel_defs()


# --------------------------------------------------------------------------- #
# Reference-implementation path (numpy / torch) — works on CPU, and is also the
# DETERMINISTIC implementation
# --------------------------------------------------------------------------- #


def _numpy_rms_norm(x, weight, eps, zero_centered, *, accumulate):
    data = np.asarray(x)
    dtype = np.dtype(accumulate)
    gamma = np.asarray(weight, dtype=dtype)
    if zero_centered:
        gamma = gamma + 1.0
    acc = data.astype(dtype, copy=False)
    variance = np.mean(acc * acc, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + eps)
    return (acc * inv_rms * gamma).astype(data.dtype, copy=False)


def _torch_rms_norm(x, weight, eps, zero_centered, *, accumulate):
    """torch path: composed entirely of differentiable torch ops (for the L3 training side)."""
    acc = x.to(getattr(_torch, accumulate))
    variance = acc.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = _torch.rsqrt(variance + eps)
    gamma = weight.to(getattr(_torch, accumulate))
    if zero_centered:
        gamma = gamma + 1.0
    return (acc * inv_rms * gamma).to(x.dtype)


def _reference_path(x, weight, eps, zero_centered, *, accumulate):
    if _torch is not None and isinstance(x, _torch.Tensor):
        return _torch_rms_norm(x, weight, eps, zero_centered, accumulate=accumulate)
    return _numpy_rms_norm(x, weight, eps, zero_centered, accumulate=accumulate)


# --------------------------------------------------------------------------- #
# Triton kernel launch
# --------------------------------------------------------------------------- #


def _triton_rms_norm(x, weight, eps, zero_centered, *, acc_fp32):
    x = x.contiguous()
    weight = weight.contiguous()
    out = _torch.empty_like(x)
    n_cols = x.shape[-1]
    rows = x.numel() // n_cols
    block = min(triton.next_power_of_2(n_cols), MAX_BLOCK)
    _rms_norm_kernel[(rows,)](
        x,
        weight,
        out,
        n_cols,
        eps,
        ZERO_CENTERED=zero_centered,
        ACC_FP32=acc_fp32,
        BLOCK=block,
        num_warps=max(1, block // 128),
    )
    return out


def _dispatch(x, weight, eps, zero_centered, *, accumulate, acc_fp32):
    """Use the kernel when CUDA + Triton are available; otherwise fall back to numpy/torch."""
    if (
        _TRITON_IMPORTED
        and _torch is not None
        and isinstance(x, _torch.Tensor)
        and x.is_cuda
    ):
        return _triton_rms_norm(x, weight, eps, zero_centered, acc_fp32=acc_fp32)
    return _reference_path(x, weight, eps, zero_centered, accumulate=accumulate)


# --------------------------------------------------------------------------- #
# Reference implementation (CI numerical baseline + training-side fallback path)
# --------------------------------------------------------------------------- #


def rms_norm_ref(x, weight, eps=1e-6, *, zero_centered=False):
    """Ground truth: torch tensors go through torch, everything else through float64 numpy."""
    return _reference_path(x, weight, eps, zero_centered, accumulate="float64")


# --------------------------------------------------------------------------- #
# Implementations of the three contract tiers
# --------------------------------------------------------------------------- #


def rms_norm_det(x, weight, eps=1e-6, *, zero_centered=False):
    """``DETERMINISTIC``: float64 ordered accumulation — trades speed for the highest
    reproducibility."""
    return _reference_path(x, weight, eps, zero_centered, accumulate="float64")


def rms_norm_fast(x, weight, eps=1e-6, *, zero_centered=False):
    """``FAST``: accumulate in the input dtype (faster and less exact under fp16/bf16)."""
    return _dispatch(
        x, weight, eps, zero_centered, accumulate="float32", acc_fp32=False
    )


def _rms_norm_impl(x, weight, eps=1e-6, *, zero_centered=False):
    """Default implementation (``HIGH_PRECISION``): Triton fp32 accumulation on CUDA."""
    return _dispatch(
        x, weight, eps, zero_centered, accumulate="float64", acc_fp32=True
    )


#: AOT prewarm shapes — (rows, hidden, dtype). Cover the typical shapes on both the
#: training and the decode sides.
WARM_SHAPES: tuple[tuple[int, int, str], ...] = (
    (4096, 4096, "bfloat16"),
    (128, 4096, "bfloat16"),
    (1, 4096, "bfloat16"),
    (128, 4096, "float32"),
)


def _prewarm_shapes():
    """Sample inputs for AOT build time — used only for compilation, not for computation."""
    if not (_TRITON_IMPORTED and _torch is not None):
        return []
    entries = []
    for rows, hidden, dtype_name in WARM_SHAPES:
        dtype = getattr(_torch, dtype_name)
        x = _torch.empty((rows, hidden), device="cuda", dtype=dtype)
        weight = _torch.empty(hidden, device="cuda", dtype=dtype)
        out = _torch.empty_like(x)
        block = min(triton.next_power_of_2(hidden), MAX_BLOCK)
        for acc_fp32 in (True, False):  # the HIGH_PRECISION and FAST variants
            entries.append(
                (
                    _rms_norm_kernel,
                    (rows,),
                    (x, weight, out, hidden, 1e-6, False, acc_fp32, block),
                )
            )
    return entries


if _rms_norm_kernel is not None:
    declare_triton(_rms_norm_impl, _rms_norm_kernel, prewarm=_prewarm_shapes)

rms_norm = kernel(
    name="rms_norm",
    contract=Contract.HIGH_PRECISION,
    variants={
        Contract.DETERMINISTIC: rms_norm_det,
        Contract.FAST: rms_norm_fast,
    },
    reference=rms_norm_ref,
    preconditions=(RequiresDtypeIn("float32", "float16", "bfloat16"),),
    backward="auto",
    supports_dynamic_shape=True,
    tags=("norm",),
)(_rms_norm_impl)
