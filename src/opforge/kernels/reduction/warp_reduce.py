"""``warp_reduce_sum`` — a C++/CUDA escape-hatch integration example.

## Why it must be **class A** (missing primitive)

This operator does not care about "adding a row up", but about **the shape of the
reduction tree itself**: it uses ``__shfl_down_sync`` to manually unroll a fixed
``16 → 8 → 4 → 2 → 1`` butterfly tree, so that **the reduction order becomes part of the
contract** — reproducible across runs and across GPUs.

Triton's ``tl.sum`` uses its own reduction implementation; **the order is an
implementation detail that you cannot specify**. Turning "the reduction order" into a
contract you can depend on means dropping down to CUDA — that is the literal meaning of
``EscalationReason.MISSING_PRIMITIVE``.

Counterexamples (these two should not take the escape hatch):
* merely "wanting a faster sum" → ``PERF_GAP``, and it must come with benchmark evidence;
* merely "wanting to process N tensors at once" → ``CALL_SHAPE_MISMATCH``, which is
  rejected outright.

## Note

Execution is taken over by the C++ backend, so the decorated function's **body is never
called**; the parameters are kept only so that ``@kernel`` can infer the parameter list
and schema. The first compile takes about a minute, after which it hits torch's extension
cache (0 seconds).
"""

from __future__ import annotations

from opforge import Contract, kernel
from opforge.lang.cpp_backend import declare_cpp
from opforge.lang.escalation import EscalationReason, cpp_kernel

_MODULE = "opforge_warp_reduce_sum"

#: The fixed shuffle-down reduction tree: 16 → 8 → 4 → 2 → 1.
#: **The order is part of the contract, do not change it.**
_CUDA_SOURCES = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum_exact(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

__global__ void warp_reduce_sum_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int rows, int cols) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= rows) return;

    const float* row = x + (size_t)warp_id * cols;

    // Each lane first accumulates its own share with stride = 32 (fixed order),
    // then does the butterfly reduction.
    float acc = 0.0f;
    for (int c = lane; c < cols; c += 32) acc += row[c];
    acc = warp_reduce_sum_exact(acc);
    if (lane == 0) out[warp_id] = acc;
}

torch::Tensor warp_reduce_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "warp_reduce_sum requires a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "warp_reduce_sum requires float32");
    TORCH_CHECK(x.dim() == 2, "warp_reduce_sum expects a (rows, cols) tensor");
    auto xc = x.contiguous();
    const int rows = static_cast<int>(xc.size(0));
    const int cols = static_cast<int>(xc.size(1));
    auto out = torch::empty({rows}, xc.options());

    const int threads = 256;                     // 8 warps / block
    const int warps_per_block = threads / 32;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    warp_reduce_sum_kernel<<<blocks, threads>>>(
        xc.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
    return out;
}
"""

#: Host-side declaration — pybind binding generation needs to see the function signature
#: (across translation units).
_CPP_SOURCES = "torch::Tensor warp_reduce_sum(torch::Tensor x);"


def warp_reduce_sum_ref(x):
    """Ground truth: float64 sum.

    Note that it is **not bitwise-identical** to the kernel — the two use different
    reduction orders. What this operator offers is "a fixed, reproducible order", not
    "agreement with torch"; the accuracy test compares within a tolerance.
    """
    import torch

    return x.double().sum(dim=-1).to(torch.float32)


@cpp_kernel(
    reason=EscalationReason.MISSING_PRIMITIVE,
    primitives=["__shfl_down_sync", "explicit reduction-tree order"],
)
def _warp_reduce_sum_impl(x):
    """Execution is taken over by the C++ backend; this function body is never called."""
    raise RuntimeError(
        "warp_reduce_sum must be executed through the cpp backend "
        "(declare_cpp + backend='cpp')"
    )


declare_cpp(
    _warp_reduce_sum_impl,
    module=_MODULE,
    cuda_sources=_CUDA_SOURCES,
    cpp_sources=_CPP_SOURCES,
    functions=["warp_reduce_sum"],
)

warp_reduce_sum = kernel(
    name="warp_reduce_sum",
    contract=Contract.DETERMINISTIC,
    reference=warp_reduce_sum_ref,
    backend="cpp",
    tags=("reduction", "gpu"),
)(_warp_reduce_sum_impl)
