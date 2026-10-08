"""The GPU path of ``rms_norm`` (a real Triton kernel). Skipped entirely without a
GPU / when Triton is unavailable."""

from __future__ import annotations

import unittest

from opforge import Contract, KernelCache, fallback_counts, reset_fallback_counts
from opforge.kernels.norm import rms_norm
from support import requires_torch


def _requires_gpu_triton():
    torch = requires_torch()
    from opforge.lang.triton_backend import triton_usable

    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA device")
    if not triton_usable():
        raise unittest.SkipTest(
            "triton driver unavailable (see tools/gpu_env.sh: LD_PRELOAD the real libcuda)"
        )
    return torch


def _inputs(torch, shape=(64, 512), dtype=None, seed=0):
    dtype = dtype or torch.float32
    torch.manual_seed(seed)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    return x, w


def test_triton_backend_is_selected_on_gpu():
    _requires_gpu_triton()
    from opforge.lang import select_backend

    assert select_backend(rms_norm.spec).name == "triton"
    assert rms_norm.spec.backend == "auto", "the same code must also run in a no-Triton environment"


def test_build_precompiles_at_build_time():
    """The AOT half: compilation happens at **build time**, and the artifact carries the
    prewarm results."""
    _requires_gpu_triton()
    from pathlib import Path
    import tempfile

    from opforge.kernel import KernelHandle

    handle = KernelHandle(
        rms_norm.spec, cache=KernelCache(Path(tempfile.mkdtemp()), persist=False)
    )
    artifact = handle.build()
    assert type(artifact).__name__ == "TritonArtifact"
    assert artifact.prewarm_results, "prewarm compilation should complete at build time"
    assert all(r == "ok" for r in artifact.prewarm_results), artifact.prewarm_results


def test_matches_reference_within_fp32_tolerance():
    torch = _requires_gpu_triton()
    x, w = _inputs(torch)
    out = rms_norm(x, w)
    assert out.shape == x.shape
    assert torch.allclose(out, rms_norm.reference(x, w), rtol=1e-3, atol=1e-3)


def test_zero_centered_path():
    torch = _requires_gpu_triton()
    x, w = _inputs(torch, shape=(8, 256))
    out = rms_norm(x, w, zero_centered=True)
    assert torch.allclose(out, rms_norm.reference(x, w, zero_centered=True), rtol=1e-3, atol=1e-3)


def test_fp16_and_bfloat16_dtypes():
    torch = _requires_gpu_triton()
    for dtype in (torch.float16, torch.bfloat16):
        x, w = _inputs(torch, shape=(16, 128), dtype=dtype)
        out = rms_norm(x, w)
        assert out.dtype == dtype
        assert torch.allclose(
            out.float(), rms_norm.reference(x, w).float(), rtol=1e-2, atol=1e-2
        )


def test_non_power_of_two_hidden_size_is_masked_correctly():
    torch = _requires_gpu_triton()
    x, w = _inputs(torch, shape=(32, 300))  # 300 is not a power of two
    assert torch.allclose(rms_norm(x, w), rms_norm.reference(x, w), rtol=1e-3, atol=1e-3)


def test_hidden_larger_than_block_loops_correctly():
    """When n_cols exceeds the width of a single tl.arange, the in-kernel loop must be correct."""
    torch = _requires_gpu_triton()
    x, w = _inputs(torch, shape=(4, 8192))  # > MAX_BLOCK(4096)
    assert torch.allclose(rms_norm(x, w), rms_norm.reference(x, w), rtol=1e-3, atol=1e-3)


def test_contracts_are_real_not_labels():
    """FAST accumulates in the input dtype on bf16 -- it must really differ numerically
    from HIGH_PRECISION."""
    torch = _requires_gpu_triton()
    x, w = _inputs(torch, shape=(64, 512), dtype=torch.bfloat16)
    hp = rms_norm(x, w, contract=Contract.HIGH_PRECISION)
    fast = rms_norm(x, w, contract=Contract.FAST)
    assert not torch.equal(hp, fast), (
        "FAST and HIGH_PRECISION should not produce exactly the same bits"
    )

    truth = rms_norm.reference(x, w).float()
    hp_err = (hp.float() - truth).abs().max().item()
    fast_err = (fast.float() - truth).abs().max().item()
    assert fast_err >= hp_err, f"FAST must be no more precise: hp={hp_err:.3e} fast={fast_err:.3e}"


def test_deterministic_tier_falls_back_to_the_reference_path():
    """DETERMINISTIC is the fp64 path and does not go through the Triton kernel."""
    torch = _requires_gpu_triton()
    x, w = _inputs(torch, shape=(8, 256))
    det = rms_norm(x, w, contract=Contract.DETERMINISTIC)
    assert torch.allclose(det, rms_norm.reference(x, w), rtol=1e-5, atol=1e-5)


def test_no_fallback_when_all_three_tiers_are_implemented():
    """rms_norm has all three tiers -- any request should hit exactly, so the fallback
    count must be zero."""
    torch = _requires_gpu_triton()
    reset_fallback_counts()
    x, w = _inputs(torch, shape=(4, 128))
    for contract in Contract:
        rms_norm(x, w, contract=contract)
    assert fallback_counts() == {}, fallback_counts()


def test_runs_under_cuda_graph_capture():
    """The Triton kernel should be capturable by CUDA Graph -- a key prerequisite on the
    inference side."""
    torch = _requires_gpu_triton()
    x, w = _inputs(torch, shape=(8, 256))
    rms_norm(x, w)  # prewarm, to avoid compiling during capture
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rms_norm(x, w)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, rms_norm.reference(x, w), rtol=1e-3, atol=1e-3)
