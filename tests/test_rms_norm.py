"""Reference op ``rms_norm``: precision, contract tiers, dispatch by input type."""

from __future__ import annotations

import unittest

import numpy as np

from opforge import Contract, Unavailable
from opforge.kernels.norm import rms_norm
from opforge.kernels.norm.rms_norm import rms_norm_ref


def _inputs(shape=(8, 128), dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(shape).astype(dtype),
        rng.standard_normal(shape[-1]).astype(dtype),
    )


def test_high_precision_tiers_match_reference_exactly():
    x, w = _inputs()
    for contract in (Contract.DETERMINISTIC, Contract.HIGH_PRECISION):
        got = rms_norm(x, w, contract=contract)
        assert np.array_equal(got, rms_norm_ref(x, w)), contract


def test_fast_tier_is_close_but_not_required_to_be_identical():
    x, w = _inputs()
    got = rms_norm(x, w, contract=Contract.FAST)
    assert np.allclose(got, rms_norm_ref(x, w), rtol=1e-5, atol=1e-5)


def test_zero_centered_matches_reference():
    x, w = _inputs()
    got = rms_norm(x, w, zero_centered=True)
    assert np.array_equal(got, rms_norm_ref(x, w, zero_centered=True))


def test_matches_the_manual_formula():
    x, w = _inputs(shape=(2, 4))
    got = rms_norm(x, w, eps=1e-6)
    acc = x.astype(np.float64)
    variance = np.mean(acc * acc, axis=-1, keepdims=True)
    expected = (acc / np.sqrt(variance + 1e-6) * w.astype(np.float64)).astype(x.dtype)
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_output_dtype_follows_input():
    for dtype in (np.float32, np.float16):
        x, w = _inputs(dtype=dtype)
        assert rms_norm(x, w).dtype == dtype


def test_eps_is_honoured():
    x, w = _inputs(shape=(2, 8))
    assert not np.array_equal(rms_norm(x, w, eps=1e-6), rms_norm(x, w, eps=1e-2))


def test_int_input_is_rejected_by_precondition():
    x, w = _inputs()
    result = rms_norm(x.astype(np.int8), w)
    assert isinstance(result, Unavailable)
    assert "requires_dtype_in" in result.reason


def test_reference_uses_torch_when_given_a_torch_tensor():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise unittest.SkipTest(f"torch unavailable: {exc}") from exc

    x, w = _inputs()
    torch_out = rms_norm_ref(torch.from_numpy(x), torch.from_numpy(w))
    assert isinstance(torch_out, torch.Tensor)
    assert np.allclose(torch_out.numpy(), rms_norm_ref(x, w), atol=1e-6)


def test_default_impl_dispatches_to_torch_path():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise unittest.SkipTest(f"torch unavailable: {exc}") from exc

    x, w = _inputs()
    out = rms_norm(torch.from_numpy(x), torch.from_numpy(w))
    assert isinstance(out, torch.Tensor)
