"""Reduction operators."""

__all__ = ["warp_reduce_sum"]

try:  # the C++ escape hatch needs torch + a CUDA compiler
    from .warp_reduce import warp_reduce_sum
except ImportError:  # pragma: no cover - decided by the environment
    warp_reduce_sum = None  # type: ignore[assignment]
