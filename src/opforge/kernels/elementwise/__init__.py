"""Elementwise operators."""

__all__ = ["scale"]

try:  # the CuTeDSL path needs nvidia-cutlass-dsl; the package still works without it
    from .scale import scale
except ImportError:  # pragma: no cover - decided by the environment
    scale = None  # type: ignore[assignment]
