"""Test helpers (the filename starts with ``_``, so the collector ignores it)."""

from __future__ import annotations

import unittest


class FakeTensor:
    """Duck-typed fake tensor: only provides ``shape`` / ``dtype``.

    Used to test preconditions and launch signatures -- precisely so that this
    logic does **not depend on torch**.
    """

    def __init__(self, shape: tuple[int, ...], dtype: str = "float32") -> None:
        self.shape = tuple(shape)
        self.dtype = dtype

    def __repr__(self) -> str:  # pragma: no cover
        return f"FakeTensor{self.shape}:{self.dtype}"


class FakeArtifact:
    """A persistable fake artifact: ``export`` writes a file, ``run`` counts calls."""

    def __init__(self, tag: str = "a") -> None:
        self.tag = tag
        self.calls = 0

    def export(self, path, key) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"artifact:{self.tag}:{key.export_symbol}")

    def run(self, args, kwargs):
        self.calls += 1
        return (self.tag, args, dict(kwargs))


def requires_torch():
    """Get torch; raise ``SkipTest`` when it is unavailable or broken (both pytest
    and the bundled runner honour it)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"torch unavailable: {exc}") from exc
    return torch
