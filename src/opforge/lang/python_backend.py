"""Pure-Python reference backend — no compilation.

It exists for two reasons:

1. **Make the whole chain executable in a CPU-only environment**: contract parsing →
   preconditions → cache → launch all work end to end, except that the "artifact" is
   the Python callable itself;
2. **A fallback path**: when the L3 adapter layer falls back in place onto the
   operator's ``reference``, it goes through this backend.

The price is that it is **not persistable** (there is no compiled artifact to write to
disk), so for it the cache layer degrades to a purely in-process cache — hit/miss,
seal-compilation and the other semantics stay exactly the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from ..contract import Contract
from .backend import register_backend


class PythonArtifact:
    """Wrap a Python callable as an artifact."""

    __slots__ = ("_impl", "spec_name", "contract", "calls")

    def __init__(self, impl: Callable, spec_name: str, contract: Contract) -> None:
        self._impl = impl
        self.spec_name = spec_name
        self.contract = contract
        self.calls = 0

    def run(self, args: tuple, kwargs: Mapping[str, Any]) -> Any:
        self.calls += 1
        return self._impl(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<PythonArtifact {self.spec_name} contract={self.contract.name}>"


class PythonBackend:
    """Pass-through backend: ``build`` returns the callable, ``load`` is always ``None``."""

    name = "python"

    def supports(self, spec: Any, impl: Callable) -> bool:
        return True

    def build(self, spec: Any, impl: Callable, contract: Contract) -> PythonArtifact:
        return PythonArtifact(impl, spec.name, contract)

    def load(self, spec: Any, key: Any, path: Path) -> Any | None:
        return None


register_backend(PythonBackend())
