"""Triton backend.

Its difference from the `python` reference backend is that **the build stage eagerly
prewarms compilation**: it makes compilation happen at "build time" rather than at
"first request" — which is exactly the shape this project's AOT stance takes on the
Triton side.

Two cache layers, each with its own job:

* **Triton's own on-disk cache** (`TRITON_CACHE_DIR`) handles cross-process
  serialization — switching processes only needs a "load";
* `opforge.launch.Launcher` handles the in-process "signature → launch entry" cache —
  saving the binding cost of every launch.

**Prewarming needs to know which shapes to compile**, so an operator declares "prewarm
with these sample shapes" via `__opforge_triton_prewarm__`. Without that declaration it
degrades to compiling on the first call.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .backend import BACKEND_MARKER_ATTR, register_backend

_PREWARM_ATTR = "__opforge_triton_prewarm__"
_KERNELS_ATTR = "__opforge_triton_kernels__"


def triton_usable() -> bool:
    """Whether Triton can load its CUDA driver extension.

    In some containers `libcuda.so.1` is loaded but not in the global symbol scope,
    and this returns ``False``; for the fix see `tools/gpu_env.sh`.
    """
    try:
        from triton.backends.nvidia.driver import CudaUtils

        CudaUtils()
    except Exception:
        return False
    return True


class TritonArtifact:
    """Triton operator artifact: a thin wrapper over the host implementation plus a
    record of prewarm results."""

    __slots__ = ("_impl", "spec_name", "contract", "prewarm_results", "calls")

    def __init__(
        self,
        impl: Callable,
        spec_name: str,
        contract: Any,
        prewarm_results: tuple[str, ...] = (),
    ) -> None:
        self._impl = impl
        self.spec_name = spec_name
        self.contract = contract
        self.prewarm_results = prewarm_results
        self.calls = 0

    def run(self, args: tuple, kwargs: dict) -> Any:
        self.calls += 1
        return self._impl(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TritonArtifact {self.spec_name} prewarm={self.prewarm_results}>"


class TritonBackend:
    """Compile/prewarm operators that declare a Triton kernel into an artifact."""

    name = "triton"

    def supports(self, spec: Any, impl: Callable) -> bool:
        return bool(getattr(impl, _KERNELS_ATTR, ())) and triton_usable()

    def build(self, spec: Any, impl: Callable, contract: Any) -> TritonArtifact:
        return TritonArtifact(
            impl,
            spec.name,
            contract,
            prewarm_results=self._prewarm(impl),
        )

    def load(self, spec: Any, key: Any, path: Any) -> Any | None:
        """Triton's persistence is handled by its own on-disk cache; not duplicated here."""
        return None

    @staticmethod
    def _prewarm(impl: Callable) -> tuple[str, ...]:
        """Trigger compilation for the sample shapes declared by the operator.

        A failed prewarm is **not fatal**: record the reason and defer the actual
        compilation to the first call; this way "one shape cannot be prewarmed" does
        not make the whole operator unusable.
        """
        prewarm = getattr(impl, _PREWARM_ATTR, None)
        if prewarm is None:
            return ()
        results: list[str] = []
        for entry in prewarm():
            kernel, grid, args = _unpack_prewarm_entry(entry)
            try:
                kernel.warmup(*args, grid=grid)
            except Exception as exc:  # pragma: no cover - environment-dependent
                results.append(f"deferred:{type(exc).__name__}")
            else:
                results.append("ok")
        return tuple(results)


def _unpack_prewarm_entry(entry: Sequence[Any]) -> tuple[Any, Any, tuple]:
    kernel, grid, *rest = entry
    args = tuple(rest[0]) if rest and isinstance(rest[0], (list, tuple)) else tuple(rest)
    return kernel, grid, args


def declare_triton(impl: Callable, kernel: Any, *, prewarm: Callable | None = None) -> Callable:
    """Mark a host function as a Triton implementation.

    Args:
        impl: the host function (the one actually called).
        kernel: the ``@triton.jit`` kernel it uses.
        prewarm: optional provider returning ``[(kernel, grid, args), ...]`` sample
            shapes.
    """
    setattr(impl, _KERNELS_ATTR, (kernel,))
    setattr(impl, BACKEND_MARKER_ATTR, "triton")
    if prewarm is not None:
        setattr(impl, _PREWARM_ATTR, prewarm)
    return impl


register_backend(TritonBackend())
