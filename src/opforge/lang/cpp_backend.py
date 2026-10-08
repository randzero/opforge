"""C++/PTX escape-hatch backend (category A).

**Use it only when you should**: the reason must be registered via
:mod:`opforge.lang.escalation`, and only the ``MISSING_PRIMITIVE`` / ``PERF_GAP``
categories are accepted — ``CALL_SHAPE_MISMATCH`` is explicitly rejected (that kind of
problem should be solved inside the DSL with an idiom).

Compilation goes through ``torch.utils.cpp_extension.load_inline``: the artifact is
cached in torch's extensions directory, and **the second load takes 0 seconds** — the
same idea as Triton's on-disk cache (see `tools/build_aot.py`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .backend import BACKEND_MARKER_ATTR, register_backend

_MARKER_ATTR = "__opforge_cpp__"


@dataclass(frozen=True)
class CppSource:
    """A C++/CUDA escape-hatch implementation waiting to be compiled."""

    module: str
    """Extension module name — must be stable and unique; it is the key of the
    compile cache."""
    cuda_sources: str
    """CUDA source."""
    cpp_sources: str = ""
    """Host-side source (usually only needs the function declarations, used for pybind
    generation)."""
    functions: tuple[str, ...] = ()
    """Names of the functions to export."""
    entry: str | None = None
    """The entry the operator actually calls; when ``None``, ``functions[0]`` is used."""
    extra_cuda_cflags: tuple[str, ...] = ("-O3",)

    @property
    def call_entry(self) -> str:
        return self.entry or self.functions[0]


def cpp_compile_available() -> bool:
    """Whether this machine can compile C++/CUDA extensions (has nvcc, or at least
    can compile C++)."""
    try:
        from torch.utils import cpp_extension  # noqa: F401
    except Exception:
        return False
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if os.path.exists(os.path.join(cuda_home, "bin", "nvcc")):
        return True
    from shutil import which

    return which("nvcc") is not None


class CppArtifact:
    """A compiled C++ extension artifact."""

    __slots__ = ("_module", "_entry", "spec_name", "contract", "prewarm_results", "calls")

    def __init__(
        self,
        module: Any,
        entry: Callable,
        spec_name: str,
        contract: Any,
    ) -> None:
        self._module = module
        self._entry = entry
        self.spec_name = spec_name
        self.contract = contract
        self.prewarm_results: tuple[str, ...] = ("ok",)
        self.calls = 0

    def run(self, args: tuple, kwargs: Mapping[str, Any]) -> Any:
        self.calls += 1
        return self._entry(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CppArtifact {self.spec_name} entry={getattr(self._entry, '__name__', '?')}>"


class CppBackend:
    """Compile the registered CUDA source into a callable artifact."""

    name = "cpp"

    def supports(self, spec: Any, impl: Callable) -> bool:
        return bool(getattr(impl, _MARKER_ATTR, None)) and cpp_compile_available()

    def build(self, spec: Any, impl: Callable, contract: Any) -> CppArtifact:
        source: CppSource = getattr(impl, _MARKER_ATTR)
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            name=source.module,
            cpp_sources=source.cpp_sources or "",
            cuda_sources=source.cuda_sources,
            functions=list(source.functions),
            extra_cuda_cflags=list(source.extra_cuda_cflags),
            verbose=False,
        )
        entry = getattr(module, source.call_entry)
        return CppArtifact(module, entry, spec.name, contract)

    def load(self, spec: Any, key: Any, path: Any) -> Any | None:
        """Cross-process reuse of the extension is handled by
        ``torch.utils.cpp_extension``'s own cache (keyed by module name + source hash);
        not duplicated here."""
        return None


def declare_cpp(
    impl: Callable,
    *,
    module: str,
    cuda_sources: str,
    cpp_sources: str = "",
    functions: Sequence[str] = (),
    entry: str | None = None,
    extra_cuda_cflags: Sequence[str] = ("-O3",),
) -> Callable:
    """Register an operator as a C++/CUDA escape-hatch implementation.

    Note: after registration **the operator body itself is never called** — execution
    goes through the compiled extension entry. The decorated function's **signature is
    still used** (it determines the argument list / schema), so do not delete its
    parameters.
    """
    if not functions:
        raise ValueError("declare_cpp requires at least one exported function name")
    setattr(
        impl,
        _MARKER_ATTR,
        CppSource(
            module=module,
            cuda_sources=cuda_sources,
            cpp_sources=cpp_sources,
            functions=tuple(functions),
            entry=entry,
            extra_cuda_cflags=tuple(extra_cuda_cflags),
        ),
    )
    setattr(impl, BACKEND_MARKER_ATTR, "cpp")
    return impl


register_backend(CppBackend())
