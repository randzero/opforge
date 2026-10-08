"""CuTeDSL backend — a second kind of DSL, used to verify that the `Backend` protocol
is truly pluggable.

Three key differences from the Triton backend:

1. **The compile entry point needs example inputs**: you need
   `cute.compile(host_fn, *example_args)` to specialize the compilation. So here
   `build()` *is* **AOT compilation**, and the artifact is a
   `CudaDialectJitCompiledFunction`. Operators supply example inputs via
   `__opforge_cutedsl_examples__` — the same concept as Triton's
   `__opforge_triton_prewarm__` ("I have to know which shapes to compile").
2. **Compilation and execution have different environment requirements**: compilation
   only needs the toolchain; execution additionally needs a matching driver version
   (see `cutedsl_launchable()`). So `supports()` only checks "can it compile"; whether
   it can run is gated by tests and preconditions respectively.
3. **Cross-process reuse is an "official interface", not "copying a cache directory"**:
   Triton can only treat its cache directory as an interface (the hash directory names
   and file naming are internal implementation and may break on upgrade); CuTeDSL has
   the official ``export_to_c`` + ``cute.runtime.load_module``, where both the artifact
   and the loader are stable APIs, and the manifest is fully under our control — this
   is the cleanest path in this project's "write with JIT, ship with AOT".

**Requirements on operators**: the host function must be type-annotated
(``x: cute.Tensor`` / ``alpha: cutlass.Float32``), and the example inputs must be
wrapped into ``cute.Tensor`` with ``from_dlpack``. ``export_to_c`` generates a C ABI
struct per argument type, and a bare ``torch.Tensor`` argument cannot get through that
step. At runtime you still pass ``torch.Tensor`` directly.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .backend import BACKEND_MARKER_ATTR, register_backend

_EXAMPLES_ATTR = "__opforge_cutedsl_examples__"
_MARKER_ATTR = "__opforge_cutedsl__"


def cutedsl_usable() -> bool:
    """Whether CuTeDSL can **compile** (only requires the toolchain to be importable,
    not that it can execute)."""
    try:
        import cutlass.cute  # noqa: F401
    except Exception:
        return False
    return True


def cutedsl_launchable() -> bool:
    """Whether CuTeDSL can **execute** — compile an empty kernel and try to launch it.

    Relatively expensive (it really compiles once) and used only by cold paths such as
    tests. A common failure cause is the driver version:
    `cudaErrorInsufficientDriver` means cuda-python requires a newer CUDA version than
    the driver supports.
    """
    if not cutedsl_usable():
        return False
    try:
        import cutlass.cute as cute

        @cute.kernel
        def _probe_kernel():
            return

        @cute.jit
        def _probe_host():
            _probe_kernel().launch(grid=[1, 1, 1], block=[1, 1, 1])

        cute.compile(_probe_host)()
    except Exception:
        return False
    return True


class CuTeDSLArtifact:
    """A compiled CuTeDSL artifact.

    ``_compiled`` may be either the current result of ``cute.compile`` or a
    ``CudaDialectJitCompiledFunction`` loaded back from disk — the two have the same
    call shape (``fn(*torch_args)``), so the load path does not need a separate
    artifact type.
    """

    __slots__ = ("_compiled", "spec_name", "contract", "prewarm_results", "calls")

    def __init__(
        self,
        compiled: Callable,
        spec_name: str,
        contract: Any,
        prewarm_results: tuple[str, ...] = (),
    ) -> None:
        self._compiled = compiled
        self.spec_name = spec_name
        self.contract = contract
        self.prewarm_results = prewarm_results
        self.calls = 0

    def run(self, args: tuple, kwargs: dict) -> Any:
        self.calls += 1
        return self._compiled(*args, **kwargs)

    def export(self, path: Path, key: Any) -> None:
        """Write the compiled artifact to disk as a **cross-process-loadable object file**.

        It goes through the official ``export_to_c``: that writes ``<name>.h`` +
        ``<name>.o``; the header is meant for C/C++ AOT users to link against, and
        opforge's load path only consumes the object, so only the ``.o`` bytes are kept
        here. The exported symbol is ``key.export_symbol`` — loading looks the function
        up by the same name.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self._compiled.export_to_c(tmp, "kernel", function_prefix=key.export_symbol)
            shutil.copyfile(Path(tmp) / "kernel.o", path)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CuTeDSLArtifact {self.spec_name} prewarm={self.prewarm_results}>"


class CuTeDSLBackend:
    """Compile operators declaring a CuTeDSL host function into an artifact."""

    name = "cutedsl"

    def supports(self, spec: Any, impl: Callable) -> bool:
        return bool(getattr(impl, _MARKER_ATTR, False)) and cutedsl_usable()

    def build(self, spec: Any, impl: Callable, contract: Any) -> CuTeDSLArtifact:
        examples = getattr(impl, _EXAMPLES_ATTR, None)
        if examples is None:
            raise ValueError(
                f"{spec.name}: a CuTeDSL operator must declare example inputs via "
                f"`declare_cutedsl(..., examples=...)` — cute.compile needs them to specialize"
            )
        import cutlass.cute as cute

        compiled = cute.compile(impl, *examples())
        return CuTeDSLArtifact(compiled, spec.name, contract, prewarm_results=("ok",))

    def load(self, spec: Any, key: Any, path: Path) -> Any | None:
        """Load an existing artifact from disk — this is the CuTeDSL side's
        **cross-process reuse**.

        ``cute.runtime.load_module`` JITLinks the exported object inside the process,
        then pulls out the host launch entry by ``key.export_symbol``. The manifest
        validation is done by the cache layer **before** this method is called
        (fingerprint + sha256), so reaching here is no longer a question of "is it
        usable" but a real environment issue (DSL version, missing runtime library) —
        let it raise, do not silently fall back to recompiling.
        """
        if not cutedsl_usable():
            return None
        from cutlass.cute.runtime import load_module

        module = load_module(str(path))
        compiled = module[key.export_symbol]
        return CuTeDSLArtifact(compiled, spec.name, key.contract, prewarm_results=("loaded",))


def declare_cutedsl(
    impl: Callable,
    *,
    examples: Callable[[], tuple] | None = None,
) -> Callable:
    """Mark a host function as a CuTeDSL implementation.

    Args:
        impl: the ``@cute.jit`` host function (the object of `cute.compile`).
        examples: a provider that returns a tuple of example inputs — used for
            compile-time specialization and also serving as the AOT declaration of
            "which shapes to compile". **Omitting it still registers the operator, but
            `build()` will error** — the error happens at build time rather than at
            runtime, which is deliberate.
    """
    setattr(impl, _MARKER_ATTR, True)
    setattr(impl, BACKEND_MARKER_ATTR, "cutedsl")
    if examples is not None:
        setattr(impl, _EXAMPLES_ATTR, examples)
    return impl


register_backend(CuTeDSLBackend())
