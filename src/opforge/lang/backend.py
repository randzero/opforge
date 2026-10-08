"""L0 · backend protocol and registry.

A backend = the ability to "turn an operator description into a loadable artifact".
Built-in:

* ``cutedsl`` — operators needing layout / TMA / warp-level control;
* ``triton``  — regular block-shaped operators;
* ``cpp``     — the **escape hatch** (must be registered via
  :mod:`opforge.lang.escalation`);
* ``python``  — a pure-Python reference backend, **no compilation**, used to run and
  test the whole chain (contract → preconditions → cache → launch) end to end in a
  CPU-only environment.

The backend selection order is fixed (:data:`PROBE_ORDER`) so that "whoever registers
first wins" cannot introduce non-determinism.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, runtime_checkable

from ..contract import Contract
from ..errors import KernelSpecError

if TYPE_CHECKING:  # pragma: no cover
    from ..cache import BuildKey
    from ..kernel.spec import KernelSpec


@runtime_checkable
class Artifact(Protocol):
    """A backend artifact: can be executed directly in the cached call shape."""

    def run(self, args: tuple, kwargs: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class Backend(Protocol):
    """Backend protocol."""

    name: str

    def supports(self, spec: "KernelSpec", impl: Callable) -> bool:
        """Whether this backend can build ``impl``.

        Note that ``impl`` is **the implementation being built**, not necessarily
        ``spec.default_impl`` — different contract variants of the same operator can
        use different backends (see :func:`select_backend`). A backend should inspect
        the marker on ``impl`` (e.g. the Triton backend looks at
        ``__opforge_triton_kernels__``), not the default implementation inside ``spec``.
        """
        ...

    def build(self, spec: "KernelSpec", impl: Callable, contract: Contract) -> Any:
        """Build the artifact; the artifact must provide ``run(args, kwargs)``."""
        ...

    def load(self, spec: "KernelSpec", key: "BuildKey", path: Path) -> Any | None:
        """Load an existing artifact from disk; return ``None`` when unsupported or unavailable."""
        ...


PROBE_ORDER: tuple[str, ...] = ("cutedsl", "triton", "cpp", "python")

#: The marker that ``declare_triton`` / ``declare_cutedsl`` / ``declare_cpp`` put on
#: an implementation. It says "this implementation is meant for some backend" — a
#: **soft** constraint: when that backend is unavailable the kernel may be downgraded.
BACKEND_MARKER_ATTR = "__opforge_backend__"

_REGISTRY: dict[str, Backend] = {}


def register_backend(backend: Backend, *, replace: bool = False) -> None:
    """Register a backend. A duplicate name errors unless ``replace`` is set,
    to avoid silent overwrites."""
    name = backend.name
    if name in _REGISTRY and not replace:
        raise KernelSpecError(f"backend {name!r} is already registered")
    _REGISTRY[name] = backend


def get_backend(name: str) -> Backend:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KernelSpecError(
            f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def registered_backends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def select_backend(spec: "KernelSpec", impl: Callable | None = None) -> Backend:
    """Choose a backend for ``impl`` (falls back to ``spec.default_impl`` when ``None``).

    Three tiers, from "best fit for this implementation" to "most general":

    1. **The backend the implementation declares** — the :data:`BACKEND_MARKER_ATTR`
       that ``declare_triton`` / ``declare_cutedsl`` / ``declare_cpp`` put on ``impl``.
       This is a **soft** constraint: when the declared backend cannot build it on
       this platform (typically: no Triton driver in a CPU-only environment), keep
       probing so that **the same operator can downgrade to an available backend**
       (e.g. the python reference backend).
    2. **``spec.backend != "auto"``** — a **hard** constraint: the operator explicitly
       requires a backend; if unmet it errors, and it never silently swaps backends
       (swapping backends = swapping numeric semantics).
    3. **Probe in :data:`PROBE_ORDER`** — the deterministic fallback for ``"auto"``.

    Why select by ``impl`` rather than by ``spec``: **different contract variants of
    the same operator may use different backends** (e.g. the default tier goes through
    Triton, the FAST tier through CuTeDSL), so the backend is a property of the
    implementation, not of the operator.
    """
    target = impl if impl is not None else spec.default_impl
    declared = getattr(target, BACKEND_MARKER_ATTR, None)

    if declared is not None:
        backend = get_backend(declared)
        if backend.supports(spec, target):
            if spec.backend not in ("auto", declared):
                raise KernelSpecError(
                    f"{spec.name}: impl declares backend {declared!r} but the kernel "
                    f"requires {spec.backend!r}"
                )
            return backend
        # The declared backend is unavailable on this platform; if the operator
        # also hard-requires it, it cannot be downgraded.
        if spec.backend == declared:
            raise KernelSpecError(
                f"{spec.name}: backend {declared!r} cannot build this kernel on this "
                f"platform"
            )

    if spec.backend != "auto":
        backend = get_backend(spec.backend)
        if not backend.supports(spec, target):
            raise KernelSpecError(
                f"{spec.name}: backend {spec.backend!r} does not support this kernel"
            )
        return backend

    for name in PROBE_ORDER:
        backend = _REGISTRY.get(name)
        if backend is not None and backend.supports(spec, target):
            return backend
    raise KernelSpecError(
        f"{spec.name}: no registered backend supports this kernel "
        f"(registered: {registered_backends()})"
    )
