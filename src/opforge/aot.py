"""AOT precompilation: move "compilation" from **runtime** to **build time**.

This project's thesis is "write JIT, publish AOT". Writing JIT is already in place;
publishing AOT needs three things together:

1. **operators must declare which shapes to prewarm** -- otherwise the build step
   does not know what to compile. See
   `lang.triton_backend.declare_triton(..., prewarm=...)`;
2. **a build-time entry point** -- `warm()` / `tools/build_aot.py`, which compiles
   every operator once and persists it to disk;
3. **a relocatable cache directory** -- fill `TRITON_CACHE_DIR` (or each backend's
   own directory) during image build, mount the same directory at runtime, and the
   image starts warm out of the box.

At runtime, if the cache is complete, `KernelCache.seal()` lets you assert that "no
compilation happens for the rest of the process lifetime".
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .kernel import KernelHandle
from .lang.backend import select_backend

DEFAULT_KERNELS_PACKAGE = "opforge.kernels"


ImportErrorHook = Callable[[str, ImportError], None]


def iter_kernels(
    package: str = DEFAULT_KERNELS_PACKAGE,
    *,
    on_import_error: ImportErrorHook | None = None,
) -> tuple[KernelHandle, ...]:
    """Collect every importable operator in the package (sorted by name, stable order).

    When an operator module fails to import (for example its DSL is not installed)
    it does **not** raise -- that merely means "this operator does not exist in this
    environment". Pass ``on_import_error`` to observe such gaps (``warm()`` uses it
    to report ``skip:`` entries).
    """
    root = importlib.import_module(package)
    found: dict[str, KernelHandle] = {}
    for modinfo in pkgutil.walk_packages(root.__path__, prefix=package + "."):
        try:
            module = importlib.import_module(modinfo.name)
        except ImportError as exc:
            if on_import_error is not None:
                on_import_error(modinfo.name, exc)
            continue
        for value in vars(module).values():
            if isinstance(value, KernelHandle):
                found.setdefault(value.name, value)
    return tuple(found[name] for name in sorted(found))


@dataclass(frozen=True)
class WarmEntry:
    """The prewarm result of a single operator.

    ``status`` values:

    * ``ok`` -- prewarm compilation succeeded;
    * ``loaded`` -- hit a persisted artifact and loaded it directly (no compile) --
      exactly the state "write JIT, publish AOT" aims for;
    * ``no-prewarm`` -- this backend does not need prewarming (e.g. the pure Python
      reference backend);
    * ``deferred(n/m)`` -- declared m shapes and n of them did not get baked,
      deferred to the first call;
    * ``skip:<reason>`` -- the backend is unavailable (e.g. no GPU / Triton driver
      failed to load).
    """

    kernel: str
    backend: str
    contract: str
    status: str

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "loaded", "no-prewarm")


@dataclass(frozen=True)
class WarmReport:
    entries: tuple[WarmEntry, ...]

    @property
    def ok(self) -> bool:
        """True only if all entries succeeded (including "prewarm not needed"); ``skip:*`` fails."""
        return all(entry.ok for entry in self.entries)

    @property
    def compiled(self) -> int:
        """Number of operators that were **actually compiled** this run."""
        return sum(1 for entry in self.entries if entry.status == "ok")

    @property
    def loaded(self) -> int:
        """Number of operators that **hit a persisted artifact** (no compile) this run."""
        return sum(1 for entry in self.entries if entry.status == "loaded")

    def format(self) -> str:
        width = max((len(e.kernel) for e in self.entries), default=0)
        backend_width = max((len(e.backend) for e in self.entries), default=0)
        lines = [
            f"  {e.kernel:<{width}}  {e.backend:<{backend_width}}  {e.contract:<15}  {e.status}"
            for e in self.entries
        ]
        header = (
            f"{len(self.entries)} kernels, {self.compiled} compiled, "
            f"{self.loaded} loaded, ok={self.ok}"
        )
        return "\n".join([header, *lines])


def warm(kernels: Sequence[KernelHandle] | None = None) -> WarmReport:
    """Build every operator once under its **default contract**, triggering each
    one's prewarm compilation.

    Deliberately does not raise: an operator that is unavailable in the current
    environment (no GPU, missing dependency) should surface as a ``skip:`` entry in
    the report rather than crashing the whole build-time job. Whether it passed is up
    to the caller, via ``report.ok``.
    """
    entries: list[WarmEntry] = []

    if kernels is None:
        def _on_import_error(module_name: str, exc: ImportError) -> None:
            label = module_name.rsplit(".", 1)[-1]
            entries.append(WarmEntry(label, "?", "-", f"skip:ImportError"))

        handles: Iterable[KernelHandle] = iter_kernels(on_import_error=_on_import_error)
    else:
        handles = kernels

    for handle in handles:
        contract_name = handle.spec.contract.name

        try:
            backend = select_backend(handle.spec)
        except Exception as exc:
            entries.append(
                WarmEntry(handle.name, "?", contract_name, f"skip:{type(exc).__name__}")
            )
            continue

        try:
            artifact = handle.build()
        except Exception as exc:
            entries.append(
                WarmEntry(handle.name, backend.name, contract_name, f"skip:{type(exc).__name__}")
            )
            continue

        entries.append(
            WarmEntry(
                handle.name,
                backend.name,
                contract_name,
                _status_of(getattr(artifact, "prewarm_results", None)),
            )
        )

    return WarmReport(tuple(entries))


def _status_of(prewarm_results: Any) -> str:
    if not prewarm_results:
        return "no-prewarm"
    if tuple(prewarm_results) == ("loaded",):
        return "loaded"
    failed = [item for item in prewarm_results if item != "ok"]
    if not failed:
        return "ok"
    return f"deferred({len(failed)}/{len(prewarm_results)})"
