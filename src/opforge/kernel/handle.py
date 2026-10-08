"""L1 · the operator handle — the complete pipeline for one call.

    call ─► contract resolution ─► preconditions ─► cache hit / build ─► launch

Failure is expressed as an **explicit result** rather than being mixed into exceptions:

* contract cannot be satisfied -> raise :class:`~opforge.errors.ContractUnsatisfied`
  (**hard failure**: carrying on would silently change the numerical semantics);
* preconditions unmet -> return :class:`~opforge.result.Unavailable`
  (**normal fallback**, not an exception).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

from ..cache.fingerprint import BuildKey, source_fingerprint, target_fingerprint
from ..cache.store import KernelCache, get_cache
from ..context import current_exec_context
from ..contract import CallContext, Contract, check_all, resolve
from ..lang.backend import select_backend
from ..launch.launcher import LaunchSpec, Launcher, get_launcher
from ..result import Unavailable
from .spec import KernelSpec


def _find_reference(spec: KernelSpec) -> Callable | None:
    """Find the reference in the operator's module by naming convention.

    Convention: ``<operator name>_ref`` first, then ``<default impl function name>_ref``.
    Lazy resolution (not at decoration time) is necessary — the reference is often defined
    **after** the decorated function.
    """
    module = inspect.getmodule(spec.default_impl)
    if module is None:
        return None
    candidates = (f"{spec.name}_ref", f"{getattr(spec.default_impl, '__name__', '')}_ref")
    for attr in candidates:
        fn = getattr(module, attr, None)
        if inspect.isfunction(fn):
            return fn
    return None


class KernelHandle:
    """The return value of ``@kernel``: both an operator object and directly callable."""

    def __init__(
        self,
        spec: KernelSpec,
        *,
        cache: KernelCache | None = None,
        launcher: Launcher | None = None,
    ) -> None:
        self.spec = spec
        self._cache_override = cache
        self._launcher_override = launcher
        self._reference = spec.reference
        self._reference_searched = spec.reference is not None
        self._source_fp: str | None = None

    # -- read-only views -------------------------------------------------- #

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def contract(self) -> Contract:
        return self.spec.contract

    @property
    def reference(self) -> Callable | None:
        """Pure PyTorch ground-truth implementation (resolved lazily)."""
        if not self._reference_searched:
            self._reference = _find_reference(self.spec)
            self._reference_searched = True
        return self._reference

    @property
    def backward(self) -> Callable | None | str:
        """Hand-written backward implementation; ``"auto"`` means use the reference's autograd."""
        return self.spec.backward

    # -- dependencies (injectable, for tests and multiple instances) ------- #

    @property
    def cache(self) -> KernelCache:
        return self._cache_override if self._cache_override is not None else get_cache()

    @property
    def launcher(self) -> Launcher:
        return (
            self._launcher_override
            if self._launcher_override is not None
            else get_launcher()
        )

    # -- call ------------------------------------------------------------- #

    def __call__(
        self,
        *args: Any,
        contract: Contract | None = None,
        exec_context: Any = None,
        **kwargs: Any,
    ) -> Any:
        requested = self.spec.contract if contract is None else contract
        if not isinstance(requested, Contract):
            raise TypeError(f"contract must be a Contract, got {requested!r}")

        resolved = resolve(requested, self.spec.impls, op_name=self.spec.name)

        call = CallContext(
            exec=exec_context if exec_context is not None else current_exec_context(),
            args=args,
            kwargs=kwargs,
        )
        verdict = check_all(self.spec.preconditions, call)
        if not verdict.ok:
            return Unavailable(self.spec.name, verdict.reason)

        return self.launcher(
            LaunchSpec(
                # kernel_id must be unique **per contract tier**: different tiers are
                # different artifacts, and sharing one id would let their launch caches
                # bleed into each other.
                kernel_id=f"{self.spec.name}:{resolved.actual.suffix}",
                artifact=self._artifact_for(resolved),
                runtime_args=args,
                runtime_kwargs=kwargs,
            )
        )

    def build(self, contract: Contract | None = None) -> Any:
        """Build (or fetch a cached) artifact of this operator under the given contract.

        **This is the path AOT precompilation takes** — it triggers `Backend.build`, which
        in turn triggers the operator's prewarm compilation; artifacts are persisted by
        each backend itself (Triton writes to `TRITON_CACHE_DIR`).

        Args:
            contract: the target contract; with ``None`` the operator's default is used.
        """
        requested = self.spec.contract if contract is None else contract
        resolved = resolve(requested, self.spec.impls, op_name=self.spec.name)
        return self._artifact_for(resolved)

    def _artifact_for(self, resolved: Any) -> Any:
        key = self.build_key(resolved.actual)
        return self.cache.get_or_build(
            key,
            lambda: self._build(resolved.impl, resolved.actual),
            load=self._load,
        )

    def ref(self, *args: Any, **kwargs: Any) -> Any:
        """Call the reference directly (L3's fallback path, and CI's numerical ground truth)."""
        reference = self.reference
        if reference is None:
            raise NotImplementedError(f"{self.spec.name}: no reference implementation")
        return reference(*args, **kwargs)

    # -- cache key -------------------------------------------------------- #

    def _source_fingerprint(self) -> str:
        """Source fingerprint — computed once per process.

        Deliberately folds in the source of **all variants + the reference**: any change
        invalidates every contract entry of this operator. This is the conservative choice
        (better to invalidate too much than to reuse a wrong artifact).
        """
        if self._source_fp is None:
            self._source_fp = source_fingerprint(
                self.spec.sources(),
                namespace=self.spec.name,
                extra=self.spec.fingerprint_fields(),
            )
        return self._source_fp

    def build_key(self, contract: Contract) -> BuildKey:
        return BuildKey(
            source_fingerprint=self._source_fingerprint(),
            target_fingerprint=target_fingerprint(),
            namespace=self.spec.name,
            contract=contract,
        )

    # -- build / load ----------------------------------------------------- #

    def _build(self, impl: Callable, contract: Contract) -> Any:
        return select_backend(self.spec, impl).build(self.spec, impl, contract)

    def _load(self, key: BuildKey, path: Path) -> Any:
        # The load path also selects the backend by **the implementation matching that
        # contract** — variants may land on different backends.
        impl = self.spec.impls.get(key.contract, self.spec.default_impl)
        return select_backend(self.spec, impl).load(self.spec, key, path)

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"<KernelHandle {self.spec.name} default={self.spec.contract.name} "
            f"impls={[c.name for c in self.spec.available_contracts]}>"
        )
