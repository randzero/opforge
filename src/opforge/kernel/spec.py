"""L1 · operator metadata.

``KernelSpec`` is "an operator's entire architectural identity": name, parameters,
default contract, variants, preconditions, reference, and the declarations that affect
the cache key. It **contains no framework objects** — which is why L1 can be shared by
both the training and the inference side.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..contract import Contract, Precondition


@dataclass(frozen=True)
class ParamSpec:
    """A description of one parameter (introspection and docs only; not used for calls)."""

    name: str
    kind: str
    has_default: bool
    default: Any = None


def describe_params(fn: Callable) -> tuple[ParamSpec, ...]:
    """Describe ``fn``'s parameters in signature order."""
    parameters = inspect.signature(fn).parameters.values()
    return tuple(
        ParamSpec(
            name=p.name,
            kind=p.kind.name,
            has_default=p.default is not inspect.Parameter.empty,
            default=None if p.default is inspect.Parameter.empty else p.default,
        )
        for p in parameters
    )


def capture_impl_sources(
    name: str,
    default_impl: Callable,
    variants: Mapping[Contract, Callable],
) -> dict[str, str]:
    """Capture each implementation's source text **at declaration time** for the source fingerprint.

    It must be "at declaration time" rather than "on first use": the preprocessor of some
    DSLs (CuTeDSL) **rewrites** the source of functions that have gone through
    ``cute.compile``, so a later ``inspect.getsource`` returns the rewritten text — that
    would make the same operator compute two different cache keys "before / after
    compilation", breaking cross-process reuse outright (and newly created handles within
    the same process would recompile as well).
    """
    from ..cache.fingerprint import function_source

    out = {f"{name}.default": function_source(default_impl)}
    for contract, impl in sorted(variants.items(), key=lambda kv: kv[0].value):
        out[f"{name}.{contract.suffix}"] = function_source(impl)
    return out


@dataclass(frozen=True)
class KernelSpec:
    """An operator declaration."""

    name: str
    default_impl: Callable
    """The decorated function itself (= the implementation for the default contract tier)."""

    contract: Contract
    """The default contract. Callers may override with ``contract=`` (stricter direction only)."""

    params: tuple[ParamSpec, ...] = ()
    preconditions: tuple[Precondition, ...] = ()
    variants: Mapping[Contract, Callable] = field(default_factory=dict)
    reference: Callable | None = None
    backward: Callable | None | str = None
    supports_dynamic_shape: bool = False
    min_alignment: int | None = None
    backend: str = "auto"
    tags: tuple[str, ...] = ()
    module: str = ""
    impl_sources: Mapping[str, str] = field(default_factory=dict)
    """Source text of each implementation (default tier + variants) **as declared**, see
    :func:`capture_impl_sources`."""

    @property
    def impls(self) -> dict[Contract, Callable]:
        """Contract -> implementation. The default tier comes from ``default_impl``,
        the rest from ``variants``."""
        merged: dict[Contract, Callable] = {self.contract: self.default_impl}
        merged.update(self.variants)
        return merged

    @property
    def available_contracts(self) -> tuple[Contract, ...]:
        return tuple(sorted(self.impls))

    def sources(self) -> dict[str, str]:
        """All source text that feeds the source fingerprint (default impl + variants + reference).

        The implementation part uses the text captured at declaration time
        (:attr:`impl_sources`) — it may be rewritten by a DSL compiler, so reading it on
        demand is unstable. The reference is pure Python, is never rewritten, and is often
        defined **after** the decorated function, so it is read on demand. A spec without
        ``impl_sources`` (constructed by hand) degrades to reading on demand, with
        behaviour identical to earlier versions.
        """
        from ..cache.fingerprint import function_source

        if self.impl_sources:
            out = dict(self.impl_sources)
        else:
            out = {f"{self.name}.default": function_source(self.default_impl)}
            for contract, impl in sorted(self.variants.items(), key=lambda kv: kv[0].value):
                out[f"{self.name}.{contract.suffix}"] = function_source(impl)
        if self.reference is not None:
            out[f"{self.name}.reference"] = function_source(self.reference)
        return out

    def fingerprint_fields(self) -> dict[str, Any]:
        """Declarations other than the source that feed the source fingerprint."""
        return {
            "backend": self.backend,
            "preconditions": ",".join(
                getattr(p, "name", type(p).__name__) for p in self.preconditions
            ),
            "dynamic_shape": self.supports_dynamic_shape,
            "min_alignment": self.min_alignment,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"<KernelSpec {self.name} contract={self.contract.name} "
            f"impls={[c.name for c in self.available_contracts]}>"
        )
