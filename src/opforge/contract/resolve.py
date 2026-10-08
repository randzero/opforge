"""Variant resolution -- the single most central rule in this project.

**Fallback may only go in the [stricter] direction, and it must be visible.**

Motivation: in systems with a ``bitwise`` boolean parameter, the most common
production bug is "some call site forgot to pass it, so it silently took the
nondeterministic path". Constraining fallback to one direction and making it
observable leaves such bugs nowhere to hide.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Mapping

from ..errors import ContractUnsatisfied
from .contract import Contract, LOOSEST_FIRST


@dataclass(frozen=True)
class Resolved:
    """The result of one contract resolution."""

    requested: Contract
    actual: Contract
    impl: Callable

    @property
    def fell_back(self) -> bool:
        """Whether the actual contract differs from the request (= a fallback occurred)."""
        return self.actual != self.requested


FallbackHook = Callable[[str, Resolved], None]

_lock = threading.Lock()
_counts: dict[tuple[str, Contract, Contract], int] = {}


def _default_on_fallback(op_name: str, resolved: Resolved) -> None:
    with _lock:
        key = (op_name, resolved.requested, resolved.actual)
        _counts[key] = _counts.get(key, 0) + 1


def fallback_counts() -> dict[tuple[str, Contract, Contract], int]:
    """Fallback counts since the last :func:`reset_fallback_counts`.

    Production should export this as a metric
    (``opforge_contract_fallback_total{op,requested,actual}``), corresponding to
    invariant I4.
    """
    with _lock:
        return dict(_counts)


def reset_fallback_counts() -> None:
    """Clear the fallback counts (for tests and process startup)."""
    with _lock:
        _counts.clear()


def resolve(
    requested: Contract,
    impls: Mapping[Contract, Callable],
    *,
    op_name: str = "<anonymous>",
    on_fallback: FallbackHook | None = None,
) -> Resolved:
    """Select, from ``impls``, an implementation that satisfies ``requested``.

    Rule (invariant I3): the returned implementation is **at least as strict as the
    requested one**. The scan goes loosest-to-strictest, so the cheapest implementation
    is preferred; if nothing is found it tightens step by step, and if even the
    strictest has no implementation it raises :class:`ContractUnsatisfied`.
    """
    if not impls:
        raise ContractUnsatisfied(op_name, requested, ())

    for candidate in LOOSEST_FIRST:
        if candidate.value <= requested.value and candidate in impls:
            resolved = Resolved(requested=requested, actual=candidate, impl=impls[candidate])
            if resolved.fell_back:
                (on_fallback or _default_on_fallback)(op_name, resolved)
            return resolved

    raise ContractUnsatisfied(op_name, requested, tuple(sorted(impls)))
