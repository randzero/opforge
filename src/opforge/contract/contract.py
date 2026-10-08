"""Numeric contracts.

Core thesis: an operator's numeric semantics are a **first-class declarative
attribute**, not boolean switches (`bitwise=True`) or environment variables
(`*_LEVEL`) scattered along the call chain.
"""

from __future__ import annotations

from enum import IntEnum


class Contract(IntEnum):
    """Numeric contract. **The smaller the enum value, the stricter.**

    ``DETERMINISTIC ⊒ HIGH_PRECISION ⊒ FAST``

    A "stricter" implementation can satisfy a "looser" request; never the other way
    around -- this one-directionality is the whole point of
    :func:`opforge.contract.resolve.resolve` (invariant I3).
    """

    DETERMINISTIC = 0
    """**Bitwise identical** across ranks / across runs.

    Forbids nondeterministic reduction order, atomic accumulation, and
    scheduling-dependent splits.
    """

    HIGH_PRECISION = 1
    """Accumulate in fp32 (or higher precision). Nondeterministic reduction order is allowed."""

    FAST = 2
    """No extra numeric guarantees. Nondeterministic split-k and low-precision
    accumulation allowed."""

    @property
    def suffix(self) -> str:
        """Variant suffix: ``_det`` / ``_hp`` / ``_fast``."""
        return _SUFFIXES[self]

    def is_at_least_as_strict_as(self, other: "Contract") -> bool:
        """Whether this contract is **at least as strict as** ``other``."""
        return self.value <= other.value


_SUFFIXES: dict[Contract, str] = {
    Contract.DETERMINISTIC: "det",
    Contract.HIGH_PRECISION: "hp",
    Contract.FAST: "fast",
}

#: From strictest to loosest. Docs and assertions follow this order.
STRICTEST_FIRST: tuple[Contract, ...] = (
    Contract.DETERMINISTIC,
    Contract.HIGH_PRECISION,
    Contract.FAST,
)

#: From loosest to strictest. :func:`resolve`'s scan order -- first hit wins, so the
#: cheapest implementation is preferred.
LOOSEST_FIRST: tuple[Contract, ...] = tuple(reversed(STRICTEST_FIRST))
