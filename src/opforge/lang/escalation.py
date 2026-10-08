"""Escape-hatch registration (category E).

Only two categories of reason are allowed:

* ``MISSING_PRIMITIVE`` — the DSL has no corresponding primitive (category A, e.g. TMA
  descriptors, named barriers, inline PTX);
* ``PERF_GAP`` — the DSL can express it but cannot reach the performance target
  (**must be accompanied by benchmark evidence**).

``CALL_SHAPE_MISMATCH`` is **explicitly rejected**: that kind of problem ("one kernel
consuming N heterogeneous tensors") should be absorbed with a DSL idiom (e.g. bucketing
by dtype) rather than dragging users into a whole nvcc toolchain.

See `docs/design.md` §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from ..errors import EscalationNotAllowed


class EscalationReason(Enum):
    """Reason for requesting escalation down to C++/PTX."""

    MISSING_PRIMITIVE = "missing_primitive"
    PERF_GAP = "perf_gap"
    CALL_SHAPE_MISMATCH = "call_shape_mismatch"


@dataclass(frozen=True)
class Escalation:
    """A registered escape-hatch record."""

    name: str
    reason: EscalationReason
    primitives: tuple[str, ...]
    benchmark: str | None


_REGISTRY: dict[str, Escalation] = {}


def cpp_kernel(
    *,
    reason: EscalationReason,
    primitives: Sequence[str] = (),
    benchmark: str | None = None,
    name: str | None = None,
) -> Callable[[Callable], Callable]:
    """Register a C++/PTX implementation as an escape hatch.

    Args:
        reason: see :class:`EscalationReason`.
        primitives: the names of the missing primitives (with ``MISSING_PRIMITIVE``,
            used to explain "why the DSL cannot express it").
        benchmark: the benchmark-evidence identifier that ``PERF_GAP`` must provide.
        name: overrides the registration name (defaults to the function name).

    Raises:
        EscalationNotAllowed: when the reason is ``CALL_SHAPE_MISMATCH``, or
            ``PERF_GAP`` without evidence.
    """
    if not isinstance(reason, EscalationReason):
        raise TypeError(f"reason must be an EscalationReason, got {reason!r}")

    def decorate(fn: Callable) -> Callable:
        key = name or getattr(fn, "__name__", "<anonymous>")

        if reason is EscalationReason.CALL_SHAPE_MISMATCH:
            raise EscalationNotAllowed(
                f"{key}: 'call shape mismatch' is not a valid reason for the C++ escape "
                f"hatch; solve it inside the DSL (e.g. dtype bucketing) so users do not "
                f"need an nvcc toolchain"
            )
        if reason is EscalationReason.PERF_GAP and not benchmark:
            raise EscalationNotAllowed(
                f"{key}: a PERF_GAP escalation must carry benchmark evidence "
                f"(pass benchmark=...)"
            )

        record = Escalation(
            name=key,
            reason=reason,
            primitives=tuple(primitives),
            benchmark=benchmark,
        )
        _REGISTRY[key] = record
        fn.__opforge_escalation__ = record  # type: ignore[attr-defined]
        return fn

    return decorate


def escalations() -> Mapping[str, Escalation]:
    """Registered escape hatches (for invariant check I8 to tally the ratio)."""
    return dict(_REGISTRY)


def clear_escalations() -> None:
    """Clear the registry (for tests)."""
    _REGISTRY.clear()
