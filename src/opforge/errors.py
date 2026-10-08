"""opforge's exception types.

Convention: **"the operator is unavailable in this scenario" is not an exception** --
that is a normal state (e.g. during CUDA Graph capture), and the caller needs a
fallback path. See `contract.preconditions.PreconditionResult`.

Exceptions are reserved for "configuration / contract-level errors": cases the caller
should never ignore regardless.
"""

from __future__ import annotations

from typing import Any


class OpforgeError(Exception):
    """Base class for all opforge exceptions."""


class KernelSpecError(OpforgeError):
    """The operator declaration itself is invalid (duplicate variant, missing
    contract, missing reference, etc.)."""


class ContractUnsatisfied(OpforgeError):
    """No implementation satisfies the requested numeric contract.

    This is a **hard failure**, not a recoverable "unavailable": when DETERMINISTIC is
    requested but only a FAST implementation exists, continuing would silently change
    the numeric semantics.
    """

    def __init__(self, op_name: str, requested: Any, available: Any) -> None:
        super().__init__(
            f"{op_name}: requested contract {requested!r} cannot be satisfied; "
            f"available implementations: {available!r}"
        )
        self.op_name = op_name
        self.requested = requested
        self.available = available


class ManifestError(OpforgeError):
    """The artifact manifest is missing, a field does not match, or the digest mismatches.

    Deliberately an exception rather than a **silent** fallback to JIT: using a
    mismatched artifact shows up as numeric errors far downstream, which is far worse
    than failing on the spot.
    """


class CompilationSealed(OpforgeError):
    """Compilation still happened after the compilation was sealed."""


class EscalationNotAllowed(OpforgeError):
    """An escape-hatch registration was rejected.

    Typical case: requesting a C++ escape hatch for a "missing primitive" when the real
    problem is a "call-shape mismatch" (which can be absorbed inside the DSL with
    idioms such as bucketing). See `lang.escalation`.
    """
