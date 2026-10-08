"""L2 · contract layer: numeric contracts, variant resolution, preconditions.

This layer is where opforge differs most from other operator libraries. It has **zero
dependencies**: it does not import torch / triton and never touches the GPU, so it can
be fully tested in any environment (including CPU-only CI).
"""

from .contract import LOOSEST_FIRST, STRICTEST_FIRST, Contract
from .preconditions import (
    CallContext,
    ExecContext,
    Precondition,
    PreconditionResult,
    RequiresAligned,
    RequiresAllRanksLockstep,
    RequiresDtypeIn,
    RequiresMinLeadingDim,
    RequiresNoCapture,
    check_all,
)
from .resolve import (
    Resolved,
    fallback_counts,
    reset_fallback_counts,
    resolve,
)

__all__ = [
    "Contract",
    "LOOSEST_FIRST",
    "STRICTEST_FIRST",
    "Resolved",
    "resolve",
    "fallback_counts",
    "reset_fallback_counts",
    "ExecContext",
    "CallContext",
    "Precondition",
    "PreconditionResult",
    "check_all",
    "RequiresAllRanksLockstep",
    "RequiresNoCapture",
    "RequiresAligned",
    "RequiresMinLeadingDim",
    "RequiresDtypeIn",
]
