"""opforge -- the operator forge.

**Write JIT, publish AOT, declare contracts explicitly**; one set of kernels serves
both training and inference.

Layers (see ``docs/architecture.md`` for details) ::

    L0 language layer   backend protocol + escape-hatch registry
    L1 kernel layer     operator declaration (@kernel) and the call pipeline
    L2 contract layer   numeric contracts, variant resolution, preconditions
    L2 cache layer      two-level fingerprints, manifest, sealed compilation
    L2 launch layer     cache a launch closure per signature
    L3 adapter layer    training / inference integration (outside this core package)

The **core** of this package (everything above except L3) has zero third-party
dependencies and can be fully tested in a CPU-only environment. Operator
implementations (``opforge.kernels``) need numpy; the L3 adapter layer needs torch.
"""

from __future__ import annotations

__version__ = "0.0.1.dev0"

from .cache import (
    BuildKey,
    KernelCache,
    Manifest,
    default_cache_root,
    get_cache,
    source_fingerprint,
    target_fingerprint,
)
from .context import current_exec_context, set_exec_context, use_exec_context
from .contract import (
    CallContext,
    Contract,
    ExecContext,
    Precondition,
    PreconditionResult,
    RequiresAligned,
    RequiresAllRanksLockstep,
    RequiresDtypeIn,
    RequiresMinLeadingDim,
    RequiresNoCapture,
    Resolved,
    check_all,
    fallback_counts,
    reset_fallback_counts,
    resolve,
)
from .errors import (
    CompilationSealed,
    ContractUnsatisfied,
    EscalationNotAllowed,
    KernelSpecError,
    ManifestError,
    OpforgeError,
)
from .kernel import KernelHandle, KernelSpec, kernel
from .lang import (
    Artifact,
    Backend,
    Escalation,
    EscalationReason,
    cpp_kernel,
    escalations,
    get_backend,
    register_backend,
    registered_backends,
    select_backend,
)
from .launch import LaunchSpec, Launcher, get_launcher
from .result import Unavailable

__all__ = [
    "__version__",
    # L1
    "kernel",
    "KernelHandle",
    "KernelSpec",
    # L2 contract
    "Contract",
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
    # L2 cache
    "BuildKey",
    "Manifest",
    "KernelCache",
    "default_cache_root",
    "get_cache",
    # L2 launch
    "LaunchSpec",
    "Launcher",
    "get_launcher",
    # L0
    "Artifact",
    "Backend",
    "register_backend",
    "get_backend",
    "registered_backends",
    "select_backend",
    "cpp_kernel",
    "Escalation",
    "EscalationReason",
    "escalations",
    # context
    "current_exec_context",
    "set_exec_context",
    "use_exec_context",
    # result
    "Unavailable",
    # errors
    "OpforgeError",
    "KernelSpecError",
    "ContractUnsatisfied",
    "ManifestError",
    "CompilationSealed",
    "EscalationNotAllowed",
]
