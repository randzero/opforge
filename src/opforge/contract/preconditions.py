"""Preconditions -- turn "under what circumstances is the operator unavailable" into
something a program can check.

Motivation: if conditions like these live only in comments, they turn into production
incidents such as "a multi-GPU deadlock during CUDA Graph capture". See
`docs/design.md` §3.3.

Conventions:

* ``check()`` must be a **cheap pure function** -- no CUDA synchronization, no
  allocation, no side effects;
* tensor checks use **duck typing** (looking only at ``.shape`` / ``.dtype``), so the
  core layer can run and be tested without torch / numpy;
* when unsatisfied it returns ``PreconditionResult.unavailable(...)``, **not** an
  exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ExecContext:
    """The execution environment. Provided by the L3 adapter layer; the core layer only
    reads it and never probes it itself."""

    device: str = "cpu"
    world_size: int = 1
    rank: int = 0
    is_capturing: bool = False
    """Whether a CUDA Graph capture is in progress."""

    is_warming_up: bool = False
    """Whether a warmup / probe (dummy run) is in progress."""


@dataclass(frozen=True)
class CallContext:
    """The full context of a call: execution environment + actual arguments."""

    exec: ExecContext = field(default_factory=ExecContext)
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def primary_tensor(self) -> Any | None:
        """Convention: the first positional argument is the primary input tensor (if any)."""
        return self.args[0] if self.args else None


@dataclass(frozen=True)
class PreconditionResult:
    ok: bool
    reason: str = ""

    @classmethod
    def satisfied(cls) -> "PreconditionResult":
        return cls(True)

    @classmethod
    def unavailable(cls, reason: str) -> "PreconditionResult":
        return cls(False, reason)


@runtime_checkable
class Precondition(Protocol):
    """The precondition protocol."""

    name: str

    def check(self, call: CallContext) -> PreconditionResult: ...


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(d) for d in shape)
    except (TypeError, ValueError):
        return None


def _dtype_name(value: Any) -> str | None:
    """Normalize a dtype name: ``torch.float32`` -> ``float32``, ``np.dtype('float32')``
    -> ``float32``."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).rsplit(".", 1)[-1]


# --------------------------------------------------------------------------- #
# built-in conditions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RequiresAllRanksLockstep:
    """Multicast / spin-barrier style operators: require all ranks to arrive in lockstep.

    During capture / warmup the ranks are not in lockstep. **The consequence of not
    checking is a deadlock** (all GPUs at 100% SM, 0% memory bandwidth), not a
    diagnosable error.
    """

    name: str = "requires_all_ranks_lockstep"

    def check(self, call: CallContext) -> PreconditionResult:
        ctx = call.exec
        if ctx.world_size > 1 and (ctx.is_capturing or ctx.is_warming_up):
            return PreconditionResult.unavailable(
                "lockstep collective is unsafe during capture or warmup"
            )
        return PreconditionResult.satisfied()


@dataclass(frozen=True)
class RequiresNoCapture:
    """The operator is unavailable during CUDA Graph capture (e.g. it allocates memory
    or synchronizes)."""

    name: str = "requires_no_capture"

    def check(self, call: CallContext) -> PreconditionResult:
        if call.exec.is_capturing:
            return PreconditionResult.unavailable("op is not capture-safe")
        return PreconditionResult.satisfied()


@dataclass(frozen=True)
class RequiresAligned:
    """The last dim of the primary input tensor must be a multiple of ``multiple_of``."""

    multiple_of: int
    name: str = "requires_aligned"

    def check(self, call: CallContext) -> PreconditionResult:
        shape = _shape_of(call.primary_tensor())
        if not shape:
            return PreconditionResult.satisfied()
        last = shape[-1]
        if last % self.multiple_of:
            return PreconditionResult.unavailable(
                f"last dim {last} is not a multiple of {self.multiple_of}"
            )
        return PreconditionResult.satisfied()


@dataclass(frozen=True)
class RequiresMinLeadingDim:
    """Dim 0 of the primary input tensor must be at least ``minimum``."""

    minimum: int
    name: str = "requires_min_leading_dim"

    def check(self, call: CallContext) -> PreconditionResult:
        shape = _shape_of(call.primary_tensor())
        if not shape:
            return PreconditionResult.satisfied()
        if shape[0] < self.minimum:
            return PreconditionResult.unavailable(
                f"leading dim {shape[0]} < minimum {self.minimum}"
            )
        return PreconditionResult.satisfied()


@dataclass(frozen=True)
class RequiresDtypeIn:
    """The primary input tensor's dtype must be in the allowlist (compared by short name,
    e.g. ``float32``)."""

    dtypes: frozenset[str]
    name: str = "requires_dtype_in"

    def __init__(self, *dtypes: str) -> None:
        normalized = frozenset(d.rsplit(".", 1)[-1] for d in dtypes)
        object.__setattr__(self, "dtypes", normalized)

    def check(self, call: CallContext) -> PreconditionResult:
        dtype = _dtype_name(call.primary_tensor())
        if dtype is None:
            return PreconditionResult.satisfied()
        if dtype not in self.dtypes:
            return PreconditionResult.unavailable(
                f"dtype {dtype} not in {sorted(self.dtypes)}"
            )
        return PreconditionResult.satisfied()


def check_all(
    preconditions: Sequence[Precondition],
    call: CallContext,
) -> PreconditionResult:
    """Check in order; returns as soon as one is unsatisfied, with the condition name
    prepended to the reason."""
    for precondition in preconditions:
        result = precondition.check(call)
        if not result.ok:
            return PreconditionResult.unavailable(
                f"{precondition.name}: {result.reason}"
            )
    return PreconditionResult.satisfied()
