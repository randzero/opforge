"""Cross-layer return types.

``Unavailable`` is a core opforge convention: **"the operator is unavailable in this
scenario" is a normal return value, not an exception**. The caller (the L3 adapter
layer) uses it to fall back to a more conservative path instead of being interrupted
by an exception.

See `docs/architecture.md` §2.3 and §3.4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Unavailable:
    """The operator is unavailable for the current call; see ``reason``.

    ``reason`` usually comes from a
    :class:`~opforge.contract.preconditions.Precondition` (with the condition name
    prepended), e.g. ``requires_all_ranks_lockstep: ...``.
    """

    op_name: str
    reason: str

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:  # pragma: no cover - convenient for logging
        return f"<Unavailable {self.op_name}: {self.reason}>"
