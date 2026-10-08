"""Process / thread-level propagation of the execution context.

The L3 adapter layer uses :func:`use_exec_context` to declare "we are now capturing /
warming up"; the core layer (preconditions) only **reads** it and never probes the
environment itself -- this keeps the core framework-agnostic and testable.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from .contract.preconditions import ExecContext

_state = threading.local()


def current_exec_context() -> ExecContext:
    """The current thread's execution context; returns the default (single device, not
    capturing, not warming up) when unset."""
    value = getattr(_state, "value", None)
    return value if value is not None else ExecContext()


def set_exec_context(context: ExecContext | None) -> None:
    """Set explicitly (``None`` = clear, fall back to the default)."""
    _state.value = context


@contextmanager
def use_exec_context(context: ExecContext) -> Iterator[ExecContext]:
    """Use the given execution context inside the with block, restoring it on exit."""
    previous = getattr(_state, "value", None)
    _state.value = context
    try:
        yield context
    finally:
        _state.value = previous
