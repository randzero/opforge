"""L0 · language layer: backend protocol and escape-hatch registration.

Importing this package performs the built-in backend registration (currently the
`python` reference backend; `triton` / `cutedsl` / `cpp` are registered by their own
implementation modules when available).
"""

from .backend import (
    PROBE_ORDER,
    Artifact,
    Backend,
    get_backend,
    register_backend,
    registered_backends,
    select_backend,
)
from .escalation import (
    Escalation,
    EscalationReason,
    clear_escalations,
    cpp_kernel,
    escalations,
)
from . import cpp_backend as _cpp_backend  # noqa: F401  import registers it (no torch)
from . import cutedsl_backend as _cutedsl_backend  # noqa: F401  import registers it (no cutlass)
from . import python_backend as _python_backend  # noqa: F401  import registers it
from . import triton_backend as _triton_backend  # noqa: F401  import registers it (no triton)

__all__ = [
    "Artifact",
    "Backend",
    "PROBE_ORDER",
    "register_backend",
    "get_backend",
    "registered_backends",
    "select_backend",
    "Escalation",
    "EscalationReason",
    "cpp_kernel",
    "escalations",
    "clear_escalations",
]
