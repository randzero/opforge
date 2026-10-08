"""L2 · launch layer: cut "the fixed overhead of every launch" down to one dict lookup.

What is cached is "[**signature**] -> a directly executable call closure". It is a
different thing from the cache layer (compiled artifacts):

* the cache layer saves **compilation** (expensive, persisted, cross-process);
* the launch layer saves **the per-call binding overhead** (cheap, in-process, gone
  with the process).

Hard rule (invariant I5): **runtime dynamic dimensions must not enter the signature** --
numeric values such as token counts and sequence lengths entering the signature would
blow up the in-process dict and force repeated rebuilds under high traffic.

See `docs/architecture.md` §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping


def arg_signature(value: Any) -> tuple:
    """The signature of a single argument: only **dtype / shape / type**, **never the value**."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            dims = tuple(int(d) for d in shape)
        except (TypeError, ValueError):
            dims = ("?",)
        return ("tensor", str(getattr(value, "dtype", None)), dims)

    # scalar: record only the type, not the value -- the concrete embodiment of
    # "dynamic dimensions do not enter the signature".
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ("scalar", type(value).__name__)

    if isinstance(value, (list, tuple)):
        return ("seq", tuple(arg_signature(v) for v in value))

    if isinstance(value, Mapping):
        return ("map", tuple(sorted((k, arg_signature(v)) for k, v in value.items())))

    return ("other", type(value).__name__)


def default_signature(args: tuple, kwargs: Mapping[str, Any]) -> Hashable:
    """Default signature: signatures of positional args + signatures of kwargs sorted by name."""
    return (
        tuple(arg_signature(a) for a in args),
        tuple((k, arg_signature(kwargs[k])) for k in sorted(kwargs)),
    )


@dataclass
class LaunchSpec:
    """A single launch request."""

    kernel_id: str
    artifact: Any
    """The artifact produced by the cache layer; must provide ``run(args, kwargs)``."""

    runtime_args: tuple = ()
    runtime_kwargs: Mapping[str, Any] = field(default_factory=dict)
    compute_signature: Callable[[tuple, Mapping[str, Any]], Hashable] | None = None

    def signature(self) -> Hashable:
        fn = self.compute_signature or default_signature
        return (self.kernel_id, fn(self.runtime_args, self.runtime_kwargs))


class Launcher:
    """Cache "a directly executable call closure" per signature."""

    def __init__(self) -> None:
        self._runners: dict[Hashable, Callable[[tuple, Mapping[str, Any]], Any]] = {}

    def __call__(self, spec: LaunchSpec) -> Any:
        key = spec.signature()
        run = self._runners.get(key)
        if run is None:
            # What is cached is "the artifact entry resolved for this signature", **not**
            # this call's arguments -- the signature deliberately excludes scalar values,
            # and closing over the arguments would let calls with different epsilons, etc.,
            # contaminate each other.
            run = spec.artifact.run
            self._runners[key] = run
        return run(spec.runtime_args, spec.runtime_kwargs)

    @property
    def cached_signatures(self) -> int:
        """Number of cached signatures (for tests and metrics)."""
        return len(self._runners)


_default_launcher = Launcher()


def get_launcher() -> Launcher:
    """The process-wide default launcher."""
    return _default_launcher
