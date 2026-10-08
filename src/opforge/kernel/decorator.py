"""L1 · the ``@kernel`` decorator.

``contract`` is a **required argument with no default**. This is deliberate: it forces
the author to think through the operator's numerical semantics the moment they write it
down, rather than leaving a blank for someone else to guess at.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..cache.store import KernelCache
from ..contract import Contract, Precondition
from ..errors import KernelSpecError
from ..launch.launcher import Launcher
from .handle import KernelHandle
from .spec import KernelSpec, capture_impl_sources, describe_params


def kernel(
    *,
    name: str,
    contract: Contract,
    preconditions: Sequence[Precondition] = (),
    variants: Mapping[Contract, Callable] | None = None,
    reference: Callable | None = None,
    backward: Callable | None | str = None,
    supports_dynamic_shape: bool = False,
    min_alignment: int | None = None,
    backend: str = "auto",
    tags: Sequence[str] = (),
    cache: KernelCache | None = None,
    launcher: Launcher | None = None,
) -> Callable[[Callable], KernelHandle]:
    """Declare an operator.

    Args:
        name: globally unique operator name; also the cache namespace.
        contract: the **default contract** (required). Callers may override it with
            ``contract=``, but only towards a stricter tier.
        preconditions: entry preconditions; when unmet the operator returns ``Unavailable``.
        variants: implementations for the other contract tiers, ``{Contract: callable}``.
            Must not duplicate ``contract``.
        reference: pure PyTorch ground truth. When left empty, looked up lazily by the
            ``<name>_ref`` naming convention.
        backward: hand-written backward; ``"auto"`` means derive it from the reference's
            autograd.
        supports_dynamic_shape: whether shapes may contain symbolic dimensions
            (conservative default ``False``).
        min_alignment: element alignment requirement; unmet it is caught by a precondition.
        backend: ``"auto"`` or the name of a registered backend.
        tags: for CI grouping / filtering.
        cache: inject a cache instance (defaults to the process-level cache).
        launcher: inject a launcher (defaults to the process-level launcher).

    Returns:
        :class:`KernelHandle`.

    Raises:
        KernelSpecError: the declaration is invalid (wrong contract type, a variant
            duplicating the default contract, an empty name, etc.).
    """
    if not isinstance(contract, Contract):
        raise KernelSpecError(f"contract must be a Contract, got {contract!r}")
    if not name:
        raise KernelSpecError("kernel name must be a non-empty string")
    if min_alignment is not None and min_alignment <= 0:
        raise KernelSpecError(f"{name}: min_alignment must be positive, got {min_alignment}")

    variants_map: dict[Contract, Callable] = dict(variants or {})
    if contract in variants_map:
        raise KernelSpecError(
            f"{name}: the default contract {contract.name} must not also appear in "
            f"`variants` — the decorated function already is that implementation"
        )
    for variant_contract in variants_map:
        if not isinstance(variant_contract, Contract):
            raise KernelSpecError(
                f"{name}: variant keys must be Contract members, got {variant_contract!r}"
            )

    def decorate(fn: Callable) -> KernelHandle:
        spec = KernelSpec(
            name=name,
            default_impl=fn,
            contract=contract,
            params=describe_params(fn),
            preconditions=tuple(preconditions),
            variants=variants_map,
            reference=reference,
            backward=backward,
            supports_dynamic_shape=supports_dynamic_shape,
            min_alignment=min_alignment,
            backend=backend,
            tags=tuple(tags),
            module=getattr(fn, "__module__", ""),
            # Capture the source at declaration time: a DSL compiler rewrites the
            # function object later, so reading it on demand would yield different text.
            impl_sources=capture_impl_sources(name, fn, variants_map),
        )
        handle = KernelHandle(spec, cache=cache, launcher=launcher)

        # Make the operator object "look like" the decorated function under introspection.
        handle.__doc__ = fn.__doc__
        handle.__name__ = getattr(fn, "__name__", name)
        handle.__qualname__ = getattr(fn, "__qualname__", name)
        handle.__module__ = getattr(fn, "__module__", "")
        return handle

    return decorate
