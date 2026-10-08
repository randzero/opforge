"""L3 · inference-side adaptation.

Register opforge operators as ``torch.library`` custom operators (the ``opforge``
namespace), so they can be traced by ``torch.compile``; optionally register a **fake**
(shape inference only, no kernel run).

Adaptations for **execution modes** such as CUDA Graph, dynamic shapes, and paged KV
belong to this layer only and do not sink down into L1/L2.

This layer **must not** be depended upon in reverse by L0-L2 (invariant I2).
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401  (reserved for later backend extensions)
from typing import Any, Callable, Sequence

from ..contract import Contract
from ..errors import KernelSpecError
from ..kernel import KernelHandle
from ..kernel.spec import KernelSpec
from ..result import Unavailable

_PY_TO_TORCH: dict[type, str] = {int: "int", float: "float", bool: "bool", str: "str"}


def _require_torch():  # noqa: ANN202 - the return value is the torch module
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "opforge.inference requires torch; install `opforge[inference]`"
        ) from exc
    return torch


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def _type_name(param: Any) -> str:
    """Without an annotation, infer from the default value's type; Tensor when no default."""
    if param.has_default and param.default is not None:
        return _PY_TO_TORCH.get(type(param.default), "Tensor")
    return "Tensor"


def infer_schema(
    spec: KernelSpec,
    *,
    mutates_args: Sequence[str] = (),
    returns: str = "Tensor",
) -> str:
    """Infer a TorchScript schema from :class:`KernelSpec`'s parameter table."""
    mutable = set(mutates_args)
    pieces: list[str] = []
    keyword_marker_added = False

    for param in spec.params:
        if param.kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            raise KernelSpecError(
                f"{spec.name}: *args / **kwargs cannot be expressed as a custom op schema"
            )
        if param.kind == "KEYWORD_ONLY" and not keyword_marker_added:
            pieces.append("*")
            keyword_marker_added = True

        type_name = _type_name(param)
        if param.name in mutable:
            type_name = f"{type_name}(a!)"

        piece = f"{type_name} {param.name}"
        if param.has_default and param.default is not None:
            piece += f"={_literal(param.default)}"
        pieces.append(piece)

    return f"({', '.join(pieces)}) -> {returns}"


def to_custom_op(
    op: KernelHandle,
    *,
    variant: Contract = Contract.FAST,
    mutates_args: Sequence[str] = (),
    fake: Callable | None = None,
    returns: str = "Tensor",
    namespace: str = "opforge",
) -> Any:
    """Register ``op`` as a custom operator visible to ``torch.compile``.

    Args:
        op: an operator handle.
        variant: the contract level used on the inference side (default ``FAST``).
        mutates_args: names of arguments that are **written in place**. Must be
            declared truthfully -- a wrong declaration breaks functionalization at
            compile time (this is an L3 contract, not an L0-L2 concern).
        fake: an optional fake implementation; it only infers shapes and does not take
            part in contract selection.
        returns: the schema's return type.
        namespace: the torch library namespace.

    Returns:
        The ``torch.library.Library`` object used for registration (must be kept alive).
    """
    torch = _require_torch()
    library = torch.library.Library(namespace, "FRAGMENT")
    library.define(f"{op.name}{infer_schema(op.spec, mutates_args=mutates_args, returns=returns)}")

    def _impl(*args: Any, **kwargs: Any) -> Any:
        out = op(*args, contract=variant, **kwargs)
        if isinstance(out, Unavailable):
            return op.ref(*args, **kwargs)
        return out

    library.impl(op.name, _impl, "CompositeExplicitAutograd")

    fake_impl = fake if fake is not None else _default_fake(returns)
    if fake_impl is not None:
        library._register_fake(op.name, fake_impl)
        fake_impl.__opforge_fake__ = True  # type: ignore[attr-defined]
    return library


def _default_fake(returns: str):
    """Provide a default fake for operators where "output shape = the first tensor
    argument's shape".

    **An operator without a fake cannot be traced by ``torch.compile``** -- during
    tracing all tensors are meta, so without a fake it will run the real implementation
    and blow up on device/memory. So a conservative default is given here; operators
    with more complex shape rules should pass ``fake=`` explicitly.
    """
    if returns != "Tensor":
        return None

    def _fake(*args: Any, **kwargs: Any) -> Any:
        for candidate in args:
            if hasattr(candidate, "new_empty") and hasattr(candidate, "dtype"):
                return candidate.new_empty(candidate.shape)
        raise AssertionError(
            "cannot derive a fake output shape: no tensor-like argument; pass fake= explicitly"
        )

    return _fake
