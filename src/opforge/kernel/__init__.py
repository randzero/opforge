"""L1 · Kernel layer: operator declaration and the call pipeline."""

from .decorator import kernel
from .handle import KernelHandle
from .spec import KernelSpec, ParamSpec, describe_params

__all__ = ["kernel", "KernelHandle", "KernelSpec", "ParamSpec", "describe_params"]
