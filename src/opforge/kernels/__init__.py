"""L1 · operator implementations, organized into directories by domain.

Importing this package registers all built-in operators (each of them a
:class:`~opforge.kernel.handle.KernelHandle`).

    kernels/
      norm/          normalization
      elementwise/   elementwise
      reduction/     reduction
      activation/    activation (to be added)
      quant/         quantization (to be added)
      gemm/          matmul (to be added)
      moe/           mixture of experts (to be added)
      attention/     attention (to be added)
"""

from . import elementwise as elementwise
from . import norm as norm
from . import reduction as reduction

__all__ = ["norm", "elementwise", "reduction"]
