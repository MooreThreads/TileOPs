"""Helpers shared by row-wise normalization ops.

BatchNorm uses spatial reduction rather than dim=-1, so it does not
inherit from this module's ``RowNormOp`` helper.
"""

import math
from abc import abstractmethod
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.utils import backend_tensor_error, is_backend_tensor

from ..op_base import Op

__all__ = ["ALIGNMENT", "RowNormOp"]

ALIGNMENT = 256


def align_up(n: int, alignment: int) -> int:
    """Return ``n`` rounded up to the next multiple of ``alignment``."""
    return ((n + alignment - 1) // alignment) * alignment


def normalized_shape_to_n(normalized_shape: Sequence[int]) -> int:
    """Return the product of ``normalized_shape``.

    Args:
        normalized_shape: Manifest-style trailing-axis shape; must be
            non-empty.

    Returns:
        Product of every entry in ``normalized_shape``.

    Raises:
        ValueError: If ``normalized_shape`` is empty.
    """
    normalized_shape = tuple(int(d) for d in normalized_shape)
    if len(normalized_shape) == 0:
        raise ValueError("normalized_shape must be non-empty")
    return math.prod(normalized_shape)


class RowNormOp(Op):
    """Base class for row-wise norm ops with a single hidden dimension."""

    _kernel_key: str
    _kernel_cls: type

    def __init__(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
        eps: float = 1e-6,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.M = M
        self.N = N
        self.dtype = dtype
        self.eps = eps
        self.N_padded = self._align_up(N, ALIGNMENT)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._kernel_key](
            M, N, eps, dtype, tune=tune,
        )

    @staticmethod
    def _align_up(n: int, alignment: int) -> int:
        """Round *n* up to the nearest multiple of *alignment*."""
        return ((n + alignment - 1) // alignment) * alignment

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {self._kernel_key: self._kernel_cls}

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = self.dtype.itemsize
        return (
            4 * self.M * self.N,
            (2 * self.M * self.N + self.N) * elem_bytes,
        )

    # -- Validation helpers --------------------------------------------------

    def _validate_cuda_dtype(self, name: str, t: torch.Tensor) -> None:
        """Raise ValueError if *t* is not on the active backend or has wrong dtype."""
        if not is_backend_tensor(t):
            raise ValueError(backend_tensor_error(name))
        if t.dtype != self.dtype:
            raise ValueError(
                f"Expected {name}.dtype {self.dtype}, got {t.dtype}"
            )

    def _validate_1d(self, name: str, t: torch.Tensor) -> None:
        """Raise ValueError if *t* is not 1-D with size N."""
        if t.ndim != 1:
            raise ValueError(f"Expected {name} to be 1D, got {t.ndim}D")
        if t.shape[0] != self.N:
            raise ValueError(
                f"Expected {name} dim {self.N}, got {t.shape[0]}"
            )

    def _validate_hidden_dim(self, x: torch.Tensor) -> None:
        """Raise ValueError if x's last dim != N."""
        if x.shape[-1] != self.N:
            raise ValueError(
                f"Expected hidden dim {self.N}, got {x.shape[-1]}"
            )

    # -- Reshape / pad helpers -----------------------------------------------

    def _flatten_and_check_M(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten leading dims to (M, N), validate M, return contiguous 2-D."""
        x = x.contiguous().reshape(-1, self.N)
        if x.shape[0] != self.M:
            raise ValueError(
                f"Expected M={self.M} (product of leading dims), got {x.shape[0]}"
            )
        return x

    def _pad_row(self, t: torch.Tensor) -> torch.Tensor:
        """Pad a 2-D row tensor along dim=-1 to N_padded."""
        return F.pad(t, (0, self.N_padded - self.N))

    def _pad_vec(self, t: torch.Tensor) -> torch.Tensor:
        """Pad a 1-D vector to N_padded."""
        return F.pad(t, (0, self.N_padded - self.N))

    @property
    def _needs_pad(self) -> bool:
        return self.N_padded != self.N

    def _trim_and_reshape(
        self, y: torch.Tensor, orig_shape: tuple
    ) -> torch.Tensor:
        """Trim padding from dim=-1 and restore *orig_shape*."""
        if self._needs_pad:
            y = y[:, : self.N]
        return y.reshape(orig_shape)

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> object:
        """Subclasses must implement their own forward with concrete args."""
