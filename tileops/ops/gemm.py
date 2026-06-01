from typing import Dict, Optional

import torch

from tileops.kernels.gemm import GemmKernel, GemvKernel
from tileops.kernels.kernel_base import Kernel
from tileops.utils import backend_tensor_error, get_sm_version, is_backend_tensor

from .op_base import Op

__all__ = ["GemmOp"]


class GemmOp(Op):

    def __init__(self,
                 m: int,
                 n: int,
                 k: int,
                 trans_a: bool = False,
                 trans_b: bool = False,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.M = m
        self.N = n
        self.K = k
        self.trans_a = trans_a
        self.trans_b = trans_b

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.use_gemv_kernel = False
        self.gemv_mode = ""

        is_gemv_lhs_row_case = (m == 1 and not self.trans_a and self.trans_b)
        is_gemv_rhs_col_case = (n == 1 and not self.trans_a and not self.trans_b)
        gemv_kernel_type = self.kernel_map.get("gemv_kernel")
        if (is_gemv_lhs_row_case or is_gemv_rhs_col_case) and gemv_kernel_type is not None:
            self.use_gemv_kernel = True
            if is_gemv_lhs_row_case:
                self.gemv_mode = "lhs_row"
                self.kernel = gemv_kernel_type(n, k, self.dtype, tune=tune)
            else:
                self.gemv_mode = "rhs_col"
                self.kernel = gemv_kernel_type(m, k, self.dtype, tune=tune)

        if not self.use_gemv_kernel:
            self.kernel = self.kernel_map["gemm_kernel"](
                m, n, k, self.dtype, tune=tune, trans_a=self.trans_a, trans_b=self.trans_b)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernels = {"gemm_kernel": GemmKernel}
        if get_sm_version() in GemvKernel.supported_archs:
            kernels["gemv_kernel"] = GemvKernel
        return kernels

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not is_backend_tensor(a):
            raise ValueError(backend_tensor_error("a"))
        if not is_backend_tensor(b):
            raise ValueError(backend_tensor_error("b"))
        if a.dtype != self.dtype:
            raise ValueError(f"Expected a.dtype {self.dtype}, got {a.dtype}")
        if b.dtype != self.dtype:
            raise ValueError(f"Expected b.dtype {self.dtype}, got {b.dtype}")
        if a.ndim != 2:
            raise ValueError(f"Expected a to be 2D, got {a.ndim}D")
        if b.ndim != 2:
            raise ValueError(f"Expected b to be 2D, got {b.ndim}D")

        expected_a_shape = (self.K, self.M) if self.trans_a else (self.M, self.K)
        expected_b_shape = (self.N, self.K) if self.trans_b else (self.K, self.N)
        if tuple(a.shape) != expected_a_shape:
            raise ValueError(
                f"Expected a.shape {expected_a_shape}, got {tuple(a.shape)}"
            )
        if tuple(b.shape) != expected_b_shape:
            raise ValueError(
                f"Expected b.shape {expected_b_shape}, got {tuple(b.shape)}"
            )

        if self.use_gemv_kernel:
            if self.gemv_mode == "lhs_row":
                a_vec = a.reshape(-1)
                return self.kernel(a_vec, b).reshape(1, self.N)
            elif self.gemv_mode == "rhs_col":
                b_vec = b.reshape(-1)
                return self.kernel(b_vec, a).reshape(self.M, 1)
        return self.kernel(a, b)
