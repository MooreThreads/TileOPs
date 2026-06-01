from tileops.utils import get_backend_name

DEVICE = get_backend_name()
import torch

from workloads.workload_base import WorkloadBase


class FusedAddLayerNormTest(WorkloadBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=DEVICE)
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device=DEVICE)
        weight = torch.randn(self.n, dtype=self.dtype, device=DEVICE)
        bias = torch.randn(self.n, dtype=self.dtype, device=DEVICE)
        return x, residual, weight, bias
