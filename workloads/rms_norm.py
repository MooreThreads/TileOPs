import torch

from tileops.utils import get_backend_name
from workloads.workload_base import WorkloadBase


class RMSNormTest(WorkloadBase):

    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-6):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = get_backend_name()
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=device)
        weight = torch.randn(self.n, dtype=self.dtype, device=device)
        return x, weight
