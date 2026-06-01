from tileops.utils import get_backend_name

DEVICE = get_backend_name()
import torch

from workloads.workload_base import WorkloadBase


class MlaDecodeTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                 dim_pe: int, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_pe = dim_pe
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device=DEVICE, dtype=self.dtype)
        Q_pe = torch.randn(self.batch, self.heads, self.dim_pe, device=DEVICE, dtype=self.dtype)
        K = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim,
            device=DEVICE,
            dtype=self.dtype)
        K_pe = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim_pe,
            device=DEVICE,
            dtype=self.dtype)
        return Q, Q_pe, K, K_pe
