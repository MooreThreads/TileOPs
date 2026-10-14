import torch

from tileops.utils import get_backend_name
from workloads.workload_base import WorkloadBase, benchmark_compare_active

DEVICE = get_backend_name()


class FFTWorkload(WorkloadBase):
    def __init__(self, n: int, dtype: torch.dtype, batch_shape: tuple = ()):
        self.n = n
        self.dtype = dtype
        self.batch_shape = batch_shape

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.batch_shape, self.n, device=DEVICE, dtype=self.dtype)
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if benchmark_compare_active():
            return torch.fft.fft(x, dim=-1)
        return torch.fft.fft(x.cpu(), dim=-1).to(x.device)
