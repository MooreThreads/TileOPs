from tileops.utils import get_backend_name

DEVICE = get_backend_name()
import torch

from workloads.workload_base import WorkloadBase


class FFTTest(WorkloadBase):

    def __init__(self, n: int, dtype: torch.dtype, batch_shape: tuple = ()):
        self.n = n
        self.dtype = dtype
        self.batch_shape = batch_shape

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.batch_shape, self.n, device=DEVICE, dtype=self.dtype)
        return (x,)
