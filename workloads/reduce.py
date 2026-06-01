from tileops.utils import get_backend_name

DEVICE = get_backend_name()
import torch

from workloads.workload_base import RandnTest, WorkloadBase


class SumTest(RandnTest):
    """Workload definition for SumFwdOp."""


class MeanTest(RandnTest):
    """Workload definition for MeanFwdOp."""


class AmaxTest(RandnTest):
    """Workload definition for AmaxFwdOp."""


class AminTest(RandnTest):
    """Workload definition for AminFwdOp."""


class ProdTest(WorkloadBase):
    """Workload definition for ProdFwdOp.

    Uses small-range values (0.99..1.0) to avoid overflow in product reduction.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.rand(*self.shape, dtype=self.dtype, device=DEVICE) * 0.01 + 0.99
        return (x,)


class StdTest(RandnTest):
    """Workload definition for StdFwdOp."""


class VarTest(RandnTest):
    """Workload definition for VarFwdOp."""


class VarMeanTest(RandnTest):
    """Workload definition for VarMeanFwdOp."""
