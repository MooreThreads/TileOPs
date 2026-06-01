import pytest
import torch
import torch.nn.functional as F

from tileops.ops.gemm import GemmOp
from tileops.ops.norm.layer_norm import LayerNormFwdOp
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.ops.reduction.softmax import SoftmaxFwdOp
from tileops.utils import get_backend_name, is_available, synchronize


pytestmark = pytest.mark.smoke

DEVICE = get_backend_name()


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


def test_musa_backend_subset_smoke() -> None:
    if not is_available():
        pytest.skip(f"{DEVICE} backend is unavailable")

    torch.manual_seed(1235)

    gemm_a = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
    gemm_b = torch.randn(64, 64, device=DEVICE, dtype=torch.float16)
    gemm = GemmOp(64, 64, 64, dtype=torch.float16)
    gemm_out = gemm(gemm_a, gemm_b)
    gemm_ref = torch.matmul(gemm_a, gemm_b)
    _assert_close(gemm_out, gemm_ref, atol=1e-3, rtol=1e-3)
    synchronize()

    norm_x = torch.randn(4, 128, device=DEVICE, dtype=torch.float16)
    norm_weight = torch.randn(128, device=DEVICE, dtype=torch.float16)
    norm_bias = torch.randn(128, device=DEVICE, dtype=torch.float16)
    layer_norm = LayerNormFwdOp(M=4, N=128, dtype=torch.float16)
    layer_norm_out = layer_norm(norm_x, norm_weight, norm_bias)
    layer_norm_ref = F.layer_norm(
        norm_x.float(),
        (128,),
        weight=norm_weight.float(),
        bias=norm_bias.float(),
        eps=1e-5,
    ).to(torch.float16)
    _assert_close(layer_norm_out, layer_norm_ref, atol=1e-3, rtol=1e-3)
    synchronize()

    rms_x = torch.randn(4, 128, device=DEVICE, dtype=torch.float16)
    rms_weight = torch.randn(128, device=DEVICE, dtype=torch.float16)
    rms_norm = RMSNormFwdOp(M=4, N=128, dtype=torch.float16)
    rms_out = rms_norm(rms_x, rms_weight)
    rms_x_f32 = rms_x.float()
    rms_ref = (
        rms_x_f32
        / torch.sqrt(rms_x_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        * rms_weight.float()
    ).to(torch.float16)
    _assert_close(rms_out, rms_ref, atol=1e-2, rtol=1e-2)
    synchronize()

    softmax_x = torch.randn(4, 128, device=DEVICE, dtype=torch.float16)
    softmax = SoftmaxFwdOp(dtype=torch.float16, dim=-1)
    softmax_out = softmax(softmax_x)
    softmax_ref = F.softmax(softmax_x.float(), dim=-1).to(torch.float16)
    _assert_close(softmax_out, softmax_ref, atol=1e-3, rtol=1e-3)
