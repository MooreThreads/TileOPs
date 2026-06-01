import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.rms_norm import RMSNormFwdOp
from tileops.utils import get_backend_name
from workloads.rms_norm import RMSNormTest as _RMSNormTestWorkload

DEVICE = get_backend_name()


class RMSNormTest(_RMSNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_f32 / rms) * weight.float()).to(x.dtype)


class RMSNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype, tune", [
            # Standard aligned shapes (AC required)
            pytest.param(1024, 4096, torch.float16, False, marks=[pytest.mark.smoke, pytest.mark.packaging]),
            pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(8192, 8192, torch.float16, False, marks=pytest.mark.full),
            pytest.param(8192, 8192, torch.bfloat16, False, marks=pytest.mark.full),
            # Non-aligned N (AC required)
            pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
            # Tail-M: M not divisible by block_m (proves T.copy partial block safety)
            pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@RMSNormFixture
def test_rms_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = RMSNormTest(m, n, dtype)
    op = RMSNormFwdOp(normalized_shape=(n,), dtype=dtype)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    rtol = atol
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class RMSNormNonContigFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@RMSNormNonContigFixture
def test_rms_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=DEVICE)
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device=DEVICE)

    op = RMSNormFwdOp(normalized_shape=(n,), dtype=dtype)

    # Reference on contiguous copy
    eps = 1e-6
    x_ref = x.contiguous()
    x_f32 = x_ref.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"Non-contiguous test failed, max err: {(y - y_ref).abs().max()}"


class RMSNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@RMSNorm3DFixture
def test_rms_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=DEVICE)
    weight = torch.randn(hidden, dtype=dtype, device=DEVICE)

    op = RMSNormFwdOp(normalized_shape=(hidden,), dtype=dtype)

    # Reference
    eps = 1e-6
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    y = op(x, weight)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, y_ref, atol=atol, rtol=atol), \
        f"3D test failed, max err: {(y - y_ref).abs().max()}"


@pytest.mark.smoke
def test_rms_norm_rejects_shape_mismatch() -> None:
    op = RMSNormFwdOp(M=32, N=64, dtype=torch.float16)
    x = torch.randn(32, 63, dtype=torch.float16, device=DEVICE)
    weight = torch.randn(64, dtype=torch.float16, device=DEVICE)

    with pytest.raises(ValueError, match="Expected hidden dim"):
        op(x, weight)


@pytest.mark.smoke
def test_rms_norm_rejects_non_backend_tensor() -> None:
    op = RMSNormFwdOp(M=32, N=64, dtype=torch.float16)
    x = torch.randn(32, 64, dtype=torch.float16, device="cpu")
    weight = torch.randn(64, dtype=torch.float16, device=DEVICE)

    with pytest.raises(ValueError, match="MUSA tensor"):
        op(x, weight)


class RMSNormMUSASmokeFixture(FixtureBase):
    PARAMS = [
        ("shape, dtype", [
            pytest.param((256, 1024), torch.float16, marks=pytest.mark.smoke, id="2d-fp16"),
            pytest.param((256, 1024), torch.bfloat16, marks=pytest.mark.smoke, id="2d-bf16"),
            pytest.param((2, 128, 1024), torch.float16, marks=pytest.mark.smoke, id="3d-fp16"),
        ]),
    ]


@RMSNormMUSASmokeFixture
def test_rms_norm_musa_smoke_proof(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Small, explicit MUSA proof cases that finish faster than the full suite."""
    *leading, hidden = shape
    rows = 1
    for dim in leading:
        rows *= dim

    x = torch.randn(*shape, dtype=dtype, device=DEVICE)
    weight = torch.randn(hidden, dtype=dtype, device=DEVICE)

    op = RMSNormFwdOp(M=rows, N=hidden, dtype=dtype)
    y = op(x, weight)

    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    y_ref = ((x_f32 / rms) * weight.float()).to(dtype)

    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=atol, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
