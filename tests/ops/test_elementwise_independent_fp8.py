"""Tests for FP8 dtype boundaries in independent elementwise kernels."""

import pytest
import torch

from tileops.utils import get_backend_name

DEVICE = get_backend_name()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_where_rejects_fp8_dtype(bad_dtype: torch.dtype) -> None:
    """WhereFwdOp must reject fp8 dtypes (manifest contract).

    The element type arrives with the tensors, so the rejection does too.
    """
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    op = WhereFwdOp()
    cond = torch.zeros(shape, device=DEVICE, dtype=torch.bool)
    x = torch.zeros(shape, device=DEVICE).to(bad_dtype)
    with pytest.raises((ValueError, TypeError)):
        op(cond, x, x)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_where_accepts_manifest_dtypes(dtype: torch.dtype) -> None:
    """WhereFwdOp constructs and runs for every manifest-declared dtype."""
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.randint(0, 2, shape, device=DEVICE).bool()
    inp = torch.randn(shape, device=DEVICE, dtype=dtype)
    other = torch.randn(shape, device=DEVICE, dtype=dtype)
    op = WhereFwdOp()
    out = op(cond, inp, other)
    ref = torch.where(cond, inp, other)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
