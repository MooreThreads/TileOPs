import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from tileops.utils import get_sm_version
from workloads.attention.deepseek import MlaDecodeWorkload


class MlaDecodeTest(MlaDecodeWorkload, TestBase):
    pass


class MlaDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune",
            [
                pytest.param(
                    32, 128, 1, 8192, 512, 64, torch.float16, False, marks=pytest.mark.smoke
                ),
            ],
        ),
    ]


@MlaDecodeFixture
def test_mla_decode(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_kv: int,
    dim: int,
    dim_pe: int,
    dtype: torch.dtype,
    tune: bool,
):
    if get_sm_version() not in (31, 90):
        pytest.skip("requires Hopper")
    test = MlaDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, tune=tune
    )
    test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-3)
