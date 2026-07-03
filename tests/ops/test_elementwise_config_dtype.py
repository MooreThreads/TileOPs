"""Regression tests for elementwise dtype/config consistency."""

from unittest.mock import patch

import pytest
import torch

from tileops.kernels.elementwise import (
    AbsFwdKernel,
    AddFwdKernel,
    EluFwdKernel,
    EqFwdKernel,
    GeFwdKernel,
    GtFwdKernel,
    HardtanhFwdKernel,
    # Independent kernels (custom-signature)
    LeakyReluFwdKernel,
    LeFwdKernel,
    LogicalAndFwdKernel,
    LogicalOrFwdKernel,
    LtFwdKernel,
    NeFwdKernel,
    PowFwdKernel,
    PreluFwdKernel,
    ReluFwdKernel,
    SiluAndMulFwdKernel,
    SoftplusFwdKernel,
)

# ---------------------------------------------------------------------------
# Fix 1: npt 3-way check in default_config
# ---------------------------------------------------------------------------


INDEPENDENT_KERNELS_SIMPLE = [LeakyReluFwdKernel, HardtanhFwdKernel]


@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_npt"),
    [
        (torch.float32, 4),
        (torch.float16, 8),
        (torch.bfloat16, 8),
        (torch.float8_e4m3fn, 16),
        (torch.float8_e5m2, 16),
    ],
)
@pytest.mark.parametrize("kernel_cls", INDEPENDENT_KERNELS_SIMPLE)
def test_independent_kernels_use_expected_default_npt(kernel_cls, dtype, expected_npt):
    """Representative independent kernels should preserve dtype-driven npt defaults."""
    kernel = kernel_cls.__new__(kernel_cls)
    kernel.dtype = dtype
    assert kernel.default_config["num_per_thread"] == expected_npt


@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_threads", "expected_npt"),
    [
        (torch.float32, 256, 4),
        (torch.float16, 128, 2),
        (torch.bfloat16, 128, 2),
        (torch.float8_e4m3fn, 256, 16),
        (torch.float8_e5m2, 256, 16),
    ],
)
@pytest.mark.parametrize("kernel_cls", [EluFwdKernel, SoftplusFwdKernel])
def test_elu_softplus_use_measured_dtype_default_config(
    kernel_cls, dtype, expected_threads, expected_npt,
):
    kernel = kernel_cls.__new__(kernel_cls)
    kernel.dtype = dtype
    assert kernel.default_config["threads"] == expected_threads
    assert kernel.default_config["num_per_thread"] == expected_npt


@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_threads", "expected_npt"),
    [
        (torch.float32, 256, 4),
        (torch.float16, 128, 8),
        (torch.bfloat16, 128, 2),
        (torch.float8_e4m3fn, 256, 16),
        (torch.float8_e5m2, 256, 16),
    ],
)
def test_prelu_uses_measured_dtype_default_config(dtype, expected_threads, expected_npt):
    kernel = PreluFwdKernel.__new__(PreluFwdKernel)
    kernel.dtype = dtype
    assert kernel.default_config["threads"] == expected_threads
    assert kernel.default_config["num_per_thread"] == expected_npt


@pytest.mark.full
def test_prelu_bf16_small_inner_size_uses_measured_default_config():
    kernel = PreluFwdKernel.__new__(PreluFwdKernel)
    kernel.dtype = torch.bfloat16
    kernel.inner_size = 28 * 28
    assert kernel.default_config["threads"] == 512
    assert kernel.default_config["num_per_thread"] == 2


@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_threads", "expected_npt"),
    [
        (torch.float32, 256, 4),
        (torch.float16, 256, 2),
        (torch.bfloat16, 256, 2),
    ],
)
def test_relu_uses_measured_dtype_default_config(dtype, expected_threads, expected_npt):
    kernel = ReluFwdKernel.__new__(ReluFwdKernel)
    kernel.dtype = dtype
    kernel.strategy = "register_copy"
    assert kernel.default_config["threads"] == expected_threads
    assert kernel.default_config["num_per_thread"] == expected_npt


# ---------------------------------------------------------------------------
# Fix 2: OUTPUT_DTYPE consistency (all should be torch.dtype, not string)
# ---------------------------------------------------------------------------


COMPARISON_KERNELS = [EqFwdKernel, NeFwdKernel, GtFwdKernel, LtFwdKernel, GeFwdKernel, LeFwdKernel]
LOGICAL_BINARY_KERNELS = [LogicalAndFwdKernel, LogicalOrFwdKernel]


@pytest.mark.full
@pytest.mark.parametrize("kernel_cls", COMPARISON_KERNELS + LOGICAL_BINARY_KERNELS)
def test_bool_like_elementwise_kernels_expose_torch_dtype_output(kernel_cls):
    """Comparison and logical kernels should declare `OUTPUT_DTYPE` as `torch.int8`."""
    assert isinstance(kernel_cls.OUTPUT_DTYPE, torch.dtype), (
        f"{kernel_cls.__name__}.OUTPUT_DTYPE is {type(kernel_cls.OUTPUT_DTYPE).__name__} "
        f"({kernel_cls.OUTPUT_DTYPE!r}), expected torch.dtype"
    )
    assert torch.int8 == kernel_cls.OUTPUT_DTYPE


# ---------------------------------------------------------------------------
# Fix 3: output_dtype attribute on all three base kernel types
# ---------------------------------------------------------------------------


@pytest.mark.full
def test_unary_kernel_sets_output_dtype_in_init():
    """Unary kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(ReluFwdKernel, "_build_kernel", return_value=None),
        patch.object(ReluFwdKernel, "init_config"),
    ):
        kernel = ReluFwdKernel(N_total=1024, dtype=torch.float16)
    assert kernel.output_dtype == torch.float16


@pytest.mark.full
def test_binary_kernel_sets_output_dtype_in_init():
    """Binary kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(AddFwdKernel, "_build_kernel", return_value=None),
        patch.object(AddFwdKernel, "init_config"),
    ):
        kernel = AddFwdKernel(
            N_total=1024, dtype=torch.float16,
            coalesced_shape=(1024,), a_strides=(1,), b_strides=(1,),
            a_numel=1024, b_numel=1024,
        )
    assert kernel.output_dtype == torch.float16


@pytest.mark.full
def test_fused_gated_kernel_sets_output_dtype_in_init():
    """Fused-gated kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(SiluAndMulFwdKernel, "_build_kernel", return_value=None),
        patch.object(SiluAndMulFwdKernel, "init_config"),
    ):
        kernel = SiluAndMulFwdKernel(M=32, N=1024, dtype=torch.float16)
    assert kernel.output_dtype == torch.float16


@pytest.mark.full
def test_unary_default_config_preserves_strategy_npt_split():
    """Unary kernels should keep the explicit_parallel/register_copy npt split."""
    with (
        patch.object(AbsFwdKernel, "_build_kernel", return_value=None),
        patch.object(AbsFwdKernel, "init_config"),
    ):
        explicit = AbsFwdKernel(N_total=1024, dtype=torch.float16, strategy="explicit_parallel")
        register = AbsFwdKernel(N_total=1024, dtype=torch.float16, strategy="register_copy")
    assert explicit.default_config["num_per_thread"] == 4
    assert register.default_config["num_per_thread"] == 8


@pytest.mark.full
@pytest.mark.parametrize("kernel_cls", [AddFwdKernel, PowFwdKernel, GeFwdKernel])
def test_binary_default_config_preserves_strategy_npt_split(kernel_cls):
    """Binary kernels should keep the explicit_parallel/register_copy npt split."""
    common_kwargs = {
        "N_total": 1024,
        "dtype": torch.float16,
        "coalesced_shape": (1024,),
        "a_strides": (1,),
        "b_strides": (1,),
        "a_numel": 1024,
        "b_numel": 1024,
    }
    with (
        patch.object(kernel_cls, "_build_kernel", return_value=None),
        patch.object(kernel_cls, "init_config"),
    ):
        explicit = kernel_cls(strategy="explicit_parallel", **common_kwargs)
        register = kernel_cls(strategy="register_copy", **common_kwargs)
    assert explicit.default_config["num_per_thread"] == 4
    assert register.default_config["threads"] == 256
    assert register.default_config["num_per_thread"] == 8


@pytest.mark.full
def test_fused_gated_default_config_preserves_strategy_npt_split():
    """Fused-gated kernels should keep the direct/explicit_parallel npt split."""
    with (
        patch.object(SiluAndMulFwdKernel, "_build_kernel", return_value=None),
        patch.object(SiluAndMulFwdKernel, "init_config"),
    ):
        direct = SiluAndMulFwdKernel(M=32, N=1024, dtype=torch.float16, strategy="direct")
        explicit = SiluAndMulFwdKernel(M=32, N=1024, dtype=torch.float16, strategy="explicit_parallel")
    assert direct.default_config["num_per_thread"] == 8
    assert explicit.default_config["num_per_thread"] == 4
