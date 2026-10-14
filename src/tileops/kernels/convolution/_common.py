"""What the 1d, 2d and 3d kernels share: the autotune search space and the launch path."""

import itertools
from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_backend_name, get_sm_version

# Panel width handed to ``T.use_swizzle``: the number of blocks along the grid's fast
# axis reordered together so their tiles share L2. A power of two, because the block
# remap divides by it and a divide the compiler cannot turn into a shift shows on a
# short kernel; the narrowest one, because the long 1d grids lose monotonically as it
# widens. Whether to swizzle at all is a searched config.
CONV_SWIZZLE_PANEL = 2


def conv_num_stages() -> int:
    """Pipeline depth this target's shared memory holds: three on SM90, two before."""
    return 3 if get_sm_version() == 90 else 2


def conv_autotune_configs(
    dtype,
    *,
    block_m=(32, 64, 128),
    block_n=(64, 128, 256),
    block_k=(32, 64, 128),
    num_stages=(2, 3),
    threads=(128, 256),
    enable_rasterization=(False, True),
) -> list[dict]:
    """Search space filtered to combinations that fit in shared memory.

    ``enable_rasterization`` turns on a swizzle that orders blocks for L2 locality. It
    is searched rather than fixed because which way wins follows the grid a shape
    produces, and both ways win on some shapes. Callers narrow the other axes to keep
    the search the size it was before this one joined it.
    """
    limit = get_shared_memory_limit_bytes()
    valid = []
    for bm, bn, bk, ns, th, rast in itertools.product(
        block_m,
        block_n,
        block_k,
        num_stages,
        threads,
        enable_rasterization,
    ):
        if conv_shared_memory_bytes(bm, bn, bk, ns, dtype) > limit:
            continue
        valid.append(
            {
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "num_stages": ns,
                "threads": th,
                "enable_rasterization": rast,
            }
        )
    return valid


def get_shared_memory_limit_bytes() -> int:
    backend = getattr(torch, get_backend_name())
    return backend.get_device_properties(backend.current_device()).shared_memory_per_block_optin


def _fit_block_k(block_k: int, limit: int) -> int:
    for candidate in (128, 64, 32, 16, 8, 4, 2, 1):
        if candidate <= block_k and candidate <= limit and limit % candidate == 0:
            return candidate
    return limit


def _fit_block_n(block_n: int, limit: int) -> int:
    for candidate in (256, 128, 64, 32, 16, 8):
        if candidate <= block_n and candidate <= limit:
            return candidate
    return limit


def fit_conv_config(
    config: dict,
    *,
    block_m_limit: int,
    block_n_limit: int,
    block_k_limit: int,
    block_m_alignment: int = 1,
) -> dict:
    fitted = dict(config)
    fitted["block_m"] = min(fitted["block_m"], block_m_limit)
    if fitted["block_m"] % block_m_alignment:
        fitted["block_m"] = (
            (fitted["block_m"] + block_m_alignment - 1) // block_m_alignment
        ) * block_m_alignment
    fitted["block_n"] = _fit_block_n(fitted["block_n"], block_n_limit)
    fitted["block_k"] = _fit_block_k(fitted["block_k"], block_k_limit)
    return fitted


def conv_shared_memory_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    per_stage_bytes = (block_m * block_k + block_k * block_n) * dtype_bytes
    out_shared_bytes = block_m * block_n * dtype_bytes
    return per_stage_bytes * max(1, num_stages) + out_shared_bytes


def _launch(
    kernel: Kernel,
    *tensors: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run *kernel*'s compiled program, passing *bias* only when the call carries one.

    The program was traced for one side of ``has_bias``: the no-bias variant has no bias
    parameter at all, so what the call carries has to agree with what was built.

    Args:
        kernel: The kernel whose ``self.kernel`` builder and ``self.config`` are used.
        *tensors: The program's inputs in prim_func order, bias excluded.
        bias: The bias this call carries, or ``None``.

    Returns:
        The output tensor the program allocated.

    Raises:
        ValueError: The call's bias presence differs from the one the kernel was built for.
    """
    if (bias is not None) != kernel.has_bias:
        built, given = ("with", "without") if kernel.has_bias else ("without", "with")
        raise ValueError(
            f"{type(kernel).__name__} was built {built} a bias and was called {given} one; "
            f"bias presence is part of what the program is compiled for, so the op layer "
            f"builds one kernel per side"
        )
    name = type(kernel).__name__
    if name in {"Conv1dKernel", "Conv1dPointwiseKernel"}:
        config = fit_conv_config(
            kernel.config,
            block_m_limit=kernel.c_out,
            block_n_limit=kernel.out_l,
            block_k_limit=kernel.k_total,
        )
    elif name == "GroupConv1dKernel":
        out_l = (
            kernel.l_in
            + kernel.pad_left
            + kernel.pad_right
            - kernel.dilation_l * (kernel.kernel_l - 1)
            - 1
        ) // kernel.stride_l + 1
        config = fit_conv_config(
            kernel.config,
            block_m_limit=kernel.c_out_g,
            block_n_limit=out_l,
            block_k_limit=kernel.c_in_g * kernel.kernel_l,
            block_m_alignment=16 if not kernel.use_direct else 1,
        )
    elif name == "Conv2dSymmetricKernel":
        config = fit_conv_config(
            kernel.config,
            block_m_limit=kernel.m,
            block_n_limit=kernel.c_out,
            block_k_limit=kernel.k_total,
        )
    elif name == "Conv2d1x1Kernel":
        config = fit_conv_config(
            kernel.config,
            block_m_limit=kernel.c_out,
            block_n_limit=kernel.h * kernel.w,
            block_k_limit=kernel.c_in,
        )
    elif name in {"Conv2dKernel", "GroupConv2dKernel"}:
        config = fit_conv_config(
            kernel.config,
            block_m_limit=kernel.c_out_g if name == "GroupConv2dKernel" else kernel.c_out,
            block_n_limit=kernel.out_h * kernel.out_w,
            block_k_limit=kernel.k_total,
            block_m_alignment=8 if name == "GroupConv2dKernel" and not kernel.use_direct else 1,
        )
    elif name in {"Conv3dKernel", "GroupConv3dKernel"}:
        config = fit_conv_config(
            kernel.config,
            block_m_limit=kernel.c_out_g if name == "GroupConv3dKernel" else kernel.c_out,
            block_n_limit=kernel.out_d * kernel.out_h * kernel.out_w,
            block_k_limit=kernel.k_total,
            block_m_alignment=8 if name == "GroupConv3dKernel" else 1,
        )
    else:
        config = kernel.config
    program = kernel.kernel(
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["num_stages"],
        config["threads"],
        config["enable_rasterization"],
    )
    if bias is None:
        return program(*tensors)
    return program(*tensors, bias)
