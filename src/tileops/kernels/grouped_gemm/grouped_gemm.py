import functools
import itertools
import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.grouped_gemm.call import GroupedGemmCall
from tileops.kernels.grouped_gemm.grouped_gemm_persistent import grouped_gemm_entry
from tileops.kernels.grouped_tiling import GroupTiling
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_backend_name

__all__ = [
    "GroupedGemmKernel",
]

# Default configs per layout variant (transpose_a, transpose_b)
_DEFAULT_CONFIGS = {
    (False, True): {"block_m": 64, "block_n": 256, "block_k": 64, "num_stages": 2, "threads": 128},
    (False, False): {
        "block_m": 256,
        "block_n": 256,
        "block_k": 64,
        "num_stages": 2,
        "threads": 256,
    },
    (True, False): {"block_m": 32, "block_n": 128, "block_k": 128, "num_stages": 2, "threads": 128},
    (True, True): {"block_m": 64, "block_n": 256, "block_k": 64, "num_stages": 2, "threads": 128},
}


@functools.lru_cache(maxsize=32)
def _grouped_gemm_kernel(batch_sum, batch_count, N, K, transpose_a, transpose_b, dtype="float16"):
    accum_dtype = "float"

    # Region copies lower to the dense GEMM TMA only with these passes. The
    # same passes split a guarded element copy across warps, so they stay off
    # unless this exact shape takes the region path.
    rows_per_group, row_rem = divmod(batch_sum, batch_count)
    region_tma_passes = (
        get_backend_name() == "musa"
        and not transpose_b
        and row_rem == 0
        and (
            (not transpose_a and rows_per_group % 256 == 0 and N % 256 == 0 and K % 64 == 0)
            or (transpose_a and rows_per_group % 128 == 0 and N % 128 == 0 and K % 64 == 0)
        )
    )
    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: get_backend_name() != "musa",
    }
    if region_tma_passes:
        pass_configs[tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH] = False
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC] = True

    @tilelang.jit(
        out_idx=[2],
        pass_configs=pass_configs,
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _grouped_gemm_func(block_m, block_n, block_k, num_stages, threads):
        if not transpose_a:
            # NT / NN pattern: iterate over K, batch-offset lookup
            A_shape = (batch_sum, K)
            C_shape = (batch_sum, N)
            A_shared_shape = (block_m, block_k)
            if transpose_b:
                B_shape = (batch_count, N, K)
                B_shared_shape = (block_n, block_k)
            else:
                B_shape = (batch_count, K, N)
                B_shared_shape = (block_k, block_n)

            # Tiles enumerated per group; CTAs past the real count exit.
            _tiling = GroupTiling(batch_count, block_m)
            _num_pid_m = _tiling.tile_upper_bound(batch_sum)
            _num_pid_n = math.ceil(N / block_n)

            @T.prim_func
            def _grouped_gemm_main(
                A: T.Tensor(A_shape, dtype),  # type: ignore
                B: T.Tensor(B_shape, dtype),  # type: ignore
                C: T.Tensor(C_shape, dtype),  # type: ignore
                batch_sizes: T.Tensor([batch_count], "int32"),
                batch_offsets: T.Tensor([batch_count], "int32"),
                batch_padded_offsets: T.Tensor([batch_count], "int32"),
            ):
                with T.Kernel(_num_pid_m * _num_pid_n, threads=threads) as (pid,):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_m, block_n], accum_dtype)
                    s_cum = T.alloc_shared([batch_count + 1], "int32")
                    lo = T.alloc_local([1], "int32")
                    hi = T.alloc_local([1], "int32")
                    row = T.alloc_local([1], "int32")
                    cur_batch_idx = T.alloc_local([1], "int32")

                    _tiling.cumsum(batch_sizes, s_cum)

                    # M-major ordering: each M-tile processes all N-tiles
                    m_tile = pid // _num_pid_n
                    n_start = (pid % _num_pid_n) * block_n

                    if m_tile < s_cum[batch_count]:
                        _tiling.decode(m_tile, s_cum, lo, hi, cur_batch_idx, row)
                        m_start = batch_offsets[cur_batch_idx[0]] + row[0]
                        actual_rows = T.min(block_m, batch_sizes[cur_batch_idx[0]] - row[0])
                        actual_cols = T.min(block_n, N - n_start)
                        T.clear(C_local)

                        # NN groups are contiguous [K, N] matrices. A full tile
                        # copied as a region lowers to the same TMA the dense
                        # GEMM uses; the guarded element copy does not. The
                        # predicate has to be a Python constant: a TIR compare
                        # stays inside the copy and the TMA lowering bails out.
                        nn_tma = (
                            get_backend_name() == "musa"
                            and (not transpose_b)
                            and batch_sum % batch_count == 0
                            and (batch_sum // batch_count) % block_m == 0
                            and N % block_n == 0
                            and K % block_k == 0
                        )
                        for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                            if nn_tma:
                                # The group index has to come from the block index.
                                # A value decoded into a local is copied into the
                                # producer warp uninitialized, so the B TMA reads
                                # the wrong group.
                                T.copy(
                                    A[m_start : m_start + block_m, k * block_k : (k + 1) * block_k],
                                    A_shared,
                                )
                                T.copy(
                                    B[
                                        pid // _num_pid_n,
                                        k * block_k : (k + 1) * block_k,
                                        n_start : n_start + block_n,
                                    ],
                                    B_shared,
                                )
                            else:
                                for i, j in T.Parallel(block_m, block_k):
                                    A_shared[i, j] = T.if_then_else(
                                        i < actual_rows and j < K - k * block_k,
                                        A[m_start + i, k * block_k + j],
                                        0,
                                    )
                                if transpose_b:
                                    for i, j in T.Parallel(block_n, block_k):
                                        B_shared[i, j] = T.if_then_else(
                                            j < K - k * block_k and i < actual_cols,
                                            B[cur_batch_idx[0], n_start + i, k * block_k + j],
                                            0,
                                        )
                                else:
                                    for i, j in T.Parallel(block_k, block_n):
                                        B_shared[i, j] = T.if_then_else(
                                            i < K - k * block_k and j < actual_cols,
                                            B[cur_batch_idx[0], k * block_k + i, n_start + j],
                                            0,
                                        )
                            T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_b)
                        for i, j in T.Parallel(block_m, block_n):
                            if i < actual_rows and j < actual_cols:
                                C[m_start + i, n_start + j] = C_local[i, j]

        else:
            # TN / TT pattern: iterate per batch, outputs 3D tensor
            A_shape = (batch_sum, N)
            C_shape = (batch_count, N, K)
            A_shared_shape = (block_m, block_n)
            if transpose_b:
                B_shape = (K, batch_sum)
                B_shared_shape = (block_k, block_m)
            else:
                B_shape = (batch_sum, K)
                B_shared_shape = (block_m, block_k)

            @T.prim_func
            def _grouped_gemm_main(
                A: T.Tensor(A_shape, dtype),  # type: ignore
                B: T.Tensor(B_shape, dtype),  # type: ignore
                C: T.Tensor(C_shape, dtype),  # type: ignore
                batch_sizes: T.Tensor([batch_count], "int32"),
                batch_offsets: T.Tensor([batch_count], "int32"),
                batch_padded_offsets: T.Tensor([batch_count], "int32"),
            ):
                with T.Kernel(
                    batch_count, T.ceildiv(N, block_n) * T.ceildiv(K, block_k), threads=threads
                ) as (bx, by):
                    A_shared = T.alloc_shared(A_shared_shape, dtype)
                    B_shared = T.alloc_shared(B_shared_shape, dtype)
                    C_local = T.alloc_fragment([block_n, block_k], accum_dtype)

                    n_block_idx = by // T.ceildiv(K, block_k)
                    k_block_idx = by % T.ceildiv(K, block_k)
                    n_start = n_block_idx * block_n
                    k_start = k_block_idx * block_k
                    actual_N = T.min(block_n, N - n_start)
                    actual_K = T.min(block_k, K - k_start)
                    T.clear(C_local)

                    batch_start = batch_offsets[bx]
                    batch_size = batch_sizes[bx]

                    # TN groups are contiguous [rows, N] and [rows, K]. A full
                    # tile copied as a region lowers to the same TMA a dense
                    # A^T @ B uses. The predicate is a Python constant so the
                    # TMA lowering sees a complete tile instead of a compare.
                    tn_tma = (
                        get_backend_name() == "musa"
                        and (not transpose_b)
                        and batch_sum % batch_count == 0
                        and (batch_sum // batch_count) % block_m == 0
                        and N % block_n == 0
                        and K % block_k == 0
                    )
                    for m in T.Pipelined(T.ceildiv(batch_size, block_m), num_stages=num_stages):
                        m_start = batch_start + m * block_m
                        actual_rows = T.min(block_m, batch_size - m * block_m)
                        if tn_tma:
                            T.copy(
                                A[m_start : m_start + block_m, n_start : n_start + block_n],
                                A_shared,
                            )
                            T.copy(
                                B[m_start : m_start + block_m, k_start : k_start + block_k],
                                B_shared,
                            )
                        else:
                            for i, j in T.Parallel(block_m, block_n):
                                A_shared[i, j] = T.if_then_else(
                                    i < actual_rows and j < actual_N, A[m_start + i, n_start + j], 0
                                )
                            if transpose_b:
                                for i, j in T.Parallel(block_k, block_m):
                                    B_shared[i, j] = T.if_then_else(
                                        i < actual_K and j < actual_rows,
                                        B[k_start + i, m_start + j],
                                        0,
                                    )
                            else:
                                for i, j in T.Parallel(block_m, block_k):
                                    B_shared[i, j] = T.if_then_else(
                                        i < actual_rows and j < actual_K,
                                        B[m_start + i, k_start + j],
                                        0,
                                    )
                        T.gemm(
                            A_shared, B_shared, C_local, transpose_A=True, transpose_B=transpose_b
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        if i < actual_N and j < actual_K:
                            C[bx, n_start + i, k_start + j] = C_local[i, j]

        return _grouped_gemm_main

    return _grouped_gemm_func


class GroupedGemmKernel(Kernel):
    supported_archs: list[int] = [31, 80, 86, 89, 90]
    general: bool = True

    @classmethod
    def entry_for(cls, call: GroupedGemmCall) -> Entry:
        return grouped_gemm_entry(cls, call)

    def __init__(
        self,
        batch_sum,
        batch_count,
        N,
        K,
        dtype,
        transpose_a: bool = False,
        transpose_b: bool = True,
        config: Optional[dict] = None,
        tune=False,
    ):
        super().__init__()
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

        self.kernel = _grouped_gemm_kernel(
            self.batch_sum,
            self.batch_count,
            self.N,
            self.K,
            self.transpose_a,
            self.transpose_b,
            self.dtype_str,
        )

        self.init_config(config, tune)

    @property
    def autotune_supply_prog(self):
        """Supply autotuning the batch metadata a real call carries.

        Both templates read a group's row count out of this metadata. Both offset
        vectors take the tight prefix sum, which puts
        ``batch_padded_offsets[-1] + batch_sizes[-1]`` at ``batch_sum`` and so
        keeps every tile in the K-loop.
        """
        from tilelang.utils.device import get_current_device
        from tilelang.utils.tensor import get_tensor_supply

        default_supply = get_tensor_supply(tilelang.TensorSupplyType.Auto)
        batch_count = self.batch_count
        base, extra = divmod(self.batch_sum, batch_count)

        def supply_prog(params):
            device = get_current_device()
            sizes = torch.full((batch_count,), base, dtype=torch.int32, device=device)
            sizes[:extra] += 1
            offsets = torch.zeros(batch_count, dtype=torch.int32, device=device)
            offsets[1:] = torch.cumsum(sizes[:-1], dim=0)

            # Matched by position among themselves: the prim_func takes
            # batch_sizes, then batch_offsets, then batch_padded_offsets.
            is_metadata = [
                str(p.dtype) == "int32" and list(p.shape) == [batch_count] for p in params
            ]
            if sum(is_metadata) != 3:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects 3 int32 [{batch_count}] "
                    f"parameters (batch_sizes, batch_offsets, batch_padded_offsets), "
                    f"got {sum(is_metadata)}"
                )

            seen = 0
            inputs = []
            for param, metadata in zip(params, is_metadata, strict=True):
                if metadata:
                    inputs.append(sizes if seen == 0 else offsets)
                    seen += 1
                else:
                    inputs.append(default_supply(param))
            return inputs

        return supply_prog

    @property
    def default_config(self) -> dict:
        return _DEFAULT_CONFIGS[(self.transpose_a, self.transpose_b)]

    @property
    def autotune_configs(self) -> list[dict]:
        if get_backend_name() == "musa":
            # The full grid is thousands of tiles and most fail to compile on
            # MP31. Each layout gets the tile measured for it. The 64x128x64
            # tile wins TN but faults the device on NN, so it stays there only.
            if self.transpose_a and not self.transpose_b:
                # Region TMA. 128x128x64 stages 2 was the fastest correct tile
                # on the 16x256x4096x256 TN shape.
                return [
                    {"block_m": 128, "block_n": 128, "block_k": 64, "num_stages": 2, "threads": 256}
                ]
            if not self.transpose_a and not self.transpose_b:
                return [
                    {"block_m": 256, "block_n": 256, "block_k": 64, "num_stages": 2, "threads": 256}
                ]
            if not self.transpose_a and self.transpose_b:
                # NT's measured tile. 4 stages overflowed shared memory here,
                # so it keeps 2; the 128x256x64 shape is what won the layout.
                return [
                    {"block_m": 128, "block_n": 256, "block_k": 64, "num_stages": 2, "threads": 256}
                ]
            return [dict(self.default_config)]

        block_m = [32, 64, 128, 256]
        block_n = [32, 64, 128, 256]
        block_k = [32, 64, 128, 256]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_m, block_n, block_k, num_stages, threads))

        return [
            {"block_m": c[0], "block_n": c[1], "block_k": c[2], "num_stages": c[3], "threads": c[4]}
            for c in _configs
        ]

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
        batch_padded_offsets: torch.Tensor,
    ) -> torch.Tensor:
        kernel = _grouped_gemm_kernel(
            self.batch_sum,
            self.batch_count,
            self.N,
            self.K,
            self.transpose_a,
            self.transpose_b,
            self.dtype_str,
        )(
            self.config["block_m"],
            self.config["block_n"],
            self.config["block_k"],
            self.config["num_stages"],
            self.config["threads"],
        )
        return kernel(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
