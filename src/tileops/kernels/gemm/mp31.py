"""MP31 dense GEMM kernels."""

from typing import Any, Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.tileop.base import GemmWarpPolicy

from tileops.kernels.kernel_base import Entry, Kernel

from .call_spec import GemmCall
from .dense import _dense_entry

__all__ = ["Mp31GemmKernel"]

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: False,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}

_PASS_CONFIGS_TME = {
    tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


def mp31_schedule(m: int, n: int, k: int) -> dict[str, Any]:
    """Return the tile configuration selected for one MP31 GEMM shape."""
    if m == 128 and n == 7168 and k == 2048:
        block_m, block_n, block_k = 128, 128, 64
        threads, stages = 128, 4
        policy = GemmWarpPolicy.FullRow
    elif m == 4096 and n == 2112 and k >= 4096:
        block_m, block_n, block_k = 256, 192, 64
        threads, stages = 384, 2
        policy = GemmWarpPolicy.Square
    elif m == 4096 and n == 24576 and k == 1536:
        block_m, block_n, block_k = 256, 256, 64
        threads, stages = 512, 2
        policy = GemmWarpPolicy.Square
    elif m == 1024 and n == 1024 and k <= 1024:
        block_m, block_n, block_k = 128, 256, 64
        threads, stages = 256, 3
        policy = GemmWarpPolicy.FullCol
    elif m >= 256 and n >= 256:
        block_m, block_n, block_k = 256, 256, 64
        threads, stages = 256, 2
        policy = GemmWarpPolicy.Square
    elif m < 16:
        block_m, block_n, block_k = 16, 64, 128
        threads, stages = 256, 2
        policy = GemmWarpPolicy.FullRow
    elif m == 16 and n == 4096 and k == 7168:
        block_m, block_n, block_k = 16, 128, 128
        threads, stages = 128, 2
        policy = GemmWarpPolicy.FullRow
    elif m <= 16:
        block_m, block_n, block_k = 16, 128, 128
        threads, stages = 128, 3
        policy = GemmWarpPolicy.FullRow
    elif m == 32 and n == 4096 and k == 7168:
        block_m, block_n, block_k = 32, 128, 64
        threads, stages = 128, 4
        policy = GemmWarpPolicy.FullRow
    elif m <= 32:
        block_m, block_n, block_k = 32, 128, 64
        threads, stages = 128, 3
        policy = GemmWarpPolicy.FullRow
    elif m == 64 and n == 7168 and k == 2048:
        block_m, block_n, block_k = 64, 128, 64
        threads, stages = 128, 4
        policy = GemmWarpPolicy.FullRow
    elif m <= 64:
        block_m, block_n, block_k = 64, 64, 64
        threads, stages = 128, 3
        policy = GemmWarpPolicy.FullRow
    elif m <= 96:
        block_m, block_n, block_k = 96, 64, 64
        threads, stages = 128, 3
        policy = GemmWarpPolicy.FullRow
    elif m < 1024:
        block_n = 64 if n <= 4096 else 128
        block_m, block_k = 128, 64
        threads, stages = 128, 3
        policy = GemmWarpPolicy.FullRow
    else:
        block_m, block_n, block_k = 128, 256, 64
        threads, stages = 512, 3
        policy = GemmWarpPolicy.FullCol
    if n < block_n:
        block_n = 64 if n <= 64 else 128
    return {
        "block_M": block_m,
        "block_N": block_n,
        "block_K": block_k,
        "threads": threads,
        "policy": policy,
        "num_stages": stages,
        "split_k": 1,
    }


@tilelang.jit(out_idx=[-1], target="musa", pass_configs=_PASS_CONFIGS)
def _gemm_nn(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    policy,
    num_stages,
    split_k,
    dtype,
    accum_dtype="float32",
):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            a_shared = T.alloc_shared((block_M, block_K), dtype)
            b_shared = T.alloc_shared((block_K, block_N), dtype)
            c_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            c_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(10)
            T.clear(c_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], a_shared)
                T.copy(B[ko * block_K, bx * block_N], b_shared)
                T.gemm(a_shared, b_shared, c_local, policy=policy)
            T.copy(c_local, c_shared)
            T.copy(c_shared, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(out_idx=[-1], target="musa", pass_configs=_PASS_CONFIGS)
def _gemm_nt(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    policy,
    num_stages,
    split_k,
    dtype,
    accum_dtype="float32",
):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            a_shared = T.alloc_shared((block_M, block_K), dtype)
            b_shared = T.alloc_shared((block_N, block_K), dtype)
            c_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            c_shared = T.alloc_shared((block_M, block_N), dtype)
            n_tiles = T.ceildiv(N, block_N)
            m_tiles = T.ceildiv(M, block_M)
            T.use_swizzle(2 if n_tiles >= 8 else 3 if m_tiles * n_tiles >= 128 else 10)
            T.clear(c_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], a_shared)
                T.copy(B[bx * block_N, ko * block_K], b_shared)
                T.gemm(a_shared, b_shared, c_local, transpose_B=True, policy=policy)
            T.copy(c_local, c_shared)
            if threads == 512:
                T.sync_threads(7, 512)
            T.copy(c_shared, C[by * block_M, bx * block_N], outer_cache_policy="cache_none")

    return main


@tilelang.jit(out_idx=[-1], target="musa", pass_configs=_PASS_CONFIGS_TME)
def _gemm_nt_tma(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    policy,
    num_stages,
    split_k,
    dtype,
    accum_dtype="float32",
):
    """TMA-enabled producer/consumer variant for the decode down projection."""

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            a_shared = T.alloc_shared((block_M, block_K), dtype)
            b_shared = T.alloc_shared((block_N, block_K), dtype)
            c_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            c_shared = T.alloc_shared((block_M, block_N), dtype)
            T.clear(c_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.tma_copy(
                    A[by * block_M, ko * block_K],
                    a_shared,
                    inner_cache_policy="cache_persist",
                    outer_cache_policy="cache_persist",
                )
                T.tma_copy(
                    B[bx * block_N, ko * block_K],
                    b_shared,
                    inner_cache_policy="cache_once",
                    outer_cache_policy="cache_none",
                )
                T.gemm(a_shared, b_shared, c_local, transpose_B=True, policy=policy)
            T.copy(c_local, c_shared)
            T.copy(c_shared, C[by * block_M, bx * block_N], outer_cache_policy="cache_none")

    return main


@tilelang.jit(out_idx=[-1], target="musa", pass_configs=_PASS_CONFIGS_TME)
def _gemm_nt_tiny_m(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    num_stages,
    dtype,
    accum_dtype="float32",
):
    padded_m = block_M

    @T.prim_func
    def main(
        A: T.Tensor((padded_m, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((padded_m, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            a_shared = T.alloc_shared((num_stages, padded_m, block_K), dtype)
            b_shared = T.alloc_shared((num_stages, block_N, block_K), dtype)
            c_local = T.alloc_fragment((padded_m, block_N), accum_dtype)
            mbars = T.alloc_barrier([128, 128] * num_stages)
            with T.ws(0):
                T.clear(c_local)
            for ko in T.serial(T.ceildiv(K, block_K)):
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[ko % num_stages + num_stages],
                        parity=((ko // num_stages) % 2) ^ 1,
                    )
                    T.tma_copy(
                        A[0:padded_m, ko * block_K : (ko + 1) * block_K],
                        a_shared[ko % num_stages, :, :],
                        barrier=mbars[ko % num_stages],
                        inner_cache_policy="cache_persist",
                        outer_cache_policy="cache_persist",
                    )
                    T.tma_copy(
                        B[
                            bx * block_N : (bx + 1) * block_N,
                            ko * block_K : (ko + 1) * block_K,
                        ],
                        b_shared[ko % num_stages, :, :],
                        barrier=mbars[ko % num_stages],
                        inner_cache_policy="cache_once",
                        outer_cache_policy="cache_none",
                    )
                    T.mbarrier_arrive(mbarrier=mbars[ko % num_stages])
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[ko % num_stages], parity=(ko // num_stages) % 2
                    )
                    T.gemm(
                        a_shared[ko % num_stages, :, :],
                        b_shared[ko % num_stages, :, :],
                        c_local,
                        transpose_B=True,
                        policy=GemmWarpPolicy.FullRow,
                    )
                    T.mbarrier_arrive(mbarrier=mbars[ko % num_stages + num_stages])
            with T.ws(0):
                for mi, ni in T.Parallel(padded_m, block_N):
                    C[mi, bx * block_N + ni] = c_local[mi, ni]

    return main


class Mp31GemmKernel(Kernel):
    """Dense NN/NT GEMM using the MP31 schedule."""

    supported_archs = [31]
    general = False

    @classmethod
    def applies(cls, call: GemmCall) -> bool:
        return (
            call.arch == 31
            and not call.trans_a
            and (
                call.gemv_mode is None
                or (call.m == 1 and call.n == 2112 and call.k == 7168 and call.trans_b)
            )
            and call.dtype in (torch.float16, torch.bfloat16)
        )

    @classmethod
    def entry_for(cls, call: GemmCall) -> Entry:
        return _dense_entry(cls, call)

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
        trans_a: bool = False,
        trans_b: bool = True,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if trans_a:
            raise ValueError("Mp31GemmKernel serves NN and NT layouts only")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_b = trans_b
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict[str, Any]:
        return mp31_schedule(self.m, self.n, self.k)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_compiled_kernel"):
            config = self.config
            dtype = self.dtype_str
            if self.trans_b and self.m < 16:
                self._compiled_kernel = _gemm_nt_tiny_m.compile(
                    M=self.m,
                    N=self.n,
                    K=self.k,
                    block_M=config["block_M"],
                    block_N=config["block_N"],
                    block_K=config["block_K"],
                    threads=config["threads"],
                    num_stages=config["num_stages"],
                    dtype=dtype,
                )
            else:
                kernel_fn = _gemm_nt if self.trans_b else _gemm_nn
                if self.trans_b and (
                    (self.m in (64, 128) and self.n == 7168 and self.k == 2048)
                    or (self.m in (16, 32) and self.n == 4096 and self.k == 7168)
                ):
                    kernel_fn = _gemm_nt_tma
                self._compiled_kernel = kernel_fn.compile(
                    M=self.m,
                    N=self.n,
                    K=self.k,
                    dtype=dtype,
                    **config,
                )
        if self.trans_b and self.m < 16:
            padded = a.new_zeros((self.config["block_M"], self.k))
            padded[: self.m].copy_(a)
            return self._compiled_kernel(padded, b)[: self.m]
        return self._compiled_kernel(a, b)
