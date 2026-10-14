import functools
from typing import Any, Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_backend_name

from .online_softmax import LOG2E


@functools.lru_cache(maxsize=32)
def _nsa_topk_varlen_kernel(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    chunk_num: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bc: int,
    bs: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    scale_log2 = scale * LOG2E
    head_kv = heads // group
    # The shared-memory tiles span the whole head dimension: a narrower tile
    # would silently truncate the QK contraction.
    bk = dim
    use_musa = get_backend_name() == "musa"
    # MP31 SQMMA wants the N tile to be a multiple of 64. The manifest bc is 32,
    # so the GEMM tile is padded and the ranking only reads the real columns.
    gemm_bc = bc if not use_musa or bc % 64 == 0 else ((bc + 63) // 64) * 64

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _nsa_topk_varlen_func(threads: int):
        @T.macro
        def odd_even_sort(indices, values, size):
            eps = 1e-5
            if use_musa:
                # The MUSA sync lowering moves a barrier that sits between a
                # shared load and its store to before the load. Compute the
                # swap into registers first, then barrier, then write.
                tx = T.get_thread_binding()
                ev1 = T.alloc_local([1], accum_dtype)
                ev2 = T.alloc_local([1], accum_dtype)
                ei1 = T.alloc_local([1], T.int32)
                ei2 = T.alloc_local([1], T.int32)
                eswap = T.alloc_local([1], T.bool)
                ov1 = T.alloc_local([1], accum_dtype)
                ov2 = T.alloc_local([1], accum_dtype)
                oi1 = T.alloc_local([1], T.int32)
                oi2 = T.alloc_local([1], T.int32)
                oswap = T.alloc_local([1], T.bool)
                for _ in T.serial(size):
                    if tx < size // 2:
                        ev1[0] = values[tx * 2]
                        ev2[0] = values[tx * 2 + 1]
                        ei1[0] = indices[tx * 2]
                        ei2[0] = indices[tx * 2 + 1]
                        # Quantize like the reference's round(score / eps) so a
                        # score that only differs in the last fp16 bit does not
                        # flip the order.
                        q1 = T.cast(T.round(ev1[0] / eps), T.int32)
                        q2 = T.cast(T.round(ev2[0] / eps), T.int32)
                        eswap[0] = (q1 < q2) or (q1 == q2 and ei1[0] < ei2[0])
                    T.sync_threads()
                    if tx < size // 2 and eswap[0]:
                        values[tx * 2] = ev2[0]
                        values[tx * 2 + 1] = ev1[0]
                        indices[tx * 2] = ei2[0]
                        indices[tx * 2 + 1] = ei1[0]
                    T.sync_threads()
                    if tx < (size - 1) // 2:
                        ov1[0] = values[tx * 2 + 1]
                        ov2[0] = values[tx * 2 + 2]
                        oi1[0] = indices[tx * 2 + 1]
                        oi2[0] = indices[tx * 2 + 2]
                        q1 = T.cast(T.round(ov1[0] / eps), T.int32)
                        q2 = T.cast(T.round(ov2[0] / eps), T.int32)
                        oswap[0] = (q1 < q2) or (q1 == q2 and oi1[0] < oi2[0])
                    T.sync_threads()
                    if tx < (size - 1) // 2 and oswap[0]:
                        values[tx * 2 + 1] = ov2[0]
                        values[tx * 2 + 2] = ov1[0]
                        indices[tx * 2 + 1] = oi2[0]
                        indices[tx * 2 + 2] = oi1[0]
                    T.sync_threads()
            else:
                for _ in T.serial(size):
                    for i in T.Parallel(size // 2):
                        v1 = values[i * 2]
                        v2 = values[i * 2 + 1]
                        idx1 = indices[i * 2]
                        idx2 = indices[i * 2 + 1]
                        v_diff = T.abs(v1 - v2)
                        is_equal = v_diff < eps
                        if v1 < v2 or (is_equal and idx1 < idx2):
                            values[i * 2], values[i * 2 + 1] = v2, v1
                            indices[i * 2], indices[i * 2 + 1] = idx2, idx1
                    T.sync_threads()
                    for i in T.Parallel((size - 1) // 2):
                        v1 = values[i * 2 + 1]
                        v2 = values[i * 2 + 2]
                        idx1 = indices[i * 2 + 1]
                        idx2 = indices[i * 2 + 2]
                        v_diff = T.abs(v1 - v2)
                        is_equal = v_diff < eps
                        if v1 < v2 or (is_equal and idx1 < idx2):
                            values[i * 2 + 1], values[i * 2 + 2] = v2, v1
                            indices[i * 2 + 1], indices[i * 2 + 2] = idx2, idx1
                    T.sync_threads()

        @T.prim_func
        def _parallel_nsa_topk_varlen_main(
            q: T.Tensor((c_seq_len, heads, dim), dtype),
            k_cmp: T.Tensor((chunk_num, head_kv, dim), dtype),
            lse_in: T.Tensor((c_seq_len, heads), dtype),  # todo: lse_in is none.
            offsets: T.Tensor((seq_num + 1,), T.int32),
            chunk_offsets: T.Tensor((seq_num + 1,), T.int32),
            token_indices: T.Tensor((c_seq_len, 2), T.int32),
            block_indices: T.Tensor((c_seq_len, head_kv, selected_block_num), T.int32),
        ):
            _ = lse_in
            with T.Kernel(c_seq_len, head_kv, threads=threads) as (bx, by):
                q_shared = T.alloc_shared([group, bk], dtype)
                k_shared = T.alloc_shared([gemm_bc, bk], dtype)

                pool_scores_s = T.alloc_shared([bc * 2], accum_dtype)
                pool_indices_s = T.alloc_shared([bc * 2], T.int32)

                i_c, i_h = bx, by
                i_n, i_t = token_indices[i_c, 0], token_indices[i_c, 1]

                bos = offsets[i_n]
                boc = chunk_offsets[i_n]
                nc = (i_t + 1) // bs

                if use_musa:
                    for i, j in T.Parallel(group, bk):
                        q_shared[i, j] = q[bos + i_t, i_h * group + i, j]
                else:
                    T.copy(q[bos + i_t, i_h * group : (i_h + 1) * group, :bk], q_shared)

                b_lse = T.alloc_fragment([group], accum_dtype)
                acc_s = T.alloc_fragment([group, gemm_bc], accum_dtype)
                scores_max = T.alloc_fragment([group], accum_dtype)
                scores_max_prev = T.alloc_fragment([group], accum_dtype)
                scores_scale = T.alloc_fragment([group], accum_dtype)
                scores_sum = T.alloc_fragment([group], accum_dtype)
                logsum = T.alloc_fragment([group], accum_dtype)

                T.fill(scores_max, -T.infinity(accum_dtype))
                T.fill(logsum, 0.0)

                for p in T.Parallel(bc * 2):
                    pool_scores_s[p] = -T.infinity(accum_dtype)
                    pool_indices_s[p] = 0

                # step1: LSE calculation
                for i_loop in T.serial(T.ceildiv(nc, bc)):
                    curr_bc = T.min(bc, nc - i_loop * bc)
                    if use_musa:
                        for i, j in T.Parallel(gemm_bc, bk):
                            k_shared[i, j] = T.if_then_else(
                                i < curr_bc,
                                k_cmp[boc + i_loop * bc + i, i_h, j],
                                T.cast(0, dtype),
                            )
                    else:
                        T.copy(
                            k_cmp[boc + i_loop * bc : boc + i_loop * bc + curr_bc, i_h, :bk],
                            k_shared[:curr_bc, :bk],
                        )

                    for g_m, c_m in T.Parallel(group, gemm_bc):
                        acc_s[g_m, c_m] = T.if_then_else(
                            c_m < curr_bc, 0.0, -T.infinity(accum_dtype)
                        )

                    T.gemm(
                        q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow
                    )

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for i in T.Parallel(group):
                        scores_scale[i] = T.if_then_else(
                            scores_max[i] > -T.infinity(accum_dtype),
                            T.exp2(scores_max_prev[i] * scale_log2 - scores_max[i] * scale_log2),
                            0.0,
                        )

                    for i, j in T.Parallel(group, gemm_bc):
                        acc_s[i, j] = T.if_then_else(
                            acc_s[i, j] > -T.infinity(accum_dtype),
                            T.exp2(acc_s[i, j] * scale_log2 - scores_max[i] * scale_log2),
                            0.0,
                        )

                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(group):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i in T.Parallel(group):
                    if nc == 0 or logsum[i] <= 0:
                        b_lse[i] = 0.0
                    else:
                        b_lse[i] = (scores_max[i] * scale_log2 + T.log2(logsum[i])) / LOG2E

                # step2: Importance Scores alignment and streaming Top-K
                if not use_musa:
                    T.sync_threads()
                nc_topk = i_t // bs + 1
                k_shared_topk = T.alloc_shared([gemm_bc, bk], dtype)

                for i_tk in T.serial(T.ceildiv(nc_topk, bc)):
                    curr_bc_tk = T.min(bc, nc_topk - i_tk * bc)
                    if use_musa:
                        for i, j in T.Parallel(gemm_bc, bk):
                            k_shared_topk[i, j] = T.if_then_else(
                                i < curr_bc_tk,
                                k_cmp[boc + i_tk * bc + i, i_h, j],
                                T.cast(0, dtype),
                            )
                    else:
                        T.copy(
                            k_cmp[boc + i_tk * bc : boc + i_tk * bc + curr_bc_tk, i_h, :bk],
                            k_shared_topk[:curr_bc_tk, :bk],
                        )

                    for g_m2, c_m2 in T.Parallel(group, gemm_bc):
                        acc_s[g_m2, c_m2] = T.if_then_else(
                            c_m2 < curr_bc_tk, 0.0, -T.infinity(accum_dtype)
                        )

                    T.gemm(
                        q_shared,
                        k_shared_topk,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for g_idx, c_idx in T.Parallel(group, gemm_bc):
                        curr_blk = i_tk * bc + c_idx
                        is_curr = curr_blk == i_t // bs
                        is_hist = curr_blk < i_t // bs
                        imp = T.if_then_else(
                            is_curr,
                            1.0,
                            T.if_then_else(
                                is_hist,
                                T.exp2((acc_s[g_idx, c_idx] * scale - b_lse[g_idx]) * LOG2E),
                                0.0,
                            ),
                        )
                        acc_s[g_idx, c_idx] = imp

                    b_i_current = T.alloc_fragment([gemm_bc], accum_dtype)
                    T.reduce_sum(acc_s, b_i_current, dim=0)

                    for c_in in T.Parallel(bc):
                        pool_scores_s[bc + c_in] = T.if_then_else(
                            c_in < curr_bc_tk, b_i_current[c_in], -T.infinity(accum_dtype)
                        )
                        pool_indices_s[bc + c_in] = T.if_then_else(
                            c_in < curr_bc_tk, i_tk * bc + c_in + 1, 0
                        )

                    T.sync_threads()
                    odd_even_sort(pool_indices_s, pool_scores_s, bc * 2)

                for s_out in T.Parallel(selected_block_num):
                    idx_final = pool_indices_s[s_out]
                    block_indices[i_c, i_h, s_out] = idx_final - 1

        return _parallel_nsa_topk_varlen_main

    return _nsa_topk_varlen_func


def _nsa_topk_varlen_run(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    chunk_num: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bc: int,
    bs: int,
    dtype: str,
    accum_dtype: str,
    threads: int,
    q: torch.Tensor,
    k_cmp: torch.Tensor,
    lse_in: torch.Tensor,
    offsets: torch.Tensor,
    chunk_offsets: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    return _nsa_topk_varlen_kernel(
        seq_num,
        c_seq_len,
        heads,
        dim,
        chunk_num,
        group,
        scale,
        selected_block_num,
        bc,
        bs,
        dtype,
        accum_dtype,
    )(threads)(q, k_cmp, lse_in, offsets, chunk_offsets, token_indices)


def _(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    chunk_num: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bc: int,
    bs: int,
    dtype: str,
    accum_dtype: str,
    threads: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    _ = (seq_num, dim, chunk_num, scale, bc, bs, dtype, accum_dtype, threads)
    # One int32 block id per token, KV head and kept block: the selection is per KV head.
    return torch.empty(
        [c_seq_len, heads // group, selected_block_num],
        dtype=torch.int32,
        device=inputs[0].device,
    )


class NSATopkVarlenKernel(Kernel):
    supported_archs: list[int] = [31, 80, 86, 89, 90]

    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim: int,
        chunk_num: int,
        group: int,
        scale: float,
        selected_block_num: int,
        bc: int,
        bs: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim = dim
        self.chunk_num = chunk_num
        self.group = group
        self.scale = scale
        self.selected_block_num = selected_block_num
        self.bc = bc
        self.bs = bs
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.accum_dtype_str = self.dtype_to_str(self.accum_dtype)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "threads": 128 if get_backend_name() == "musa" else 32,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        threads = [32]
        return [{"threads": t} for t in threads]

    def forward(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        lse_in: torch.Tensor,
        offsets: torch.Tensor,
        chunk_offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        return _nsa_topk_varlen_run(
            self.seq_num,
            self.c_seq_len,
            self.heads,
            self.dim,
            self.chunk_num,
            self.group,
            self.scale,
            self.selected_block_num,
            self.bc,
            self.bs,
            self.dtype_str,
            self.accum_dtype_str,
            self.config["threads"],
            q.to(self.dtype),
            k_cmp.to(self.dtype),
            lse_in.to(self.dtype),
            offsets.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )
