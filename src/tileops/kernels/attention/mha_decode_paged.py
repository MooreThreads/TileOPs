import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import AttentionCall
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_backend_name

from .online_softmax import (
    make_log2e_scale,
    make_online_softmax,
    make_online_softmax_with_shared_state,
    make_rescale,
)

__all__ = ["MHADecodePagedKernel"]

DEVICE = get_backend_name()

# JIT kernel: no-split variant (paged)


@functools.lru_cache(maxsize=32)
def _mha_decode_no_split_kernel(
    batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype
):
    scale = make_log2e_scale(dim)
    accum_dtype = "float"
    use_musa = get_backend_name() == "musa"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_M, block_N, num_stages, threads):
        shape_q = [batch, seqlen_q, heads, dim]
        shape_kv = [seqlen_kv, heads, dim]

        online_softmax = make_online_softmax(scale, accum_dtype, block_M, block_N)
        online_softmax_musa = make_online_softmax_with_shared_state(
            scale, accum_dtype, block_M, block_N
        )
        rescale = make_rescale(block_M, dim)

        @T.prim_func
        def mha_decode_no_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
            Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(T.ceildiv(seqlen_q, block_M), heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                seqlen_kv = real_seqlen_kv[bz]

                Q_shared = T.alloc_shared([block_M, dim], dtype)
                # Two KV slots so the next page copy is in flight while the
                # current tile runs QK, softmax, and PV. A single buffer cannot
                # overlap: the generated copy always waits for group 0.
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                if use_musa:
                    acc_s_half_shared = T.alloc_shared([block_M, block_N], dtype)
                    scores_max_shared = T.alloc_shared([block_M], accum_dtype)
                    scores_max_prev_shared = T.alloc_shared([block_M], accum_dtype)
                    scores_scale_shared = T.alloc_shared([block_M], accum_dtype)
                    scores_sum_shared = T.alloc_shared([block_M], accum_dtype)
                    logsum_shared = T.alloc_shared([block_M], accum_dtype)

                if use_musa:
                    for i, j in T.Parallel(block_M, dim):
                        q_idx = bx * block_M + i
                        Q_shared[i, j] = T.if_then_else(
                            q_idx < seqlen_q,
                            Q[bz, q_idx, by, j],
                            T.cast(0, dtype),
                        )
                else:
                    T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                T.clear(acc_o)
                if use_musa:
                    T.fill(logsum_shared, 0)
                    T.fill(scores_max_shared, -T.infinity(accum_dtype))
                else:
                    T.clear(logsum)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                # Walk the tokens this batch actually holds. seqlen_kv is the
                # allocated cache, which is longer than real_seqlen_kv whenever
                # a page is only partly filled.
                loop_range = (
                    T.ceildiv((bx + 1) * block_M, block_N)
                    if is_causal
                    else T.ceildiv(seqlen_kv, block_N)
                )

                num_blockn_in_page = page_size // block_N

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    page_idx = k // num_blockn_in_page
                    block_idx_in_page = k % num_blockn_in_page
                    blockn_num_offset = (
                        block_table[bz, page_idx] * num_blockn_in_page + block_idx_in_page
                    )

                    T.copy(
                        K[
                            blockn_num_offset * block_N : (blockn_num_offset + 1) * block_N,
                            by,
                            :,
                        ],
                        K_shared,
                        disable_tma=use_musa,
                    )
                    T.copy(
                        V[
                            blockn_num_offset * block_N : (blockn_num_offset + 1) * block_N,
                            by,
                            :,
                        ],
                        V_shared,
                        disable_tma=use_musa,
                    )
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                acc_s[i, j],
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                k * block_N + j < seqlen_kv,
                                acc_s[i, j],
                                -T.infinity(acc_s.dtype),
                            )
                    if use_musa:
                        online_softmax_musa(
                            acc_s,
                            scores_max,
                            scores_sum,
                            scores_max_shared,
                            scores_max_prev_shared,
                            scores_scale_shared,
                            scores_sum_shared,
                            logsum_shared,
                        )
                        rescale(acc_o, scores_scale_shared)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s_cast[i, j] = T.cast(acc_s[i, j], dtype)
                        T.copy(acc_s_cast, acc_s_half_shared)
                        T.gemm(acc_s_half_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    else:
                        online_softmax(
                            acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
                        )
                        T.copy(acc_s, acc_s_cast)
                        rescale(acc_o, scores_scale)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                if use_musa:
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = T.if_then_else(
                            logsum_shared[i] == 0, 0, acc_o[i, j] / logsum_shared[i]
                        )
                    for i, j in T.Parallel(block_M, dim):
                        q_idx = bx * block_M + i
                        if q_idx < seqlen_q:
                            Output[bz, q_idx, by, j] = acc_o[i, j]
                    for i in T.Parallel(block_M):
                        logsum_safe = T.if_then_else(logsum_shared[i] == 0, 1, logsum_shared[i])
                        logsum_shared[i] = T.log2(logsum_safe) + scores_max_shared[i] * scale
                else:
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                    T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])
                    for i in T.Parallel(block_M):
                        logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                        logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale

        return mha_decode_no_split

    return _func


# JIT kernel: split variant (paged, split + combine)


@functools.lru_cache(maxsize=32)
def _mha_decode_split_kernel(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype):
    scale = make_log2e_scale(dim)
    accum_dtype = "float"
    use_musa = get_backend_name() == "musa"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_M, block_N, num_split, num_stages, threads):
        shape_q = [batch, seqlen_q, heads, dim]
        shape_kv = [seqlen_kv, heads, dim]
        part_shape = [batch, seqlen_q, heads, num_split, dim]
        rescale = make_rescale(block_M, dim)

        @T.macro
        def MMA0(
            K: T.Tensor(shape_kv, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k_global: T.int32,
            k: T.int32,
            mid: T.int32,
            hid: T.int32,
            bid: T.int32,
            sid: T.int32,
        ):
            seqlen_kv = real_seqlen_kv[bid]
            if use_musa:
                for i, j in T.Parallel(block_N, dim):
                    K_shared[i, j] = K[k * block_N + i, hid, j]
            else:
                T.copy(K[k * block_N : (k + 1) * block_N, hid, :], K_shared)
            # TODO: Handle causal split case
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        mid * block_M + i >= k_global * block_N + j, 0, -T.infinity(acc_s.dtype)
                    )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        k_global * block_N + j < seqlen_kv, 0, -T.infinity(acc_s.dtype)
                    )
            if use_musa:
                for i, j in T.Parallel(block_M, block_N):
                    for d in T.Serial(dim):
                        acc_s[i, j] += T.cast(Q_shared[i, d], accum_dtype) * T.cast(
                            K_shared[j, d], accum_dtype
                        )
            else:
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(shape_kv, dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            real_seqlen_kv: T.int32,
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k_global: T.int32,
            k: T.int32,
            hid: T.int32,
            bid: T.int32,
            sid: T.int32,
        ):
            if use_musa:
                for i, j in T.Parallel(block_N, dim):
                    V_shared[i, j] = V[k * block_N + i, hid, j]
            else:
                T.copy(V[k * block_N : (k + 1) * block_N, hid, :], V_shared)
            if use_musa:
                for i, j in T.Parallel(block_M, dim):
                    for k in T.Serial(block_N):
                        acc_o[i, j] += T.cast(acc_s_cast[i, k], accum_dtype) * T.cast(
                            V_shared[k, j], accum_dtype
                        )
            else:
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            scores_max: T.FragmentBuffer([block_M], accum_dtype),
            scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
            scores_sum: T.FragmentBuffer([block_M], accum_dtype),
            logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_scale[i] = T.if_then_else(
                    scores_max[i] == -T.infinity(accum_dtype),
                    0,
                    T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                )
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def _mha_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
            glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            split_length: T.Tensor([batch, num_split], "int32"),
        ):
            with T.Kernel(
                T.ceildiv(seqlen_q, block_M), heads * batch, num_split, threads=threads
            ) as (
                bx,
                by,
                bz,
            ):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                # CUDA MMA0/MMA1 still take these. MUSA uses the two slots below.
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                if use_musa:
                    acc_s_half_shared = T.alloc_shared([block_M, block_N], dtype)
                    score_shared = T.alloc_shared([block_M, block_N], accum_dtype)
                    scores_max_shared = T.alloc_shared([block_M], accum_dtype)
                    scores_scale_shared = T.alloc_shared([block_M], accum_dtype)
                    logsum_shared = T.alloc_shared([block_M], accum_dtype)

                mid = bx
                hid = by % heads
                bid = by // heads
                sid = bz
                split_length_shared = T.alloc_shared([num_split], "int32")
                T.copy(split_length[bid, :], split_length_shared, disable_tma=True)
                # A one-token query is shorter than the WMMA tile. MUSA rejects a
                # copy whose tile is larger than seqlen_q, so the padded rows load
                # as zeros instead of a block_M slice.
                if use_musa:
                    for i, j in T.Parallel(block_M, dim):
                        q_idx = mid * block_M + i
                        Q_shared[i, j] = T.if_then_else(
                            q_idx < seqlen_q,
                            Q[bid, q_idx, hid, j],
                            T.cast(0, dtype),
                        )
                else:
                    T.copy(
                        Q[bid, mid * block_M : (mid + 1) * block_M, hid, :],
                        Q_shared,
                        disable_tma=True,
                    )
                T.fill(acc_o, 0)
                if use_musa:
                    T.fill(logsum_shared, 0)
                    T.fill(scores_max_shared, -T.infinity(accum_dtype))
                else:
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))

                real_seqlen_kv_b = real_seqlen_kv[bid]
                # Per-batch loop_range: only iterate blocks within real_seqlen_kv[bid],
                # so shorter batches don't run empty splits -> all -inf -> NaN in combine.
                start_block_sid = T.if_then_else(
                    sid > 0, split_length_shared[sid - 1] // block_N, 0
                )
                end_block_valid = T.ceildiv(real_seqlen_kv_b, block_N)
                blocks_valid_this_split = end_block_valid - start_block_sid
                blocks_in_split = T.if_then_else(
                    sid > 0,
                    T.ceildiv(split_length_shared[sid] - split_length_shared[sid - 1], block_N),
                    T.ceildiv(split_length_shared[0], block_N),
                )
                loop_range = T.if_then_else(
                    blocks_valid_this_split <= 0,
                    0,
                    T.if_then_else(
                        blocks_valid_this_split <= blocks_in_split,
                        blocks_valid_this_split,
                        blocks_in_split,
                    ),
                )
                num_blockn_in_page = page_size // block_N
                offset = 0 if sid == 0 else split_length_shared[sid - 1] // block_N

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    k_global = k
                    k_global += offset

                    page_idx = k_global // num_blockn_in_page
                    block_idx_in_page = k_global % num_blockn_in_page
                    blockn_num_offset = (
                        block_table[bid, page_idx] * num_blockn_in_page + block_idx_in_page
                    )

                    if use_musa:
                        T.copy(
                            K[
                                blockn_num_offset * block_N : (blockn_num_offset + 1) * block_N,
                                hid,
                                :,
                            ],
                            K_shared,
                            disable_tma=True,
                        )
                        T.copy(
                            V[
                                blockn_num_offset * block_N : (blockn_num_offset + 1) * block_N,
                                hid,
                                :,
                            ],
                            V_shared,
                            disable_tma=True,
                        )
                        T.clear(acc_s)
                        T.gemm(
                            Q_shared,
                            K_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        # Move the GEMM fragment to shared before any thread
                        # indexes it. A raw acc_s[row, col] at 128 threads
                        # conflicts with the WMMA fragment layout.
                        for i, j in T.Parallel(block_M, block_N):
                            score_shared[i, j] = T.if_then_else(
                                k_global * block_N + j < real_seqlen_kv[bid],
                                acc_s[i, j],
                                -T.infinity(accum_dtype),
                            )
                        tx = T.get_thread_binding()
                        lane = tx % 32
                        warp = tx // 32
                        row_max = T.alloc_local([1], accum_dtype)
                        row_sum = T.alloc_local([1], accum_dtype)
                        if warp < 2:
                            row = warp * 8 + lane // 8
                            row_max[0] = -T.infinity(accum_dtype)
                            for c in T.serial(8):
                                row_max[0] = T.max(
                                    row_max[0], score_shared[row, (lane % 8) + c * 8]
                                )
                            for shift in T.unroll(3):
                                row_max[0] = T.max(row_max[0], T.shfl_xor(row_max[0], 4 >> shift))
                        if (warp < 2) and ((lane % 8) == 0):
                            row = warp * 8 + lane // 8
                            prev = scores_max_shared[row]
                            cur = T.max(row_max[0], prev)
                            scores_max_shared[row] = T.max(cur, T.cast(-1e38, accum_dtype))
                            scores_scale_shared[row] = T.exp2(
                                prev * scale - scores_max_shared[row] * scale
                            )
                        T.sync_threads()
                        if warp < 2:
                            row = warp * 8 + lane // 8
                            row_sum[0] = T.cast(0, accum_dtype)
                            for c in T.serial(8):
                                col = (lane % 8) + c * 8
                                val = T.exp2(
                                    score_shared[row, col] * scale - scores_max_shared[row] * scale
                                )
                                score_shared[row, col] = val
                                row_sum[0] += val
                            for shift in T.unroll(3):
                                row_sum[0] += T.shfl_xor(row_sum[0], 4 >> shift)
                        if (warp < 2) and ((lane % 8) == 0):
                            row = warp * 8 + lane // 8
                            logsum_shared[row] = (
                                logsum_shared[row] * scores_scale_shared[row] + row_sum[0]
                            )
                        T.sync_threads()
                        for i, j in T.Parallel(block_M, dim):
                            acc_o[i, j] *= scores_scale_shared[i]
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s_cast[i, j] = T.cast(score_shared[i, j], dtype)
                        T.copy(acc_s_cast, acc_s_half_shared)
                        T.gemm(acc_s_half_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    else:
                        MMA0(
                            K,
                            Q_shared,
                            K_shared,
                            real_seqlen_kv,
                            acc_s,
                            k_global,
                            blockn_num_offset,
                            mid,
                            hid,
                            bid,
                            sid,
                        )
                        Softmax(
                            acc_s,
                            acc_s_cast,
                            scores_max,
                            scores_max_prev,
                            scores_scale,
                            scores_sum,
                            logsum,
                        )
                        rescale(acc_o, scores_scale)
                        MMA1(
                            V,
                            V_shared,
                            real_seqlen_kv,
                            acc_s_cast,
                            acc_o,
                            k_global,
                            blockn_num_offset,
                            hid,
                            bid,
                            sid,
                        )

                if use_musa:
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = T.if_then_else(
                            logsum_shared[i] == 0, 0, acc_o[i, j] / logsum_shared[i]
                        )
                    for i in T.Parallel(block_M):
                        logsum_safe = T.if_then_else(logsum_shared[i] == 0, 1, logsum_shared[i])
                        logsum_shared[i] = T.log2(logsum_safe) + scores_max_shared[i] * scale
                else:
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                    for i in T.Parallel(block_M):
                        logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                        logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale
                T.copy(acc_o, O_shared)
                if use_musa:
                    for i in T.Parallel(block_M):
                        q_idx = mid * block_M + i
                        if q_idx < seqlen_q:
                            glse[bid, hid, sid, q_idx] = logsum_shared[i]
                    for i, j in T.Parallel(block_M, dim):
                        q_idx = mid * block_M + i
                        if q_idx < seqlen_q:
                            Output_partial[bid, q_idx, hid, sid, j] = O_shared[i, j]
                else:
                    T.copy(logsum, glse[bid, hid, sid, mid * block_M : (mid + 1) * block_M])
                    T.copy(
                        O_shared,
                        Output_partial[bid, mid * block_M : (mid + 1) * block_M, hid, sid, :],
                        disable_tma=True,
                    )

        @T.macro
        def combine(
            glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),
            Output_partial: T.Tensor(part_shape, dtype),
            Output: T.Tensor(shape_q, dtype),
        ):
            with T.Kernel(T.ceildiv(seqlen_q, block_M), heads, batch, threads=threads) as (
                bx,
                by,
                bz,
            ):
                po_local = T.alloc_fragment([block_M, dim], dtype)
                po_shared = T.alloc_shared([block_M, dim], dtype)
                o_accum_local = T.alloc_fragment([block_M, dim], accum_dtype)
                o_shared = T.alloc_shared([block_M, dim], dtype)
                lse_shared = T.alloc_shared([num_split, block_M], dtype)
                lse_local = T.alloc_fragment([num_split, block_M], dtype)
                lse_local_split = T.alloc_fragment([block_M], accum_dtype)
                lse_logsum_local = T.alloc_fragment([block_M], accum_dtype)
                lse_max_local = T.alloc_fragment([block_M], accum_dtype)
                scale_local = T.alloc_fragment([block_M], accum_dtype)

                if not use_musa:
                    T.annotate_layout(
                        {
                            o_accum_local: T.Fragment(
                                [block_M, dim], forward_thread_fn=lambda i, j: i
                            ),
                            o_shared: tilelang.layout.make_swizzled_layout(o_shared),
                            po_shared: tilelang.layout.make_swizzled_layout(po_shared),
                            lse_local: T.Fragment(
                                [num_split, block_M], forward_thread_fn=lambda _k, i: i
                            ),
                        }
                    )

                if use_musa:
                    # One thread owns a whole row of the GEMM fragment, so the
                    # default combine walks dim serially. Keep the scale in
                    # shared and let every lane of the warp cover the head dim.
                    lse_max_shared = T.alloc_shared([block_M], accum_dtype)
                    lse_logsum_shared = T.alloc_shared([block_M], accum_dtype)
                    scale_shared = T.alloc_shared([block_M], accum_dtype)
                    for i, j in T.Parallel(block_M, dim):
                        o_shared[i, j] = T.cast(0, accum_dtype)
                    for i in T.Parallel(block_M):
                        lse_max_shared[i] = -T.infinity(accum_dtype)
                        lse_logsum_shared[i] = T.cast(0, accum_dtype)
                    for k, i in T.Parallel(num_split, block_M):
                        q_idx = bx * block_M + i
                        lse_shared[k, i] = T.if_then_else(
                            q_idx < seqlen_q,
                            glse[bz, by, k, q_idx],
                            -T.infinity(dtype),
                        )
                    for k in T.serial(num_split):
                        for i in T.Parallel(block_M):
                            lse_max_shared[i] = T.max(lse_max_shared[i], lse_shared[k, i])
                    for k in T.serial(num_split):
                        for i in T.Parallel(block_M):
                            lse_logsum_shared[i] += T.exp2(lse_shared[k, i] - lse_max_shared[i])
                    for i in T.Parallel(block_M):
                        lse_logsum_shared[i] = T.log2(lse_logsum_shared[i]) + lse_max_shared[i]
                    for k in T.serial(num_split):
                        for i in T.Parallel(block_M):
                            scale_shared[i] = T.exp2(lse_shared[k, i] - lse_logsum_shared[i])
                        for i, j in T.Parallel(block_M, dim):
                            q_idx = bx * block_M + i
                            partial = T.if_then_else(
                                q_idx < seqlen_q,
                                Output_partial[bz, q_idx, by, k, j],
                                T.cast(0, dtype),
                            )
                            o_shared[i, j] += partial * scale_shared[i]
                    for i, j in T.Parallel(block_M, dim):
                        q_idx = bx * block_M + i
                        if q_idx < seqlen_q:
                            Output[bz, q_idx, by, j] = T.cast(o_shared[i, j], dtype)
                else:
                    T.clear(lse_logsum_local)
                    T.clear(o_accum_local)
                    T.copy(
                        glse[
                            bz,
                            by,
                            :,
                            bx * block_M : (bx + 1) * block_M,
                        ],
                        lse_shared,
                        disable_tma=True,
                    )
                    T.copy(lse_shared, lse_local)
                    T.reduce_max(lse_local, lse_max_local, dim=0, clear=False)
                    for k in T.Pipelined(num_split):
                        T.copy(lse_shared[k, :], lse_local_split)
                        for i in T.Parallel(block_M):
                            lse_logsum_local[i] += T.exp2(lse_local_split[i] - lse_max_local[i])
                    for i in T.Parallel(block_M):
                        lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]
                    for k in T.Pipelined(num_split, num_stages=1):
                        T.copy(
                            Output_partial[bz, bx * block_M : (bx + 1) * block_M, by, k, :],
                            po_shared,
                            disable_tma=True,
                        )
                        T.copy(po_shared, po_local)
                        T.copy(lse_shared[k, :], lse_local_split)
                        for i in T.Parallel(block_M):
                            scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
                        for i, j in T.Parallel(block_M, dim):
                            o_accum_local[i, j] += po_local[i, j] * scale_local[i]
                    T.copy(o_accum_local, o_shared)
                    T.copy(
                        o_shared,
                        Output[bz, bx * block_M : (bx + 1) * block_M, by, :],
                        disable_tma=True,
                    )

        @T.prim_func
        def mha_decode_split(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            real_seqlen_kv: T.Tensor([batch], T.int32),
            block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
            glse: T.Tensor([batch, heads, num_split, seqlen_q], dtype),
            Output_partial: T.Tensor(part_shape, dtype),  # [batch, seqlen_q, heads, num_split, dim]
            split_length: T.Tensor([batch, num_split], "int32"),
            Output: T.Tensor(shape_q, dtype),
        ):
            _mha_decode_split(
                Q, K, V, real_seqlen_kv, block_table, glse, Output_partial, split_length
            )
            combine(glse, Output_partial, Output)

        return mha_decode_split

    return _func


# Custom ops (torch.compile compatible wrappers)


# Use distinct op names so paged and non-paged (mha_decode.py) do not overwrite
# each other in torch.library; otherwise the first-registered impl "changes" the
# registry and later parametrized tests can hit the wrong implementation.
def _mha_decode_paged_no_split_run(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: str,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return _mha_decode_no_split_kernel(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype
    )(block_M, block_N, num_stages, threads)(Q, K, V, real_seqlen_kv, block_table)


def _(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: str,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(Q)


def _mha_decode_paged_split_run(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: str,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    num_split: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
    acc_split_length: torch.Tensor,
) -> torch.Tensor:
    return _mha_decode_split_kernel(
        batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype
    )(block_M, block_N, num_split, num_stages, threads)(
        Q, K, V, real_seqlen_kv, block_table, glse, Output_partial, acc_split_length
    )


def _(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    page_size: int,
    is_causal: bool,
    dtype: str,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    num_split: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    real_seqlen_kv: torch.Tensor,
    block_table: torch.Tensor,
    glse: torch.Tensor,
    Output_partial: torch.Tensor,
    acc_split_length: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(Q)


def paged_decode_entry(cls: type, call: AttentionCall) -> Entry:
    """The entry for an MHA paged-decode candidate: both take the same arguments.

    The device index is in the identity because the kernel is compiled for the
    architecture it is built on.
    """
    index = call.device.index if call.device is not None else None
    args = (
        call.batch,
        call.heads,
        call.max_seqlen_q,
        call.seqlen_kv,
        call.dim,
        call.page_size,
        call.is_causal,
        call.dtype,
    )
    return (*args, call.tune, index), lambda: cls(*args, tune=call.tune, device_index=index)


class MHADecodePagedKernel(Kernel):
    supported_archs: list[int] = [31, 80, 89, 90]
    # The implementation behind the specialised one for this key: it serves any
    # paged decode call, including the query lengths, head dims and page sizes
    # the warp-specialized SM90 kernel does not claim.
    general: bool = True

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        return paged_decode_entry(cls, call)

    def __init__(
        self,
        batch,
        heads,
        seqlen_q,
        seqlen_kv,
        dim,
        page_size,
        is_causal,
        dtype: str = "bfloat16",
        config: Optional[dict] = None,
        tune=False,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.page_size = page_size

        self.no_split_jit = _mha_decode_no_split_kernel(
            self.batch,
            self.heads,
            self.seqlen_q,
            self.seqlen_kv,
            self.dim,
            self.page_size,
            self.is_causal,
            self.dtype_str,
        )
        self.split_jit = _mha_decode_split_kernel(
            self.batch,
            self.heads,
            self.seqlen_q,
            self.seqlen_kv,
            self.dim,
            self.page_size,
            self.is_causal,
            self.dtype_str,
        )

        # autotune targets the split kernel
        self.kernel = self.split_jit
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    def _make_supply_prog(self):
        """Create a supply_prog that handles int32 tensor parameters for paged attention."""
        from tilelang.utils.tensor import get_tensor_supply as _get_tensor_supply

        default_supply = _get_tensor_supply(tilelang.TensorSupplyType.Auto)
        seqlen_kv = self.seqlen_kv
        batch = self.batch
        page_size = self.page_size
        num_pages = seqlen_kv // page_size

        def supply_prog(params):
            inputs = []
            for param in params:
                if str(param.dtype) == "int32":
                    shape = param.shape
                    if len(shape) == 1 and shape[0] == batch:
                        # real_seqlen_kv: [batch]
                        inputs.append(
                            torch.full((batch,), seqlen_kv, dtype=torch.int32, device=DEVICE)
                        )
                    elif len(shape) == 2 and shape[1] == num_pages:
                        # block_table: [batch, num_pages] — sequential page indices
                        t = (
                            torch.arange(num_pages, dtype=torch.int32, device=DEVICE)
                            .unsqueeze(0)
                            .expand(batch, -1)
                            .contiguous()
                        )
                        inputs.append(t)
                    elif len(shape) == 2:
                        # acc_split_length (cumulative): [batch, num_split]
                        num_split = shape[1]
                        base = seqlen_kv // num_split
                        t = torch.full(shape, base, dtype=torch.int32, device=DEVICE)
                        t[:, -1] += seqlen_kv % num_split
                        t = torch.cumsum(t, dim=1).to(torch.int32)
                        inputs.append(t)
                    else:
                        inputs.append(default_supply(param))
                else:
                    inputs.append(default_supply(param))
            return inputs

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    @property
    def default_config(self) -> dict:
        if get_backend_name() == "musa":
            # PH1 WMMA's smallest fp16 atom is 16x8. A one-token decode with
            # block_M=64 lowers a scalar fold that mcc does not finish inside
            # the smoke timeout, so the tile stays on the atom and uses T.gemm.
            # threads stays at 32: a 64-thread reduction on PH1 allocates
            # mbarrier_0, which this split kernel does not receive as an API
            # argument. block_N=128 was slower than 64 on the paged decode bench.
            block_m = 16 if self.seqlen_q <= 16 else min(64, self.seqlen_q)
            # Four warps. Softmax is a warp shuffle, not T.reduce, so PH1 does
            # not allocate mbarrier_0.
            return {
                "block_M": block_m,
                "block_N": 64 if self.dim <= 128 else 32,
                "num_split": 4,
                "num_stages": 1,
                "threads": 128,
            }
        return {
            "block_M": 128,
            "block_N": 64 if self.dim <= 128 else 32,
            "num_split": 4,
            "num_stages": 2,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        if get_backend_name() == "musa":
            return [dict(self.default_config)]

        block_M = [64, 128]
        block_N = [64, 128]
        num_split = [2, 4, 8]
        num_stages = [2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_split, num_stages, threads))

        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "num_split": c[2],
                "num_stages": c[3],
                "threads": c[4],
            }
            for c in _configs
        ]
        return configs

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ):
        block_M = self.config["block_M"]
        block_N = self.config["block_N"]
        num_split = self.config["num_split"]
        num_stages = self.config["num_stages"]
        threads = self.config["threads"]

        # A single program that scans the whole cache is slower than four
        # short splits on these shapes: measured median 0.060x versus 0.183x.
        real_max = (
            real_seqlen_kv.max().item() if real_seqlen_kv.dim() > 0 else real_seqlen_kv.item()
        )
        threshold = num_split * block_N
        if real_max < threshold:
            return _mha_decode_paged_no_split_run(
                self.batch,
                self.heads,
                self.seqlen_q,
                self.seqlen_kv,
                self.dim,
                self.page_size,
                self.is_causal,
                self.dtype_str,
                block_M,
                block_N,
                num_stages,
                threads,
                Q,
                K,
                V,
                real_seqlen_kv,
                block_table,
            )

        # Split path: compute cumulative per-split lengths
        chunk_size = real_max // (num_split * block_N) * block_N
        split_length = torch.full(
            (self.batch, num_split), chunk_size, dtype=torch.int32, device=Q.device
        )
        split_length[:, -1] = int(real_max - (num_split - 1) * chunk_size)
        acc_split_length = torch.cumsum(split_length, dim=1).to(torch.int32)

        glse = torch.zeros(
            (self.batch, self.heads, num_split, self.seqlen_q), dtype=self.dtype, device=Q.device
        ).contiguous()
        Output_partial = torch.zeros(
            (self.batch, self.seqlen_q, self.heads, num_split, self.dim),
            dtype=self.dtype,
            device=Q.device,
        ).contiguous()

        return _mha_decode_paged_split_run(
            self.batch,
            self.heads,
            self.seqlen_q,
            self.seqlen_kv,
            self.dim,
            self.page_size,
            self.is_causal,
            self.dtype_str,
            block_M,
            block_N,
            num_stages,
            threads,
            num_split,
            Q,
            K,
            V,
            real_seqlen_kv,
            block_table,
            glse,
            Output_partial,
            acc_split_length,
        )
