# tests/ops per-file status (20260531)

Total finished: 78 / 78
PASS: 30
PASS_WITH_TESTS: 27
PASS_ALL_SKIPPED: 3
PARTIAL_PASS_FAIL: 16
FAIL: 48
INTERRUPTED: 1

Notes:
- `tests/ops/test_gemm.py` was interrupted by request and was not rerun in the 5-GPU remaining-file pass.
- The first pass ran files sequentially until `test_gemm.py`; the remaining files were run in parallel on MUSA devices `0,1,2,4,5`.

Ordering:
- Full PASS files first.
- Partial pass/fail files next.
- Full FAIL files after that.
- Interrupted files last.

| file | status | rc | seconds | summary |
|---|---:|---:|---:|---|
| `tests/ops/test_activation.py` | `PASS` | `0` | `16.43` | 56 passed in 16.43s |
| `tests/ops/test_batch_norm.py` | `PASS` | `0` | `8` | 19 skipped in 1.71s |
| `tests/ops/test_binary_arith.py` | `PASS` | `0` | `522` | 147 passed in 515.58s (0:08:35) |
| `tests/ops/test_bitwise.py` | `PASS` | `0` | `86` | 29 passed in 78.98s (0:01:18) |
| `tests/ops/test_comparison.py` | `PASS` | `0` | `214` | 60 passed in 206.77s (0:03:26) |
| `tests/ops/test_deltanet_recurrence.py` | `PASS` | `0` | `40` | 6 passed, 8 skipped in 32.74s |
| `tests/ops/test_elementwise_binary_broadcast.py` | `PASS` | `0` | `74` | 21 passed in 67.33s (0:01:07) |
| `tests/ops/test_elementwise_caching_autotune.py` | `PASS` | `0` | `14.22` | 30 passed in 14.22s |
| `tests/ops/test_elementwise_compile.py` | `PASS` | `0` | `25.40` | 80 passed in 25.40s |
| `tests/ops/test_elementwise_config_dtype.py` | `PASS` | `0` | `8` | 34 passed in 1.02s |
| `tests/ops/test_elementwise_fp8.py` | `PASS` | `0` | `7` | 15 passed in 0.97s |
| `tests/ops/test_elementwise_independent_fp8.py` | `PASS` | `0` | `33` | 5 passed in 26.21s |
| `tests/ops/test_elementwise_unary_activation_alignment.py` | `PASS` | `0` | `15.50` | 36 passed in 15.50s |
| `tests/ops/test_engram.py` | `PASS` | `0` | `188` | 9 passed, 7 skipped in 181.08s (0:03:01) |
| `tests/ops/test_fp8_quant.py` | `PASS` | `0` | `48` | 5 passed in 41.28s |
| `tests/ops/test_fused_gated.py` | `PASS` | `0` | `82` | 28 passed in 76.06s (0:01:16) |
| `tests/ops/test_gated_deltanet_recurrence.py` | `PASS` | `0` | `39` | 6 passed, 8 skipped in 32.75s |
| `tests/ops/test_gla_recurrence.py` | `PASS` | `0` | `36` | 6 passed, 15 skipped in 28.60s |
| `tests/ops/test_grouped_gemm.py` | `PASS` | `0` | `73` | 4 passed in 66.18s (0:01:06) |
| `tests/ops/test_kernel_map_install.py` | `PASS` | `0` | `27` | 5 passed in 19.76s |
| `tests/ops/test_logical.py` | `PASS` | `0` | `137` | 33 passed in 129.83s (0:02:09) |
| `tests/ops/test_mhc_post.py` | `PASS` | `0` | `30` | 3 passed in 23.06s |
| `tests/ops/test_moe_fused_moe_distributed.py` | `PASS` | `0` | `9` | 8 skipped in 1.76s |
| `tests/ops/test_moe_shared_fused_moe_distributed.py` | `PASS` | `0` | `8` | 3 skipped in 1.72s |
| `tests/ops/test_pool.py` | `PASS` | `0` | `78` | 43 passed, 2 skipped in 70.72s (0:01:10) |
| `tests/ops/test_reduction_primitives.py` | `PASS` | `0` | `8` | 34 passed in 1.86s |
| `tests/ops/test_special_elementwise.py` | `PASS` | `0` | `16.60` | 77 passed in 16.60s |
| `tests/ops/test_special_elementwise_conformance.py` | `PASS` | `0` | `142` | 81 passed in 134.72s (0:02:14) |
| `tests/ops/test_topk_selector.py` | `PASS` | `0` | `69` | 4 passed in 61.31s (0:01:01) |
| `tests/ops/test_unary_math.py` | `PASS` | `0` | `16.90` | 139 passed in 16.90s |
| `tests/ops/test_argreduce.py` | `FAIL` | `1` | `26` | 80 failed, 4 passed in 18.38s |
| `tests/ops/test_convolution.py` | `FAIL` | `1` | `249` | 1 failed, 35 passed in 242.30s (0:04:02) |
| `tests/ops/test_dropout.py` | `FAIL` | `1` | `40` | 22 failed, 6 passed in 33.33s |
| `tests/ops/test_group_norm.py` | `FAIL` | `1` | `10` | 29 failed, 2 passed, 3 skipped in 3.63s |
| `tests/ops/test_instance_norm.py` | `FAIL` | `1` | `10` | 29 failed, 8 passed, 2 skipped in 3.61s |
| `tests/ops/test_layer_norm.py` | `FAIL` | `1` | `278` | 30 failed, 3 passed in 270.76s (0:04:30) |
| `tests/ops/test_mamba.py` | `FAIL` | `1` | `26` | 24 failed, 1 passed in 18.45s |
| `tests/ops/test_moe_experts_nopad.py` | `FAIL` | `1` | `31` | 4 failed, 11 passed in 23.59s |
| `tests/ops/test_moe_fused_moe.py` | `FAIL` | `1` | `9` | 18 failed, 1 passed, 4 skipped in 2.28s |
| `tests/ops/test_norm_ops.py` | `FAIL` | `1` | `17` | 2 failed, 1 passed, 2 skipped in 10.39s |
| `tests/ops/test_normalization_alignment.py` | `FAIL` | `1` | `9` | 1 failed, 2 passed, 2 skipped in 1.87s |
| `tests/ops/test_reduce_dim_none.py` | `FAIL` | `1` | `25` | 63 failed, 2 passed in 17.13s |
| `tests/ops/test_reduce_multidim.py` | `FAIL` | `1` | `64` | 73 failed, 9 passed in 55.91s |
| `tests/ops/test_reduction_defaults.py` | `FAIL` | `1` | `17` | 21 failed, 6 passed in 9.04s |
| `tests/ops/test_rope.py` | `FAIL` | `1` | `19` | 51 failed, 1 passed in 12.36s |
| `tests/ops/test_softmax.py` | `FAIL` | `1` | `1051` | 44 failed, 95 passed, 1 skipped in 1043.33s (0:17:23) |
| `tests/ops/test_ada_layer_norm.py` | `FAIL` | `1` | `17` | 14 failed in 10.71s |
| `tests/ops/test_ada_layer_norm_zero.py` | `FAIL` | `1` | `17` | 14 failed in 10.84s |
| `tests/ops/test_cumulative.py` | `FAIL` | `1` | `18` | 38 failed in 10.71s |
| `tests/ops/test_deltanet_chunkwise_bwd.py` | `FAIL` | `1` | `20` | 6 failed in 14.23s |
| `tests/ops/test_deltanet_fwd.py` | `FAIL` | `1` | `30` | 8 failed in 23.17s |
| `tests/ops/test_fft.py` | `FAIL` | `1` | `8` | 9 failed in 1.87s |
| `tests/ops/test_fp8_lighting_indexer.py` | `FAIL` | `1` | `8` | 1 failed in 1.86s |
| `tests/ops/test_fused_add_layer_norm.py` | `FAIL` | `1` | `18` | 17 failed in 10.52s |
| `tests/ops/test_fused_add_rms_norm.py` | `FAIL` | `1` | `22` | 14 failed in 14.94s |
| `tests/ops/test_gated_deltanet_chunkwise_bwd.py` | `FAIL` | `1` | `24` | 6 failed in 17.11s |
| `tests/ops/test_gated_deltanet_fwd.py` | `FAIL` | `1` | `34` | 8 failed in 26.56s |
| `tests/ops/test_gla_chunkwise_bwd.py` | `FAIL` | `1` | `28` | 6 failed in 20.72s |
| `tests/ops/test_gla_chunkwise_fwd.py` | `FAIL` | `1` | `30` | 7 failed in 22.68s |
| `tests/ops/test_logical_reduce.py` | `FAIL` | `1` | `28` | 117 failed, 5 skipped in 19.85s |
| `tests/ops/test_mhc_pre.py` | `FAIL` | `1` | `126` | 3 failed in 117.88s (0:01:57) |
| `tests/ops/test_moe_fused_topk.py` | `FAIL` | `1` | `24` | 13 failed in 16.01s |
| `tests/ops/test_moe_grouped_gemm_nopad.py` | `FAIL` | `1` | `8` | 3 failed in 1.82s |
| `tests/ops/test_moe_permute.py` | `FAIL` | `1` | `18` | 9 failed in 10.64s |
| `tests/ops/test_moe_permute_align.py` | `FAIL` | `1` | `12` | 15 failed in 4.62s |
| `tests/ops/test_moe_permute_nopad.py` | `FAIL` | `1` | `17` | 3 failed in 10.18s |
| `tests/ops/test_moe_shared_fused_moe.py` | `FAIL` | `1` | `14` | 4 failed in 6.97s |
| `tests/ops/test_moe_unpermute.py` | `FAIL` | `1` | `18` | 9 failed in 10.40s |
| `tests/ops/test_reduce.py` | `FAIL` | `1` | `29` | 167 failed in 21.90s |
| `tests/ops/test_reduce_arithmetic_conformance.py` | `FAIL` | `1` | `20` | 88 failed in 12.87s |
| `tests/ops/test_reduce_boolean_conformance.py` | `FAIL` | `1` | `19` | 54 failed in 11.84s |
| `tests/ops/test_reduce_scalar_conformance.py` | `FAIL` | `1` | `18` | 108 failed in 10.61s |
| `tests/ops/test_reduce_variance_conformance.py` | `FAIL` | `1` | `24` | 171 failed in 16.24s |
| `tests/ops/test_reduction_scalar_input.py` | `FAIL` | `1` | `14` | 59 failed in 7.37s |
| `tests/ops/test_rms_norm.py` | `FAIL` | `1` | `21` | 21 failed in 14.64s |
| `tests/ops/test_vector_norm.py` | `FAIL` | `1` | `21` | 109 failed in 14.23s |
| `tests/ops/test_welford_non_aligned.py` | `FAIL` | `1` | `26` | 72 failed in 18.28s |
| `tests/ops/test_gemm.py` | `INTERRUPTED` | `143` | `1439` | no pytest summary; interrupted |
