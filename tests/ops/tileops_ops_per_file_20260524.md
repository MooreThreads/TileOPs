# tests/ops per-file status (20260524)

Total finished: 93 / 93
PASS: 13
PASS_WITH_TESTS: 10
PASS_ALL_SKIPPED: 3
PARTIAL_PASS_FAIL: 34
FAIL: 80

| file | status | rc | seconds | summary |
|---|---:|---:|---:|---|
| `tests/ops/test_elementwise_config_dtype.py` | `PASS` | `0` | `7.1` | 34 passed in 1.00s |
| `tests/ops/test_elementwise_fp8.py` | `PASS` | `0` | `7.1` | 15 passed in 0.96s |
| `tests/ops/test_elementwise_independent_fp8.py` | `PASS` | `0` | `7.9` | 2 passed, 3 skipped in 1.74s |
| `tests/ops/test_engram.py` | `PASS` | `0` | `24.7` | 9 passed, 7 skipped in 17.73s |
| `tests/ops/test_fp8_quant.py` | `PASS` | `0` | `20.1` | 5 passed in 13.23s |
| `tests/ops/test_grouped_gemm.py` | `PASS` | `0` | `71.4` | 4 passed in 64.39s (0:01:04) |
| `tests/ops/test_mhc_post.py` | `PASS` | `0` | `20.5` | 3 passed in 13.76s |
| `tests/ops/test_pool.py` | `PASS` | `0` | `25.5` | 43 passed, 2 skipped in 18.49s |
| `tests/ops/test_reduction_primitives.py` | `PASS` | `0` | `8.1` | 34 passed in 1.87s |
| `tests/ops/test_topk_selector.py` | `PASS` | `0` | `40.2` | 4 passed in 33.01s |
| `tests/ops/test_batch_norm.py` | `PASS` | `0` | `7.9` | 19 skipped in 1.73s |
| `tests/ops/test_moe_fused_moe_distributed.py` | `PASS` | `0` | `7.9` | 8 skipped in 1.73s |
| `tests/ops/test_moe_shared_fused_moe_distributed.py` | `PASS` | `0` | `7.9` | 3 skipped in 1.72s |
| `tests/ops/attention/test_gqa.py` | `FAIL` | `1` | `49.3` | 45 failed, 6 passed in 42.57s |
| `tests/ops/attention/test_gqa_prefill_paged.py` | `FAIL` | `1` | `18.2` | 16 failed, 1 passed in 11.57s |
| `tests/ops/attention/test_gqa_sliding_window.py` | `FAIL` | `1` | `18.2` | 15 failed, 6 passed in 11.57s |
| `tests/ops/attention/test_mean_pooling.py` | `FAIL` | `1` | `21.6` | 5 failed, 1 passed in 14.78s |
| `tests/ops/test_activation.py` | `FAIL` | `1` | `32.8` | 54 failed, 2 passed in 26.42s |
| `tests/ops/test_argreduce.py` | `FAIL` | `1` | `17.7` | 80 failed, 4 passed in 11.26s |
| `tests/ops/test_binary_arith.py` | `FAIL` | `1` | `25.4` | 112 failed, 24 passed, 11 skipped in 18.85s |
| `tests/ops/test_bitwise.py` | `FAIL` | `1` | `13.3` | 21 failed, 8 passed in 7.05s |
| `tests/ops/test_comparison.py` | `FAIL` | `1` | `19.4` | 54 failed, 6 passed in 12.69s |
| `tests/ops/test_convolution.py` | `FAIL` | `1` | `84.9` | 1 failed, 35 passed in 77.89s (0:01:17) |
| `tests/ops/test_deltanet_recurrence.py` | `FAIL` | `1` | `22.1` | 3 failed, 3 passed, 8 skipped in 15.24s |
| `tests/ops/test_dropout.py` | `FAIL` | `1` | `21.9` | 22 failed, 6 passed in 15.05s |
| `tests/ops/test_elementwise_caching_autotune.py` | `FAIL` | `1` | `29.2` | 6 failed, 24 passed in 22.44s |
| `tests/ops/test_elementwise_unary_activation_alignment.py` | `FAIL` | `1` | `12.7` | 1 failed, 11 passed, 24 skipped in 6.45s |
| `tests/ops/test_fused_gated.py` | `FAIL` | `1` | `17.3` | 23 failed, 5 passed in 11.02s |
| `tests/ops/test_gated_deltanet_recurrence.py` | `FAIL` | `1` | `21.9` | 1 failed, 5 passed, 8 skipped in 15.12s |
| `tests/ops/test_gemm.py` | `FAIL` | `1` | `21.6` | 28 failed, 2 passed, 28 warnings in 14.86s |
| `tests/ops/test_group_norm.py` | `FAIL` | `1` | `9.7` | 29 failed, 2 passed, 3 skipped in 3.51s |
| `tests/ops/test_instance_norm.py` | `FAIL` | `1` | `9.6` | 29 failed, 8 passed, 2 skipped in 3.50s |
| `tests/ops/test_kernel_map_install.py` | `FAIL` | `1` | `11.8` | 1 failed, 4 passed in 5.68s |
| `tests/ops/test_layer_norm.py` | `FAIL` | `1` | `26.1` | 29 failed, 4 passed in 19.21s |
| `tests/ops/test_mamba.py` | `FAIL` | `1` | `23.6` | 24 failed, 1 passed in 16.52s |
| `tests/ops/test_moe_experts_nopad.py` | `FAIL` | `1` | `14.7` | 4 failed, 11 passed in 8.31s |
| `tests/ops/test_moe_fused_moe.py` | `FAIL` | `1` | `8.5` | 18 failed, 1 passed, 4 skipped in 2.23s |
| `tests/ops/test_norm_ops.py` | `FAIL` | `1` | `10.0` | 2 failed, 1 passed, 2 skipped in 3.80s |
| `tests/ops/test_normalization_alignment.py` | `FAIL` | `1` | `8.1` | 1 failed, 2 passed, 2 skipped in 1.83s |
| `tests/ops/test_reduce_dim_none.py` | `FAIL` | `1` | `18.1` | 63 failed, 2 passed in 11.47s |
| `tests/ops/test_reduce_multidim.py` | `FAIL` | `1` | `27.3` | 73 failed, 9 passed in 20.14s |
| `tests/ops/test_reduction_defaults.py` | `FAIL` | `1` | `10.0` | 21 failed, 6 passed in 3.78s |
| `tests/ops/test_rope.py` | `FAIL` | `1` | `16.1` | 51 failed, 1 passed in 9.55s |
| `tests/ops/test_softmax.py` | `FAIL` | `1` | `34.3` | 33 failed, 106 passed, 1 skipped in 27.17s |
| `tests/ops/test_special_elementwise.py` | `FAIL` | `1` | `33.2` | 68 failed, 13 passed in 26.74s |
| `tests/ops/test_special_elementwise_conformance.py` | `FAIL` | `1` | `37.6` | 64 failed, 17 passed in 31.04s |
| `tests/ops/test_unary_math.py` | `FAIL` | `1` | `23.5` | 133 failed, 6 passed in 16.87s |
| `tests/ops/attention/test_deepseek_dsa_decode.py` | `FAIL` | `1` | `8.1` | 1 failed in 1.82s |
| `tests/ops/attention/test_deepseek_mla_decode.py` | `FAIL` | `1` | `17.8` | 1 failed in 11.34s |
| `tests/ops/attention/test_deepseek_nsa.py` | `FAIL` | `1` | `8.1` | 3 failed in 1.85s |
| `tests/ops/attention/test_deepseek_nsa_cmp.py` | `FAIL` | `1` | `7.9` | 1 failed in 1.81s |
| `tests/ops/attention/test_deepseek_nsa_topk.py` | `FAIL` | `1` | `7.9` | 2 failed in 1.81s |
| `tests/ops/attention/test_gqa_decode.py` | `FAIL` | `1` | `19.7` | 3 failed, 3 warnings in 13.14s |
| `tests/ops/attention/test_gqa_decode_paged.py` | `FAIL` | `1` | `31.9` | 7 failed, 7 warnings in 25.04s |
| `tests/ops/attention/test_gqa_sliding_window_varlen.py` | `FAIL` | `1` | `17.3` | 14 failed in 10.63s |
| `tests/ops/attention/test_mha.py` | `FAIL` | `1` | `45.7` | 4 failed, 4 skipped in 39.05s |
| `tests/ops/attention/test_mha_decode.py` | `FAIL` | `1` | `17.5` | 3 failed in 11.26s |
| `tests/ops/attention/test_mha_decode_paged.py` | `FAIL` | `1` | `30.2` | 4 failed in 23.40s |
| `tests/ops/test_ada_layer_norm.py` | `FAIL` | `1` | `15.6` | 14 failed in 9.04s |
| `tests/ops/test_ada_layer_norm_zero.py` | `FAIL` | `1` | `15.5` | 14 failed in 9.00s |
| `tests/ops/test_cumulative.py` | `FAIL` | `1` | `14.2` | 38 failed in 7.87s |
| `tests/ops/test_deltanet_chunkwise_bwd.py` | `FAIL` | `1` | `20.3` | 6 failed in 14.13s |
| `tests/ops/test_deltanet_fwd.py` | `FAIL` | `1` | `29.7` | 8 failed in 23.25s |
| `tests/ops/test_elementwise_binary_broadcast.py` | `FAIL` | `1` | `13.3` | 21 failed in 6.91s |
| `tests/ops/test_elementwise_compile.py` | `FAIL` | `1` | `44.8` | 74 failed, 6 skipped in 38.18s |
| `tests/ops/test_fft.py` | `FAIL` | `1` | `8.2` | 9 failed in 1.88s |
| `tests/ops/test_fp8_lighting_indexer.py` | `FAIL` | `1` | `7.9` | 1 failed in 1.84s |
| `tests/ops/test_fused_add_layer_norm.py` | `FAIL` | `1` | `17.2` | 17 failed in 10.43s |
| `tests/ops/test_fused_add_rms_norm.py` | `FAIL` | `1` | `19.3` | 14 failed in 12.42s |
| `tests/ops/test_gated_deltanet_chunkwise_bwd.py` | `FAIL` | `1` | `23.4` | 6 failed in 16.99s |
| `tests/ops/test_gated_deltanet_fwd.py` | `FAIL` | `1` | `33.6` | 8 failed in 26.83s |
| `tests/ops/test_gla_chunkwise_bwd.py` | `FAIL` | `1` | `28.1` | 6 failed in 20.98s |
| `tests/ops/test_gla_chunkwise_fwd.py` | `FAIL` | `1` | `23.0` | 7 failed in 16.28s |
| `tests/ops/test_gla_recurrence.py` | `FAIL` | `1` | `22.0` | 6 failed, 15 skipped in 15.18s |
| `tests/ops/test_logical.py` | `FAIL` | `1` | `17.9` | 33 failed in 11.52s |
| `tests/ops/test_logical_reduce.py` | `FAIL` | `1` | `20.7` | 117 failed, 5 skipped in 13.95s |
| `tests/ops/test_mhc_pre.py` | `FAIL` | `1` | `26.0` | 3 failed in 18.95s |
| `tests/ops/test_moe_fused_topk.py` | `FAIL` | `1` | `22.4` | 13 failed in 15.34s |
| `tests/ops/test_moe_grouped_gemm_nopad.py` | `FAIL` | `1` | `8.0` | 3 failed in 1.80s |
| `tests/ops/test_moe_permute.py` | `FAIL` | `1` | `10.4` | 9 failed in 4.11s |
| `tests/ops/test_moe_permute_align.py` | `FAIL` | `1` | `10.6` | 15 failed in 4.36s |
| `tests/ops/test_moe_permute_nopad.py` | `FAIL` | `1` | `8.0` | 3 failed in 1.93s |
| `tests/ops/test_moe_shared_fused_moe.py` | `FAIL` | `1` | `10.7` | 4 failed in 4.48s |
| `tests/ops/test_moe_unpermute.py` | `FAIL` | `1` | `15.0` | 9 failed in 8.36s |
| `tests/ops/test_reduce.py` | `FAIL` | `1` | `22.6` | 167 failed in 15.79s |
| `tests/ops/test_reduce_arithmetic_conformance.py` | `FAIL` | `1` | `9.9` | 88 failed in 3.76s |
| `tests/ops/test_reduce_boolean_conformance.py` | `FAIL` | `1` | `12.6` | 54 failed in 6.15s |
| `tests/ops/test_reduce_scalar_conformance.py` | `FAIL` | `1` | `10.0` | 108 failed in 3.98s |
| `tests/ops/test_reduce_variance_conformance.py` | `FAIL` | `1` | `12.0` | 171 failed in 6.01s |
| `tests/ops/test_reduction_scalar_input.py` | `FAIL` | `1` | `9.0` | 59 failed in 2.97s |
| `tests/ops/test_rms_norm.py` | `FAIL` | `1` | `19.2` | 21 failed in 12.33s |
| `tests/ops/test_vector_norm.py` | `FAIL` | `1` | `15.3` | 109 failed in 8.86s |
| `tests/ops/test_welford_non_aligned.py` | `FAIL` | `1` | `16.4` | 72 failed in 9.76s |
