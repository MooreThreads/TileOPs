# tests root per-file status (20260524_rerun_each_file)

Excluded second-level dirs: `tests/ops`, `tests/kernels`, `tests/perf`

Total finished: 16 / 16
PASS: 14
FAIL: 2

| file | status | rc | seconds | summary |
|---|---:|---:|---:|---|
| `tests/test_autotune.py` | `PASS` | `0` | `7.8` | 1 skipped in 1.70s |
| `tests/test_backend_utils.py` | `PASS` | `0` | `7.1` | 4 passed in 0.98s |
| `tests/test_benchmark_record.py` | `PASS` | `0` | `6.0` | 6 passed in 0.13s |
| `tests/test_ci_venv_hash.py` | `PASS` | `0` | `6.0` | 13 passed in 0.13s |
| `tests/test_compile.py` | `PASS` | `0` | `7.9` | 3 skipped in 1.70s |
| `tests/test_dtype_codegen.py` | `PASS` | `0` | `7.9` | 1 passed in 1.71s |
| `tests/test_gpu_smoke_policy.py` | `PASS` | `0` | `6.1` | 6 passed in 0.23s |
| `tests/test_op_base.py` | `PASS` | `0` | `7.9` | 10 passed in 1.73s |
| `tests/test_ops_manifest.py` | `PASS` | `0` | `6.8` | 21 passed in 0.86s |
| `tests/test_reclaim_action.py` | `PASS` | `0` | `6.1` | 13 passed in 0.20s |
| `tests/test_roofline_codegen.py` | `PASS` | `0` | `7.8` | 3 passed in 1.71s |
| `tests/test_tier_validation.py` | `PASS` | `0` | `6.0` | 21 passed in 0.15s |
| `tests/test_validate_manifest.py` | `PASS` | `0` | `23.5` | 221 passed, 1 warning in 17.31s |
| `tests/test_workloads_to_params.py` | `PASS` | `0` | `7.0` | 6 passed in 0.84s |
| `tests/test_base.py` | `FAIL` | `5` | `5.9` | no tests ran in 0.10s |
| `tests/test_musa_backend_subset.py` | `FAIL` | `1` | `21.6` | 1 failed, 2 warnings in 15.08s |
