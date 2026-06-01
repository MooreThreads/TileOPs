# TileOPs Current Status

## Goal

当前目标是先完成“能稳定证明后端接入方法”的主流子集，而不是一次性把整个 TileOPs 全量迁到 MUSA。

当前已验证的主流子集包括：

- GEMM
- LayerNorm
- RMSNorm
- Softmax

验证环境：

- Docker 容器：`<your-docker-container>`
- 验证设备：`export MUSA_VISIBLE_DEVICES=4`
- TileLang 运行时来自：`<tilelang_musa build dir>`

## Completed Work

### 1. Backend abstraction added

新增统一后端入口：

- `tileops/utils/backend.py`

该模块已经收口以下能力：

- backend 名称获取
- TileLang target 获取
- backend 可用性判断
- seed / synchronize / empty_cache
- current device / device name
- compute capability / compute version
- profiler activity / device type
- backend tensor 判断与错误信息

相关导出和兼容层也已更新：

- `tileops/utils/__init__.py`
- `tileops/utils/utils.py`

### 2. Package import path made safer

为避免过早 import 导致 `tilelang/tvm_ffi` 相关爆炸，已将：

- `tileops/__init__.py`

改成 lazy import 风格，只在真正访问 `ops` 时再导入。

### 3. Common runtime paths switched to backend-aware logic

以下公共路径已切到 backend-aware 逻辑：

- `tileops/ops/op_base.py`
- `tests/conftest.py`
- `benchmarks/conftest.py`
- `benchmarks/benchmark_base.py`

首批算子的输入校验 / 错误信息也已改成 backend-aware：

- `tileops/ops/norm/norm_base.py`
- `tileops/ops/norm/layer_norm.py`
- `tileops/ops/reduction/_softmax_base.py`

### 4. First kernel subset moved to MUSA target

以下 kernel 已切到 `target=get_tilelang_target()`，并使用 MP31 方向配置：

- `tileops/kernels/gemm.py`
- `tileops/kernels/norm/layer_norm.py`
- `tileops/kernels/norm/rms_norm.py`
- `tileops/kernels/reduction/softmax.py`
- `tileops/kernels/reduction/_primitives.py`

### 5. GEMM aligned toward tilelang_musa benchmark source

按用户要求，GEMM 已参考：

- `tilelang_musa/musa_tests/benchmark/test_tme_mma_perf.py`
- `tilelang_musa/musa_tests/benchmark/test_tme_mma2.py`

当前 GEMM 的 MUSA 方向改动包括：

- 使用 `T.use_swizzle(panel_size=4, order="col")`
- 默认配置切到：
  - `block_m=256`
  - `block_n=256`
  - `block_k=64`
  - `num_stages=3`
  - `threads=512`
- autotune 配置加入 `threads=512`

### 6. First-subset workloads/tests no longer hard-code CUDA

已迁移首批 workload / tests 中显式 `device="cuda"` 的路径，覆盖：

- `workloads/gemm.py`
- `workloads/layer_norm.py`
- `workloads/rms_norm.py`
- `workloads/workload_base.py`
- `tests/ops/test_layer_norm.py`
- `tests/ops/test_rms_norm.py`
- `tests/ops/test_softmax.py`
- `tests/ops/test_gemm.py`

## Validation Artifacts

### 1. Backend utility unit test

新增：

- `tests/test_backend_utils.py`

覆盖内容包括：

- MUSA compute version / sm version
- backend tensor 判断
- shared memory budget 查询
- backend unavailable fallback

### 2. Stable subset smoke test

新增：

- `tests/test_musa_backend_subset.py`

该文件顺序验证以下四个接入点：

- `GemmOp`
- `LayerNormFwdOp`
- `RMSNormFwdOp`
- `SoftmaxFwdOp`

这个 smoke 文件的目的不是替代全量 pytest，而是作为“TileOPs 主流子集已打通”的稳定证明入口。

## Verified Results

以下验证已在 Docker 中、绑定卡 4 完成：

```bash
docker exec <your-docker-container> bash -lc 'export MUSA_VISIBLE_DEVICES=4; cd <repo-root> && pytest -q tests/test_backend_utils.py'
docker exec <your-docker-container> bash -lc 'export MUSA_VISIBLE_DEVICES=4; cd <repo-root> && pytest -q tests/test_musa_backend_subset.py -s'
```

结果：

- `tests/test_backend_utils.py`: `4 passed in 0.94s`
- `tests/test_musa_backend_subset.py`: `1 passed in 21.40s`

其中 `tests/test_musa_backend_subset.py` 的运行日志已确认：

- GEMM 在 MUSA 上完成编译与执行
- LayerNorm / RMSNorm / Softmax 在同一条后端链路下成功运行

## Current Boundary

当前可以确认的是：

- TileOPs 已具备基础 MUSA backend 抽象
- 主流子集的 kernel/op/workload/test 链路已经打通
- 可以在指定 Docker + 指定 MUSA 卡上稳定复现接入成功

当前还不能宣称的是：

- 全仓所有算子都已完成 MUSA 迁移
- 全量 `tests/ops` 已全部通过
- 所有历史 `cuda` 假设都已清理完毕

换句话说，当前状态是：

> “TileOPs 的接入方法已经被主流子集稳定证明，但仍处于分阶段迁移中，而不是全量完工状态。”

## Remaining Work

下一阶段建议按以下顺序推进：

1. 继续扩展 backend-aware 改造范围，把更多 `device="cuda"` / `torch.cuda` 路径迁到统一抽象层。
2. 扩大已支持算子集合，优先处理与当前主流子集相邻、依赖模式类似的算子。
3. 为更多算子补“最小稳定 smoke”，避免一开始就陷入全量 pytest 的冷启动和环境噪声。
4. 在主流 correctness 稳定后，再逐步恢复 benchmark / autotune / profiling 链路。
5. 如果 GEMM 还要进一步优化，可继续向 `tilelang_musa/musa_tests/benchmark` 的实现细节靠拢，例如进一步核对 `T.gemm` 的 policy 选择。

## Recommended Acceptance Baseline

当前建议把下面两个命令作为 TileOPs 第一阶段验收基线：

```bash
docker exec <your-docker-container> bash -lc 'export MUSA_VISIBLE_DEVICES=4; cd <repo-root> && pytest -q tests/test_backend_utils.py'
docker exec <your-docker-container> bash -lc 'export MUSA_VISIBLE_DEVICES=4; cd <repo-root> && pytest -q tests/test_musa_backend_subset.py -s'
```

只要这两个入口稳定通过，就说明：

- backend abstraction 仍然成立
- 主流子集的 MUSA 接入链路仍然成立
- Docker 验证环境仍然可用
