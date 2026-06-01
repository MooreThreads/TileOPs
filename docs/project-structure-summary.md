# TileOPs 项目结构总结

本文基于当前仓库代码结构整理，目标是快速回答三个问题：

1. 这个项目的核心分层是什么
2. 主要目录分别负责什么
3. 如果要继续开发或阅读代码，应该从哪里进入

## 1. 项目定位

`TileOPs` 是一个基于 `TileLang` 的、面向 LLM 训练和推理场景的 GPU 算子库。它的核心设计不是“先写 kernel 再补外围”，而是：

- 先用 `ops_manifest.yaml` 声明算子规格
- 再实现 `Op` 与 `Kernel`
- 再用 `tests/` 做正确性验证
- 用 `benchmarks/` 做性能评估
- 用 `tileops/perf/` 和 `docs/` 承接 roofline 与设计文档

仓库当前最重要的架构约束是两层分离：

- `Op` 层：Python 侧入口，负责参数校验、形状处理、dtype 检查、kernel 调度
- `Kernel` 层：TileLang kernel 实现，负责设备端执行与 autotune

这个分层在 [`README.md`](../README.md)、[`docs/architecture.md`](architecture.md)、[`docs/ops-design.md`](ops-design.md) 和实际基类 [`tileops/ops/op_base.py`](../tileops/ops/op_base.py)、[`tileops/kernels/kernel_base.py`](../tileops/kernels/kernel_base.py) 中是一致的。

## 2. 顶层目录结构

项目顶层可以按“规范、实现、验证、性能、工程化”五类来理解：

```text
TileOPs/
├── tileops/         # 核心 Python 包
├── workloads/       # 测试/基准共享的 workload 定义层
├── tests/           # 正确性测试
├── benchmarks/      # 性能测试与硬件基准
├── docs/            # 架构、接口、流程、性能文档
├── scripts/         # CI/报告/校验辅助脚本
├── .github/         # GitHub Actions 工作流
├── .claude/         # 仓库内开发规范与 agent/domain rule
├── Makefile         # 常用安装、测试、benchmark 命令
└── pyproject.toml   # 打包、依赖、pytest/ruff 配置
```

几个直接影响开发体验的入口：

- [`pyproject.toml`](../pyproject.toml)：定义基础依赖 `torch`、`tilelang==0.1.8`、`einops`、`pyyaml`，以及 `dev`、`bench` 两组扩展依赖
- [`Makefile`](../Makefile)：封装 `install`、`test-smoke`、`test-full`、`test-nightly`、`bench`
- `.github/workflows/`：分成 `preflight`、`gpu-smoke`、`nightly` 三条主链，说明这个项目明确区分 CPU 合约检查、GPU smoke、nightly 长跑

## 3. `tileops/` 是真正的核心实现层

`tileops/` 是 Python package 本体，当前大致由 5 个部分组成：

```text
tileops/
├── __init__.py
├── ops_manifest.yaml
├── manifest.py
├── kernels/
├── ops/
├── perf/
└── utils/
```

### 3.1 `ops_manifest.yaml` + `manifest.py`

这是项目的规格入口。

- `ops_manifest.yaml`：集中声明算子接口、workloads、roofline 信息
- `manifest.py`：提供程序化访问，目前最直接的入口是 `load_workloads(op_name)`

这说明 manifest 不只是文档，而是测试和 benchmark 的共享数据源。

## 4. `tileops/ops/`：用户侧算子接口层

这里是 L2 层。职责是：

- 定义用户可调用的 Op 类
- 做 shape/dtype 校验
- 处理动态 shape 的 kernel cache key
- 调用具体 kernel

关键基类是 [`tileops/ops/op_base.py`](../tileops/ops/op_base.py)：

- `default_kernel_map`：声明当前 op 对应哪些 kernel
- `dispatch_kernel()`：根据 GPU 架构挑选可用 kernel
- `_infer_output_shapes()`：输出 shape 推导接口
- `_validate_dtypes()`：输入 dtype 验证接口
- `eval_roofline()`：roofline 计算接口
- `_cache_key()`：动态 shape 下的 kernel cache 关键点

从 [`tileops/ops/__init__.py`](../tileops/ops/__init__.py) 看，当前项目已经暴露了大量算子族，主要包括：

- `norm`：`RMSNorm`、`LayerNorm`、`GroupNorm`、`BatchNorm`、`InstanceNorm`
- `reduction`：`sum`、`mean`、`softmax`、`argmax`、`cumsum`、`logsumexp` 等
- `attention`：`MHA`、`GQA`、decode/paged/sliding-window、DeepSeek NSA/DSA/MLA
- `moe`：permute、unpermute、fused topk、shared fused moe、grouped gemm
- 其他：`gemm`、`grouped_gemm`、`dropout`、`rope`、`fft`、`mamba`、`deltanet`、`gla`

按当前目录统计，`tileops/ops/` 下约有 `70` 个 Python 文件，已经不是 demo 规模，而是按算子族分模块组织的库。

## 5. `tileops/kernels/`：TileLang kernel 实现层

这里是 L1 层，对应具体 GPU kernel。职责比 `Op` 更底层：

- 持有 TileLang program
- 管理 kernel config
- 定义 autotune 配置
- 只关心设备侧执行，不关心用户接口语义

基类是 [`tileops/kernels/kernel_base.py`](../tileops/kernels/kernel_base.py)：

- `init_config()`：合并默认配置、手动配置或 autotune
- `default_config`：默认配置入口
- `autotune()`：接入 `tilelang.autotuner.autotune`
- `forward()`：统一执行接口

子目录按算子族拆分得比较清楚：

- `attention/`
- `reduction/`
- `norm/`
- `moe/`
- `pool/`
- `mamba/`
- `deltanet/`
- `gated_deltanet/`
- `engram/`
- `gla/`
- 以及若干单文件 kernel，如 `gemm.py`、`convolution.py`、`elementwise.py`、`rope.py`

当前 `tileops/kernels/` 下约有 `88` 个 Python 文件，是仓库里最重的实现目录之一。

## 6. `workloads/`：测试与 benchmark 共用的输入规格层

这是这个项目很值得注意的一层。它不是实现层，也不是 benchmark 本身，而是共享 workload 层。

在 [`docs/trust-model.md`](trust-model.md) 里，这一层被明确为：

- 由 Test 阶段拥有
- 供 tests 与 benchmarks 共用
- 只能定义输入生成与参数化
- 不能塞 reference program、性能逻辑、benchmark baseline

关键基类在 [`workloads/workload_base.py`](../workloads/workload_base.py)：

- `WorkloadBase`：要求实现 `gen_inputs()`
- `RandnTest`：一类通用随机输入 workload
- `FixtureMeta` / `FixtureBase`：把 workload 参数直接变成 pytest 参数化装饰器

`workloads/` 下当前大约有 `48` 个 Python 文件，基本一类算子对应一个 workload 文件，和 `tests/ops/`、`benchmarks/ops/` 形成镜像关系。

## 7. `tests/`：正确性验证层

`tests/` 不是简单的单元测试集合，而是一个明确分层的 correctness framework。

关键点：

- [`tests/test_base.py`](../tests/test_base.py) 中的 `TestBase` 继承 `WorkloadBase`
- `TestBase` 要求子类提供 `ref_program()`
- `check()` 负责运行 op 与 reference，并比较输出

这说明测试层的模型是：

```text
workload.gen_inputs() -> op(*inputs) -> ref_program(*inputs) -> compare
```

目录上又分为两类：

- `tests/ops/`：绝大多数算子的 correctness tests
- `tests/ops/attention/`：attention 子类专门拆分
- 顶层 `tests/test_*.py`：框架、manifest、benchmark record、CI policy、打包/校验类测试

按当前统计：

- `tests/ops/` 约 `81` 个 Python 文件
- 其中 `tests/ops/attention/` 约 `15` 个

这说明项目既测算子功能，也测仓库治理规则。

## 8. `benchmarks/`：性能评估与硬件画像层

`benchmarks/` 不是附属目录，而是 TileOPs 的一条主流程。

关键基类是 [`benchmarks/benchmark_base.py`](../benchmarks/benchmark_base.py)。从这个文件可以看出几个实际特征：

- benchmark 框架有显式 workload protocol
- 用 `load_workloads()` 从 manifest 取 workload
- 内建纯 kernel timing 路径
- 采用 L2 flush、warmup、repeat、trial 的规范 benchmark 流程
- 支持记录环境元数据

目录进一步分成：

- `benchmarks/ops/`：按 op 维度的性能 benchmark
- `benchmarks/ops/attention/`：attention 家族 benchmark
- `benchmarks/kernels/`：更底层的 kernel benchmark
- `benchmarks/hardware/`：硬件微基准，目前能看到 `memory/hbm_bandwidth.py` 和 `hbm_saturation.cu`
- `benchmarks/tests/`：benchmark framework 自己的 contract tests

按当前统计：

- `benchmarks/ops/` 约 `68` 个 Python 文件
- 其中 `benchmarks/ops/attention/` 约 `15` 个

可以看出这个仓库把“可测性能”当成一等公民，而不是单独几份脚本。

## 9. `tileops/perf/`：roofline 与性能模型支撑

`docs/architecture.md` 里的设计目标是让 benchmark 原始耗时与 roofline 公式、硬件 profile 接起来。

当前代码里，`tileops/perf/` 已有：

- `formulas.py`
- `profile.py`
- `profiles/h200.yaml`

也就是说，这部分已经有“性能建模骨架”，但从代码现状看仍是部分落地状态，不是一个完全闭环的成熟子系统。阅读时建议把它理解为：

- 设计目标已经明确
- 一部分协议和配置已经落库
- 某些文档中的理想目录形态还没有全部完全实现

## 10. `docs/`：不是附属说明，而是设计规范的一部分

`docs/README.md` 明确写了一个重要原则：

> 文档和 manifest 是 authoritative spec，代码要去符合 spec。

所以 `docs/` 在这个项目里不是补充材料，而是规范本体。建议优先读：

1. `docs/architecture.md`
2. `docs/manifest.md`
3. `docs/ops-design.md`
4. `docs/trust-model.md`
5. `docs/testing.md`

如果只看代码、不看这些文档，会漏掉这个仓库最核心的治理思想：不同阶段各自拥有不同责任边界。

## 11. `scripts/`、`.github/`、`.claude/`：工程化与治理层

### `scripts/`

主要是辅助工具和 nightly/CI 支撑脚本，例如：

- `validate_manifest.py`
- `warmup_kernel_cache.py`
- `nightly_report.py`
- `gpu_smoke_report.py`
- `ci_venv_hash.py`
- `scripts/ci/setup_nightly_venv.sh`
- `scripts/ci/verify_nightly_runner.sh`

### `.github/workflows/`

当前至少有三类关键工作流：

- `preflight.yml`
  - PR 标题校验
  - `pre-commit`
  - `gitleaks`
  - `actionlint`
  - manifest 校验
  - benchmark contract tests
- `gpu-smoke.yml`
  - 在 GPU runner 上跑 smoke 级别检查，并且带有 fork/安全策略判断
- `nightly.yml`
  - cache warmup
  - benchmark
  - 全量 op tests
  - 夜间报告与性能历史

### `.claude/`

这是项目内部的 agent/开发规则仓库，文档中多次引用它的 domain rule。说明 TileOPs 不只是“写算子”，而是在尝试把 AI agent 参与开发的流程也规约化。

## 12. 代码阅读建议

如果你是第一次进入这个项目，建议按下面顺序读：

1. [`README.md`](../README.md)
2. [`docs/architecture.md`](architecture.md)
3. [`docs/trust-model.md`](trust-model.md)
4. [`tileops/ops_manifest.yaml`](../tileops/ops_manifest.yaml)
5. [`tileops/ops/op_base.py`](../tileops/ops/op_base.py)
6. [`tileops/kernels/kernel_base.py`](../tileops/kernels/kernel_base.py)
7. `workloads/` 中某个你关心的 workload 文件
8. 对应的 `tileops/ops/...`、`tileops/kernels/...`
9. 对应的 `tests/ops/...`
10. 对应的 `benchmarks/ops/...`

如果你是要改某个具体算子，更有效的阅读顺序是：

```text
manifest -> workload -> op -> kernel -> test -> benchmark
```

## 13. 一句话总结

`TileOPs` 的本质不是“一个 TileLang kernel 集合”，而是一个以 `manifest` 为中心、把 `实现 / 正确性 / benchmark / roofline / CI` 组织成统一流程的 spec-driven GPU operator 平台。

对阅读者来说，真正要抓住的不是某个单独目录，而是这条主链：

```text
ops_manifest.yaml
  -> workloads
  -> tileops/ops
  -> tileops/kernels
  -> tests
  -> benchmarks
  -> perf/docs/CI
```
