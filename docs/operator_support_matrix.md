# TileOPs 算子支持矩阵

最近更新：2026-06-11

这个文件是 TileOPs MUSA 算子支持状态和 Q2 性能状态的统一入口。
原始日志、完整 CSV、一次性实验报告默认放在本地 `docs/reports/` 下，但不要求随 patch 提交；
这里记录可长期保留的当前结论和最近一次 patch 的变化。

## 口径定义

### 支持状态

`已支持` 表示：

- 该 op 在 `tileops/manifest` 中的状态是 `implemented`。
- 最新 case-aware MUSA smoke 报告能把至少一个 PASS case 映射到该 op。
- 该 op 映射到的 case 中没有 FAIL 或 UNKNOWN。

`部分通过` 表示该 op 至少有一个 case 通过，但同时存在失败或未知 case。

`待支持` 包括 `spec-only`、全部 skip、无映射 case、以及全部失败的 op。

### 性能指标

Q2 性能范围是本地 tracking 表 `docs/reports/q2_tileops_50op_tracking_20260607.csv` 中选定的 50 个 elementwise op。

主指标采用 op 平均 shape SOL，不做 bytes/耗时加权，也不让 case 多的 op 占更大权重。
计算时先对每个 op 的 shape/dtype case 直接平均，再对 50 个 op 直接平均：

```text
op_avg_shape_sol =
  mean_over_cases(s5000_case_tbps / (h20_case_tbps * 1.6 / 4.0))

avg_op_shape_sol =
  mean_over_ops(op_avg_shape_sol)
```

补充参考指标：

- SOL 达标阈值：`>= 0.8`
- Torch 加速比达标阈值：`torch_ms / tileops_ms >= 1.0`

## 当前快照

| 项目                      |     当前结果 | 对比 / 备注                        |
| ------------------------- | -----------: | ---------------------------------- |
| Manifest op 总数          |          132 | 来自最新 manifest-backed 支持报告  |
| 已支持 op                 |     65 / 132 | 严格 case-aware smoke 口径         |
| 部分通过 op               |     15 / 132 | 有些 case 通过，有些失败或未知     |
| 待支持 op                 |     52 / 132 | 包括 `spec-only` 和无 case 的 op   |
| Q2 选定 op 支持           |      50 / 50 | 选定 Q2 op 已全部支持              |
| Q2 manifest benchmark     | 255 / 255 ok | 上一版 S5000 baseline 为 254 / 255 |
| Q2 manifest 全通过 op     |      50 / 50 | 上一版 S5000 baseline 为 49 / 50   |
| 主指标：op 平均 shape SOL |        1.132 | 上一版为 1.076                     |
| op 平均 shape SOL 达标    |      40 / 50 | 上一版为 38 / 50                   |
| Torch 加速比 >= 1.0       |      34 / 50 | 补充指标                           |

本地证据文件（不随本次 patch 提交）：

- 支持报告：
  `docs/reports/musa_ops_support_report_20260610_current_smoke_case.md`
- 当前 S5000 benchmark：
  `docs/reports/q2_50_s5000_manifest_current_20260610.csv`
- H20 benchmark：
  `docs/reports/q2_50_h20_manifest_20260608_130806.csv`
- 按 op 聚合后的性能数据：
  `docs/reports/q2_50_current_perf_summary_20260610.csv`
- 聚合摘要：
  `docs/reports/q2_50_current_perf_summary_20260610.md`

## 按 Family 统计

| Family        | 已支持 / 总数 | 部分通过 | 待支持 | 备注                                            |
| ------------- | ------------: | -------: | -----: | ----------------------------------------------- |
| attention     |        0 / 15 |        4 |     11 | 一些 GQA smoke case 通过，但还没有完整支持的 op |
| convolution   |         0 / 2 |        0 |      2 | 当前 conv 行还是 `spec-only`                    |
| elementwise   |       65 / 71 |        0 |      6 | Q2 当前主线支持 family                          |
| moe           |        0 / 11 |        1 |     10 | 一个 MoE op 有部分 smoke 覆盖                   |
| normalization |        0 / 12 |        5 |      7 | 严格 smoke 口径下只有部分覆盖，没有完整支持     |
| reduction     |        0 / 19 |        5 |     14 | 多个 reduce op 只有部分 case 通过               |
| scan          |         0 / 2 |        0 |      2 | 待支持                                          |

## 最近 Patch 摘要

Patch：`414daf7 Optimize several elementwise ops`

### 修改内容

- `PowFwdOp`：在当前 manifest/test 合同下，对正数 base 使用
  `exp2(log2(a) * b)` 快路径。
- `EluFwdOp`：fp16/bf16 默认配置改为 `threads=128, npt=2`，负分支使用
  `exp2(x * log2e)`。
- `SoftplusFwdOp`：fp16/bf16 默认配置改为 `threads=128, npt=2`。
- `PreluFwdOp`：使用 channel-grid launch，去掉每个元素上的 channel
  div/mod，并使用实测得到的 dtype 默认配置。
- `ReluFwdOp`：fp16/bf16 register-copy 默认配置改为 `threads=256, npt=2`。
- 已删除或拒绝的改动：shared binary `npt=2`、NanToNum `128/1`、scalar
  hoist 类改动；原因是收益太小、回退明显或缺少足够性能证据。

### 支持变化

| 指标                       |      之前 |      现在 |
| -------------------------- | --------: | --------: |
| Q2 选定 op 支持            |   50 / 50 |   50 / 50 |
| Q2 manifest benchmark case | 254 / 255 | 255 / 255 |
| Q2 manifest 全通过 op      |   49 / 50 |   50 / 50 |

新增跑干净的 benchmark case 是
`NegFwdOp / elementwise-256M / torch.bfloat16`；上一版 baseline 记录的是 warmup hang。

### 性能变化

| 指标                      |    之前 |    现在 |
| ------------------------- | ------: | ------: |
| 主指标：op 平均 shape SOL |   1.076 |   1.132 |
| op 平均 shape SOL 达标    | 38 / 50 | 40 / 50 |

主要正收益：

| Op            | 上一版 SOL | 当前 SOL |   变化 | Torch 加速比 |
| ------------- | ---------: | -------: | -----: | -----------: |
| PowFwdOp      |      0.702 |    2.107 | +1.406 |        1.900 |
| EluFwdOp      |      0.312 |    0.906 | +0.594 |        1.052 |
| PreluFwdOp    |      0.113 |    0.556 | +0.443 |        1.808 |
| SoftplusFwdOp |      0.477 |    0.756 | +0.280 |        1.086 |
| ReluFwdOp     |      1.010 |    1.108 | +0.097 |        1.052 |

当前主要性能缺口：

| Op             | op 平均 shape SOL | Torch 加速比 | 备注                                     |
| -------------- | ----------------: | -----------: | ---------------------------------------- |
| PreluFwdOp     |             0.556 |        1.808 | 已明显提升，但仍低于目标                 |
| NanToNumFwdOp  |             0.567 |        0.462 | `npt>1` 被 TileLang MUSA vectorizer 阻塞 |
| LeakyReluFwdOp |             0.660 |        1.004 | 低于 SOL 目标                            |
| HardtanhFwdOp  |             0.674 |        1.011 | 低于 SOL 目标                            |
| SubFwdOp       |             0.717 |        1.000 | 简单 binary baseline 仍低于目标          |
| MulFwdOp       |             0.723 |        1.005 | 简单 binary baseline 仍低于目标          |
| AddFwdOp       |             0.724 |        1.014 | 简单 binary baseline 仍低于目标          |
| LerpFwdOp      |             0.727 |        1.358 | 低于 SOL 目标                            |
| SoftplusFwdOp  |             0.756 |        1.086 | config 有收益，但数学路径仍偏低          |

Torch 加速比低于 1.0：

| Op                                                        | Torch/TileOps | op 平均 shape SOL |
| --------------------------------------------------------- | ------------: | ----------------: |
| NanToNumFwdOp                                             |         0.462 |             0.567 |
| MaximumFwdOp                                              |         0.780 |             0.824 |
| MinimumFwdOp                                              |         0.788 |             0.834 |
| EqFwdOp / NeFwdOp / LtFwdOp / LeFwdOp / GeFwdOp / GtFwdOp |         ~0.79 |            >=1.89 |

## Patch 历史

| 日期       | Patch                                      | 支持变化                                 | 性能变化                                      | 本地证据                                 |
| ---------- | ------------------------------------------ | ---------------------------------------- | --------------------------------------------- | ---------------------------------------- |
| 2026-06-10 | `414daf7 Optimize several elementwise ops` | Q2 benchmark 254/255 -> 255/255 cases ok | 主 SOL 1.076 -> 1.132；达标 op 38/50 -> 40/50 | `q2_50_current_perf_summary_20260610.md` |

后续每次提交都在这里追加一行，并同步刷新上面的当前快照。
