# TileOPs Operator Support Matrix

Last updated: 2026-04-23

## Scope

This document answers three different questions separately:

1. What operators exist in the repository
2. Which operators have explicit MUSA support and correctness evidence
3. Which operators are still pending migration

## Current Snapshot

- Top-level `tileops.ops.__all__` currently exports `96` symbols from [tileops/ops/__init__.py](../tileops/ops/__init__.py).
- There are `3` additional `engram` operators present in the repo but not exported from top-level `tileops.ops`:
  - `EngramGateConvFwdOp`
  - `EngramGateConvBwdOp`
  - `EngramDecodeOp`
- Based on the current repo state, the operators with explicit MUSA correctness evidence in this fork are the `engram` operators plus `GemmOp` (including its `gemv` fallback paths), `LayerNormFwdOp`, `RMSNormFwdOp`, `SoftmaxFwdOp`, and `LogSoftmaxFwdOp`.
- A small set of operators already shows backend-abstraction work in code or tests, but has not been validated on MUSA in this migration branch.

## Status Legend

| Status | Meaning |
| --- | --- |
| `MUSA verified` | Has explicit MUSA-oriented code path and was validated in docker correctness runs |
| `Backend-ready candidate` | Code or tests already use backend abstraction such as `get_backend_name()` or `is_backend_tensor()`, but no MUSA validation evidence was collected in this branch |
| `Pending` | Still appears CUDA-first, or no MUSA migration evidence has been added yet |

## Operator Families

| Family | Representative operators | Repo count | MUSA status | Evidence / notes |
| --- | --- | ---: | --- | --- |
| `Engram` | `EngramGateConvFwdOp`, `EngramGateConvBwdOp`, `EngramDecodeOp` | 3 | `MUSA verified` | Backend tensor checks are in [tileops/ops/engram.py](../tileops/ops/engram.py#L22), [tileops/ops/engram.py](../tileops/ops/engram.py#L130), [tileops/ops/engram_decode.py](../tileops/ops/engram_decode.py#L21); tests include expanded full cases in [tests/ops/test_engram.py](../tests/ops/test_engram.py#L70) and [tests/ops/test_engram.py](../tests/ops/test_engram.py#L237) |
| `Attention` | `MultiHeadAttentionFwdOp`, `GroupedQueryAttentionFwdOp`, `NSAFwdVarlenOp`, `DeepSeekSparseAttentionDecodeWithKVCacheFwdOp`, `MultiHeadLatentAttentionDecodeWithKVCacheFwdOp` | 16 | `Pending` | Present under [tileops/ops/attention/__init__.py](../tileops/ops/attention/__init__.py) |
| `Norm` | `LayerNormFwdOp`, `RMSNormFwdOp`, `BatchNormFwdOp`, `BatchNormBwdOp`, `GroupNormFwdOp`, `InstanceNormFwdOp`, `FusedAddLayerNormFwdOp`, `FusedAddRMSNormFwdOp`, `AdaLayerNormFwdOp`, `AdaLayerNormZeroFwdOp`, `RowNormOp` | 11 | `Partially verified` | `LayerNormFwdOp` and `RMSNormFwdOp` are MUSA verified. Other norm ops still contain many `is_cuda` checks according to repo scan |
| `Reduction` | `SoftmaxFwdOp`, `LogSoftmaxFwdOp`, `LogSumExpFwdOp`, `SumFwdOp`, `MeanFwdOp`, `VarFwdOp`, `ArgmaxFwdOp`, `CumsumFwdOp`, `L2NormFwdOp` | 19 | `Partially verified` | `SoftmaxFwdOp` and `LogSoftmaxFwdOp` are MUSA verified. `LogSumExpFwdOp` is still blocked by MUSA arch gating in [tileops/kernels/reduction/logsumexp.py](../tileops/kernels/reduction/logsumexp.py#L275) |
| `Elementwise` | `UnaryOp`, `BinaryOp`, `FusedGatedOp` | 3 | `Pending` | Still contains direct `.is_cuda` checks and CUDA device carriers in [tileops/ops/elementwise.py](../tileops/ops/elementwise.py) |
| `Convolution / Pool / Rope` | `Conv1dFwdOp`, `Conv1dBiasFwdOp`, `Conv2dOp`, `Conv3dOp`, `AvgPool1dOp`, `AvgPool2dOp`, `AvgPool3dOp`, `RopeLlama31Op`, `RopeNeoxOp`, `RopeYarnOp` | 12 | `Pending` | Operators exist, but no MUSA migration evidence has been added in this branch |
| `GEMM / Grouped GEMM` | `GemmOp`, `GroupedGemmOp` | 2 | `Partially verified` | `GemmOp` is MUSA verified, including `gemv` fallback branches; `GroupedGemmOp` is still pending. Workload and tests are backend-aware in [workloads/gemm.py](../workloads/gemm.py#L7) and [tests/ops/test_gemm.py](../tests/ops/test_gemm.py#L13) |
| `DeltaNet family` | `DeltaNetOp`, `DeltaNetFwdOp`, `DeltaNetBwdOp`, `DeltaNetDecodeOp`, `GatedDeltaNetOp`, `GatedDeltaNetFwdOp`, `GatedDeltaNetBwdOp`, `GatedDeltaNetDecodeOp`, `GLAFwdOp`, `GLABwdOp`, `GLADecodeOp` | 11 | `Pending` | Workloads are still largely CUDA-first |
| `SSD family` | `SSDChunkScanFwdOp`, `SSDChunkStateFwdOp`, `SSDDecodeOp`, `SSDStatePassingFwdOp` | 4 | `Pending` | Current op code still uses `.is_cuda` checks in multiple files |
| `FP8 utilities` | `FP8QuantOp`, `FP8LightingIndexerOp` | 2 | `Pending` | No MUSA migration evidence in this fork |
| `MHC / FFT / Dropout / DaCumsum / TopK selector` | `MHCPreOp`, `MHCPostOp`, `FFTC2COp`, `DropoutOp`, `DaCumsumFwdOp`, `TopkSelectorOp` | 6 | `Pending` | Present in repo, but not yet migrated |
| `MoE package` | `FusedTopKOp`, `MoePermuteAlignFwdOp`, `MoePermuteNopadFwdOp`, `MoePermutePaddedFwdOp`, `MoeUnpermuteFwdOp`, `MoeGroupedGemmNopadFwdOp`, `FusedMoeExpertsFwdOp`, `FusedMoeExpertsPaddedFwdOp`, `FusedMoe`, `SharedFusedMoE` | 10 | `Pending` | Top-level `tileops.ops` currently exports only `MoePermuteAlignFwdOp`; the rest are under [tileops/ops/moe/__init__.py](../tileops/ops/moe/__init__.py) |

## What Is Supported Now

At the moment, the MUSA-supported subset in this fork is:

| Operator | Status | Notes |
| --- | --- | --- |
| `EngramGateConvFwdOp` | `Supported` | Correctness passed on MUSA smoke and full shapes |
| `EngramGateConvBwdOp` | `Supported` | Backward numerical issue was fixed and then revalidated on MUSA |
| `EngramDecodeOp` | `Supported` | Single-step and multi-step decode correctness passed on MUSA |
| `GemmOp` | `Supported` | Mainstream `gemm` plus `gemv` fallback paths passed MUSA smoke/proof cases |
| `LayerNormFwdOp` | `Supported` | Main op path, backend validation path, and stable benchmark entry were verified on MUSA |
| `RMSNormFwdOp` | `Supported` | Main op path, backend validation path, and lightweight MUSA correctness/benchmark proofs were verified |
| `SoftmaxFwdOp` | `Supported` | Direct op path and lightweight MUSA correctness/benchmark proofs were verified |
| `LogSoftmaxFwdOp` | `Supported` | Direct op path and lightweight MUSA correctness/benchmark proofs were verified |

## What Is Closest To Support

These are the best next candidates because the repo already contains backend abstraction in at least part of the path:

| Candidate | Why it is close |
| --- | --- |
| `LogSumExpFwdOp` | Shared op base is backend-ready, but the kernel class still rejects MUSA arch 31 in `supported_archs` |
| `GroupedGemmOp` | Natural next step after `GemmOp` because the standalone GEMM path is now proven on MUSA |
| `Fused / companion norm ops` | `LayerNormFwdOp` and `RMSNormFwdOp` are now proven, so fused norm variants can reuse the same bring-up pattern |

## Remaining Migration Backlog

If the goal is to build a practical `TileOPs`, the remaining work is approximately:

1. Finish top-level export cleanup:
   `engram` exists but is not re-exported from [tileops/ops/__init__.py](../tileops/ops/__init__.py)
2. Promote backend-ready candidates to real support:
   `GroupedGemm`, `LogSumExp`, fused norms
3. Remove CUDA-only assumptions from op/workload/test layers:
   direct `device="cuda"`, `.is_cuda`, and `torch.cuda.*`
4. Patch kernel `supported_archs` and TileLang target wiring for MUSA
5. Add docker-based correctness evidence family by family
6. Only then do performance tuning

## Recommended Tracking Order

Recommended migration order for the next phase:

1. `LogSumExpFwdOp`
2. `GroupedGemmOp`
3. Higher-level attention, DeltaNet, SSD, and MoE families

## Migration Gantt / Todo

### Column Legend

| Column | Meaning |
| --- | --- |
| `Op` | Op-layer interface, backend checks, dispatch, top-level export |
| `Workload` | Input generation and local reference path |
| `Test` | Smoke/full correctness coverage in docker |
| `Kernel` | TileLang kernel target, supported arch, lowering correctness |
| `Benchmark` | Stable perf harness on MUSA |

### Progress Labels

| Label | Meaning |
| --- | --- |
| `Done` | Completed and validated in the current fork |
| `In progress` | Partially migrated, but not yet complete |
| `Todo` | Not started or not yet evidenced |
| `N/A` | Not a near-term requirement for that row |

### Execution Plan

| Priority | Family | Target window | Op | Workload | Test | Kernel | Benchmark | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `P0` | `Engram` | `Done` | `Done` | `Done` | `Done` | `Done` | `In progress` | Correctness is complete on MUSA; decode still has optimization headroom |
| `P0` | `GEMM` | `Done` | `Done` | `Done` | `Done` | `Done` | `Done` | `GemmOp` main path and `gemv` fallback paths are now proven on MUSA; reusable benchmark entry exists in `benchmarks/ops/bench_gemm.py` |
| `P0` | `LayerNorm` | `Done` | `Done` | `Done` | `Done` | `Done` | `Done` | `LayerNormFwdOp` main path is proven on MUSA; benchmark entry is stabilized by disabling autotune-by-default |
| `P0` | `RMSNorm` | `Done` | `Done` | `Done` | `Done` | `Done` | `Done` | `RMSNormFwdOp` main path is proven on MUSA; benchmark entry is stabilized by disabling autotune-by-default |
| `P1` | `Softmax / LogSoftmax` | `Done` | `Done` | `Done` | `Done` | `Done` | `Done` | Shared softmax op base is backend-aware, and both ops now have lightweight MUSA correctness plus stable benchmark proof entries |
| `P1` | `LogSumExp` | `W2` | `In progress` | `Todo` | `Blocked` | `Blocked` | `Todo` | Current blocker is MUSA arch gating in the kernel class rather than an op-layer mismatch |
| `P2` | `GroupedGemm` | `W2` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | Natural extension now that standalone GEMM is formally supported |
| `P2` | `Elementwise` | `W3` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | Many direct CUDA assumptions remain in op code |
| `P2` | `Norm extras` | `W3` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | `BatchNorm`, `GroupNorm`, `InstanceNorm`, fused norms, Ada norms |
| `P3` | `Convolution / Pool / Rope` | `W4` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | Medium-value family, but lower urgency than GEMM and Norm basics |
| `P3` | `FP8 utilities` | `W4` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | Likely needs extra dtype and lowering validation |
| `P3` | `MHC / FFT / Dropout / DaCumsum / TopK selector` | `W5` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | Mixed utility bucket, lower priority unless directly needed |
| `P4` | `DeltaNet / GatedDeltaNet / GLA` | `W6+` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | More complicated recurrence kernels; postpone until base families are stable |
| `P4` | `SSD family` | `W6+` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | Still CUDA-first in current code shape |
| `P4` | `Attention family` | `W7+` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | High complexity and validation cost; should come after GEMM/norm/reduction base paths |
| `P4` | `MoE package` | `W8+` | `Todo` | `Todo` | `Todo` | `Todo` | `Todo` | Broadest surface area and likely multi-kernel dependency chain |

### Family Checklist Template

Use this checklist each time a family starts:

| Step | Exit criterion |
| --- | --- |
| `Op` | No hard-coded CUDA validation path remains in the targeted op entry |
| `Workload` | Inputs and references allocate on active backend |
| `Test` | At least one smoke and one mainstream full case pass in docker |
| `Kernel` | MUSA target and supported arch are wired and compile cleanly |
| `Benchmark` | At least one stable perf command exists and runs on pinned MUSA device |

### Next Concrete Sprint

Recommended next sprint items:

| Item | Goal |
| --- | --- |
| `LogSumExpFwdOp` | Remove the current MUSA arch gate, then validate correctness and benchmark path |
| `GroupedGemmOp` | Reuse the now-verified GEMM bring-up path for the grouped variant |
| `FusedAddRMSNormFwdOp` | Reuse the now-verified RMSNorm bring-up path for the fused variant |
