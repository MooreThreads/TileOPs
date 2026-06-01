<div align="center">
  <img src="https://raw.githubusercontent.com/MooreThreads/TileOPs/main/assets/logo.png" width="350"/>
  <h1>TileOPs</h1>
  <p><strong>Public MUSA fork of TileOPs for LLM operators — designed for AI agents to build, evaluate, and optimize</strong></p>
  <p>Built on <a href="https://github.com/tile-ai/tilelang">TileLang</a> and the MUSA backend</p>
  <!-- <p>
    <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version" height="20"></a>
  </p> -->
  <p>
    <a href="https://github.com/MooreThreads/TileOPs/tree/main/tileops/manifest"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMooreThreads%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="Spec coverage"></a>
    <a href="https://github.com/MooreThreads/TileOPs/tree/main/benchmarks"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FMooreThreads%2FTileOPs%2Fstats%2Fmanifest-benchmark.json" alt="Bench coverage"></a>
  </p>
  <p>
    <a href="#installation"><b>Installation</b></a> |
    <a href="#quick-start"><b>Quick Start</b></a> |
    <a href="#documentation"><b>Docs</b></a>
  </p>
</div>

> **Status**: TileOPs is the public MUSA fork of TileOPs. APIs may change.

## Overview

TileOPs is the public MUSA fork of TileOPs for LLM training and inference. It is built on the MUSA-enabled TileLang stack. Beyond providing a growing collection of production-quality operators, TileOPs explores a **spec-driven development model** where AI agents can read declarative operator specifications, generate kernel implementations, and evaluate them against hardware-theoretical performance bounds — with minimal human scaffolding.

### Architecture

Every operator is split into two layers with a strict boundary:

- **Op** (L2) — stateless Python entry point. Handles validation, dtype casting, and memory layout. Compatible with graph capture and `torch.compile`.
- **Kernel** (L1) — TileLang GPU implementation targeting MUSA hardware.

This separation keeps user-facing behavior independent of GPU strategy, allowing agents and developers to modify either layer without side effects on the other.

### Key Properties

- **Spec-driven** — each operator is declared in a machine-readable manifest (`tileops/manifest/`) that specifies signatures, workloads, and roofline formulas, serving as the entry point for both agent code generation and automated validation
- **Roofline-evaluated** — kernel performance is measured against Speed-of-Light hardware bounds, not relative baselines
- **Auto-tuning** — built-in search over tile sizes, pipelines, and scheduling parameters
- **Lightweight** — depends only on TileLang, PyTorch, and einops

## Installation

TileOPs is intended to run on a MUSA-capable GPU with a MUSA-enabled PyTorch and TileLang environment.

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.1
- MUSA toolkit / runtime
- MUSA GPU: first-stage support targets **MP31**
- MUSA-enabled PyTorch
- MUSA-enabled [TileLang](https://github.com/tile-ai/tilelang)

### From package index

```bash
pip install tileops
```

### From source

```bash
git clone https://github.com/MooreThreads/TileOPs.git
cd TileOPs
make install    # dev dependencies + pre-commit hooks
```

> [!NOTE]
> If MUSA PyTorch and TileLang are already installed system-wide and you encounter build issues:
> `PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -v && pre-commit install`

Verify:

```bash
python -m pytest tests/ -q    # requires a MUSA GPU
```

## Quick Start

```python
import torch
from tileops.ops import GemmOp

M, N, K = 1024, 1024, 512
dtype = torch.float16

gemm = GemmOp(M, N, K, dtype=dtype)

A = torch.randn(M, K, device="musa", dtype=dtype)
B = torch.randn(K, N, device="musa", dtype=dtype)

C = gemm(A, B)
```

## Documentation

Design docs and development guides are in [`docs/`](docs/). The full API reference and performance tables are published at [TileOPs.github.io](https://github.com/MooreThreads/TileOPs.github.io).

## Contributing

See [docs/](docs/) for design docs. Branch and commit conventions are in [`.claude/conventions/types.sh`](.claude/conventions/types.sh).

## License

TileOPs is released under the [MIT License](LICENSE).
