"""Shared skip markers and backend-specific benchmark skip reasons."""

import os

import pytest
import torch

from .backend import get_backend_name

DEVICE = get_backend_name()

skip_requires_vllm = pytest.mark.skipif(DEVICE == "musa", reason="Requires vLLM")
skip_missing_baseline = pytest.mark.skipif(DEVICE == "musa", reason="missing comparison baseline")
skip_corner_case = pytest.mark.skipif(DEVICE == "musa", reason="corner case mismatch on MUSA")
skip_requires_distributed_launch = pytest.mark.skipif(
    DEVICE == "musa" and "RANK" not in os.environ,
    reason="requires distributed launch",
)
skip_requires_cuda_graph = pytest.mark.skipif(DEVICE == "musa", reason="Requires CUDA graph")
skip_requires_cooperative_launch = pytest.mark.skipif(
    DEVICE == "musa", reason="Requires cooperative launch"
)
_DEVICE_BACKEND = getattr(torch, DEVICE, torch.cuda)
skip_requires_devices = pytest.mark.skipif(
    DEVICE == "musa" and _DEVICE_BACKEND.device_count() < 2, reason="Requires devices"
)
skip_special_test_for_h200 = pytest.mark.skipif(
    DEVICE == "musa", reason="Requires special test for H200"
)
