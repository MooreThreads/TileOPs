from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tileops.kernels.reduction._primitives import SHARED_MEMORY_BUDGET_BYTES, device_smem_budget
from tileops.utils import (
    backend_tensor_error,
    get_compute_capability,
    get_compute_version,
    get_sm_version,
    is_available,
    is_backend_tensor,
)


pytestmark = pytest.mark.smoke


class _FakeMusaModule:
    def __init__(self, *, available: bool = True, compute_version: str = "3.1") -> None:
        self._available = available
        self._compute_version = compute_version
        self.seed = None

    def is_available(self) -> bool:
        return self._available

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, device_index: int) -> SimpleNamespace:
        return SimpleNamespace(
            compute_version=self._compute_version,
            shared_memory_per_block_optin=128 * 1024,
            shared_memory_per_block=64 * 1024,
            L2_cache_size=50 * 1024 * 1024,
        )

    def get_device_name(self, device_index: int) -> str:
        return "Fake MUSA GPU"


class _FakeTensor:
    def __init__(self, *, device_type: str = "musa", is_musa: bool = False) -> None:
        self.device = SimpleNamespace(type=device_type)
        self.is_musa = is_musa


def test_backend_helpers_use_musa_compute_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "musa", _FakeMusaModule(compute_version="3.1"), raising=False)

    assert is_available() is True
    assert get_compute_capability() == (3, 1)
    assert get_compute_version() == 31
    assert get_sm_version() == 31


def test_is_backend_tensor_accepts_musa_device_type() -> None:
    assert is_backend_tensor(_FakeTensor(device_type="musa")) is True
    assert is_backend_tensor(_FakeTensor(device_type="cpu")) is False
    assert backend_tensor_error("x") == "x must be a MUSA tensor"


def test_device_smem_budget_uses_backend_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "musa", _FakeMusaModule(), raising=False)

    assert device_smem_budget() == 128 * 1024


def test_device_smem_budget_falls_back_when_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "musa", _FakeMusaModule(available=False), raising=False)

    assert device_smem_budget() == SHARED_MEMORY_BUDGET_BYTES
