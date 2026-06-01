from __future__ import annotations

import os
from typing import Any

import torch

_DEFAULT_BACKEND = os.environ.get("TILEOPS_BACKEND", "musa").strip().lower() or "musa"
_DEFAULT_TARGET = os.environ.get("TILEOPS_TARGET", _DEFAULT_BACKEND).strip().lower() or "musa"


def get_backend_name() -> str:
    return _DEFAULT_BACKEND


def get_tilelang_target() -> str:
    return _DEFAULT_TARGET


def _torch_backend_module() -> Any:
    backend = get_backend_name()
    if hasattr(torch, backend):
        return getattr(torch, backend)
    if hasattr(torch, "cuda"):
        return torch.cuda
    raise RuntimeError(f"torch does not expose backend module {backend!r}")


def is_available() -> bool:
    backend = get_backend_name()
    if hasattr(torch, backend):
        return bool(getattr(torch, backend).is_available())
    return False


def manual_seed_all(seed: int) -> None:
    if is_available():
        _torch_backend_module().manual_seed_all(seed)


def synchronize() -> None:
    if is_available():
        _torch_backend_module().synchronize()


def empty_cache() -> None:
    if is_available():
        _torch_backend_module().empty_cache()


def current_device() -> int:
    return int(_torch_backend_module().current_device())


def get_device_name(device_index: int = 0) -> str:
    return str(_torch_backend_module().get_device_name(device_index))


def _parse_compute_version_string(value: str) -> tuple[int, int]:
    major_str, minor_str = value.split(".")
    return int(major_str), int(minor_str)


def get_compute_capability(device_index: int | None = None) -> tuple[int, int]:
    if device_index is None:
        device_index = current_device()
    backend_mod = _torch_backend_module()
    props = backend_mod.get_device_properties(device_index)
    if hasattr(props, "major") and hasattr(props, "minor"):
        return int(props.major), int(props.minor)
    if hasattr(props, "compute_version"):
        return _parse_compute_version_string(str(props.compute_version))
    raise RuntimeError("device properties do not expose a compute capability")


def get_compute_version() -> int:
    major, minor = get_compute_capability()
    return major * 10 + minor


def is_backend_tensor(tensor: torch.Tensor) -> bool:
    backend = get_backend_name()
    if getattr(tensor, f"is_{backend}", False):
        return True
    return tensor.device.type == backend


def backend_tensor_error(name: str) -> str:
    return f"{name} must be a {get_backend_name().upper()} tensor"


def get_profiler_activity():
    profiler = getattr(torch, "profiler", None)
    if profiler is None:
        raise RuntimeError("torch.profiler is unavailable")

    activities = profiler.ProfilerActivity
    backend = get_backend_name()
    if backend == "musa" and hasattr(activities, "PrivateUse1"):
        return activities.PrivateUse1
    if backend == "cuda" and hasattr(activities, "CUDA"):
        return activities.CUDA
    if hasattr(activities, "CUDA"):
        return activities.CUDA
    raise RuntimeError(f"no profiler activity found for backend {backend!r}")


def get_profiler_device_type():
    from torch.autograd.profiler import DeviceType

    backend = get_backend_name()
    if backend == "musa" and hasattr(DeviceType, "PrivateUse1"):
        return DeviceType.PrivateUse1
    if backend == "cuda" and hasattr(DeviceType, "CUDA"):
        return DeviceType.CUDA
    if hasattr(DeviceType, "CUDA"):
        return DeviceType.CUDA
    raise RuntimeError(f"no profiler device type found for backend {backend!r}")
