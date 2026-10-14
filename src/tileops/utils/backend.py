from __future__ import annotations

import ctypes
import os
from typing import Any, Callable

import torch

_DEFAULT_BACKEND = os.environ.get("TILEOPS_BACKEND", "musa").strip().lower() or "musa"
_DEFAULT_TARGET = os.environ.get("TILEOPS_TARGET", _DEFAULT_BACKEND).strip().lower() or "musa"


def get_backend_name() -> str:
    return _DEFAULT_BACKEND


_MUPTI = None
_MUPTI_BUFFERS = []
_MUPTI_RECORDS = []
_MUPTI_EXTERNAL = {}
_MUPTI_BUFFER_SIZE = 8 * 1024 * 1024
_MUPTI_PREPARE_ID = 1 << 32
_MUPTI_REQUEST_CALLBACK = None
_MUPTI_COMPLETE_CALLBACK = None


class _MuptiKernel(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("cache_config", ctypes.c_uint8),
        ("shared_memory_config", ctypes.c_uint8),
        ("registers", ctypes.c_uint16),
        ("partitioned_global_cache_requested", ctypes.c_uint32),
        ("partitioned_global_cache_executed", ctypes.c_uint32),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("completed", ctypes.c_uint64),
        ("device_id", ctypes.c_uint32),
        ("context_id", ctypes.c_uint32),
        ("stream_id", ctypes.c_uint32),
        ("grid_x", ctypes.c_int32),
        ("grid_y", ctypes.c_int32),
        ("grid_z", ctypes.c_int32),
        ("block_x", ctypes.c_int32),
        ("block_y", ctypes.c_int32),
        ("block_z", ctypes.c_int32),
        ("static_shared_memory", ctypes.c_int32),
        ("dynamic_shared_memory", ctypes.c_int32),
        ("local_memory_per_thread", ctypes.c_uint32),
        ("local_memory_total", ctypes.c_uint32),
        ("correlation_id", ctypes.c_uint32),
        ("grid_id", ctypes.c_int64),
        ("name", ctypes.c_void_p),
    ]


class _MuptiExternal(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("external_kind", ctypes.c_uint32),
        ("external_id", ctypes.c_uint64),
        ("correlation_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


_MUPTI_REQUEST = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
)
_MUPTI_COMPLETE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_size_t,
)


def _musa_mupti_request(buffer, size, max_records) -> None:
    storage = ctypes.create_string_buffer(_MUPTI_BUFFER_SIZE)
    _MUPTI_BUFFERS.append(storage)
    buffer[0] = ctypes.cast(storage, ctypes.POINTER(ctypes.c_uint8))
    size[0] = _MUPTI_BUFFER_SIZE
    max_records[0] = 0


def _musa_mupti_complete(context, stream_id, buffer, size, valid_size) -> None:
    record = ctypes.c_void_p()
    while _MUPTI.muptiActivityGetNextRecord(buffer, valid_size, ctypes.byref(record)) == 0:
        kind = ctypes.c_uint32.from_address(record.value).value
        if kind == 10:
            item = ctypes.cast(record, ctypes.POINTER(_MuptiKernel)).contents
            _MUPTI_RECORDS.append((item.start, item.end, item.correlation_id))
        elif kind == 39:
            item = ctypes.cast(record, ctypes.POINTER(_MuptiExternal)).contents
            _MUPTI_EXTERNAL[item.correlation_id] = item.external_id


def _musa_mupti_load():
    global _MUPTI, _MUPTI_REQUEST_CALLBACK, _MUPTI_COMPLETE_CALLBACK
    if _MUPTI is not None:
        return _MUPTI
    root = os.environ.get("MUSA_HOME", "/usr/local/musa-5.2.0")
    _MUPTI = ctypes.CDLL(os.path.join(root, "lib/libmupti.so"))
    _MUPTI.muptiActivityRegisterCallbacks.argtypes = [_MUPTI_REQUEST, _MUPTI_COMPLETE]
    _MUPTI.muptiActivityEnable.argtypes = [ctypes.c_uint32]
    _MUPTI.muptiActivityDisable.argtypes = [ctypes.c_uint32]
    _MUPTI.muptiActivityFlushAll.argtypes = [ctypes.c_uint32]
    _MUPTI.muptiActivityPushExternalCorrelationId.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
    _MUPTI.muptiActivityPopExternalCorrelationId.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    _MUPTI.muptiActivityGetNextRecord.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _MUPTI_REQUEST_CALLBACK = _MUPTI_REQUEST(_musa_mupti_request)
    _MUPTI_COMPLETE_CALLBACK = _MUPTI_COMPLETE(_musa_mupti_complete)
    _MUPTI.muptiActivityRegisterCallbacks(_MUPTI_REQUEST_CALLBACK, _MUPTI_COMPLETE_CALLBACK)
    return _MUPTI


def collect_musa_mupti_repeats(
    run_one: Callable[[int], None],
    n_repeat: int,
    prepare_one: Callable[[int], None],
):
    lib = _musa_mupti_load()
    _MUPTI_RECORDS.clear()
    _MUPTI_EXTERNAL.clear()
    for kind in (10, 39, 4, 5):
        lib.muptiActivityEnable(kind)
    try:
        for i in range(n_repeat):
            lib.muptiActivityPushExternalCorrelationId(3, _MUPTI_PREPARE_ID)
            prepare_one(i)
            lib.muptiActivityPopExternalCorrelationId(3, None)
            lib.muptiActivityPushExternalCorrelationId(3, i)
            run_one(i)
            torch.musa.synchronize()
            lib.muptiActivityPopExternalCorrelationId(3, None)
        lib.muptiActivityFlushAll(1)
    finally:
        for kind in (5, 4, 39, 10):
            lib.muptiActivityDisable(kind)
        _MUPTI_BUFFERS.clear()
    kernels = []
    iteration_of = {}
    for start, end, correlation_id in _MUPTI_RECORDS:
        iteration = _MUPTI_EXTERNAL.get(correlation_id)
        if iteration is None or iteration == _MUPTI_PREPARE_ID:
            continue
        kernels.append(
            {
                "name": "musa_kernel",
                "start_ns": start,
                "end_ns": end,
                "correlation_id": iteration,
            }
        )
        iteration_of[iteration] = iteration
    return kernels, iteration_of, 0


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
    return bool(hasattr(torch, backend) and getattr(torch, backend).is_available())


def manual_seed_all(seed: int) -> None:
    if is_available():
        _torch_backend_module().manual_seed_all(seed)


def synchronize() -> None:
    if is_available():
        _torch_backend_module().synchronize()


def empty_cache() -> None:
    if is_available():
        _torch_backend_module().empty_cache()


_CTX_RESET_PERSISTING_L2_CACHE: ctypes._FuncPtr | None = None


def reset_persisting_l2_cache() -> None:
    global _CTX_RESET_PERSISTING_L2_CACHE
    if _CTX_RESET_PERSISTING_L2_CACHE is None:
        lib = ctypes.CDLL("libmusart.so")
        fn_name = "musaCtxResetPersistingL2Cache"
        _CTX_RESET_PERSISTING_L2_CACHE = getattr(lib, fn_name)
    _CTX_RESET_PERSISTING_L2_CACHE()


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
    props = _torch_backend_module().get_device_properties(device_index)
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
