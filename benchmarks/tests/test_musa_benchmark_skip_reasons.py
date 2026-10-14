import pytest

from tileops.utils import skip_reasons


@pytest.mark.parametrize("device", ["musa", "cuda"])
def test_shared_skip_markers_require_musa(monkeypatch, device):
    import importlib

    from tileops.utils import backend

    with monkeypatch.context() as patch:
        patch.setattr(backend, "get_backend_name", lambda: device)
        importlib.reload(skip_reasons)
        try:
            markers = [
                value
                for name, value in vars(skip_reasons).items()
                if name.startswith("skip_") and isinstance(value, pytest.MarkDecorator)
            ]
            assert markers
            assert all(marker.name == "skipif" for marker in markers)
            if device != "musa":
                assert all(not marker.args[0] for marker in markers)
            assert skip_reasons.skip_requires_vllm.args[0] == (device == "musa")
            assert skip_reasons.skip_requires_vllm.kwargs["reason"] == "Requires vLLM"
            assert skip_reasons.skip_missing_baseline.args[0] == (device == "musa")
            assert (
                skip_reasons.skip_missing_baseline.kwargs["reason"] == "missing comparison baseline"
            )
            expected = {
                "skip_requires_devices": "Requires devices",
                "skip_requires_cuda_graph": "Requires CUDA graph",
                "skip_special_test_for_h200": "Requires special test for H200",
                "skip_requires_cooperative_launch": "Requires cooperative launch",
            }
            for name, reason in expected.items():
                assert getattr(skip_reasons, name).kwargs["reason"] == reason
        finally:
            patch.undo()
            importlib.reload(skip_reasons)
