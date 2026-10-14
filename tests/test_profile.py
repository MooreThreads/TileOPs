import pytest

from tileops.perf.profile import find_profile, get_profile_path, load_profile


class TestGetProfilePath:
    @pytest.mark.smoke
    def test_h200_exists(self):
        path = get_profile_path("h200")
        assert path.exists()
        assert path.suffix == ".yaml"
        assert path.parent.name == "nvidia"

    @pytest.mark.smoke
    def test_vendor_qualified_names_resolve(self):
        assert get_profile_path("nvidia/h200").name == "h200.yaml"
        assert get_profile_path("nvidia/h20_3e").name == "h20_3e.yaml"
        assert get_profile_path("mthreads/S5000").name == "S5000.yaml"
        assert load_profile("nvidia/h200")["gpu"] == "NVIDIA H200"
        assert load_profile("mthreads/S5000")["gpu"] == "MTT S5000"

    @pytest.mark.smoke
    def test_unknown_gpu_raises(self):
        with pytest.raises(FileNotFoundError):
            get_profile_path("nonexistent_gpu")


class TestLoadProfile:
    @pytest.mark.smoke
    def test_h200_top_level_keys(self):
        profile = load_profile("h200")
        assert profile["gpu"] == "NVIDIA H200"
        assert "hbm" in profile
        assert "tensor_core" in profile

    @pytest.mark.smoke
    def test_s5000_scaffold_is_loadable_but_not_auto_selected(self):
        profile = load_profile("S5000")
        assert profile["gpu"] == "MTT S5000"
        assert profile["status"] == "scaffold"
        assert find_profile("MTT S5000") is None
