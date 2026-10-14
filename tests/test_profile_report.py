import struct
from pathlib import Path

import pytest

from scripts.profile_report import (
    HardwareProfile,
    RooflineCeiling,
    build_roofline_points,
    parse_profile_log,
    render_html,
    render_png,
    write_profile_visualizations,
)

SAMPLE_PROFILE = """# TileOPs Benchmark Report
Generated: 2026-09-03 20:00:00

## Environment

- **Torch version**: 2.9.1
- **GPU model**: MTT S5000

## SiluAndMulFwdOp

### tileops

| M | N | dtype | device_busy_ms | latency_ms | tflops | bandwidth_tbs | flops | bytes | compute_roof | timing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 14336 | torch.float16 | 0.1638 | 0.1642 | 1.0800 | 0.5370 | 176160768 | 180000000 | cuda_core.fp32 | mupti |
| 1 | 14336 | torch.bfloat16 | 0.0035 | 0.0037 | 0.0246 | 0.0246 | 86016 | 86016 | cuda_core.fp32 | mupti |

### torch-ref

| M | N | dtype | device_busy_ms | latency_ms | tflops | bandwidth_tbs | flops | bytes | compute_roof | timing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 14336 | torch.float16 | 0.3900 | 0.3910 | 0.4538 | 0.2255 | 176160768 | 180000000 | cuda_core.fp32 | mupti |
| 1 | 14336 | torch.bfloat16 | 0.0119 | 0.0121 | 0.0072 | 0.0072 | 86016 | 86016 | cuda_core.fp32 | mupti |
"""


@pytest.mark.smoke
def test_parse_profile_log_extracts_rows_and_numeric_metrics() -> None:
    report = parse_profile_log(SAMPLE_PROFILE)

    assert report.environment == ["Torch version: 2.9.1", "GPU model: MTT S5000"]
    assert len(report.rows) == 4
    tileops = report.rows[0]
    assert tileops.op == "SiluAndMulFwdOp"
    assert tileops.tag == "tileops"
    assert tileops.params == {"M": "2048", "N": "14336", "dtype": "torch.float16"}
    assert tileops.metrics["device_busy_ms"] == pytest.approx(0.1638)
    assert tileops.metrics["flops"] == pytest.approx(176160768)


@pytest.mark.smoke
def test_build_roofline_points_computes_intensity_bound_and_efficiency() -> None:
    report = parse_profile_log(SAMPLE_PROFILE)
    profile = HardwareProfile(
        gpu="Test GPU",
        hbm=RooflineCeiling(theoretical=2.0e12, effective=1.8e12),
        roofs={"cuda_core.fp32": RooflineCeiling(theoretical=10.0e12, effective=8.0e12)},
    )

    points = build_roofline_points(report, profile)

    assert len(points) == 4
    point = points[0]
    assert point.arithmetic_intensity == pytest.approx(176160768 / 180000000)
    assert point.achieved_tflops == pytest.approx(176160768 / 0.1638e-3 / 1e12)
    assert point.bound == "memory"
    assert point.sol_efficiency == pytest.approx(
        point.achieved_tflops / (1.8e12 * point.arithmetic_intensity / 1e12)
    )


@pytest.mark.smoke
def test_render_html_contains_roofline_analysis() -> None:
    report = parse_profile_log(SAMPLE_PROFILE)
    profile = HardwareProfile(
        gpu="Test GPU",
        hbm=RooflineCeiling(theoretical=2.0e12, effective=1.8e12),
        roofs={"cuda_core.fp32": RooflineCeiling(theoretical=10.0e12, effective=8.0e12)},
    )
    html = render_html(report, profile)

    assert "TileOPs Benchmark Visual Report" in html
    assert "SiluAndMulFwdOp" in html
    assert "Roofline analysis" in html
    assert "arithmetic intensity" in html.lower()
    assert "HBM/compute effective" in html
    assert "cuda_core.fp32 effective" in html
    assert "memory-bound" in html
    assert "SOL efficiency" in html
    assert "<svg" in html
    assert "cdn.plot.ly" not in html
    assert 'points="112.0,-' not in html


@pytest.mark.smoke
def test_render_png_writes_valid_png_with_nonempty_pixels(tmp_path: Path) -> None:
    report = parse_profile_log(SAMPLE_PROFILE)
    profile = HardwareProfile(
        gpu="Test GPU",
        hbm=RooflineCeiling(theoretical=2.0e12, effective=1.8e12),
        roofs={"cuda_core.fp32": RooflineCeiling(theoretical=10.0e12, effective=8.0e12)},
    )
    output = tmp_path / "profile_report.png"

    render_png(report, output, profile)

    data = output.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 1_000
    _, height = struct.unpack(">II", data[16:24])
    assert height >= 700


@pytest.mark.smoke
def test_write_profile_visualizations_writes_html_and_png(tmp_path: Path) -> None:
    input_path = tmp_path / "profile_run.log"
    html_path = tmp_path / "report.html"
    png_path = tmp_path / "report.png"
    input_path.write_text(SAMPLE_PROFILE, encoding="utf-8")

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        """gpu: Test GPU
hbm:
  theoretical: 2.0e12
  calibration: 0.9
cuda_core:
  fp32:
    theoretical: 10.0e12
    calibration: 0.8
""",
        encoding="utf-8",
    )

    written = write_profile_visualizations(
        input_path, html_path, png_path, profile_path=profile_path
    )

    assert written == (html_path, png_path)
    assert html_path.exists()
    assert png_path.exists()
    assert "Hardware profile: Test GPU" in html_path.read_text(encoding="utf-8")
    assert "cuda_core.fp32 effective" in html_path.read_text(encoding="utf-8")


@pytest.mark.smoke
def test_write_profile_visualizations_accepts_vendor_profile(tmp_path: Path) -> None:
    input_path = tmp_path / "profile_run.log"
    html_path = tmp_path / "s5000-report.html"
    png_path = tmp_path / "s5000-report.png"
    input_path.write_text(SAMPLE_PROFILE, encoding="utf-8")

    profile_path = (
        Path(__file__).parents[1]
        / "src"
        / "tileops"
        / "perf"
        / "profiles"
        / "mthreads"
        / "S5000.yaml"
    )
    written = write_profile_visualizations(
        input_path, html_path, png_path, profile_path=profile_path
    )

    assert written == (html_path, png_path)
    html = html_path.read_text(encoding="utf-8")
    assert "Hardware profile: MTT S5000" in html
    assert "HBM/compute effective" in html


@pytest.mark.smoke
def test_parse_profile_log_accepts_string_path(tmp_path: Path) -> None:
    input_path = tmp_path / "profile_run.log"
    input_path.write_text(SAMPLE_PROFILE, encoding="utf-8")

    report = parse_profile_log(str(input_path))

    assert len(report.rows) == 4


@pytest.mark.smoke
def test_parse_profile_log_rejects_table_with_wrong_column_count() -> None:
    malformed = SAMPLE_PROFILE.replace(
        "| 2048 | 14336 | torch.float16 | 0.1638 | 0.1642 | 1.0800 | 0.5370 | 176160768 | 180000000 | cuda_core.fp32 | mupti |",
        "| 2048 | 14336 | torch.float16 | 0.1638 |",
        1,
    )

    with pytest.raises(ValueError, match="column count"):
        parse_profile_log(malformed)
