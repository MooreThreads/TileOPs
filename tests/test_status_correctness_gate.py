"""Tests for the benchmark correctness gate in ``scripts/status/benchmark.py``.

The benchmark only certifies an operator whose correctness is settled, so
``_correctness_gate`` keeps the operators a gate mode allows and records a reason
for the rest. These run without a GPU.
"""

import json

import pytest

from scripts.status import benchmark

pytestmark = pytest.mark.smoke


def _correctness(tmp_path, rows):
    path = tmp_path / "correctness" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ops": rows}), encoding="utf-8")
    return tmp_path


_ROWS = [
    {"op": "A", "status": "supported"},
    {"op": "B", "status": "partial"},
    {"op": "C", "status": "failed"},
]


def test_supported_gate_keeps_only_supported(tmp_path):
    run_dir = _correctness(tmp_path, _ROWS)
    eligible, excluded = benchmark._correctness_gate(run_dir, ["A", "B", "C"], "supported")
    assert eligible == ["A"]
    assert set(excluded) == {"B", "C"}
    assert "partial" in excluded["B"]


def test_none_gate_keeps_everything(tmp_path):
    eligible, excluded = benchmark._correctness_gate(tmp_path, ["A", "B"], "none")
    assert eligible == ["A", "B"]
    assert excluded == {}


def test_supported_or_partial_gate(tmp_path):
    run_dir = _correctness(tmp_path, _ROWS)
    eligible, excluded = benchmark._correctness_gate(
        run_dir, ["A", "B", "C"], "supported_or_partial"
    )
    assert eligible == ["A", "B"]
    assert set(excluded) == {"C"}


def test_gate_requires_a_correctness_summary(tmp_path):
    with pytest.raises(SystemExit):
        benchmark._correctness_gate(tmp_path, ["A"], "supported")


def test_aggregate_records_the_gate_reason():
    manifest = {"A": {"family": "x", "status": "implemented", "source": {"bench": "b.py"}}}
    rows = benchmark.aggregate(
        manifest, ["A"], set(), [], {"A": "correctness partial; gate supported"}
    )
    assert rows[0]["status"] == "not_run"
    assert rows[0]["reason"] == "correctness partial; gate supported"


# TODO: need a better strategy to distinguish a genuinely partial operator from
# a shared correctness case mis-attributed to it. Today the gate drops every
# partial operator, even one whose own cases all pass. Replace this placeholder
# with a test that keeps such an operator once the gate can tell them apart.
@pytest.mark.skip(reason="need a better strategy to distinguish a shared-case partial")
def test_partial_operator_whose_own_cases_pass_is_kept():
    raise AssertionError("not implemented")
