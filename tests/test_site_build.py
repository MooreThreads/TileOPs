"""The index marks a hand-started run, and only that.

A night the timer started is the norm, so History has to say when one was not:
the badge beside the date is what tells a re-run from the scheduled sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.site.build_site import build

pytestmark = pytest.mark.smoke

RECORD = {
    "schema_version": 1,
    "run_id": "nightly_20260101",
    "benchmark_run_id": "nightly_20260101",
    "generated_at": "2026-01-01T02:00:00+00:00",
    "devices": [0],
    "environment": {"device_model": "MTT S5000", "torch": "2.9.1"},
    "source": {"git_head": "0" * 40, "git_branch": "master", "git_dirty": False},
    "summary": {"ops_total": 0, "correctness": {}, "benchmark": {}, "overall": {}},
    "operators": [],
}

BADGE = '<td>2026-01-01<span class="tag"'


def _index(tmp_path: Path, *, trigger: str) -> str:
    """Build a one-run site and return the index it wrote."""
    run_dir = tmp_path / "runs" / "nightly_20260101"
    run_dir.mkdir(parents=True)
    record = tmp_path / "operator_status.json"
    record.write_text(json.dumps(RECORD), encoding="utf-8")
    build(
        run_dir=run_dir,
        record_path=record,
        out=tmp_path / "site",
        profile=None,
        date="2026-01-01",
        trigger=trigger,
        keep_days=0,
    )
    return (tmp_path / "site" / "index.html").read_text(encoding="utf-8")


def test_a_hand_started_run_is_marked_beside_its_date(tmp_path: Path) -> None:
    assert BADGE in _index(tmp_path, trigger="manual")


def test_the_scheduled_night_carries_no_mark(tmp_path: Path) -> None:
    assert '<span class="tag"' not in _index(tmp_path, trigger="timer")


def test_a_run_with_no_recorded_trigger_is_left_unmarked(tmp_path: Path) -> None:
    """No badge is not a claim that the timer started it."""
    assert '<span class="tag"' not in _index(tmp_path, trigger="")
