"""Tier 1 step 8: report the campaign's answer from the holdout shard, and
track whether the validation shard has drifted away from it.

The distinction under test: the two shards are different text, so a CONSTANT
gap between them is expected and harmless -- it affects every run equally and
so changes no ranking. A GROWING gap is not: that is the search progressively
fitting the quirks of the one shard it makes every accept/reject decision
against.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from state.results_analysis import holdout_drift

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "holdout_eval.py"
_spec = importlib.util.spec_from_file_location("holdout_eval", _SCRIPT)
holdout_eval = importlib.util.module_from_spec(_spec)
sys.modules["holdout_eval"] = holdout_eval
_spec.loader.exec_module(holdout_eval)


def _rows(pairs):
    """pairs: list of (val_bpb, holdout_val_bpb), oldest first."""
    return [
        {"run_id": f"run_{i:04d}", "timestamp": f"2026-08-06T{i:02d}:00:00",
         "val_bpb": v, "holdout_val_bpb": h}
        for i, (v, h) in enumerate(pairs)
    ]


def test_a_constant_offset_is_not_drift():
    """The shards are different text; one is simply harder. A fixed gap shifts
    every run equally and changes no ranking."""
    drift = holdout_drift(_rows([(1.30, 1.32), (1.29, 1.31), (1.28, 1.30), (1.27, 1.29)]))

    assert drift["n"] == 4
    assert drift["mean_gap"] == pytest.approx(0.02)
    assert drift["growing"] is False


def test_a_growing_gap_is_flagged():
    """val_bpb keeps improving while holdout stays put -- the signature of
    fitting the validation shard rather than the language."""
    drift = holdout_drift(_rows([(1.30, 1.32), (1.28, 1.32), (1.26, 1.325), (1.24, 1.33)]))

    assert drift["growth"] > 0
    assert drift["growing"] is True


def test_noisy_but_flat_gaps_are_not_called_drift():
    """Guards against the obvious false positive: a gap that bounces around
    must not read as a trend just because the later half happens to be higher
    by less than the bouncing itself.

    Gaps here are 0.01 / 0.06 / 0.03 / 0.05 -- spread 0.0222, half-to-half
    change only 0.005, so the movement is well inside the noise."""
    drift = holdout_drift(_rows([(1.30, 1.31), (1.29, 1.35), (1.28, 1.31), (1.27, 1.32)]))

    assert drift["growth"] == pytest.approx(0.005)
    assert drift["std_gap"] > drift["growth"]
    assert drift["growing"] is False, "change between halves is within the gaps' own spread"


def test_rows_are_ordered_by_time_not_by_file_order():
    rows = _rows([(1.30, 1.32), (1.28, 1.36)])
    rows[0]["timestamp"] = "2026-08-06T09:00:00"
    rows[1]["timestamp"] = "2026-08-06T08:00:00"

    drift = holdout_drift(rows)

    assert [r["run_id"] for r in drift["runs"]] == ["run_0001", "run_0000"]


def test_runs_without_a_holdout_score_are_ignored():
    rows = _rows([(1.30, 1.32), (1.29, 1.31)])
    rows.append({"run_id": "run_0002", "timestamp": "2026-08-06T05:00:00", "val_bpb": 1.20})
    rows.append({"run_id": "run_0003", "timestamp": "2026-08-06T06:00:00",
                 "val_bpb": 1.20, "holdout_val_bpb": float("inf")})

    drift = holdout_drift(rows)

    assert drift["n"] == 2


def test_too_little_history_returns_none_rather_than_a_trend_from_one_point():
    assert holdout_drift([]) is None
    assert holdout_drift(_rows([(1.30, 1.32)])) is None


def test_the_no_history_message_explains_why_it_is_a_winners_sample(tmp_path):
    """The honest caveat: the orchestrator only scores the holdout on a new
    best, so this record is a sample of winners, which is exactly why the
    top-K re-check exists alongside it."""
    text = holdout_eval.render_drift(None)

    assert "fewer than two runs" in text
    assert "WINNERS" in text


def test_drift_report_states_plainly_whether_it_found_drift():
    growing = holdout_eval.render_drift(
        holdout_drift(_rows([(1.30, 1.32), (1.28, 1.32), (1.26, 1.325), (1.24, 1.33)])))
    flat = holdout_eval.render_drift(
        holdout_drift(_rows([(1.30, 1.32), (1.29, 1.31), (1.28, 1.30), (1.27, 1.29)])))

    assert "**DRIFT DETECTED**" in growing
    assert "No drift detected" in flat
    assert "harmless" in flat


def test_seed_pool_starts_at_the_campaigns_historical_seed():
    """So a holdout re-check's first run is directly comparable to whatever
    history already recorded for the same configuration."""
    assert holdout_eval.SEED_POOL[0] == 42
