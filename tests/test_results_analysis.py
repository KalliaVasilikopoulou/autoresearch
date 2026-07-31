"""Synthetic-data tests for state/results_analysis.py's load_results and its
SYNTHETIC_STATUSES filter. dry_run/simulated rows carry a fixed-formula
val_bpb (not a measured result -- see agents/agent1_training_specialist.py's
train_model dry_run branch and _simulate_training_result) and must never
enter a numeric aggregate that treats val_bpb as comparable across runs:
hyperparameter correlations, the Tier 1 surrogate, noise floor, elite-run
selection. All of those go through load_results(), so filtering there is
the single choke point -- these tests confirm the filter itself, plus one
regression guard showing what it protects against.
"""

from state.results_analysis import (
    SYNTHETIC_STATUSES,
    hyperparameter_correlations,
    load_results,
    top_quartile_by_val_bpb,
)
from state.results_logger import log_result


def _base_hp(matrix_lr=0.04):
    return {"n_layer": 8, "n_embd": 512, "n_head": 4, "matrix_lr": matrix_lr}


def test_synthetic_statuses_contains_dry_run_and_simulated():
    assert SYNTHETIC_STATUSES == {"dry_run", "simulated"}


def test_load_results_drops_dry_run_rows(tmp_path):
    path = tmp_path / "results.tsv"
    log_result("run_0000", _base_hp(), {"val_bpb": 0.998, "status": "dry_run"}, results_path=str(path))
    log_result("run_0001", _base_hp(), {"val_bpb": 1.1, "status": "remote_ok"}, results_path=str(path))

    rows = load_results(str(path))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_0001"


def test_load_results_drops_simulated_rows(tmp_path):
    path = tmp_path / "results.tsv"
    log_result("run_0000", _base_hp(), {"val_bpb": 0.9, "status": "simulated"}, results_path=str(path))
    log_result("run_0001", _base_hp(), {"val_bpb": 1.1, "status": "remote_ok"}, results_path=str(path))

    rows = load_results(str(path))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_0001"


def test_load_results_keeps_real_statuses(tmp_path):
    path = tmp_path / "results.tsv"
    log_result("run_0000", _base_hp(), {"val_bpb": 1.1, "status": "remote_ok"}, results_path=str(path))
    log_result("run_0001", _base_hp(), {"val_bpb": float("inf"), "status": "remote_error"}, results_path=str(path))
    log_result("run_0002", _base_hp(), {"val_bpb": 1.3, "status": "ok"}, results_path=str(path))

    rows = load_results(str(path))
    assert {r["run_id"] for r in rows} == {"run_0000", "run_0001", "run_0002"}


def test_dry_run_row_does_not_skew_hyperparameter_correlation(tmp_path):
    """Regression guard for the actual bug: dry_run's val_bpb is a function
    of iteration alone, paired here with a matrix_lr value that points the
    "wrong way" relative to the real rows' true relationship. Unfiltered,
    it would pull the correlation away from -1.0; filtered, the real rows'
    perfect (and deliberately clean) relationship is untouched.
    """
    path = tmp_path / "results.tsv"
    # Real rows: matrix_lr strictly increasing -> val_bpb strictly decreasing
    # (perfect negative correlation, by construction).
    for i, matrix_lr in enumerate([0.01, 0.02, 0.03, 0.04, 0.05]):
        log_result(f"run_{i:04d}", _base_hp(matrix_lr), {"val_bpb": 2.0 - matrix_lr, "status": "remote_ok"},
                   results_path=str(path))
    # A dry_run row with a high matrix_lr but a low (fake) val_bpb -- if
    # counted, this contradicts the real rows' relationship.
    log_result("run_0005", _base_hp(0.5), {"val_bpb": 0.5, "status": "dry_run"}, results_path=str(path))

    rows = load_results(str(path))
    correlations = hyperparameter_correlations(rows, min_n=4)
    assert correlations["matrix_lr"]["correlation"] == -1.0
    assert correlations["matrix_lr"]["n"] == 5  # the dry_run row excluded, not just outvoted


# ---------------------------------------------------------------------------
# top_quartile_by_val_bpb -- shared "what counts as elite" selection used by
# both Agent 3's hyperparameter recommendations and Agent 2's stuck-signal
# reference value.
# ---------------------------------------------------------------------------

def test_top_quartile_by_val_bpb_empty_input():
    assert top_quartile_by_val_bpb([]) == []


def test_top_quartile_by_val_bpb_returns_at_least_one():
    # len=3, 3*0.25=0.75 -> int() truncates to 0 -> max(1, 0) = 1
    candidates = [(1.5, "a"), (1.2, "b"), (1.8, "c")]
    result = top_quartile_by_val_bpb(candidates)
    assert result == [(1.2, "b")]  # the single best (lowest val_bpb)


def test_top_quartile_by_val_bpb_matches_floor_division_by_four():
    # Regression guard: this must reproduce the exact "n // 4" count Agent 3
    # used before this was extracted into a shared helper.
    candidates = [(float(i), i) for i in range(20)]  # val_bpb == payload, best (lowest) is 0
    result = top_quartile_by_val_bpb(candidates)
    assert len(result) == 20 // 4 == 5
    assert [payload for _, payload in result] == [0, 1, 2, 3, 4]


def test_top_quartile_by_val_bpb_sorted_ascending_best_first():
    candidates = [(3.0, "c"), (1.0, "a"), (2.0, "b")]
    result = top_quartile_by_val_bpb(candidates, fraction=1.0)
    assert [payload for _, payload in result] == ["a", "b", "c"]
