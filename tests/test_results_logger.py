"""Synthetic-data tests for state/results_logger.py's "device" and
"window_s_fraction" columns -- confirm they round-trip through
log_result/load_results, and that the legacy-schema rename guard still
fires correctly each time COLUMNS grows.

window_s_fraction (dev/checks.txt follow-up: the "search never narrows"
investigation) was in state/results_analysis.py's HYPERPARAM_COLUMNS --
proposed/tuned as a real search dimension -- but was never actually written
to results.tsv, so search_planner.propose_next()'s n_usable count (rows
with every HYPERPARAM_COLUMNS field present) was 0 for every historical
row, forever, and the surrogate's cold-start check never passed.
"""

from state.results_analysis import HYPERPARAM_COLUMNS, load_results
from state.results_logger import COLUMNS, log_result


def test_window_s_fraction_column_is_last_in_schema():
    assert COLUMNS[-1] == "window_s_fraction"


def test_every_hyperparameter_column_is_actually_logged():
    """Regression guard for the actual bug: every field
    state/results_analysis.py's HYPERPARAM_COLUMNS lists as a real search
    dimension must be a real results.tsv column, or search_planner's
    n_usable count silently stays at 0 forever regardless of how much data
    accumulates."""
    for param in HYPERPARAM_COLUMNS:
        assert param in COLUMNS, f"{param} is tuned (HYPERPARAM_COLUMNS) but never logged to results.tsv"


def test_log_result_writes_window_s_fraction_from_hyperparams(tmp_path):
    path = tmp_path / "results.tsv"
    log_result("run_0000", {"n_layer": 8, "window_s_fraction": 0.75}, {"val_bpb": 1.1, "status": "remote_ok"},
               results_path=str(path))

    rows = load_results(str(path))
    assert len(rows) == 1
    assert rows[0]["window_s_fraction"] == 0.75


def test_log_result_writes_device_from_metrics(tmp_path):
    path = tmp_path / "results.tsv"
    log_result("run_0000", {"n_layer": 8}, {"val_bpb": 1.1, "status": "remote_ok", "device": 3},
               results_path=str(path))

    rows = load_results(str(path))
    assert len(rows) == 1
    assert rows[0]["device"] == "3"


def test_log_result_leaves_device_blank_when_metrics_has_none(tmp_path):
    path = tmp_path / "results.tsv"
    # status "ok" (not "dry_run"/"simulated") -- load_results drops synthetic
    # statuses (see state/results_analysis.py::SYNTHETIC_STATUSES), and this
    # test's actual point is the blank "device" field, not status handling.
    log_result("run_0000", {"n_layer": 8}, {"val_bpb": 1.1, "status": "ok"},
               results_path=str(path))

    rows = load_results(str(path))
    assert len(rows) == 1
    assert "device" not in rows[0]  # blank fields are omitted by _coerce_row, not fabricated as ""


def test_legacy_header_mismatch_still_triggers_rename_after_device_column_added(tmp_path):
    path = tmp_path / "results.tsv"
    # Write a results.tsv under the OLD (pre-device-column) header shape.
    old_header = "\t".join(COLUMNS[:-1])
    path.write_text(old_header + "\n" + "2024-01-01T00:00:00\trun_0000\t8\n")

    log_result("run_0001", {"n_layer": 8}, {"val_bpb": 1.0, "status": "ok"}, results_path=str(path))

    legacy_files = list(tmp_path.glob("results.tsv.legacy-*"))
    assert len(legacy_files) == 1
    # Fresh file now has the current (device-inclusive) header + the new row only.
    rows = load_results(str(path))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run_0001"
