"""Synthetic-data tests for state/results_logger.py's "device" column
(dev/checks.txt item 1: multi-GPU parallel search) -- confirms it round-
trips through log_result/load_results, and that the existing legacy-schema
rename guard still fires correctly now that COLUMNS has grown by one.
"""

from state.results_analysis import load_results
from state.results_logger import COLUMNS, log_result


def test_device_column_is_last_in_schema():
    assert COLUMNS[-1] == "device"


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
