import json
import math

from agents import pipeline_validator
from agents.agent1_training_specialist import SEARCH_SPACE


def _clean_decision_log(iteration=0):
    return {
        "iteration": iteration,
        "path_taken": "evidence",
        "evidence_considered": 1,
        "summary_considered": False,
        "params": {k: {"before": 0.5, "after": 0.5, "changed": False, "reason": "unchanged: no signal for this parameter"}
                   for k in SEARCH_SPACE},
        "lr_clamps": {},
    }


def test_validate_agent1_decision_clean_input_has_no_issues():
    issues = pipeline_validator.validate_agent1_decision(_clean_decision_log(), evidence=[{"x": 1}], latest_summary=None)
    assert issues == []


def test_validate_agent1_decision_fatal_when_signal_but_no_log():
    issues = pipeline_validator.validate_agent1_decision(None, evidence=[{"hyperparameter_importance": {"n_layer": 0.9}}], latest_summary=None)
    assert len(issues) == 1
    assert issues[0].severity == pipeline_validator.FATAL
    assert issues[0].source == "agent1"


def test_validate_agent1_decision_no_signal_no_log_is_fine():
    # No evidence/summary at all -> no log is expected, not an error.
    issues = pipeline_validator.validate_agent1_decision(None, evidence=None, latest_summary=None)
    assert issues == []


def test_validate_agent1_decision_flags_nan_param():
    log = _clean_decision_log()
    log["params"]["matrix_lr"]["after"] = float("nan")
    issues = pipeline_validator.validate_agent1_decision(log, evidence=[{"x": 1}], latest_summary=None)
    assert any(i.severity == pipeline_validator.ERROR and "matrix_lr" in i.message for i in issues)


def test_validate_agent1_decision_boundary_pinning_warns(tmp_path):
    param, (lo, hi) = next(iter(SEARCH_SPACE.items()))
    for i in range(3):
        log = _clean_decision_log(iteration=i)
        log["params"][param]["after"] = lo
        (tmp_path / f"decision_{i:04d}.json").write_text(json.dumps(log))
    issues = pipeline_validator.validate_agent1_decision(
        _clean_decision_log(iteration=2),
        evidence=None, latest_summary=None, decisions_dir=tmp_path, lookback=3,
    )
    assert any(i.severity == pipeline_validator.WARN and param in i.message for i in issues)


def test_validate_training_result_flags_clamp_mismatch():
    metrics = {"val_bpb": 1.3, "hyperparam_clamps": {"n_embd": {"requested": 473, "clamped": 484, "bounds": [11, 8192]}}}
    issues = pipeline_validator.validate_training_result(metrics, {"n_embd": 473})
    assert len(issues) == 1
    assert issues[0].severity == pipeline_validator.ERROR
    assert "n_embd" in issues[0].message and "473" in issues[0].message and "484" in issues[0].message


def test_validate_training_result_flags_nan_val_bpb():
    issues = pipeline_validator.validate_training_result({"val_bpb": float("nan")}, {})
    assert any(i.severity == pipeline_validator.ERROR and "NaN" in i.message for i in issues)


def test_validate_training_result_clean_input_has_no_issues():
    issues = pipeline_validator.validate_training_result({"val_bpb": 1.3}, {"n_embd": 512})
    assert issues == []


def test_validate_agent2_report_flags_out_of_range_importance():
    issues = pipeline_validator.validate_agent2_report({"hyperparameter_importance": {"n_layer": 1.5}})
    assert any(i.severity == pipeline_validator.ERROR for i in issues)


def test_validate_agent2_report_flags_nan_importance():
    issues = pipeline_validator.validate_agent2_report({"hyperparameter_importance": {"n_layer": float("nan")}})
    assert any(i.severity == pipeline_validator.ERROR for i in issues)


def test_validate_agent2_report_warns_on_implausible_head_impact():
    issues = pipeline_validator.validate_agent2_report({"important_heads": [{"head": "L0_H0", "impact": 5.0}]})
    assert any(i.severity == pipeline_validator.WARN for i in issues)


def test_validate_agent2_report_clean_input_has_no_issues():
    issues = pipeline_validator.validate_agent2_report({"hyperparameter_importance": {"n_layer": 0.5}, "important_heads": [{"head": "L0_H0", "impact": 0.001}]})
    assert issues == []


def test_validate_agent3_summary_flags_out_of_bounds_recommendation():
    param = "matrix_lr"
    lo, hi = SEARCH_SPACE[param]
    issues = pipeline_validator.validate_agent3_summary({"recommended_hyperparams": {param: hi + 10}})
    assert any(i.severity == pipeline_validator.ERROR and param in i.message for i in issues)


def test_validate_agent3_summary_warns_on_empty_findings_with_history():
    issues = pipeline_validator.validate_agent3_summary({"stable_patterns": [], "conflicting_signals": []}, total_reports=15)
    assert any(i.severity == pipeline_validator.WARN for i in issues)


def test_validate_agent3_summary_no_warning_with_little_history():
    issues = pipeline_validator.validate_agent3_summary({"stable_patterns": [], "conflicting_signals": []}, total_reports=2)
    assert issues == []


def test_prune_old_runs_keeps_only_most_recent(tmp_path):
    for i in range(12):
        (tmp_path / f"run_202601{i:02d}_000000").mkdir()
    pipeline_validator.prune_old_runs(tmp_path, keep=10)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert len(remaining) == 10
    # The 2 oldest (run_20260100, run_20260101) should be gone; the 10 most recent kept.
    assert "run_20260100_000000" not in remaining
    assert "run_20260111_000000" in remaining


def test_prune_old_runs_noop_when_fewer_than_keep(tmp_path):
    for i in range(3):
        (tmp_path / f"run_202601{i:02d}_000000").mkdir()
    pipeline_validator.prune_old_runs(tmp_path, keep=10)
    assert len(list(tmp_path.iterdir())) == 3


def test_write_iteration_issues_accumulates_and_marks_suspect(tmp_path):
    issue = pipeline_validator.Issue(pipeline_validator.ERROR, "agent2", "test issue")
    pipeline_validator.write_iteration_issues(tmp_path, 0, [issue], suspect=True)
    data = json.loads((tmp_path / "iteration_0000.json").read_text())
    assert data["suspect"] is True
    assert len(data["issues"]) == 1

    # A second phase's issues in the same iteration accumulate rather than overwrite.
    issue2 = pipeline_validator.Issue(pipeline_validator.WARN, "agent3", "another issue")
    pipeline_validator.write_iteration_issues(tmp_path, 0, [issue2], suspect=False)
    data2 = json.loads((tmp_path / "iteration_0000.json").read_text())
    assert len(data2["issues"]) == 2
    assert data2["suspect"] is True  # sticky once suspect


def test_render_issues_empty_is_ok_message():
    assert "OK" in pipeline_validator.render_issues([])


def test_render_issues_includes_severity_and_source():
    issue = pipeline_validator.Issue(pipeline_validator.ERROR, "train", "something broke")
    text = pipeline_validator.render_issues([issue])
    assert "ERROR" in text and "train" in text and "something broke" in text
