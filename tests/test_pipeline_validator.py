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


def test_render_issues_includes_context_inline():
    issue = pipeline_validator.Issue(pipeline_validator.WARN, "agent2", "something odd",
                                      {"field": "attn_entropy", "index": 3})
    text = pipeline_validator.render_issues([issue])
    assert "field=attn_entropy" in text
    assert "index=3" in text


# ---------------------------------------------------------------------------
# Tier 2: token_fingerprint validation (via validate_agent2_report)
# ---------------------------------------------------------------------------

def _clean_fingerprint(n_layer=4):
    return {
        "attn_entropy": [1.0] * n_layer,
        "attn_distance": [5.0] * n_layer,
        "dla": [0.1] * n_layer,
        "x0_lambda": [2.0] * n_layer,
        "pos_saliency": [0.05] * 16,
        "induction_score": 0.2,
        "attn_distance_slope": 0.1,
    }


def test_validate_agent2_report_clean_fingerprint_has_no_issues():
    evidence = {"token_fingerprint": _clean_fingerprint()}
    assert pipeline_validator.validate_agent2_report(evidence) == []


def test_validate_agent2_report_absent_fingerprint_has_no_issues():
    assert pipeline_validator.validate_agent2_report({}) == []
    assert pipeline_validator.validate_agent2_report({"token_fingerprint": {}}) == []


def test_validate_agent2_report_flags_nan_in_fingerprint_array():
    fp = _clean_fingerprint()
    fp["attn_entropy"][2] = float("nan")
    issues = pipeline_validator.validate_agent2_report({"token_fingerprint": fp})
    assert any(i.severity == pipeline_validator.ERROR and "attn_entropy" in i.message for i in issues)


def test_validate_agent2_report_flags_negative_entropy():
    fp = _clean_fingerprint()
    fp["attn_entropy"][0] = -0.5  # entropy can never be negative
    issues = pipeline_validator.validate_agent2_report({"token_fingerprint": fp})
    assert any(i.severity == pipeline_validator.ERROR and "negative" in i.message for i in issues)


def test_validate_agent2_report_allows_negative_dla():
    # dla is a signed logit contribution -- negative is legitimate, unlike entropy/distance/pos_saliency.
    fp = _clean_fingerprint()
    fp["dla"][0] = -0.9
    issues = pipeline_validator.validate_agent2_report({"token_fingerprint": fp})
    assert issues == []


def test_validate_agent2_report_flags_empty_array_present():
    fp = _clean_fingerprint()
    fp["dla"] = []
    issues = pipeline_validator.validate_agent2_report({"token_fingerprint": fp})
    assert any(i.severity == pipeline_validator.WARN and "dla" in i.message and "empty" in i.message for i in issues)


def test_validate_agent2_report_flags_induction_score_out_of_range():
    fp = _clean_fingerprint()
    fp["induction_score"] = 1.5
    issues = pipeline_validator.validate_agent2_report({"token_fingerprint": fp})
    assert any(i.severity == pipeline_validator.ERROR and "induction_score" in i.message for i in issues)


def test_validate_agent2_report_flags_nan_attn_distance_slope():
    fp = _clean_fingerprint()
    fp["attn_distance_slope"] = float("nan")
    issues = pipeline_validator.validate_agent2_report({"token_fingerprint": fp})
    assert any(i.severity == pipeline_validator.ERROR and "attn_distance_slope" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Tier 3: fingerprint_clusters validation (via validate_agent3_summary)
# ---------------------------------------------------------------------------

def test_validate_agent3_summary_clean_clusters_have_no_issues():
    summary = {"fingerprint_clusters": {
        "overall": {"k": 2, "silhouette": 0.6, "clusters": [{"cluster_id": 1, "n": 10}, {"cluster_id": 2, "n": 8}]},
    }}
    assert pipeline_validator.validate_agent3_summary(summary) == []


def test_validate_agent3_summary_absent_clusters_have_no_issues():
    assert pipeline_validator.validate_agent3_summary({}) == []
    assert pipeline_validator.validate_agent3_summary({"fingerprint_clusters": {}}) == []


def test_validate_agent3_summary_warns_on_nonpositive_silhouette():
    summary = {"fingerprint_clusters": {"overall": {"k": 2, "silhouette": -0.1, "clusters": [{"cluster_id": 1, "n": 5}]}}}
    issues = pipeline_validator.validate_agent3_summary(summary)
    assert any(i.severity == pipeline_validator.WARN and "no better than a random split" in i.message for i in issues)


def test_validate_agent3_summary_warns_on_weak_silhouette():
    summary = {"fingerprint_clusters": {"trajectory": {"k": 2, "silhouette": 0.1, "clusters": [{"cluster_id": 1, "n": 5}]}}}
    issues = pipeline_validator.validate_agent3_summary(summary)
    assert any(i.severity == pipeline_validator.WARN and "weak" in i.message for i in issues)


def test_validate_agent3_summary_warns_on_degenerate_cluster_size():
    summary = {"fingerprint_clusters": {"overall": {"k": 2, "silhouette": 0.6, "clusters": [{"cluster_id": 1, "n": 1}, {"cluster_id": 2, "n": 9}]}}}
    issues = pipeline_validator.validate_agent3_summary(summary)
    assert any(i.severity == pipeline_validator.WARN and "only 1 member" in i.message for i in issues)


# ---------------------------------------------------------------------------
# Tier 4: fingerprint_adjustments thrashing detection (via validate_agent1_decision)
# ---------------------------------------------------------------------------

def _write_decision_with_fingerprint_adjustment(decisions_dir, iteration, param, delta):
    log = _clean_decision_log(iteration=iteration)
    log["fingerprint_adjustments"] = [{"param": param, "votes": [1 if delta > 0 else -1], "delta": delta, "new_value": 10}]
    (decisions_dir / f"decision_{iteration:04d}.json").write_text(json.dumps(log))


def test_validate_agent1_decision_flags_thrashing_fingerprint_adjustments(tmp_path):
    for i, delta in enumerate([1, -1, 1, -1]):
        _write_decision_with_fingerprint_adjustment(tmp_path, i, "n_layer", delta)
    current = _clean_decision_log(iteration=3)
    issues = pipeline_validator.validate_agent1_decision(current, evidence=None, latest_summary=None, decisions_dir=tmp_path)
    assert any(i.severity == pipeline_validator.WARN and "n_layer" in i.message and "alternating" in i.message for i in issues)


def test_validate_agent1_decision_no_thrashing_warning_for_consistent_direction(tmp_path):
    for i, delta in enumerate([1, 1, 1, 1]):
        _write_decision_with_fingerprint_adjustment(tmp_path, i, "n_layer", delta)
    current = _clean_decision_log(iteration=3)
    issues = pipeline_validator.validate_agent1_decision(current, evidence=None, latest_summary=None, decisions_dir=tmp_path)
    assert not any("alternating" in i.message for i in issues)


def test_validate_agent1_decision_no_thrashing_warning_below_min_occurrences(tmp_path):
    for i, delta in enumerate([1, -1]):  # only 2 occurrences, need >=3
        _write_decision_with_fingerprint_adjustment(tmp_path, i, "n_layer", delta)
    current = _clean_decision_log(iteration=1)
    issues = pipeline_validator.validate_agent1_decision(current, evidence=None, latest_summary=None, decisions_dir=tmp_path)
    assert not any("alternating" in i.message for i in issues)


# ---------------------------------------------------------------------------
# validate_batch_accumulation
# ---------------------------------------------------------------------------

def test_validate_batch_accumulation_warns_when_stalled():
    issues = pipeline_validator.validate_batch_accumulation(report_batch_size=10, configured_batch_size=3)
    assert any(i.severity == pipeline_validator.WARN and "stuck" in i.message for i in issues)


def test_validate_batch_accumulation_clean_within_threshold():
    assert pipeline_validator.validate_batch_accumulation(report_batch_size=2, configured_batch_size=3) == []


def test_validate_batch_accumulation_noop_when_configured_size_not_positive():
    assert pipeline_validator.validate_batch_accumulation(report_batch_size=100, configured_batch_size=0) == []
