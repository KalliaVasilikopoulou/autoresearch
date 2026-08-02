"""dev/checks.txt item 2: visualization gaps. Lightweight tests for the new
chart functions -- no existing chart function in this file had dedicated
tests before this (chart correctness here has always been verified by
manual visual inspection, not pixel assertions); these confirm the
None-vs-real-Path contract (never fabricate a chart for missing data) and
that valid synthetic data actually renders without error.
"""
from state.visualize import (
    chart_ei_candidates,
    chart_fingerprint_adjustments_trend,
    chart_interaction_matrix,
    chart_noise_floor_trend,
    chart_pipeline_issues_trend,
    chart_predicted_vs_actual,
    chart_sobol_coverage,
    chart_surrogate_sensitivity,
    chart_token_fingerprint_scalars_evolution,
    chart_val_bpb_trend,
)


# ---------------------------------------------------------------------------
# chart_predicted_vs_actual
# ---------------------------------------------------------------------------

def test_chart_predicted_vs_actual_none_when_empty(tmp_path):
    assert chart_predicted_vs_actual([], [], tmp_path / "out.png") is None


def test_chart_predicted_vs_actual_renders(tmp_path):
    actual = [1.0, 1.1, 0.9, 1.05]
    predicted = [1.02, 1.05, 0.95, 1.0]
    out = chart_predicted_vs_actual(actual, predicted, tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_surrogate_sensitivity
# ---------------------------------------------------------------------------

def test_chart_surrogate_sensitivity_none_when_empty(tmp_path):
    assert chart_surrogate_sensitivity([], [], tmp_path / "out.png") is None


def test_chart_surrogate_sensitivity_renders(tmp_path):
    ranked = [("n_layer", 0.8), ("matrix_lr", 0.5), ("weight_decay", 0.01)]
    out = chart_surrogate_sensitivity(ranked, frozen=["weight_decay"], path=tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_interaction_matrix
# ---------------------------------------------------------------------------

def test_chart_interaction_matrix_none_when_empty(tmp_path):
    assert chart_interaction_matrix({}, ["a", "b"], tmp_path / "out.png") is None


def test_chart_interaction_matrix_none_with_fewer_than_two_params(tmp_path):
    assert chart_interaction_matrix({("a", "a"): 1.0}, ["a"], tmp_path / "out.png") is None


def test_chart_interaction_matrix_renders(tmp_path):
    scores = {("a", "b"): 0.9, ("a", "c"): 0.1, ("b", "c"): 0.05}
    out = chart_interaction_matrix(scores, ["a", "b", "c"], tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_ei_candidates
# ---------------------------------------------------------------------------

def test_chart_ei_candidates_none_when_empty(tmp_path):
    assert chart_ei_candidates({}, tmp_path / "out.png") is None


def test_chart_ei_candidates_renders(tmp_path):
    diagnostics = {
        "free_params": ["n_layer", "matrix_lr"],
        "candidate_values": {"n_layer": [4, 8, 12, 16], "matrix_lr": [0.01, 0.05, 0.1, 0.15]},
        "mus": [1.0, 0.95, 0.9, 1.05],
        "sigmas": [0.1, 0.1, 0.1, 0.1],
        "eis": [0.01, 0.03, 0.05, 0.0],
        "best_idx": 2,
    }
    out = chart_ei_candidates(diagnostics, tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_sobol_coverage
# ---------------------------------------------------------------------------

def test_chart_sobol_coverage_none_when_empty(tmp_path):
    assert chart_sobol_coverage([], ["n_layer"], tmp_path / "out.png") is None


def test_chart_sobol_coverage_renders(tmp_path):
    points = [{"n_layer": v, "matrix_lr": v / 100} for v in (4, 8, 12, 16, 20)]
    out = chart_sobol_coverage(points, ["n_layer", "matrix_lr"], tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_fingerprint_adjustments_trend
# ---------------------------------------------------------------------------

def test_chart_fingerprint_adjustments_trend_none_when_empty(tmp_path):
    assert chart_fingerprint_adjustments_trend([], tmp_path / "out.png") is None


def test_chart_fingerprint_adjustments_trend_none_when_no_adjustments_present(tmp_path):
    logs = [{"iteration": 0, "params": {}}, {"iteration": 1, "params": {}}]
    assert chart_fingerprint_adjustments_trend(logs, tmp_path / "out.png") is None


def test_chart_fingerprint_adjustments_trend_renders(tmp_path):
    logs = [
        {"iteration": 0, "fingerprint_adjustments": [{"param": "n_layer", "delta": 1}]},
        {"iteration": 1, "fingerprint_adjustments": [{"param": "n_layer", "delta": -1}]},
        {"iteration": 2, "fingerprint_adjustments": []},
    ]
    out = chart_fingerprint_adjustments_trend(logs, tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_pipeline_issues_trend
# ---------------------------------------------------------------------------

def test_chart_pipeline_issues_trend_none_when_empty(tmp_path):
    assert chart_pipeline_issues_trend([], tmp_path / "out.png") is None


def test_chart_pipeline_issues_trend_renders(tmp_path):
    logs = [
        {"iteration": 0, "issues": [{"severity": "WARN"}, {"severity": "ERROR"}]},
        {"iteration": 1, "issues": []},
        {"iteration": 2, "issues": [{"severity": "FATAL"}]},
    ]
    out = chart_pipeline_issues_trend(logs, tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_token_fingerprint_scalars_evolution
# ---------------------------------------------------------------------------

def test_chart_token_fingerprint_scalars_evolution_none_when_empty(tmp_path):
    assert chart_token_fingerprint_scalars_evolution([], tmp_path / "out.png") is None


def test_chart_token_fingerprint_scalars_evolution_none_when_no_fingerprints(tmp_path):
    all_metrics = [{"val_bpb": 1.0}, {"val_bpb": 0.9}]
    assert chart_token_fingerprint_scalars_evolution(all_metrics, tmp_path / "out.png") is None


def test_chart_token_fingerprint_scalars_evolution_renders(tmp_path):
    all_metrics = [
        {"token_fingerprint": {"attn_distance_slope": 0.1, "induction_score": 0.2}},
        {"token_fingerprint": {"attn_distance_slope": -0.05, "induction_score": 0.3}},
        {"val_bpb": 1.0},  # no fingerprint this run -- must be skipped, not error
    ]
    out = chart_token_fingerprint_scalars_evolution(all_metrics, tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_noise_floor_trend
# ---------------------------------------------------------------------------

def test_chart_noise_floor_trend_none_when_empty(tmp_path):
    assert chart_noise_floor_trend([], tmp_path / "out.png") is None


def test_chart_noise_floor_trend_renders_single_point(tmp_path):
    history = [{"timestamp": "t0", "mean": 1.0, "std": 0.01, "n": 3}]
    out = chart_noise_floor_trend(history, tmp_path / "out.png")
    assert out is not None and out.exists()


def test_chart_noise_floor_trend_renders_multiple_points(tmp_path):
    history = [
        {"timestamp": "t0", "mean": 1.0, "std": 0.02, "n": 3},
        {"timestamp": "t1", "mean": 0.95, "std": 0.015, "n": 3},
    ]
    out = chart_noise_floor_trend(history, tmp_path / "out.png")
    assert out is not None and out.exists()


# ---------------------------------------------------------------------------
# chart_val_bpb_trend -- holdout overlay extension
# ---------------------------------------------------------------------------

def test_chart_val_bpb_trend_still_none_when_empty(tmp_path):
    assert chart_val_bpb_trend([], tmp_path / "nf.json", tmp_path / "out.png") is None


def test_chart_val_bpb_trend_renders_with_holdout_overlay(tmp_path):
    all_metrics = [
        {"val_bpb": 1.0, "metadata": {}},
        {"val_bpb": 0.95, "metadata": {"holdout_val_bpb": 0.97}},
        {"val_bpb": 0.9, "metadata": {}},
    ]
    out = chart_val_bpb_trend(all_metrics, tmp_path / "nf.json", tmp_path / "out.png")
    assert out is not None and out.exists()


def test_chart_val_bpb_trend_renders_without_holdout_data():
    # Regression check: the holdout overlay addition must not break the
    # pre-existing no-holdout path.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        all_metrics = [{"val_bpb": 1.0}, {"val_bpb": 0.9}]
        out = chart_val_bpb_trend(all_metrics, Path(d) / "nf.json", Path(d) / "out.png")
        assert out is not None and out.exists()


def test_chart_val_bpb_trend_renders_with_annotation(tmp_path):
    all_metrics = [{"val_bpb": 1.0 - 0.01 * i} for i in range(10)]
    annotations = [{"report_index": 4, "label": "regime change"}]
    out = chart_val_bpb_trend(all_metrics, tmp_path / "nf.json", tmp_path / "out.png", annotations=annotations)
    assert out is not None and out.exists()


def test_chart_val_bpb_trend_skips_malformed_or_out_of_range_annotations(tmp_path):
    # Must not raise or corrupt the chart -- just silently skip anything
    # that isn't a well-formed, in-range annotation.
    all_metrics = [{"val_bpb": 1.0 - 0.01 * i} for i in range(5)]
    annotations = [
        {"report_index": "not an int", "label": "bad type"},
        {"report_index": 2, "label": ""},          # empty label
        {"report_index": 999, "label": "way out of range"},
        {"label": "missing report_index"},
    ]
    out = chart_val_bpb_trend(all_metrics, tmp_path / "nf.json", tmp_path / "out.png", annotations=annotations)
    assert out is not None and out.exists()


def test_chart_val_bpb_trend_none_annotations_is_a_noop(tmp_path):
    all_metrics = [{"val_bpb": 1.0}, {"val_bpb": 0.9}]
    out = chart_val_bpb_trend(all_metrics, tmp_path / "nf.json", tmp_path / "out.png", annotations=None)
    assert out is not None and out.exists()
