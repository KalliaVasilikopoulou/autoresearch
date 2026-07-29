"""Tier 1 (state/surrogate.py): RF surrogate, EI acquisition, Sobol
cold-start, interaction detection, union-find blocking. Never had dedicated
tests before this -- synthetic-data tests only, same discipline used for
Tiers 2-4 (which each caught real bugs this way).
"""
import random

import pytest

import state.surrogate as surrogate_module
from state.surrogate import (
    SURROGATE_DEPS_AVAILABLE,
    _denormalize,
    _snap_discrete,
    blocks_from_interactions,
    coordinate_slice,
    expected_improvement,
    fit_surrogate,
    interaction_matrix,
    normalized_value,
    prune_by_noise_floor,
    propose_via_ei,
    rank_by_sensitivity,
    sensitivity_perf,
    sobol_cold_start,
)

pytestmark = pytest.mark.skipif(not SURROGATE_DEPS_AVAILABLE, reason="numpy/scikit-learn not installed")


def _linear_rows(n, seed=0, noise=0.01, coef_a=0.1, coef_b=0.05):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.uniform(0, 10)
        b = rng.uniform(0, 10)
        y = coef_a * a + coef_b * b + rng.uniform(-noise, noise)
        rows.append({"a": a, "b": b, "val_bpb": y})
    return rows


def _interaction_rows(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a = rng.uniform(-1, 1)
        b = rng.uniform(-1, 1)
        c = rng.uniform(-1, 1)  # unrelated to a/b and to y
        y = a * b + rng.uniform(-0.01, 0.01)  # pure interaction, no main effects
        rows.append({"a": a, "b": b, "c": c, "val_bpb": y})
    return rows


@pytest.fixture
def linear_surrogate():
    """y = 1.0*a + 0.01*b + tiny noise -- a matters a lot, b barely at all."""
    rows = _linear_rows(200, seed=42, noise=0.01, coef_a=1.0, coef_b=0.01)
    sm = fit_surrogate(rows, feature_columns=["a", "b"], min_n=15)
    assert sm is not None
    return sm


# ---------------------------------------------------------------------------
# fit_surrogate / SurrogateModel.predict
# ---------------------------------------------------------------------------

def test_fit_surrogate_returns_none_below_min_n():
    rows = _linear_rows(5)
    assert fit_surrogate(rows, feature_columns=["a", "b"], min_n=15) is None


def test_fit_surrogate_returns_none_without_deps(monkeypatch):
    monkeypatch.setattr(surrogate_module, "SURROGATE_DEPS_AVAILABLE", False)
    rows = _linear_rows(50)
    assert fit_surrogate(rows, feature_columns=["a", "b"], min_n=15) is None


def test_fit_surrogate_predicts_reasonably_on_synthetic_linear_data():
    rows = _linear_rows(200, seed=1, noise=0.005, coef_a=0.1, coef_b=0.05)
    sm = fit_surrogate(rows, feature_columns=["a", "b"], min_n=15)
    assert sm is not None
    assert sm.n_train == 200
    assert sm.feature_names == ("a", "b")
    mean, _std = sm.predict({"a": 5.0, "b": 5.0})
    true = 0.1 * 5.0 + 0.05 * 5.0
    assert abs(mean - true) < 0.05


def test_fit_surrogate_skips_rows_missing_features_or_val_bpb():
    rows = _linear_rows(20, seed=2)
    rows.append({"a": 1.0, "val_bpb": 0.5})  # missing "b"
    rows.append({"a": 1.0, "b": 1.0, "val_bpb": float("nan")})  # non-finite
    rows.append({"a": 1.0, "b": 1.0})  # missing val_bpb entirely
    sm = fit_surrogate(rows, feature_columns=["a", "b"], min_n=15)
    assert sm is not None
    assert sm.n_train == 20  # the 3 malformed rows never counted


def test_fit_surrogate_populates_reasonably_accurate_oob_predictions():
    rows = _linear_rows(300, seed=9, noise=0.005, coef_a=0.1, coef_b=0.05)
    sm = fit_surrogate(rows, feature_columns=["a", "b"], min_n=15)
    assert sm is not None
    assert len(sm.oob_actual) > 0
    assert len(sm.oob_actual) == len(sm.oob_predicted)
    # OOB predictions are a real held-out-style check (each row scored only
    # by trees that never saw it) -- should track actual values reasonably,
    # not just memorize them.
    mean_abs_error = sum(abs(a - p) for a, p in zip(sm.oob_actual, sm.oob_predicted)) / len(sm.oob_actual)
    assert mean_abs_error < 0.05


def test_surrogate_model_defaults_to_empty_oob_when_hand_constructed():
    # A SurrogateModel built directly (not via fit_surrogate, e.g. in a test
    # or a future caller) must not require the new fields.
    from state.surrogate import SurrogateModel
    sm = SurrogateModel(model=None, feature_names=("a",), bounds={"a": (0.0, 1.0)}, n_train=0)
    assert sm.oob_actual == ()
    assert sm.oob_predicted == ()


# ---------------------------------------------------------------------------
# normalized_value / _denormalize
# ---------------------------------------------------------------------------

def test_normalized_value_denormalize_roundtrip_linear():
    bounds = {"x": (0.0, 100.0)}
    for v in [0.0, 25.0, 50.0, 99.9]:
        t = normalized_value("x", v, bounds)
        assert _denormalize("x", t, bounds) == pytest.approx(v, abs=1e-6)


def test_normalized_value_denormalize_roundtrip_log_scale():
    bounds = {"matrix_lr": (0.005, 0.2)}
    for v in [0.005, 0.01, 0.1, 0.2]:
        t = normalized_value("matrix_lr", v, bounds)
        assert _denormalize("matrix_lr", t, bounds) == pytest.approx(v, rel=1e-6)


def test_normalized_value_degenerate_bounds_returns_half():
    assert normalized_value("x", 5.0, {"x": (10.0, 10.0)}) == 0.5


def test_normalized_value_log_scale_geometric_midpoint():
    bounds = {"matrix_lr": (0.01, 1.0)}
    # geometric midpoint of [0.01, 1.0] is sqrt(0.01*1.0) = 0.1
    assert normalized_value("matrix_lr", 0.1, bounds) == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# coordinate_slice / sensitivity_perf / rank_by_sensitivity
# ---------------------------------------------------------------------------

def test_coordinate_slice_sweeps_param_holds_others_fixed():
    bounds = {"x": (0.0, 10.0)}
    center = {"x": 3.0, "y": 99.0}

    def metric_fn(hp):
        assert hp["y"] == 99.0
        return hp["x"] * 2

    points = coordinate_slice(metric_fn, "x", center, bounds, n_points=5)
    assert len(points) == 5
    values = [p["value"] for p in points]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(10.0)
    assert points[0]["metric"] == pytest.approx(0.0)
    assert points[-1]["metric"] == pytest.approx(20.0)


def test_coordinate_slice_degenerate_range_returns_empty():
    assert coordinate_slice(lambda hp: 0.0, "x", {"x": 1.0}, {"x": (5.0, 5.0)}) == []


def test_sensitivity_perf_zero_when_metric_ignores_param():
    bounds = {"x": (0.0, 10.0)}
    assert sensitivity_perf(lambda hp: 42.0, "x", {"x": 1.0}, bounds) == 0.0


def test_sensitivity_perf_positive_and_correctly_ordered():
    bounds = {"x": (0.0, 10.0), "y": (0.0, 10.0)}
    s_x = sensitivity_perf(lambda hp: hp["x"] * 10, "x", {"x": 1.0, "y": 1.0}, bounds)
    s_y = sensitivity_perf(lambda hp: hp["y"] * 1, "y", {"x": 1.0, "y": 1.0}, bounds)
    assert s_x > 0
    assert s_y > 0
    assert s_x > s_y


def test_rank_by_sensitivity_orders_correctly(linear_surrogate):
    center = {"a": 5.0, "b": 5.0}
    ranked = rank_by_sensitivity(linear_surrogate, ["a", "b"], center, linear_surrogate.bounds)
    assert ranked[0][0] == "a"
    assert ranked[0][1] > ranked[1][1]


def test_prune_by_noise_floor_splits_correctly(linear_surrogate):
    center = {"a": 5.0, "b": 5.0}
    # a's total effect across its ~10-wide range is ~10 (coef 1.0); b's is ~0.1
    # (coef 0.01) -- sigma=0.5, k=2.0 -> threshold=1.0 sits cleanly between them.
    kept, frozen = prune_by_noise_floor(linear_surrogate, ["a", "b"], center, linear_surrogate.bounds, sigma=0.5, k=2.0)
    assert "a" in kept
    assert "b" in frozen


# ---------------------------------------------------------------------------
# _snap_discrete
# ---------------------------------------------------------------------------

def test_snap_discrete_rounds_int_params():
    out = _snap_discrete({"n_layer": 7.6, "batch_size": 1000.4})
    assert out["n_layer"] == 8
    assert out["batch_size"] == 1000


def test_snap_discrete_n_embd_head_dim_odd_snaps_up_to_even():
    # head_dim = round(100/3) = 33 (odd) -> 34 -> n_embd = 34*3 = 102
    out = _snap_discrete({"n_embd": 100, "n_head": 3})
    assert out["n_head"] == 3
    assert out["n_embd"] == 102


def test_snap_discrete_n_embd_head_dim_already_even_unchanged():
    # head_dim = 128/4 = 32 (already even) -> n_embd unchanged
    out = _snap_discrete({"n_embd": 128, "n_head": 4})
    assert out["n_embd"] == 128


# ---------------------------------------------------------------------------
# expected_improvement
# ---------------------------------------------------------------------------

def test_expected_improvement_zero_without_deps(monkeypatch):
    monkeypatch.setattr(surrogate_module, "SURROGATE_DEPS_AVAILABLE", False)
    assert expected_improvement(1.0, 0.1, 0.9) == 0.0


def test_expected_improvement_positive_when_mu_below_best():
    assert expected_improvement(mu=0.5, sigma=0.1, f_best=1.0) > 0


def test_expected_improvement_near_zero_when_mu_far_above_best():
    assert expected_improvement(mu=10.0, sigma=0.1, f_best=1.0) == pytest.approx(0.0, abs=1e-6)


def test_expected_improvement_degenerate_sigma_branch():
    assert expected_improvement(mu=0.5, sigma=0.0, f_best=1.0, xi=0.01) == pytest.approx(1.0 - 0.5 - 0.01)
    assert expected_improvement(mu=2.0, sigma=0.0, f_best=1.0, xi=0.01) == 0.0


# ---------------------------------------------------------------------------
# propose_via_ei
# ---------------------------------------------------------------------------

def test_propose_via_ei_returns_fixed_values_without_deps(monkeypatch):
    monkeypatch.setattr(surrogate_module, "SURROGATE_DEPS_AVAILABLE", False)
    result = propose_via_ei(None, f_best=1.0, bounds={}, free_params=[], fixed_values={"x": 1})
    assert result == {"x": 1}


def test_propose_via_ei_gravitates_toward_the_true_optimum(linear_surrogate):
    # y = 1.0*a + 0.01*b -- minimizing y means minimizing a (the dominant term).
    # f_best set optimistically (below anything observed) to actively pull EI
    # toward the predicted minimum rather than just exploring near incumbents.
    bounds = linear_surrogate.bounds
    candidate = propose_via_ei(
        linear_surrogate, f_best=0.0, bounds=bounds,
        free_params=["a", "b"], fixed_values={}, n_candidates=2000, seed=0,
    )
    a_lo, a_hi = bounds["a"]
    midpoint = (a_lo + a_hi) / 2
    assert candidate["a"] < midpoint  # meaningfully closer to the true optimum (a_lo) than a random midpoint draw


def test_propose_via_ei_return_diagnostics_false_is_unchanged_from_before(linear_surrogate):
    # Regression check: adding return_diagnostics must not alter the
    # existing (default) return shape any caller already depends on.
    bounds = linear_surrogate.bounds
    result = propose_via_ei(
        linear_surrogate, f_best=0.0, bounds=bounds,
        free_params=["a", "b"], fixed_values={}, n_candidates=200, seed=0,
    )
    assert isinstance(result, dict)
    assert "a" in result and "b" in result


def test_propose_via_ei_return_diagnostics_true_matches_the_chosen_candidate(linear_surrogate):
    bounds = linear_surrogate.bounds
    winner, diagnostics = propose_via_ei(
        linear_surrogate, f_best=0.0, bounds=bounds,
        free_params=["a", "b"], fixed_values={}, n_candidates=200, seed=0,
        return_diagnostics=True,
    )
    assert isinstance(winner, dict)
    assert diagnostics["free_params"] == ["a", "b"]
    assert len(diagnostics["mus"]) == 200
    assert len(diagnostics["sigmas"]) == 200
    assert len(diagnostics["eis"]) == 200
    assert len(diagnostics["candidate_values"]["a"]) == 200
    best_idx = diagnostics["best_idx"]
    # The winning candidate's free-param values must match the diagnostics'
    # own record of that same candidate -- same underlying draw, not two
    # different samples.
    assert winner["a"] == pytest.approx(diagnostics["candidate_values"]["a"][best_idx])
    assert winner["b"] == pytest.approx(diagnostics["candidate_values"]["b"][best_idx])


def test_propose_via_ei_return_diagnostics_true_without_deps(monkeypatch):
    monkeypatch.setattr(surrogate_module, "SURROGATE_DEPS_AVAILABLE", False)
    winner, diagnostics = propose_via_ei(
        None, f_best=1.0, bounds={}, free_params=[], fixed_values={"x": 1}, return_diagnostics=True,
    )
    assert winner == {"x": 1}
    assert diagnostics == {}


# ---------------------------------------------------------------------------
# sobol_cold_start
# ---------------------------------------------------------------------------

def test_sobol_cold_start_empty_without_deps(monkeypatch):
    monkeypatch.setattr(surrogate_module, "SURROGATE_DEPS_AVAILABLE", False)
    assert sobol_cold_start({"x": (0, 1)}, ["x"], 5) == []


def test_sobol_cold_start_empty_when_n_samples_not_positive():
    assert sobol_cold_start({"x": (0, 1)}, ["x"], 0) == []


def test_sobol_cold_start_covers_params_within_bounds_and_snaps_discrete():
    bounds = {"n_layer": (4, 24), "matrix_lr": (0.005, 0.2)}
    points = sobol_cold_start(bounds, ["n_layer", "matrix_lr"], 10, seed=1)
    assert len(points) == 10
    for p in points:
        assert 4 <= p["n_layer"] <= 24
        assert isinstance(p["n_layer"], int)
        assert 0.005 <= p["matrix_lr"] <= 0.2


# ---------------------------------------------------------------------------
# interaction_matrix
# ---------------------------------------------------------------------------

def test_interaction_matrix_none_below_min_n():
    rows = _interaction_rows(5)
    assert interaction_matrix(rows, feature_columns=["a", "b", "c"], min_n=15) is None


def test_interaction_matrix_none_without_deps(monkeypatch):
    monkeypatch.setattr(surrogate_module, "SURROGATE_DEPS_AVAILABLE", False)
    rows = _interaction_rows(50)
    assert interaction_matrix(rows, feature_columns=["a", "b", "c"], min_n=15) is None


def test_interaction_matrix_detects_a_real_interaction():
    rows = _interaction_rows(300, seed=2)
    scores = interaction_matrix(rows, feature_columns=["a", "b", "c"], min_n=15)
    assert scores is not None
    ab_score = scores[("a", "b")]
    other_scores = [v for k, v in scores.items() if k != ("a", "b")]
    assert ab_score > max(other_scores)


# ---------------------------------------------------------------------------
# blocks_from_interactions
# ---------------------------------------------------------------------------

def test_blocks_from_interactions_merges_strongly_interacting_pair():
    interaction_scores = {("a", "b"): 0.9, ("a", "c"): 0.05, ("b", "c"): 0.02}
    main_effect = {"a": 0.5, "b": 0.3, "c": 0.1}
    blocks = blocks_from_interactions(interaction_scores, main_effect, threshold=0.15)
    block_sets = [set(b) for b in blocks]
    assert {"a", "b"} in block_sets
    assert {"c"} in block_sets
    # sorted descending by summed main_effect: {a,b}=0.8 > {c}=0.1
    assert block_sets[0] == {"a", "b"}


def test_blocks_from_interactions_no_interactions_all_singletons():
    main_effect = {"a": 0.5, "b": 0.3}
    blocks = blocks_from_interactions({}, main_effect)
    assert sorted([set(b) for b in blocks], key=lambda s: -main_effect[next(iter(s))]) == [{"a"}, {"b"}]
