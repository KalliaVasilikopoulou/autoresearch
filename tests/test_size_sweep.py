"""Tests for scripts/size_sweep.py -- the experiment that decides whether size
is a surface Agent 4 should EXPLORE or one it can simply walk down.

Two things are worth testing here. First that the ladder really is a size
ladder: if shape drifts as size grows, the sweep silently answers a different
question. Second that the verdict distinguishes the curve shapes it has to
distinguish -- and, in particular, that it does NOT lean on how well a formula
fits, because a poor fit means either a rough surface or the wrong formula and
six points cannot tell those apart.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "size_sweep.py"
_spec = importlib.util.spec_from_file_location("size_sweep", _SCRIPT)
size_sweep = importlib.util.module_from_spec(_spec)
sys.modules["size_sweep"] = size_sweep
_spec.loader.exec_module(size_sweep)


ANCHOR = {
    "n_layer": 21, "n_embd": 960, "n_head": 6,
    "window_s_fraction": 0.3646, "embedding_lr": 0.0887,
    "unembedding_lr": 0.003956, "matrix_lr": 0.023763, "scalar_lr": 0.2101,
    "weight_decay": 0.09339, "warmup_ratio": 0.17312, "batch_size": 23180,
}

SIGMA = 0.00199  # what architecture_noise() reads off the step 4 report


# --- the ladder is a SIZE ladder ---------------------------------------------


def test_head_dim_is_identical_on_every_rung():
    """The frozen quantity that makes this a size sweep. Letting head_dim
    shrink at the narrow end would change what a single head can represent,
    which is a shape change wearing a size change's clothes."""
    rungs = size_sweep.build_ladder(ANCHOR)
    head_dims = {r["n_embd"] // r["n_head"] for r in rungs}
    assert head_dims == {ANCHOR["n_embd"] // ANCHOR["n_head"]}


def test_aspect_ratio_holds_across_the_ladder():
    """n_embd/n_layer can only be held approximately -- n_layer is an integer
    -- but it must not drift, or depth and size are confounded."""
    rungs = size_sweep.build_ladder(ANCHOR)
    aspects = [r["n_embd"] / r["n_layer"] for r in rungs]
    assert max(aspects) / min(aspects) < 1.2


def test_the_ladder_spans_orders_of_magnitude_and_tops_out_at_the_anchor():
    rungs = size_sweep.build_ladder(ANCHOR)
    assert len(rungs) == size_sweep.N_RUNGS
    assert rungs[-1]["params"] / rungs[0]["params"] > 100
    for col in ("n_layer", "n_embd", "n_head"):
        assert rungs[-1][col] == ANCHOR[col]


def test_only_architecture_moves_between_rungs():
    """The eight tunables are frozen on purpose: a rung that changed its
    learning rate as well would leave the two causes inseparable."""
    rungs = size_sweep.build_ladder(ANCHOR)
    for r in rungs:
        for col, want in ANCHOR.items():
            if col in ("n_layer", "n_embd", "n_head"):
                continue
            assert r["hyperparams"][col] == want


def test_every_rung_is_runnable_as_written():
    """train.py re-snaps n_embd so head_dim stays an even integer. A rung that
    trains at a different width than the one recorded would be plotted against
    the wrong x-axis."""
    from state import surrogate

    for r in size_sweep.build_ladder(ANCHOR):
        assert r["n_embd"] % r["n_head"] == 0
        assert (r["n_embd"] // r["n_head"]) % 2 == 0
        assert surrogate.snap_n_embd(r["n_embd"], r["n_head"]) == r["n_embd"]


def test_parameter_count_matches_the_architecture():
    """4*n_embd^2 for attention (n_kv_head == n_head here, so k and v are full
    width) plus 8*n_embd^2 for the 4x MLP, per layer."""
    assert size_sweep.non_embedding_params(21, 960) == 12 * 21 * 960 ** 2


# --- reusing a measurement instead of re-buying it ---------------------------


def _steps_for(batch_size):
    """The num_steps a run at `batch_size` would record under the budget in
    force. num_steps and batch_size are what the budget is DERIVED from
    (results_analysis.tokens_seen) -- total_tokens_M is parsed out of train.py's
    summary and then discarded, so it has never been a results.tsv column and
    cannot be the thing compared. Derived from the row's OWN batch_size, since
    _same_config also checks that one for equality."""
    snapped = (int(batch_size) // 2048) * 2048
    return round(size_sweep.current_token_budget() / snapped)


def test_a_row_matching_on_architecture_alone_is_not_reused():
    """The top rung is reused from step 4 because that run was identical in
    every respect. A row that shares the architecture but not the learning
    rates is a different measurement, and putting it on the curve would mean
    one rung was tuned differently from the rest."""
    rung = size_sweep.build_ladder(ANCHOR)[-1]
    row = dict(rung["hyperparams"], seed=size_sweep.SEED, status="remote_ok",
               num_steps=_steps_for(rung["hyperparams"]["batch_size"]))
    assert size_sweep._same_config(row, rung["hyperparams"])

    row["matrix_lr"] = ANCHOR["matrix_lr"] * 1.1
    assert not size_sweep._same_config(row, rung["hyperparams"])


def test_a_row_from_another_seed_is_not_reused():
    rung = size_sweep.build_ladder(ANCHOR)[-1]
    row = dict(rung["hyperparams"], seed=size_sweep.SEED + 1, status="remote_ok")
    assert not size_sweep._same_config(row, rung["hyperparams"])


# --- the verdict -------------------------------------------------------------


def _report(curve, tmp_path, name="s"):
    from state.results_logger import log_result

    path = str(tmp_path / f"{name}.tsv")
    for rung, val in zip(size_sweep.build_ladder(ANCHOR), curve):
        log_result(f"{size_sweep.RUN_ID_PREFIX}_{rung['label']}",
                   size_sweep._hyperparams_for(rung),
                   {"val_bpb": val, "status": "remote_ok", "budget_shortfall_pct": 0.0},
                   results_path=path)
    return size_sweep.analyze(path)


MONOTONE = [1.62, 1.48, 1.38, 1.31, 1.27, 1.2488]
HILL = [1.62, 1.45, 1.32, 1.26, 1.28, 1.34]      # one turn: a real optimum
JAGGED = [1.62, 1.35, 1.55, 1.30, 1.50, 1.28]    # alternating, well above noise
FLAT = [1.2490, 1.2485, 1.2492, 1.2483, 1.2488, 1.2486]
PLATEAU = [1.62, 1.45, 1.33, 1.27, 1.2495, 1.2488]  # falls, then stops paying


def test_a_steady_descent_is_one_hill(tmp_path):
    r = _report(MONOTONE, tmp_path)
    assert r["unimodal"] and r["readable"]
    assert r["real_sign_changes"] == 0
    assert "ONE HILL" in r["verdict"]


def test_a_single_turn_is_still_one_hill(tmp_path):
    r = _report(HILL, tmp_path)
    assert r["real_sign_changes"] == 1
    assert r["unimodal"]
    assert not r["best_is_largest"] and not r["best_is_smallest"]


def test_alternating_values_are_not_a_hill(tmp_path):
    r = _report(JAGGED, tmp_path)
    assert r["real_sign_changes"] > 1
    assert not r["unimodal"]
    assert "NOT A HILL" in r["verdict"]


def test_differences_below_the_noise_are_not_direction(tmp_path):
    """Every step here changes sign, but none of them clears the noise. Reading
    direction from those would be reading the noise -- so this reports as
    unreadable rather than as five changes of direction."""
    r = _report(FLAT, tmp_path)
    assert not r["readable"]
    assert r["real_sign_changes"] == 0
    assert "UNREADABLE" in r["verdict"]


def test_a_plateau_at_the_top_is_reported_as_stopping(tmp_path):
    """The last step is the one Agent 4 would take next, so whether size has
    stopped paying is called out separately from the overall shape."""
    r = _report(PLATEAU, tmp_path)
    assert r["unimodal"]
    assert not r["still_falling_at_the_top"]
    assert r["steps_below_noise"] >= 1

    falling = _report(MONOTONE, tmp_path, name="falling")
    assert falling["still_falling_at_the_top"]


def test_the_verdict_does_not_depend_on_how_well_the_law_fits(tmp_path):
    """THE POINT OF SEPARATING THE TWO. A textbook-clean descent can fit any
    given formula badly -- that is a fact about the formula, not about the
    surface -- so fit quality must not be able to overturn a hill."""
    r = _report(MONOTONE, tmp_path)
    assert not r["predictable"]           # the law's residuals exceed the noise
    assert "ONE HILL" in r["verdict"]     # and it changes nothing


def test_headroom_is_only_offered_when_the_ladder_is_still_descending(tmp_path):
    """Extrapolating past the largest model measured is only meaningful if the
    curve had not already flattened -- otherwise there is nothing to project."""
    assert _report(MONOTONE, tmp_path)["headroom"] is not None
    assert _report(PLATEAU, tmp_path, name="p")["headroom"] is None


def test_headroom_projects_from_the_fitted_gap_not_the_measured_one(tmp_path):
    """The two differ by the fit's residual at the top rung. Projecting from
    the measured gap would charge that residual to a change in size, which is
    not what it is."""
    r = _report(MONOTONE, tmp_path)
    law, h, top = r["scaling_law"], r["headroom"], r["rungs"][-1]

    assert h["measured_gap"] == pytest.approx(top["val_bpb"] - law["l_inf"])
    assert h["fitted_gap"] != pytest.approx(h["measured_gap"])  # the residual
    assert h["size_multiple_to_halve_it"] == pytest.approx(2 ** (1 / law["alpha"]))

    # At 2^(1/alpha) times the size, the law's own gap is exactly halved.
    bigger = law["c"] * h["params_to_halve_it"] ** -law["alpha"]
    assert bigger == pytest.approx(h["fitted_gap"] / 2, rel=1e-6)


def test_a_scaling_law_fits_a_scaling_curve(tmp_path):
    """Sanity check on the fitter itself: values generated FROM the law are
    recovered by it."""
    rungs = size_sweep.build_ladder(ANCHOR)
    params = [float(r["params"]) for r in rungs]
    truth = [1.10 + 90.0 * n ** -0.32 for n in params]
    law = size_sweep.fit_scaling_law(params, truth)
    assert law["alpha"] == pytest.approx(0.32, abs=0.02)
    assert law["l_inf"] == pytest.approx(1.10, abs=0.01)
    assert law["r2"] > 0.999


# --- the guard that step 5a's saturation rule taught us ----------------------


def test_the_noise_floor_says_where_it_came_from(tmp_path):
    """A threshold that decides how machinery gets built must never run on a
    guessed constant without saying so."""
    import json

    sigma, source = size_sweep.architecture_noise(tmp_path)  # empty dir
    assert sigma == size_sweep.FALLBACK_SIGMA
    assert "fallback" in source

    budget = size_sweep.current_token_budget()
    (tmp_path / "region_geometry.json").write_text(json.dumps(
        {"token_budget": budget,
         "a_between": {"depth_neighbour": {"std_of_gap": 0.0031}}}), encoding="utf-8")
    sigma, source = size_sweep.architecture_noise(tmp_path)
    assert sigma == pytest.approx(0.0031)
    assert "region_geometry" in source


def test_a_noise_floor_from_another_budget_is_refused(tmp_path):
    """A noise floor is not portable across budgets: sigma_seed went 0.00197 ->
    0.003215 when TOKEN_BUDGET went 12.5M -> 4.19M. Reading a stale one would
    make every "below noise" call in this report wrong in the same direction --
    understating the noise, so unresolvable steps get reported as real."""
    import json

    (tmp_path / "region_geometry.json").write_text(json.dumps(
        {"token_budget": size_sweep.current_token_budget() * 3,
         "a_between": {"depth_neighbour": {"std_of_gap": 0.0031}}}), encoding="utf-8")

    sigma, source = size_sweep.architecture_noise(tmp_path)
    assert sigma == size_sweep.FALLBACK_SIGMA
    assert "NO NOISE MEASUREMENT AT THIS BUDGET" in source


def test_a_report_predating_the_budget_stamp_is_treated_as_stale(tmp_path):
    """It cannot claim to match, so it is not assumed to. Every report on disk
    when the stamp was added was measured at 12.5M."""
    import json

    (tmp_path / "seed_variance.json").write_text(
        json.dumps({"sigma_seed": 0.00197}), encoding="utf-8")

    sigma, _ = size_sweep.architecture_noise(tmp_path)
    assert sigma == size_sweep.FALLBACK_SIGMA


def test_the_current_budgets_measurement_wins_over_an_older_experiment(tmp_path):
    """Both stamped and both current: seed_variance is preferred because it
    measures the seed effect directly, where region_geometry's A-between was
    anchor-dominated and only ever an upper bound."""
    import json

    budget = size_sweep.current_token_budget()
    (tmp_path / "region_geometry.json").write_text(json.dumps(
        {"token_budget": budget,
         "a_between": {"d": {"std_of_gap": 0.0099}}}), encoding="utf-8")
    (tmp_path / "seed_variance.json").write_text(json.dumps(
        {"token_budget": budget, "sigma_seed": 0.0032}), encoding="utf-8")

    sigma, source = size_sweep.architecture_noise(tmp_path)
    assert sigma == pytest.approx(0.0032)
    assert "seed_variance" in source


def test_a_row_from_a_different_token_budget_is_not_reused(tmp_path):
    """THE ONE THAT WOULD HAVE FAKED A HILL. When TOKEN_BUDGET was cut
    12.5M -> 4.19M the anchor's val_bpb moved 1.2486 -> 1.7063, so splicing a
    single old row into a new ladder drops a point half a bpb below the curve
    -- and the sweep would report a turn that is purely an artefact of mixed
    budgets."""
    rung = size_sweep.build_ladder(ANCHOR)[-1]

    row = dict(rung["hyperparams"], seed=size_sweep.SEED, status="remote_ok",
               num_steps=_steps_for(rung["hyperparams"]["batch_size"]))
    assert size_sweep._same_config(row, rung["hyperparams"])

    # Same config, same seed, three times the training. Derived from num_steps
    # and batch_size, so this works on rows written before any of it existed.
    row["num_steps"] = row["num_steps"] * 3
    assert not size_sweep._same_config(row, rung["hyperparams"])


def test_a_row_with_no_budget_recorded_is_not_reused(tmp_path):
    """Absent is not "matches". A row that cannot say how much training it saw
    cannot be placed on a curve whose x-axis assumes a fixed one."""
    rung = size_sweep.build_ladder(ANCHOR)[-1]
    row = dict(rung["hyperparams"], seed=size_sweep.SEED, status="remote_ok")
    assert not size_sweep._same_config(row, rung["hyperparams"])
