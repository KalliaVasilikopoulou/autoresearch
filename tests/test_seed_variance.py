"""Tests for scripts/seed_variance.py's analysis -- the part that turns 15
val_bpb numbers into the verdict "is freezing the seed sound".

The distinction being tested is the whole point of the experiment: a seed that
moves every configuration by the SAME amount is harmless (it cancels out of
every comparison the search makes), while a seed that moves each configuration
independently means single-seed val_bpb has partly been ranking seeds.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "seed_variance.py"
_spec = importlib.util.spec_from_file_location("seed_variance", _SCRIPT)
seed_variance = importlib.util.module_from_spec(_spec)
sys.modules["seed_variance"] = seed_variance
_spec.loader.exec_module(seed_variance)


SEEDS = [42, 1, 2, 3, 4]


def _shared_effect_measurements():
    """Three configs genuinely separated by 0.02, plus a per-seed offset that
    hits all three identically. Rankings can never flip here."""
    offsets = {42: 0.0, 1: +0.010, 2: -0.008, 3: +0.004, 4: -0.006}
    base = {0: 1.20, 1: 1.22, 2: 1.24}
    return {c: {s: base[c] + off for s, off in offsets.items()} for c in base}


def _independent_effect_measurements():
    """Same three configs, same 0.02 separation, but each config gets its own
    unrelated per-seed jolt that is larger than the separation."""
    jolts = {
        0: {42: 0.0, 1: +0.03, 2: -0.02, 3: +0.01, 4: -0.03},
        1: {42: -0.03, 1: +0.02, 2: +0.03, 3: -0.02, 4: +0.01},
        2: {42: +0.02, 1: -0.03, 2: +0.01, 3: +0.03, 4: -0.01},
    }
    base = {0: 1.20, 1: 1.22, 2: 1.24}
    return {c: {s: base[c] + jolts[c][s] for s in SEEDS} for c in base}


def test_shared_seed_effect_leaves_rankings_stable():
    report = seed_variance.analyze(_shared_effect_measurements(), SEEDS)

    assert report["all_pairs_consistent"] is True
    assert "RANKINGS STABLE" in report["verdict"]
    # Each config moves with the seed...
    assert report["sigma_seed"] > 0.005
    # ...but the GAP between any two does not: the offset cancels exactly.
    assert report["mean_std_of_pairwise_gap"] == pytest.approx(0.0, abs=1e-12)
    # Which is the number that actually matters, and it is reported as such.
    assert report["shared_fraction_of_seed_effect"] == pytest.approx(1.0)


def test_independent_seed_effect_flips_rankings():
    report = seed_variance.analyze(_independent_effect_measurements(), SEEDS)

    assert report["all_pairs_consistent"] is False
    assert "RANKINGS FLIP" in report["verdict"]
    # The comparison moves about as much as an individual measurement does --
    # near the sqrt(2)*sigma_seed reference for fully independent effects.
    assert report["mean_std_of_pairwise_gap"] > report["sigma_seed"]
    assert report["shared_fraction_of_seed_effect"] < 0.25


def test_a_consistent_winner_with_a_tiny_gap_is_not_called_unstable():
    """Ranking stability and effect size are separate questions. A pair whose
    winner never changes is consistent even if it wins by very little -- the
    small `separation` is what says the gap is not resolved, and conflating the
    two would report a stable search as broken."""
    measurements = {
        0: {s: 1.2000 + 0.0001 * i for i, s in enumerate(SEEDS)},
        1: {s: 1.2005 + 0.0001 * i for i, s in enumerate(SEEDS)},
    }
    report = seed_variance.analyze(measurements, SEEDS)

    assert report["all_pairs_consistent"] is True
    pair = report["pairs"][0]
    assert pair["a_wins"] == len(SEEDS)
    assert pair["mean_gap"] == pytest.approx(-0.0005)


def test_pairs_only_use_seeds_where_both_configs_produced_a_result():
    """A crashed cell must shrink the comparison, never be imputed. Config 1 is
    missing seed 3 entirely."""
    measurements = {
        0: {42: 1.20, 1: 1.21, 2: 1.22, 3: 1.23},
        1: {42: 1.25, 1: 1.26, 2: 1.27},
    }
    report = seed_variance.analyze(measurements, [42, 1, 2, 3])

    assert report["per_config"][0]["n"] == 4
    assert report["per_config"][1]["n"] == 3
    assert report["pairs"][0]["n_seeds"] == 3, "the pair is compared on shared seeds only"


def test_resolvable_gap_shrinks_with_more_seeds():
    """The actionable output: averaging k seeds narrows the smallest gap the
    experiment can distinguish, by sqrt(k)."""
    report = seed_variance.analyze(_independent_effect_measurements(), SEEDS)
    gaps = report["resolvable_gap_at_k_seeds"]

    assert gaps["1"] > gaps["2"] > gaps["3"] > gaps["5"]
    assert gaps["1"] / gaps["5"] == pytest.approx(5 ** 0.5, rel=1e-6)


def _ranked_rows(n, spacing=0.02):
    return [
        {"run_id": f"run_{i}", "status": "remote_ok", "val_bpb": 1.20 + spacing * i,
         "budget_shortfall_pct": 0.0,
         **{c: 1.0 for c in seed_variance.HYPERPARAM_COLUMNS}}
        for i in range(n)
    ]


def test_the_two_best_configs_are_always_tested():
    """The frontier pair is the comparison the search actually makes, and the
    only one where the answer has consequences. An experiment built only from
    far-apart configurations would return 'stable' under any noise level."""
    picked = seed_variance.select_configs(_ranked_rows(9), 3)

    assert [p["run_id"] for p in picked[:2]] == ["run_0", "run_1"]


def test_selection_also_reaches_well_down_the_range_as_a_power_control():
    """A pair that MUST resolve. If it flips, the experiment is at fault rather
    than the search; if it separates cleanly, the method is shown to have the
    power to detect a real difference."""
    picked = seed_variance.select_configs(_ranked_rows(9), 3)

    assert picked[2]["run_id"] == "run_6"
    assert picked[2]["val_bpb"] - picked[1]["val_bpb"] > 0.05


def test_selection_skips_the_pathological_tail():
    """Picks stop at the 75th percentile -- the worst rows are diverged configs
    whose gap to everything resolves trivially and carries no information."""
    picked = seed_variance.select_configs(_ranked_rows(21), 4)

    ranks = [int(p["run_id"].removeprefix("run_")) for p in picked]  # numeric, not lexicographic
    assert max(ranks) <= int(0.75 * 20)


def test_selection_returns_the_requested_count_even_on_a_short_history():
    """A short history collapses several quantile picks onto one index; the
    caller must still get the count it asked for rather than a silently
    smaller experiment."""
    picked = seed_variance.select_configs(_ranked_rows(4), 3)

    assert len(picked) == 3
    assert len({p["run_id"] for p in picked}) == 3


def test_incomplete_and_failed_runs_are_never_selected_as_configs():
    rows = [
        {"run_id": "truncated", "status": "remote_ok", "val_bpb": 1.10,
         "budget_shortfall_pct": 12.0, **{c: 1.0 for c in seed_variance.HYPERPARAM_COLUMNS}},
        {"run_id": "crashed", "status": "remote_error", "val_bpb": float("inf"),
         "budget_shortfall_pct": 0.0, **{c: 1.0 for c in seed_variance.HYPERPARAM_COLUMNS}},
    ] + [
        {"run_id": f"good_{i}", "status": "remote_ok", "val_bpb": 1.30 + 0.01 * i,
         "budget_shortfall_pct": 0.0, **{c: 1.0 for c in seed_variance.HYPERPARAM_COLUMNS}}
        for i in range(3)
    ]
    picked = seed_variance.select_configs(rows, 3)

    assert all(p["run_id"].startswith("good_") for p in picked)


def test_too_little_history_fails_loudly_instead_of_testing_one_config():
    with pytest.raises(SystemExit) as excinfo:
        seed_variance.select_configs([], 3)
    assert "at least 3" in str(excinfo.value)


def test_measurement_runs_skip_post_training_analysis():
    """ablation_k and the token fingerprint cost real GPU time and cannot
    change val_bpb, so an experiment dispatching 3x the usual number of runs
    turns them off."""
    row = {c: 8.0 for c in seed_variance.HYPERPARAM_COLUMNS}
    hp = seed_variance._row_to_hyperparams(row)

    assert hp["ablation_k"] == 0
    assert hp["token_xai_enabled"] is False
    assert isinstance(hp["n_layer"], int), "int params must not reach train.py as floats"
