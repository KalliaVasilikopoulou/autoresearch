"""Tier 1 steps 6 and 7, both justified by scripts/seed_variance.py's measured
result rather than by theory:

  step 6 -- Expected Improvement must aim at a value that is actually
            achievable, not at the luckiest observation. Measured: the campaign
            record 1.248540 was the kindest of 5 seeds for that config, whose
            honest mean is 1.251022.

  step 7 -- the sigma every "is this difference real" test is sized against
            must include seed variance. Measured: state/noise_floor.json's
            0.000797 was taken with the seed fixed and understates the frontier
            spread (0.00197) by ~2.5x.
"""

import json

import pytest

from agents import search_planner
from state import surrogate

pytestmark = pytest.mark.skipif(
    not surrogate.SURROGATE_DEPS_AVAILABLE, reason="scipy/scikit-learn not installed"
)

FEATURES = ("n_layer", "n_embd")


def _rows(values):
    """values: list of (n_layer, n_embd, val_bpb)."""
    return [{"n_layer": a, "n_embd": b, "val_bpb": c} for a, b, c in values]


# --- step 6: denoised EI incumbent ------------------------------------------


def test_best_predicted_mean_is_higher_than_a_lucky_minimum():
    """The core of the correction. One point is given an implausibly good
    observation; the forest smooths it toward its neighbourhood, so the
    predicted best sits above the observed best."""
    values = [(8 + i % 5, 512 + 8 * i, 1.30 + 0.001 * (i % 4)) for i in range(40)]
    values.append((10, 560, 1.20))  # the lucky draw
    rows = _rows(values)
    sm = surrogate.fit_surrogate(rows, feature_columns=FEATURES, min_n=5,
                                 exclude_compute_starved=False)

    observed_best = min(r["val_bpb"] for r in rows)
    predicted_best = surrogate.best_predicted_mean(sm, rows, feature_columns=FEATURES)

    assert observed_best == pytest.approx(1.20)
    assert predicted_best > observed_best, "the lucky draw must not survive as the target"


def test_best_predicted_mean_returns_none_rather_than_guessing():
    """No scoreable row -> the caller keeps its existing observed-best
    behaviour instead of being handed a fabricated number."""
    rows = _rows([(8, 512, 1.3)] * 20)
    sm = surrogate.fit_surrogate(rows, feature_columns=FEATURES, min_n=5,
                                 exclude_compute_starved=False)

    assert surrogate.best_predicted_mean(sm, [], feature_columns=FEATURES) is None
    assert surrogate.best_predicted_mean(
        sm, [{"val_bpb": float("inf"), "n_layer": 8, "n_embd": 512}],
        feature_columns=FEATURES) is None


def test_denoised_incumbent_is_scoped_to_a_region_when_one_is_active():
    """A region-scoped search needs a LOCAL incumbent -- that is the entire
    reason Agent1.search_region swaps best_val_bpb. A campaign-wide incumbent
    would make EI inside any non-champion region see no improvement anywhere
    and its argmax degenerate into noise."""
    champion = [{"n_layer": 8, "n_embd": 512, "val_bpb": 1.20, "region_id": "r0001"}] * 20
    laggard = [{"n_layer": 20, "n_embd": 900, "val_bpb": 1.40, "region_id": "r0002"}] * 20
    rows = champion + laggard
    sm = surrogate.fit_surrogate(rows, feature_columns=FEATURES, min_n=5,
                                 exclude_compute_starved=False)

    global_best = surrogate.best_predicted_mean(sm, rows, feature_columns=FEATURES)
    local_best = surrogate.best_predicted_mean(
        sm, [r for r in rows if r["region_id"] == "r0002"], feature_columns=FEATURES)

    assert global_best == pytest.approx(1.20, abs=0.02)
    assert local_best == pytest.approx(1.40, abs=0.02)
    assert local_best > global_best, "the laggard region must judge itself locally"


def test_agent1_actually_passes_its_region_through_to_the_planner():
    """The wiring, not just the mechanism. The test above proves that filtering
    rows to a region yields a local incumbent; this proves Agent 1 tells the
    planner WHICH region, which is a separate thing and was unverified."""
    from agents import search_planner
    from agents.agent1_training_specialist import Agent1TrainingSpecialist

    seen = {}

    def fake_propose_next(**kwargs):
        seen.update(kwargs)
        return None  # falls through to the evidence path; we only inspect the call

    specialist = Agent1TrainingSpecialist(config_path="does-not-exist.yaml")

    class _Region:
        region_id = "r0007"
        center = {"n_layer": 8, "n_embd": 512, "n_head": 4}
        # The fence is anchored here, not at the drifting centre -- see
        # Agent1._surrogate_adjustment.
        anchor = {"n_layer": 8, "n_embd": 512, "n_head": 4, "matrix_lr": 0.04}
        best_val_bpb = 1.30
        val_bpbs = [1.30]

        def planner_state_path(self, root):
            return "state/search_planner/r0007.json"

        def report_dir(self, root):
            return "reports/agent1_search_plan/r0007"

    original = search_planner.propose_next
    search_planner.propose_next = fake_propose_next
    try:
        with specialist.search_region(_Region()):
            specialist._surrogate_adjustment(iteration=3)
        assert seen["f_best_region_id"] == "r0007"

        seen.clear()
        specialist._surrogate_adjustment(iteration=4)  # no region scope active
        assert seen["f_best_region_id"] is None, "the single-search path stays global"
    finally:
        search_planner.propose_next = original


# --- step 7: sigma must include seed variance -------------------------------


def _write(tmp_path, name, payload):
    """Writes a measurement report, stamped with the token budget in force.

    The stamp is not decoration: noise is a property of how much training a run
    gets (sigma_seed went 0.00197 -> 0.003215 when TOKEN_BUDGET went 12.5M ->
    4.19M), so _load_sigma refuses a report from another budget, and an
    unstamped one reads as stale. A fixture without it is not testing the
    loader, it is testing the refusal.
    """
    from prepare import TOKEN_BUDGET

    p = tmp_path / name
    p.write_text(json.dumps({"token_budget": int(TOKEN_BUDGET), **payload}),
                 encoding="utf-8")
    return str(p)


def test_seed_variance_takes_precedence_over_the_seed_fixed_noise_floor(tmp_path):
    noise = _write(tmp_path, "noise_floor.json", {"std": 0.000797})
    seedvar = _write(tmp_path, "seed_variance.json", {"per_config": {
        "0": {"std": 0.00154}, "1": {"std": 0.00197}, "2": {"std": 0.00887},
    }})

    assert search_planner._load_sigma(noise, seedvar) == pytest.approx(0.00197)


def test_the_median_is_used_not_the_pooled_or_mean_spread(tmp_path):
    """Spread is strongly config-dependent (~6x across the real measurement),
    so mean/pooled is dragged to the noisiest config and describes nowhere.
    The median is robust and lands near the frontier, where decisions happen."""
    seedvar = _write(tmp_path, "seed_variance.json", {"per_config": {
        "0": {"std": 0.00154}, "1": {"std": 0.00197}, "2": {"std": 0.00887},
    }})
    sigma = search_planner._load_sigma(str(tmp_path / "absent.json"), seedvar)

    assert sigma == pytest.approx(0.00197)
    assert sigma < (0.00154 + 0.00197 + 0.00887) / 3, "must not be the mean"


def test_falls_back_to_the_noise_floor_when_seed_variance_is_missing(tmp_path):
    noise = _write(tmp_path, "noise_floor.json", {"std": 0.000797})

    assert search_planner._load_sigma(noise, str(tmp_path / "absent.json")) == pytest.approx(0.000797)


def test_falls_back_when_seed_variance_is_corrupt_rather_than_raising(tmp_path):
    noise = _write(tmp_path, "noise_floor.json", {"std": 0.000797})
    bad = tmp_path / "seed_variance.json"
    bad.write_text("{not json", encoding="utf-8")

    assert search_planner._load_sigma(noise, str(bad)) == pytest.approx(0.000797)


def test_zero_and_missing_stds_are_ignored_not_treated_as_a_measurement(tmp_path):
    """A config with a single run has std None; one that never varied has 0.
    Neither is evidence that the noise is zero."""
    seedvar = _write(tmp_path, "seed_variance.json", {"per_config": {
        "0": {"std": None}, "1": {"std": 0.0}, "2": {"std": 0.0031},
    }})

    assert search_planner._load_sigma("absent.json", seedvar) == pytest.approx(0.0031)


def test_last_resort_is_the_conservative_default(tmp_path):
    sigma = search_planner._load_sigma(str(tmp_path / "a.json"), str(tmp_path / "b.json"))

    assert sigma == pytest.approx(search_planner.DEFAULT_SIGMA)


def test_seed_variance_is_looked_for_beside_the_noise_floor_not_at_a_fixed_path(tmp_path):
    """Redirecting a state directory must redirect BOTH files. A hardcoded
    default meant a test (or a campaign with a custom state_dir) still picked
    up the real repo's measurement -- the redirection only half applied."""
    noise = _write(tmp_path, "noise_floor.json", {"std": 0.000797})
    _write(tmp_path, "seed_variance.json", {"per_config": {"0": {"std": 0.0042}}})

    assert search_planner._load_sigma(noise) == pytest.approx(0.0042)


def test_an_isolated_state_dir_does_not_inherit_the_repos_measurement(tmp_path):
    """The other half of the same guarantee: a state dir with only a noise
    floor must use it, not whatever seed_variance.json exists in the repo."""
    noise = _write(tmp_path, "noise_floor.json", {"std": 0.0321})

    assert search_planner._load_sigma(noise) == pytest.approx(0.0321)


def test_the_new_sigma_raises_the_freeze_bar_as_intended(tmp_path):
    """The consequence that matters: prune_by_noise_floor freezes anything
    whose total effect is under 2*sigma. Against the seed-fixed floor that bar
    was 0.0016, so parameters with an effect smaller than the run-to-run bounce
    were kept and tuned."""
    noise = _write(tmp_path, "noise_floor.json", {"std": 0.000797})
    seedvar = _write(tmp_path, "seed_variance.json", {"per_config": {
        "0": {"std": 0.00154}, "1": {"std": 0.00197}, "2": {"std": 0.00887},
    }})

    old_bar = 2 * search_planner._load_sigma(noise, str(tmp_path / "absent.json"))
    new_bar = 2 * search_planner._load_sigma(noise, seedvar)

    assert old_bar == pytest.approx(0.001594)
    assert new_bar == pytest.approx(0.00394)
    assert new_bar / old_bar == pytest.approx(2.47, rel=0.01)


# --- a noise floor is not portable across token budgets ---------------------


def test_a_sigma_measured_at_another_budget_is_refused(tmp_path):
    """sigma_seed went 0.00197 -> 0.003215 when TOKEN_BUDGET went 12.5M ->
    4.19M: less training leaves a run further from convergence and so more
    dependent on its initial weights. A stale floor is wrong in one consistent
    direction -- it UNDERSTATES the noise, so sub-noise parameters stay in the
    search and unresolvable differences get called real."""
    from prepare import TOKEN_BUDGET

    floor = _write(tmp_path, "noise_floor.json", {"std": 0.000797})
    (tmp_path / "seed_variance.json").write_text(json.dumps(
        {"token_budget": int(TOKEN_BUDGET) * 3,
         "per_config": {"0": {"std": 0.00197}}}), encoding="utf-8")

    # falls through to the noise floor rather than using another budget's sigma
    assert search_planner._load_sigma(floor) == pytest.approx(0.000797)


def test_a_report_with_no_budget_stamp_is_treated_as_stale(tmp_path):
    """Unstamped is positive evidence of age, not an absence of it: every
    report on disk when the stamp was introduced was measured at 12.5M."""
    floor = _write(tmp_path, "noise_floor.json", {"std": 0.000797})
    (tmp_path / "seed_variance.json").write_text(
        json.dumps({"per_config": {"0": {"std": 0.00197}}}), encoding="utf-8")

    assert search_planner._load_sigma(floor) == pytest.approx(0.000797)


def test_saturation_will_not_fire_on_a_stale_in_region_noise(tmp_path):
    """The rule that ABANDONS a region must not run on a number measured under
    a different amount of training -- it would understate the noise, so regions
    look further from saturation than they are and keep being searched after
    they have stopped paying. None means "do not judge", which is the same
    guard that already covers an absent measurement."""
    from prepare import TOKEN_BUDGET

    state = tmp_path / "state"
    state.mkdir()
    (state / "region_geometry.json").write_text(json.dumps(
        {"a_within": 0.001342, "token_budget": int(TOKEN_BUDGET) * 3}),
        encoding="utf-8")

    assert search_planner.measured_a_within(str(state)) is None
