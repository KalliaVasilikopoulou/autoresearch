"""Step 5a: a region is abandoned when its remaining differences become
unreadable, not merely when it stops improving.

Those are different claims. "No improvement in 15 runs" can be bad luck, and
bad luck recovers. Saturation says the variation still inside the region is
smaller than the measurement noise, so no amount of further spending can rank
what is in there -- and it says WHY, which a run counter never does.

Measured instance of the failure this prevents (scripts/region_geometry.py): at
fence radius 0.02 the real signal was 7% of the noise, so a region opened there
was saturated on arrival and every run inside it would measure nothing. At 0.05
it is 8x the noise.
"""

import json
import math

import pytest

from state.regions import SATURATED, RegionRegistry

A = 0.001342  # measured in-region noise, scripts/region_geometry.py

BASE = {
    "n_layer": 8, "n_embd": 512, "n_head": 4, "window_s_fraction": 0.75,
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192,
}


def _region(tmp_path, values):
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    r = reg.open_region(dict(BASE), at_run=0)
    for i, v in enumerate(values):
        reg.assign_run(r.region_id, f"run_{i}", v)
    return reg, r


# --- the measurement ---------------------------------------------------------


def test_real_signal_removes_the_noise_from_the_observed_spread(tmp_path):
    """Spreads add as squares, so the observed spread already contains the
    noise and has to have it subtracted, not compared against it directly."""
    _reg, r = _region(tmp_path, [1.20, 1.21, 1.22, 1.23, 1.24])

    # robust_spread, not stdev: this number decides whether to abandon a
    # region, so one unlucky run must not be able to move it. See the
    # outlier test below.
    observed = r.robust_spread(r.val_bpbs)
    expected = math.sqrt(observed ** 2 - A ** 2)

    assert r.real_signal(A) == pytest.approx(expected)
    assert r.real_signal(A) < observed, "the noise must come off"


def test_one_bad_run_cannot_move_the_saturation_verdict(tmp_path):
    """THE FAILURE THIS FIXES. real_signal used plain stdev, which the
    extremes drive -- so a BAD run widened the observed spread, RAISED the
    "real signal", and made the region look FURTHER from saturation. A bad
    result bought the region more runs.

    Measured on r0008 at n=11: run_0149 came back at 1.490004 and moved the
    signal/noise ratio 1.50x -> 3.07x on that one run."""
    clean = [1.4325, 1.4342, 1.4395, 1.4427, 1.4355, 1.4332, 1.4327, 1.4317]
    _reg, without = _region(tmp_path, clean)
    _reg2, with_outlier = _region(tmp_path / "b", clean + [1.4900])

    a, b = without.robust_spread(clean), with_outlier.robust_spread(clean + [1.4900])
    assert b / a < 1.25, "one outlier moved the spread more than 25%"

    import statistics
    sd_a = statistics.stdev(clean)
    sd_b = statistics.stdev(clean + [1.4900])
    assert sd_b / sd_a > 2.0, "fixture is wrong: stdev should blow up here"


def test_a_region_of_pure_noise_has_no_real_signal(tmp_path):
    """Five configurations that differ by less than the noise. Whatever
    separates them cannot be read, so the honest answer is zero -- never a
    negative number from the subtraction."""
    _reg, r = _region(tmp_path, [1.2000, 1.2005, 1.1996, 1.2008, 1.1999])

    assert r.real_signal(A) == 0.0
    assert r.is_saturated(A) is True


def test_a_region_with_real_structure_is_not_saturated(tmp_path):
    _reg, r = _region(tmp_path, [1.20, 1.23, 1.26, 1.21, 1.25])

    assert r.real_signal(A) > A
    assert r.is_saturated(A) is False


def test_too_few_runs_gives_none_rather_than_a_verdict(tmp_path):
    """A standard deviation from two or three points is mostly noise itself,
    and this number decides whether to abandon a region."""
    for values in ([], [1.20], [1.20, 1.25], [1.20, 1.25, 1.22]):
        _reg, r = _region(tmp_path, values)
        assert r.real_signal(A, min_runs=5) is None
        assert r.is_saturated(A, min_runs=5) is None


def test_the_measured_radii_reproduce_the_experiment(tmp_path):
    """Sanity-check the rule against what region_geometry actually measured:
    B=0.001345 at radius 0.02 is saturated, B=0.010800 at 0.05 is not."""
    import statistics

    # Construct samples with those standard deviations.
    for observed, expect_saturated in ((0.001345, True), (0.010800, False)):
        vals = [1.25 + observed * k for k in (-1.2649, -0.6325, 0.0, 0.6325, 1.2649)]
        assert statistics.stdev(vals) == pytest.approx(observed, rel=1e-3)
        _reg, r = _region(tmp_path, vals)
        assert r.is_saturated(A) is expect_saturated


# --- the verdict -------------------------------------------------------------


def _agent4(tmp_path, a_within=A):
    """Agent 4 with the measured noise in place, via the same loader Agent 1's
    freeze bar uses."""
    import yaml

    from agents.agent4_landscape_explorer import Agent4LandscapeExplorer

    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    # Stamped with the budget in force: measured_a_within refuses a report
    # from another one, because the in-region noise a saturation test needs is
    # a property of how much training a run gets.
    from prepare import TOKEN_BUDGET
    (state / "region_geometry.json").write_text(
        json.dumps({"a_within": a_within, "token_budget": int(TOKEN_BUDGET)}))
    cfg = tmp_path / "agents_config.yaml"
    # llm_mode statistics: the decisions are deterministic either way (the LLM
    # only narrates an already-made verdict), but leaving it on made this file
    # attempt a real CLI call per verdict and take 33s.
    cfg.write_text(yaml.dump({"agent4": {"min_runs_before_judgement": 5,
                                         "stuck_runs_pause": 5, "stuck_runs_retire": 15,
                                         "llm_mode": "statistics"}}))
    return Agent4LandscapeExplorer(config_path=str(cfg), state_dir=str(state),
                                   reports_dir=str(tmp_path / "reports"),
                                   root_dir=str(tmp_path))


def test_agent4_reads_the_measured_in_region_noise(tmp_path):
    agent4 = _agent4(tmp_path, a_within=0.001342)
    assert agent4._a_within() == pytest.approx(0.001342)


def test_saturation_does_not_fire_without_a_measurement(tmp_path):
    """THE SAFETY PROPERTY. Without region_geometry.json the generic sigma
    loader falls back to DEFAULT_SIGMA = 0.01 -- about 7.5x the real 0.001342 --
    and at that level almost every region looks saturated. A fresh checkout
    would then silently retire every region it opened. So the verdict is not
    reached at all unless the number behind it was measured."""
    import yaml

    from agents.agent4_landscape_explorer import Agent4LandscapeExplorer

    state = tmp_path / "bare_state"
    state.mkdir(parents=True, exist_ok=True)  # deliberately no measurement file
    cfg = tmp_path / "bare.yaml"
    cfg.write_text(yaml.dump({"agent4": {"min_runs_before_judgement": 5,
                                         "llm_mode": "statistics"}}))
    agent4 = Agent4LandscapeExplorer(config_path=str(cfg), state_dir=str(state),
                                     reports_dir=str(tmp_path / "r2"),
                                     root_dir=str(tmp_path))
    assert agent4._a_within() is None

    reg = RegionRegistry(str(state / "regions.json"))
    flat = reg.open_region(dict(BASE), at_run=0)
    _rival = reg.open_region({**BASE, "matrix_lr": 0.15}, at_run=0)
    for i, v in enumerate([1.2000, 1.2005, 1.1996, 1.2008, 1.1999, 1.2002]):
        reg.assign_run(flat.region_id, f"f{i}", v)

    assert agent4.judge(flat, reg, at_run=20) != SATURATED
    assert flat.flag != SATURATED


def test_a_saturated_region_is_retired_with_its_own_flag(tmp_path):
    """Not local_optimum -- that means "we stopped improving", which is a
    different and weaker claim than "there is nothing here we can read"."""
    agent4 = _agent4(tmp_path)
    reg = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    dead = reg.open_region(dict(BASE), at_run=0)
    rival = reg.open_region({**BASE, "matrix_lr": 0.15}, at_run=0)
    for i, v in enumerate([1.2000, 1.2005, 1.1996, 1.2008, 1.1999, 1.2002]):
        reg.assign_run(dead.region_id, f"d{i}", v)
    for i, v in enumerate([1.20, 1.23, 1.26, 1.21, 1.25, 1.22]):
        reg.assign_run(rival.region_id, f"r{i}", v)

    assert agent4.judge(dead, reg, at_run=20) == SATURATED
    assert dead.flag == SATURATED
    assert not dead.schedulable


def test_the_last_live_region_is_never_retired_for_saturation(tmp_path):
    """Same guard the other terminal verdicts respect: retiring the only live
    region would leave the campaign with nowhere to search at all.

    It may still be PAUSED, and that is the right outcome -- pausing is
    recoverable (the orchestrator resumes the best paused region when it has a
    GPU and nowhere better), whereas a terminal flag forbids ever returning.
    The guarantee is "not thrown away", not "still running".
    """
    from state.regions import LOCAL_OPTIMUM, NO_OPTIMUM

    agent4 = _agent4(tmp_path)
    reg = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    only = reg.open_region(dict(BASE), at_run=0)
    for i, v in enumerate([1.2000, 1.2005, 1.1996, 1.2008, 1.1999, 1.2002]):
        reg.assign_run(only.region_id, f"d{i}", v)

    verdict = agent4.judge(only, reg, at_run=20)
    assert verdict != SATURATED
    assert only.flag not in (SATURATED, LOCAL_OPTIMUM, NO_OPTIMUM), "not terminally retired"
    assert only.merged_into is None


def test_a_region_with_signal_left_survives_judgement(tmp_path):
    agent4 = _agent4(tmp_path)
    reg = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    good = reg.open_region(dict(BASE), at_run=0)
    _rival = reg.open_region({**BASE, "matrix_lr": 0.15}, at_run=0)
    # Improving steadily and widely spread, so neither saturated nor stalled.
    for i, v in enumerate([1.26, 1.25, 1.24, 1.23, 1.22, 1.21]):
        reg.assign_run(good.region_id, f"g{i}", v)

    assert agent4.judge(good, reg, at_run=20) == "keep"


# --- a region is judged against its OWN noise, not the campaign median -------


def _cfg(**over):
    base = {"n_layer": 12, "n_embd": 512, "n_head": 8, "window_s_fraction": 0.5,
            "embedding_lr": 0.5, "unembedding_lr": 0.004, "matrix_lr": 0.04,
            "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.05,
            "batch_size": 8192}
    base.update(over)
    return base


def _spread_configs(n, jitter):
    """`n` configurations packed within `jitter` of each other in matrix_lr,
    which is one of the eight tunables the distance is measured over."""
    return [_cfg(matrix_lr=0.04 + jitter * i / max(1, n - 1)) for i in range(n)]


def test_local_noise_reads_the_spread_between_nearest_neighbours():
    """Configurations a short distance apart differ mostly by noise, so the
    gap between them estimates it -- and it costs nothing, because those runs
    are already paid for."""
    from state.regions import local_noise

    configs = _spread_configs(8, 0.002)
    # a flat response plus a +-0.004 wobble: nothing but noise to find
    values = [1.30 + (0.004 if i % 2 else -0.004) for i in range(8)]

    est = local_noise(configs, values, radius=0.05, min_runs=5)
    assert est == pytest.approx(0.008, rel=0.35)


def test_local_noise_is_none_when_the_region_is_too_sparsely_sampled():
    """THE GUARD THAT MATTERS. The estimate reads neighbour gaps, so if the
    closest pair is still most of a region apart that gap is real variation
    rather than noise -- and an overstated noise floor retires a region for
    being sparsely sampled instead of for being finished."""
    from state.regions import local_noise

    spread_out = [_cfg(matrix_lr=lr) for lr in (0.006, 0.05, 0.1, 0.15, 0.19)]
    values = [1.40, 1.35, 1.32, 1.31, 1.30]

    assert local_noise(spread_out, values, radius=0.05, min_runs=5) is None


def test_local_noise_is_none_below_the_minimum_run_count():
    from state.regions import local_noise

    configs = _spread_configs(3, 0.002)
    assert local_noise(configs, [1.30, 1.31, 1.30], radius=0.05, min_runs=5) is None


def test_the_typical_neighbour_distance_is_used_not_the_smallest():
    """One duplicated configuration would drag the minimum to zero and wave a
    sparse region through, so the guard reads the median."""
    from state.regions import local_noise

    configs = [_cfg(matrix_lr=0.04), _cfg(matrix_lr=0.04)] + [
        _cfg(matrix_lr=lr) for lr in (0.09, 0.14, 0.19)]
    values = [1.30, 1.31, 1.33, 1.35, 1.37]

    assert local_noise(configs, values, radius=0.05, min_runs=5) is None


def test_a_regions_own_noise_beats_the_campaign_median(tmp_path):
    """The campaign figure is a median over configurations whose spreads differ
    5x, so it describes the frontier and nowhere else. A region that has
    measured its own neighbourhood should use that."""
    from state.results_logger import log_result

    agent4 = _agent4(tmp_path, a_within=0.004)      # campaign-wide
    registry = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    region = registry.open_region(_cfg(), at_run=0)

    configs = _spread_configs(6, 0.002)
    for i, (cfg, val) in enumerate(zip(configs, [1.30, 1.3002, 1.2999, 1.3001, 1.3000, 1.2998])):
        run_id = f"run_{i:04d}"
        log_result(run_id, cfg, {"val_bpb": val, "status": "remote_ok"},
                   results_path=str(tmp_path / "results.tsv"))
        registry.assign_run(region.region_id, run_id, val)

    local = agent4._a_within(region)
    assert local is not None
    assert local < 0.004, "this region is far quieter than the campaign median"


def test_without_enough_local_evidence_the_campaign_figure_is_used(tmp_path):
    """Falling back is right; inventing a local number from two runs is not."""
    agent4 = _agent4(tmp_path, a_within=0.004)
    registry = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    region = registry.open_region(_cfg(), at_run=0)
    registry.assign_run(region.region_id, "run_0000", 1.30)

    assert agent4._a_within(region) == pytest.approx(0.004)


def test_a_lone_region_can_still_migrate(tmp_path):
    """THE GUARD THAT MADE MIGRATION UNREACHABLE. It refused to migrate the only
    live region, to "never leave the campaign with nowhere to search" -- but
    migration opens its successor, so there is always somewhere to search
    afterwards. The check ran before the replacement it was worried about.

    Under the one-GPU policy the allocator can never hold more than one region
    (spare = n_gpus - len(regions) = 0), so the only live region is ALWAYS the
    sole one. A real campaign recorded 6 escapes in 6 proposals at coherence
    0.75, pointing 0.203 away from an anchor with a 0.02 fence, and migration
    could not fire on any of it."""
    agent4 = _agent4(tmp_path)
    registry = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    region = registry.open_region(_cfg(), at_run=0)
    assert len(registry.active()) == 1

    pressure = {"n_escapes": 6, "coherence": 0.75, "distance": 0.203,
                "target": _cfg(matrix_lr=0.09), "mean_direction": {"matrix_lr": 0.5}}
    successor = agent4.migrate(region, registry, pressure, at_run=10)

    assert successor is not None, "a lone region must still be able to migrate"
    assert successor.region_id != region.region_id
    assert region.successor_id == successor.region_id
    # And the campaign is not left empty -- which is what the old guard wanted.
    assert len(registry.active()) >= 1


def test_migration_leaves_the_anchor_where_it_was(tmp_path):
    """Anchor immutability is what region identity, merge detection and the
    don't-reopen-a-ruled-out-area check all rest on."""
    agent4 = _agent4(tmp_path)
    registry = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    region = registry.open_region(_cfg(), at_run=0)
    before = dict(region.anchor)

    agent4.migrate(region, registry, {
        "n_escapes": 4, "coherence": 0.8, "distance": 0.15,
        "target": _cfg(matrix_lr=0.09), "mean_direction": {"matrix_lr": 0.5}}, at_run=10)

    assert region.anchor == before
