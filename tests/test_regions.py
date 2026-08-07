"""Tests for state/regions.py -- the persistent multi-region registry.

Same discipline as tests/test_landscape.py: assert the contracts that matter
(identity is stable, distance is scale-free, merges preserve history, budget
accounting can't be gamed) rather than exact numbers out of a fitted model.
The three properties with real consequences if they break:

  - an anchor written early still means the same point later, even after the
    campaign explores past the values it was written from;
  - two regions that converge get merged, so parallelism can't silently
    collapse to two GPUs in one basin;
  - a run is attributed to the region that spent the budget, not to whichever
    anchor it drifted nearest.
"""

import json
import math

import pytest

from state.regions import (
    ACTIVE,
    LOCAL_OPTIMUM,
    MERGED,
    MIN_RUNS_FOR_ELITE_SCORE,
    NO_OPTIMUM,
    PAUSED,
    Region,
    RegionRegistry,
    distance,
    to_vector,
)
from state.results_analysis import HYPERPARAM_COLUMNS


BASE = {
    "n_layer": 8, "n_embd": 512, "n_head": 4, "window_s_fraction": 0.75,
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192,
}


def hp(**overrides):
    out = dict(BASE)
    out.update(overrides)
    return out


# -- geometry --------------------------------------------------------------


def test_vector_covers_every_searched_dimension():
    v = to_vector(BASE)
    assert len(v) == len(HYPERPARAM_COLUMNS)
    assert all(0.0 <= x <= 1.0 for x in v)


def test_missing_parameter_maps_to_midpoint_not_zero():
    """An absent parameter is unknown, not zero -- mapping it to 0.0 would
    place the configuration at the extreme edge of that axis and make it look
    maximally far from everything."""
    partial = {k: v for k, v in BASE.items() if k != "matrix_lr"}
    i = HYPERPARAM_COLUMNS.index("matrix_lr")
    assert to_vector(partial)[i] == 0.5


def test_distance_is_zero_to_self_and_scale_free():
    assert distance(BASE, BASE) == pytest.approx(0.0)
    # Opposite corners of the whole space are 1.0 apart *because* of the
    # sqrt(n_dims) division -- without it this would be sqrt(11).
    from agents.agent1_training_specialist import SEARCH_SPACE

    lo = {k: SEARCH_SPACE[k][0] for k in HYPERPARAM_COLUMNS}
    hi = {k: SEARCH_SPACE[k][1] for k in HYPERPARAM_COLUMNS}
    assert distance(lo, hi) == pytest.approx(1.0)


def test_learning_rates_are_compared_on_a_log_scale():
    """matrix_lr spans 0.005-0.2. A linear metric would call 0.005 vs 0.01
    (a doubling) a smaller step than 0.19 vs 0.2 (a 5% change)."""
    doubling = distance(hp(matrix_lr=0.005), hp(matrix_lr=0.01))
    tail = distance(hp(matrix_lr=0.19), hp(matrix_lr=0.2))
    assert doubling > tail


def test_anchor_distance_does_not_drift_as_the_campaign_explores():
    """The reason anchors normalize against SEARCH_SPACE rather than observed
    data: fit_surrogate's bounds widen with every new extreme run, and an
    anchor measured against those would quietly move."""
    a, b = hp(n_layer=6), hp(n_layer=10)
    before = distance(a, b)
    # Something far out in the space gets explored; the anchors did not move.
    _ = distance(hp(n_layer=24, batch_size=32768), a)
    assert distance(a, b) == pytest.approx(before)


# -- history and scoring ----------------------------------------------------


def test_crashed_run_consumes_budget_but_contributes_no_measurement():
    r = Region(region_id="r0001", anchor={}, center={})
    r.record("run_0", 1.30)
    r.record("run_1", float("inf"))
    r.record("run_2", None)
    assert r.n_runs == 3, "a crashed run still spent a GPU"
    assert r.n_measured == 1, "but it is not evidence about the region"
    assert r.best_val_bpb == pytest.approx(1.30)


def test_runs_since_improvement_requires_a_real_margin():
    """With sigma = 0.00919 a strict '<' counts noise as progress. The
    min_improvement argument exists to force callers to say what better
    means."""
    r = Region(region_id="r0001", anchor={}, center={})
    for v in (1.30, 1.2995, 1.2990, 1.2985):
        r.record(f"run_{v}", v)
    assert r.runs_since_improvement(0.0) == 0          # every step "improved"
    assert r.runs_since_improvement(0.00919 * 2) == 3  # none of them really did


def test_elite_score_is_a_median_below_the_quartile_threshold():
    r = Region(region_id="r0001", anchor={}, center={})
    for v in (1.20, 1.30, 1.40):
        r.record("x", v)
    assert r.n_measured < MIN_RUNS_FOR_ELITE_SCORE
    assert r.elite_score() == pytest.approx(1.30)


def test_elite_score_becomes_a_real_quartile_once_there_is_enough_data():
    r = Region(region_id="r0001", anchor={}, center={})
    for v in (1.20, 1.21, 1.30, 1.31, 1.40, 1.41, 1.50, 1.51):
        r.record("x", v)
    assert r.n_measured >= MIN_RUNS_FOR_ELITE_SCORE
    # top quartile of 8 = 2 runs -> median of {1.20, 1.21}
    assert r.elite_score() == pytest.approx(1.205)


def test_one_bad_outlier_does_not_decide_the_ranking():
    """elite_score is a median, not a mean, because the GPU allocator reads
    it -- a single OOM-adjacent finite outlier should not cost a good region
    its GPUs."""
    good = Region(region_id="r1", anchor={}, center={})
    for v in (1.21, 1.22, 1.23, 1.24, 1.75):
        good.record("x", v)
    assert good.elite_score() < 1.30


# -- registry lifecycle -----------------------------------------------------


def test_registry_round_trips_through_disk(tmp_path):
    path = tmp_path / "regions.json"
    reg = RegionRegistry(str(path))
    r = reg.open_region(hp(), at_run=3, origin="max_uncertainty")
    reg.assign_run(r.region_id, "run_0007", 1.25)
    reg.save()

    reloaded = RegionRegistry(str(path))
    got = reloaded.get(r.region_id)
    assert got is not None
    assert got.n_runs == 1
    assert got.best_val_bpb == pytest.approx(1.25)
    assert got.origin == "max_uncertainty"
    # A resumed campaign must not reuse an id it already handed out.
    assert reloaded.open_region(hp(), at_run=9).region_id != r.region_id


def test_corrupt_registry_does_not_take_down_the_campaign(tmp_path):
    path = tmp_path / "regions.json"
    path.write_text("{not json", encoding="utf-8")
    assert RegionRegistry(str(path)).regions == []


def test_active_ranks_best_first_and_sorts_unmeasured_last(tmp_path):
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    worse = reg.open_region(hp(n_layer=4), at_run=0)
    better = reg.open_region(hp(n_layer=12), at_run=0)
    fresh = reg.open_region(hp(n_layer=20), at_run=0)
    reg.assign_run(worse.region_id, "a", 1.40)
    reg.assign_run(better.region_id, "b", 1.20)

    order = [r.region_id for r in reg.active()]
    assert order == [better.region_id, worse.region_id, fresh.region_id], (
        "an unmeasured region has no evidence in its favour and must not "
        "displace a proven one from the allocation"
    )


def test_terminal_regions_are_not_schedulable(tmp_path):
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    r = reg.open_region(hp(), at_run=0)
    assert r.schedulable
    for flag in (PAUSED, NO_OPTIMUM, LOCAL_OPTIMUM):
        r.set_flag(flag, at_run=10)
        assert not r.schedulable
        assert r.flag_since_run == 10


def test_run_is_attributed_to_the_region_that_dispatched_it(tmp_path):
    """The local search drifts; attribution must not. Crediting a run to
    whichever anchor it ended up nearest would let one region's budget count
    toward another's lifecycle thresholds."""
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    a = reg.open_region(hp(n_layer=4), at_run=0)
    b = reg.open_region(hp(n_layer=20), at_run=0)
    drifted = hp(n_layer=19)  # much closer to b's anchor than to a's
    reg.assign_run(a.region_id, "run_0001", 1.25, center=drifted)

    assert reg.get(a.region_id).n_runs == 1
    assert reg.get(b.region_id).n_runs == 0
    assert reg.get(a.region_id).center["n_layer"] == 19, "center follows the search"
    assert reg.get(a.region_id).anchor["n_layer"] == 4, "anchor is identity, it does not move"


# -- merging ----------------------------------------------------------------


def test_a_merge_restores_chronological_order(tmp_path):
    """Concatenating the two histories would fabricate a chronology -- the
    absorbed region's runs appearing to happen after the survivor's, when the
    two searches ran concurrently. runs_since_improvement is order-dependent,
    so a merged region would read as stuck purely as an artifact of that."""
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    a = reg.open_region(hp(n_embd=512), at_run=0)
    b = reg.open_region(hp(n_embd=516), at_run=0)
    # Interleaved in real time: a improves late, b was worse early.
    reg.assign_run(a.region_id, "run_0000", 1.40)
    reg.assign_run(b.region_id, "run_0001", 1.60)
    reg.assign_run(a.region_id, "run_0002", 1.20)

    reg.merge_overlapping(radius=0.05, at_run=9)
    survivor = reg.get(a.region_id)
    assert survivor.val_bpbs == [1.40, 1.60, 1.20], "true order, not a + b"
    assert survivor.runs_since_improvement(0.0) == 0, "the newest run IS the best"


def test_converged_regions_merge_and_history_survives(tmp_path):
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    a = reg.open_region(hp(n_layer=8), at_run=0)
    b = reg.open_region(hp(n_layer=8, n_embd=520), at_run=1)
    reg.assign_run(a.region_id, "a0", 1.20)
    reg.assign_run(b.region_id, "b0", 1.40)

    merges = reg.merge_overlapping(radius=0.05, at_run=12)
    assert merges == [(b.region_id, a.region_id)], "the better-scoring anchor survives"
    assert reg.get(a.region_id).n_runs == 2, "the absorbed region's budget is not lost"
    assert reg.get(b.region_id).flag == MERGED
    assert reg.get(b.region_id).merged_into == a.region_id
    assert [r.region_id for r in reg.active()] == [a.region_id]


def test_distant_regions_are_left_alone(tmp_path):
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    reg.open_region(hp(n_layer=1, n_embd=128), at_run=0)
    reg.open_region(hp(n_layer=24, n_embd=1024), at_run=0)
    assert reg.merge_overlapping(radius=0.05, at_run=1) == []
    assert len(reg.active()) == 2


def test_a_ruled_out_region_is_never_resurrected_by_a_merge(tmp_path):
    """Absorbing a no_optimum region into a live one would turn a negative
    result into evidence in the live region's favour."""
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    dead = reg.open_region(hp(n_layer=8), at_run=0)
    dead.set_flag(NO_OPTIMUM, at_run=5)
    for v in (1.70, 1.72, 1.75):
        reg.assign_run(dead.region_id, "x", v)
    live = reg.open_region(hp(n_layer=8, n_embd=515), at_run=6)
    reg.assign_run(live.region_id, "y", 1.22)

    assert reg.merge_overlapping(radius=0.05, at_run=7) == []
    assert reg.get(live.region_id).n_runs == 1
    assert reg.get(live.region_id).best_val_bpb == pytest.approx(1.22)


def test_merging_is_transitive_in_one_pass(tmp_path):
    """Three regions drifting into one basin must end as one region, not two
    -- a single pass that stops after the first merge would leave a duplicate
    holding a GPU."""
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    a = reg.open_region(hp(n_embd=512), at_run=0)
    reg.open_region(hp(n_embd=516), at_run=0)
    reg.open_region(hp(n_embd=520), at_run=0)
    reg.assign_run(a.region_id, "a0", 1.20)

    reg.merge_overlapping(radius=0.05, at_run=3)
    assert len(reg.active()) == 1


# -- per-region search state ------------------------------------------------


def test_each_region_gets_its_own_planner_state_file(tmp_path):
    """The point of the whole refactor: search_planner.propose_next already
    takes state_path, so distinct paths give each region an independent cold
    start, frozen set, and block rotation."""
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    a = reg.open_region(hp(), at_run=0)
    b = reg.open_region(hp(n_layer=16), at_run=0)
    pa, pb = a.planner_state_path("state/sp"), b.planner_state_path("state/sp")
    assert pa != pb
    assert pa.endswith(f"{a.region_id}.json")
    assert a.report_dir() != b.report_dir()


# -- reporting --------------------------------------------------------------


def test_flags_snapshot_matches_the_shape_the_chart_already_reads(tmp_path):
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    r = reg.open_region(hp(), at_run=4)
    reg.assign_run(r.region_id, "run_0", 1.3)
    snap = reg.flags_snapshot()
    assert len(snap) == 1
    assert set(snap[0]) >= {"hyperparams", "flag", "since_iteration", "n_runs"}
    assert snap[0]["flag"] == ACTIVE
    assert set(snap[0]["hyperparams"]) == set(HYPERPARAM_COLUMNS)
    json.dumps(snap)  # must stay serializable for state/visualize.py


def test_merged_regions_are_excluded_from_reporting(tmp_path):
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    a = reg.open_region(hp(), at_run=0)
    reg.open_region(hp(n_embd=515), at_run=0)
    reg.assign_run(a.region_id, "a0", 1.2)
    reg.merge_overlapping(radius=0.05, at_run=1)
    assert len(reg.flags_snapshot()) == 1
    assert len(reg.summary_rows()) == 1


def test_summary_rows_carry_what_agent3_needs(tmp_path):
    reg = RegionRegistry(str(tmp_path / "regions.json"))
    r = reg.open_region(hp(), at_run=2, origin="max_ei")
    for v in (1.30, 1.28):
        reg.assign_run(r.region_id, "x", v)
    row = reg.summary_rows()[0]
    assert row["region_id"] == r.region_id
    assert row["n_runs"] == 2 and row["n_measured"] == 2
    assert row["best_val_bpb"] == pytest.approx(1.28)
    assert row["origin"] == "max_ei"
