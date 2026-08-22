"""Step 5b: a region whose search keeps trying to leave is re-anchored where it
was trying to go -- by opening a SUCCESSOR, never by moving the anchor.

The escape record costs nothing: every proposal already generates and scores
candidates outside the fence (they are simply not eligible to run), so where the
search WANTED to go is known for free. See surrogate.propose_via_ei.

The test that matters is coherence, not count. Escapes bouncing off different
walls point every which way and average to nothing -- that is a region being
explored. Sustained pressure one way survives the averaging.
"""

import json
from pathlib import Path

import pytest
import yaml

from agents.agent4_landscape_explorer import Agent4LandscapeExplorer
from state.regions import MIGRATED, RegionRegistry

BASE = {
    "n_layer": 8, "n_embd": 512, "n_head": 4, "window_s_fraction": 0.75,
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192,
}


def _agent4(tmp_path, **cfg):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump({"agent4": {"llm_mode": "statistics",
                                          "min_runs_before_judgement": 5, **cfg}}))
    return Agent4LandscapeExplorer(config_path=str(path),
                                   state_dir=str(tmp_path / "state"),
                                   reports_dir=str(tmp_path / "reports"),
                                   root_dir=str(tmp_path))


def _write_escapes(tmp_path, region, directions, escaped=True,
                   mean_inside=1.30, mean_gain=0.05):
    """One plan JSON per proposal, shaped exactly as search_planner writes it."""
    d = Path(tmp_path / "reports" / "agent1_search_plan" / region.region_id)
    d.mkdir(parents=True, exist_ok=True)
    for i, direction in enumerate(directions):
        (d / f"plan_{i:04d}.json").write_text(json.dumps({
            "iteration": i,
            "escape": {"escaped": escaped, "direction": direction,
                       "distance": 0.09, "radius": 0.05,
                       "ei_inside": 1e-4, "ei_outside": 5e-4,
                       # Predicted means, not just acquisition values: the
                       # target must be predicted BETTER (lower val_bpb), not
                       # merely less explored. Default gap is comfortably past
                       # sigma_region so these fixtures still migrate.
                       "mean_inside": mean_inside,
                       "mean_outside": mean_inside - mean_gain},
        }))


def _registry(tmp_path, n=2):
    reg = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    regions = [reg.open_region({**BASE, "matrix_lr": 0.04 + 0.05 * i}, at_run=0)
               for i in range(n)]
    # GLOBAL run ids, as the orchestrator issues them (run_{iteration:04d}).
    # escape_pressure matches plan_NNNN.json against the region's own run ids to
    # ignore plans left behind by an earlier campaign, so a region-prefixed id
    # here would make every plan look like it belonged to someone else. Each
    # region gets its own decade so two regions never claim the same run.
    for k, r in enumerate(regions):
        for i, v in enumerate([1.30, 1.28, 1.26, 1.25, 1.24, 1.23]):
            reg.assign_run(r.region_id, f"run_{k * 10 + i:04d}", v)
    return reg, regions


# --- detecting the pressure --------------------------------------------------


def test_consistent_pressure_is_detected(tmp_path):
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.30, "batch_size": 0.10}] * 4)

    p = a4.escape_pressure(r)
    assert p is not None
    assert p["n_escapes"] == 4
    assert p["coherence"] == pytest.approx(1.0)
    assert p["mean_direction"]["matrix_lr"] == pytest.approx(0.30)


def test_pressure_pointing_every_which_way_is_ignored(tmp_path):
    """The whole point. A search bouncing off different walls is being
    explored, not mis-anchored, and its escapes cancel under averaging."""
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, r, [
        {"matrix_lr": +0.30, "batch_size": 0.0},
        {"matrix_lr": -0.30, "batch_size": 0.0},
        {"matrix_lr": 0.0, "batch_size": +0.30},
        {"matrix_lr": 0.0, "batch_size": -0.30},
    ])
    assert a4.escape_pressure(r) is None


def test_too_few_escapes_is_not_pressure(tmp_path):
    a4 = _agent4(tmp_path, escape_runs_to_migrate=3)
    reg, (r, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.30}] * 2)
    assert a4.escape_pressure(r) is None


def test_candidates_that_stayed_inside_are_not_escapes(tmp_path):
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.30}] * 4, escaped=False)
    assert a4.escape_pressure(r) is None


def test_pressure_toward_somewhere_already_inside_the_fence_is_ignored(tmp_path):
    """If the target is within the radius the search may simply go there; there
    is nothing to migrate to."""
    a4 = _agent4(tmp_path, region_radius=0.5)  # fence wider than the escape
    reg, (r, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.05}] * 4)
    assert a4.escape_pressure(r) is None


def test_no_history_is_not_pressure(tmp_path):
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    assert a4.escape_pressure(r) is None


# --- acting on it ------------------------------------------------------------


def test_migration_opens_a_successor_and_never_moves_the_anchor(tmp_path):
    """Anchor immutability is what identity, merge detection and the
    don't-reopen check all rest on."""
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    anchor_before = dict(r.anchor)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.35}] * 4)

    successor = a4.migrate(r, reg, a4.escape_pressure(r), at_run=9)

    assert successor is not None
    assert r.anchor == anchor_before, "the anchor must never move"
    assert r.flag == MIGRATED
    assert r.successor_id == successor.region_id
    assert successor.anchor["matrix_lr"] > anchor_before["matrix_lr"]
    assert successor.origin == f"migrated_from_{r.region_id}"


def test_a_successor_keeps_the_same_architecture(tmp_path):
    """Escape is measured over the tunables only -- a successor is the same
    model in a different corner of its settings, not a different model."""
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.35}] * 4)

    successor = a4.migrate(r, reg, a4.escape_pressure(r), at_run=9)

    for col in ("n_layer", "n_embd", "n_head"):
        assert successor.anchor[col] == r.anchor[col]


def test_the_last_live_region_migrates_and_leaves_its_successor_behind(tmp_path):
    """WAS `..._never_migrates`, on the reasoning that closing the only live
    region would leave the campaign with nowhere to search. That reasoning was
    wrong: migration OPENS ITS SUCCESSOR before closing the original, so there
    is always somewhere to search afterwards. The guard ran before the
    replacement it was worried about and could not see it.

    It mattered because it made migration unreachable in the case that needs it
    most. Under the one-GPU policy the allocator can never hold more than one
    region (spare = n_gpus - len(regions) = 0), so the only live region is
    always the sole one -- and a real campaign recorded 6 escapes in 6
    proposals at coherence 0.75, pointing 0.203 away from an anchor with a 0.02
    fence, with migration unable to act on any of it.
    """
    a4 = _agent4(tmp_path)
    reg, (r,) = _registry(tmp_path, n=1)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.35}] * 4)

    successor = a4.migrate(r, reg, a4.escape_pressure(r), at_run=9)

    assert successor is not None
    assert r.flag == MIGRATED
    assert r.successor_id == successor.region_id
    # The property the old guard actually wanted, stated directly.
    assert reg.active(), "the campaign must still have somewhere to search"


def test_maintain_migrates_after_judging_not_before(tmp_path):
    """A region already retired must not spawn a successor: "there is nothing
    here" and "the anchor is misplaced" are different findings, and acting on
    both would open a region beside somewhere just ruled out."""
    a4 = _agent4(tmp_path)
    reg, (good, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, good, [{"matrix_lr": 0.35}] * 4)

    out = a4.maintain(reg, at_run=12)

    assert out["migrations"] == [(good.region_id, reg.regions[-1].region_id)]
    assert good.flag == MIGRATED


def test_a_retired_region_does_not_migrate(tmp_path):
    from state.regions import NO_OPTIMUM

    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    r.set_flag(NO_OPTIMUM, 0)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.35}] * 4)

    out = a4.maintain(reg, at_run=12)

    assert out["migrations"] == []
    assert r.flag == NO_OPTIMUM


def test_the_successor_pointer_survives_a_reload(tmp_path):
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.35}] * 4)
    successor = a4.migrate(r, reg, a4.escape_pressure(r), at_run=9)
    reg.save()

    reloaded = RegionRegistry(str(tmp_path / "state" / "regions.json"))
    assert reloaded.get(r.region_id).successor_id == successor.region_id


def test_a_barely_searched_region_is_not_declared_mis_anchored(tmp_path):
    """MIGRATION CHAINED WITHOUT THIS. A region with one recorded run migrated,
    its successor migrated after one more, and the next was being judged with
    zero -- the search walked from region to region in 0.07-0.10 steps and
    exploited none of them.

    Escape pressure in a brand-new region mostly reflects the global surrogate
    pointing at the campaign's best area, which EVERY new region will do
    whatever its anchor. It says something about THIS anchor only once the
    region has been searched -- the same bar judge() applies before any other
    verdict."""
    a4 = _agent4(tmp_path)
    reg, _existing = _registry(tmp_path, n=1)
    fresh = reg.open_region({**BASE, "matrix_lr": 0.12}, at_run=5)
    reg.assign_run(fresh.region_id, "run_0099", 1.40)          # one run only
    _write_escapes(tmp_path, fresh, [{"matrix_lr": 0.35}] * 6)  # ample, coherent

    assert fresh.n_measured < a4.min_runs_before_judgement
    assert a4.escape_pressure(fresh) is None


def test_a_well_searched_region_still_migrates(tmp_path):
    """The guard is a minimum, not a veto."""
    a4 = _agent4(tmp_path)
    reg, (r,) = _registry(tmp_path, n=1)      # _registry gives it 6 runs
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.35}] * 6)

    assert r.n_measured >= a4.min_runs_before_judgement
    assert a4.escape_pressure(r) is not None


def test_pressure_toward_somewhere_merely_UNEXPLORED_is_ignored(tmp_path):
    """THE RUNAWAY. EI mixes promise with uncertainty, so a candidate far from
    any data scores highly simply for being unknown. Judged on EI alone,
    migration walked outward forever: each move landed further from the
    campaign's data, where uncertainty and EI were higher still, so the next
    escape pointed further out again. Measured steps grew 0.074 -> 0.104 ->
    0.174 -> 0.206 against a 0.02 fence, and four regions running were
    abandoned at the first legal opportunity without ever being exploited.

    Comparing predicted MEANS asks the question actually at issue: is somewhere
    else better, or just less known?"""
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    # Strongly coherent, plenty of them, and the outside is predicted no better.
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.35}] * 6, mean_gain=0.0)

    assert a4.escape_pressure(r) is None


def test_pressure_toward_somewhere_predicted_better_still_migrates(tmp_path):
    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    _write_escapes(tmp_path, r, [{"matrix_lr": 0.35}] * 6, mean_gain=0.05)

    assert a4.escape_pressure(r) is not None


def test_escape_records_without_predicted_means_are_skipped(tmp_path):
    """Written before the means were recorded. Absent is not "good enough" --
    the same rule the budget stamps follow."""
    import json
    from pathlib import Path

    a4 = _agent4(tmp_path)
    reg, (r, _other) = _registry(tmp_path)
    d = Path(tmp_path / "reports" / "agent1_search_plan" / r.region_id)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(6):
        (d / f"plan_{i:04d}.json").write_text(json.dumps({
            "escape": {"escaped": True, "direction": {"matrix_lr": 0.35},
                       "distance": 0.09, "radius": 0.05}}))

    assert a4.escape_pressure(r) is None


def test_a_migration_step_is_capped(tmp_path):
    """The gate checks DIRECTION and predicted MEAN, never DISTANCE -- so a
    region could relocate 17 fence-widths on 5 runs of evidence. Measured in
    campaign 11: 0.0554, then 0.1902, then 0.3305 against a 0.02 fence.

    A cap does not forbid going far, it forbids going far IN ONE STEP: the
    successor re-anchors, measures where it landed, and migrates again if it
    still wants to."""
    # A cap of 2 fences (0.04). The default 10 does not bite on this fixture --
    # the escape target is bounded by the search space, so even a maximal pull
    # only reaches ~0.13 -- and a test that never triggers the mechanism it
    # names passes vacuously.
    a4 = _agent4(tmp_path, region_radius=0.02, max_migration_radii=2.0)
    reg = RegionRegistry(str(tmp_path / 'state' / 'regions.json'))
    r = reg.open_region(dict(BASE), at_run=0)
    for i in range(6):
        reg.assign_run(r.region_id, f'run_{i:04d}', 1.30)
    # a long coherent pull on one axis
    _write_escapes(tmp_path, r, [{'scalar_lr': 0.9}] * 4)

    p = a4.escape_pressure(r)
    assert p is not None
    cap = a4.max_migration_radii * a4.region_radius
    assert p['truncated_from'] is not None, 'the cap never fired -- vacuous test'
    assert p['truncated_from'] > cap
    assert p['distance'] <= cap * 1.05, f"step {p['distance']} exceeded cap {cap}"
    assert p['distance'] > a4.region_radius, 'truncated inside the fence, so nowhere to go'


def test_a_step_inside_the_cap_is_left_alone(tmp_path):
    a4 = _agent4(tmp_path, region_radius=0.02, max_migration_radii=10.0)
    reg = RegionRegistry(str(tmp_path / 'state' / 'regions.json'))
    r = reg.open_region(dict(BASE), at_run=0)
    for i in range(6):
        reg.assign_run(r.region_id, f'run_{i:04d}', 1.30)
    _write_escapes(tmp_path, r, [{'scalar_lr': 0.25}] * 4)

    p = a4.escape_pressure(r)
    assert p is not None and p['truncated_from'] is None


def test_the_number_the_gate_turns_on_is_recorded(tmp_path):
    """The decision log carried coherence, distance and escape count but NOT
    the predicted-mean gain the gate actually tests -- so it could show that
    migration fired and not that the margin was met. That blind spot hid four
    bugs in this same code."""
    a4 = _agent4(tmp_path, region_radius=0.02)
    reg = RegionRegistry(str(tmp_path / 'state' / 'regions.json'))
    r = reg.open_region(dict(BASE), at_run=0)
    for i in range(6):
        reg.assign_run(r.region_id, f'run_{i:04d}', 1.30)
    _write_escapes(tmp_path, r, [{'scalar_lr': 0.25}] * 4)

    p = a4.escape_pressure(r)
    assert p is not None
    assert 'mean_gain' in p and isinstance(p['mean_gain'], float)
    assert p['mean_gain'] >= p['gain_bar'], 'gate passed, so the margin was met'


def test_stale_plans_from_an_earlier_campaign_are_ignored(tmp_path):
    """THE MEASURED FAILURE. After a fresh start the live campaign wrote
    plan_0000-0029, but plan_0030-0121 from an archived campaign were still on
    disk. escape_pressure takes the last `escape_window` files BY NAME, so it
    read the stale ones, whose escapes predate mean_inside/mean_outside -- and
    returned None, which is indistinguishable from "the search does not want to
    leave". Six consecutive real escapes, gains 0.005-0.100 against a 0.0106
    bar, were never looked at."""
    a4 = _agent4(tmp_path, region_radius=0.02)
    reg = RegionRegistry(str(tmp_path / 'state' / 'regions.json'))
    r = reg.open_region(dict(BASE), at_run=0)
    for i in range(6):
        reg.assign_run(r.region_id, f'run_{i:04d}', 1.30)

    # the live campaign's escapes, run_0000-0005
    _write_escapes(tmp_path, r, [{'scalar_lr': 0.25}] * 6)
    assert a4.escape_pressure(r) is not None, 'live escapes should migrate'

    # now drop stale plans with HIGHER numbers, as an archived campaign leaves
    d = Path(tmp_path / 'reports' / 'agent1_search_plan' / r.region_id)
    for i in range(100, 106):
        (d / f'plan_{i:04d}.json').write_text(json.dumps(
            {'iteration': i, 'escape': {'escaped': True, 'direction': {'scalar_lr': -0.9},
                                        'distance': 0.5}}), encoding='utf-8')

    # they sort last, but they belong to no run this region has
    assert a4.escape_pressure(r) is not None, 'stale plans hijacked the window'
