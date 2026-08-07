"""Tests for agents/allocator.py -- how GPUs are spread across live regions.

The properties worth pinning are the ones whose failure is invisible in a
results log until several waves later: the frontier silently losing its GPU,
a region being retired because hardware was scarce rather than because it was
ruled out, and spare capacity piling onto one region while another starves.
"""

import pytest

from agents.allocator import describe, plan
from state.regions import RegionRegistry


BASE = {
    "n_layer": 8, "n_embd": 512, "n_head": 4, "window_s_fraction": 0.75,
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192,
}


@pytest.fixture
def registry(tmp_path):
    return RegionRegistry(str(tmp_path / "regions.json"))


def make_regions(registry, scores):
    """One region per score; None means "opened but never run"."""
    out = []
    for i, score in enumerate(scores):
        hp = dict(BASE)
        hp["n_layer"] = 2 + i * 3        # keep the anchors far apart
        r = registry.open_region(hp, at_run=0)
        if score is not None:
            registry.assign_run(r.region_id, f"run_{i}", score)
        out.append(r)
    return registry.active()   # ranked best-first, as the allocator expects


# -- balanced case ----------------------------------------------------------


def test_one_gpu_each_when_counts_match(registry):
    regions = make_regions(registry, [1.20, 1.30, 1.40])
    p = plan(regions, n_gpus=3, max_regions=4)
    assert sorted(x for x in p.assignments) == sorted(r.region_id for r in regions)
    assert p.to_pause == [] and p.open_new == 0


# -- rule 2: fewer GPUs than regions ----------------------------------------


def test_scarce_gpus_run_the_best_regions(registry):
    regions = make_regions(registry, [1.20, 1.30, 1.40, 1.50])
    p = plan(regions, n_gpus=2, max_regions=4)
    assert p.assignments == [regions[0].region_id, regions[1].region_id]


def test_a_region_without_a_gpu_is_paused_not_retired(registry):
    """Pausing is recoverable and keeps the region's history and planner
    state. Retiring would confuse "we ran out of hardware" with "we ruled
    this out"."""
    regions = make_regions(registry, [1.20, 1.30, 1.40, 1.50])
    p = plan(regions, n_gpus=2, max_regions=4)
    assert p.to_pause == [regions[2].region_id, regions[3].region_id]
    assert p.open_new == 0, "no capacity to spare -- must not open more"


def test_no_gpus_pauses_everything_and_assigns_nothing(registry):
    regions = make_regions(registry, [1.20, 1.30])
    p = plan(regions, n_gpus=0, max_regions=4)
    assert p.assignments == []
    assert set(p.to_pause) == {r.region_id for r in regions}


# -- rule 3: more GPUs than regions -----------------------------------------


def test_a_spare_gpu_opens_a_new_region_while_under_the_cap(registry):
    """Coverage is the scarcer thing: a second GPU on the champion buys a
    slightly faster local search, a new region buys somewhere never looked."""
    regions = make_regions(registry, [1.20, 1.30])
    p = plan(regions, n_gpus=4, max_regions=4)
    assert p.open_new == 2
    assert p.assignments.count(None) == 2
    assert p.reinforced == {}


def test_spares_reinforce_the_best_regions_once_the_cap_is_reached(registry):
    regions = make_regions(registry, [1.20, 1.30])
    p = plan(regions, n_gpus=4, max_regions=2)
    assert p.open_new == 0
    assert None not in p.assignments
    assert p.assignments.count(regions[0].region_id) == 2
    assert p.assignments.count(regions[1].region_id) == 2


def test_two_spares_do_not_both_pile_onto_the_champion(registry):
    """With 4 GPUs and 2 regions this is the difference between 3/1 and 2/2."""
    regions = make_regions(registry, [1.20, 1.30])
    p = plan(regions, n_gpus=4, max_regions=2)
    assert p.reinforced == {regions[0].region_id: 1, regions[1].region_id: 1}


def test_a_single_spare_goes_to_the_champion(registry):
    regions = make_regions(registry, [1.20, 1.30, 1.40])
    p = plan(regions, n_gpus=4, max_regions=3)
    assert p.assignments.count(regions[0].region_id) == 2


# -- rule 1: the champion is never starved ----------------------------------


@pytest.mark.parametrize("n_gpus", [1, 2, 3, 4, 5])
def test_the_best_region_always_gets_a_gpu(registry, n_gpus):
    regions = make_regions(registry, [1.20, 1.30, 1.40])
    p = plan(regions, n_gpus=n_gpus, max_regions=4)
    assert regions[0].region_id in p.assignments


def test_an_unmeasured_region_cannot_displace_a_proven_one(registry):
    """A newly-opened region has no evidence in its favour. If it sorted
    first it would take the last GPU in a scarce wave and leave the frontier
    with nothing running."""
    regions = make_regions(registry, [1.25, None, None])
    p = plan(regions, n_gpus=1, max_regions=4)
    assert p.assignments == [regions[0].region_id]
    assert regions[0].best_val_bpb == pytest.approx(1.25)


# -- cold start -------------------------------------------------------------


def test_an_empty_registry_asks_for_new_regions(registry):
    p = plan([], n_gpus=3, max_regions=4)
    assert p.assignments == [None, None, None]
    assert p.open_new == 3 and p.to_pause == []


def test_a_cold_start_never_opens_more_regions_than_the_cap(registry):
    p = plan([], n_gpus=8, max_regions=3)
    assert p.open_new == 3
    assert len(p.assignments) == 3, "surplus GPUs go unused rather than over-fragmenting"


def test_an_empty_registry_with_no_gpus_does_nothing(registry):
    p = plan([], n_gpus=0, max_regions=4)
    assert p.assignments == [] and p.open_new == 0


# -- logging ----------------------------------------------------------------


def test_describe_reports_the_split_and_the_pauses(registry):
    regions = make_regions(registry, [1.20, 1.30, 1.40, 1.50])
    p = plan(regions, n_gpus=2, max_regions=4)
    line = describe(p, regions)
    assert regions[0].region_id in line
    assert "1.2000" in line
    assert "paused" in line and regions[3].region_id in line


def test_describe_handles_regions_with_no_runs_yet(registry):
    regions = make_regions(registry, [None, None])
    line = describe(plan(regions, n_gpus=2, max_regions=4), regions)
    assert "no runs yet" in line
