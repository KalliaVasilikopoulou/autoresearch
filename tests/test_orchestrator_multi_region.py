"""Orchestrator-side tests for multi-region search: which region each GPU
serves, how a wave's slots stay independent, and how results get attributed.

Replaces tests/test_orchestrator_agent4_window.py. That file tested a model
where Agent 4 seized every slot for a bounded window and handed control back;
here every slot belongs to a region for the whole campaign and several are
searched at once.

Mirrors tests/test_orchestrator_parallel_wave.py -- remote_runner is
monkeypatched throughout, so no real SSH, no real GPU, no real training.
"""

import pytest

from agents import remote_runner
from agents.orchestrator import Orchestrator
from state.landscape import LANDSCAPE_DEPS_AVAILABLE
from state.regions import ACTIVE, LOCAL_OPTIMUM, NO_OPTIMUM, PAUSED, RegionRegistry

requires_deps = pytest.mark.skipif(
    not LANDSCAPE_DEPS_AVAILABLE, reason="scikit-learn not installed"
)

FOUR_GPUS = [
    {"index": i, "mem_used_mb": 100, "mem_total_mb": 20100, "util_pct": 1, "free_mb": 20000}
    for i in (0, 1, 2, 3)
]


def _config(tmp_path, max_regions=4, max_parallel_runs=4):
    path = tmp_path / "agents_config.yaml"
    path.write_text(f"""
agent1:
  use_llm: false
  accuracy_threshold: 0.01
  cost_limit_usd: 50.0
  training_budget_seconds: 60

agent2:
  xai_method: fast
  use_llm: false
  ablation_k: 3

agent3:
  batch_size: 100
  use_llm: false
  generate_charts: false

agent4:
  enabled: true
  llm_mode: statistics
  max_regions: {max_regions}
  min_runs_before_judgement: 5
  stuck_runs_pause: 5
  stuck_runs_retire: 15
  sigma_region: 0.0028
  retire_margin_sigma: 3.0
  region_radius: 0.05
  merge_radius: 0.025

llm:
  backend: none

orchestrator:
  parallel: true
  max_parallel_runs: {max_parallel_runs}
""".strip(), encoding="utf-8")
    return path


def _make_orchestrator(tmp_path, **kwargs):
    return Orchestrator(
        config_path=str(_config(tmp_path, **kwargs)),
        state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"),
        root_dir=str(tmp_path),
        dry_run=False,
    )


def _hyperparams(i=0):
    return {
        "n_layer": 4 + (i % 9), "n_embd": 256 + (i % 6) * 64, "n_head": 4 + (i % 3) * 2,
        "window_s_fraction": 0.2 + (i % 5) * 0.15,
        "embedding_lr": 0.05 * (1 + i % 7), "unembedding_lr": 0.001 * (1 + i % 4),
        "matrix_lr": 0.005 * (1 + i % 6), "scalar_lr": 0.02 * (1 + i % 5),
        "weight_decay": 0.01 * (i % 8), "warmup_ratio": 0.02 * (i % 6),
        "batch_size": 2048 * (1 + i % 4),
    }


def _seed_regions(orch, specs):
    """specs: list of (hyperparam index, [val_bpbs]). Returns the regions."""
    out = []
    counter = 0
    for i, values in specs:
        region = orch.registry.open_region(_hyperparams(i), at_run=0)
        for v in values:
            orch.registry.assign_run(region.region_id, f"run_{counter:04d}", v)
            counter += 1
        out.append(region)
    orch.registry.save()
    return out


# === wave planning =========================================================


def test_each_gpu_gets_a_region(tmp_path):
    orch = _make_orchestrator(tmp_path)
    _seed_regions(orch, [(0, [1.20]), (5, [1.30]), (10, [1.40]), (15, [1.50])])
    _live, plan = orch._plan_wave(n_gpus=4, at_run=10)
    assert len(plan.assignments) == 4
    assert None not in plan.assignments


def test_the_best_region_keeps_a_gpu_when_capacity_is_scarce(tmp_path):
    orch = _make_orchestrator(tmp_path)
    regions = _seed_regions(orch, [(0, [1.50]), (5, [1.20]), (10, [1.40])])
    _live, plan = orch._plan_wave(n_gpus=1, at_run=10)
    assert plan.assignments == [regions[1].region_id]


def test_regions_without_capacity_are_paused_not_retired(tmp_path):
    """Confusing "we ran out of hardware" with "we ruled this out" would
    throw away a region's history and planner state permanently."""
    from state.regions import CAPACITY_PAUSED

    orch = _make_orchestrator(tmp_path)
    regions = _seed_regions(orch, [(0, [1.20]), (5, [1.30]), (10, [1.40])])
    orch._plan_wave(n_gpus=1, at_run=10)
    for parked in regions[1:]:
        flag = orch.registry.get(parked.region_id).flag
        assert flag == CAPACITY_PAUSED
        assert flag not in (NO_OPTIMUM, LOCAL_OPTIMUM), "recoverable, not retired"
    assert orch.registry.get(regions[0].region_id).flag == ACTIVE


@requires_deps
def test_a_spare_gpu_opens_a_new_region(tmp_path):
    orch = _make_orchestrator(tmp_path)
    _seed_regions(orch, [(0, [1.20]), (5, [1.30])])
    _write_results(tmp_path, n=40)
    before = len(orch.registry.regions)
    _live, plan = orch._plan_wave(n_gpus=4, at_run=10)
    assert len(orch.registry.regions) > before
    assert None not in plan.assignments


def test_a_slot_with_no_proposable_region_falls_back_to_the_best(tmp_path):
    """Too little history to fit a surrogate, so no new region can be
    proposed -- reinforcing the champion beats idling a GPU."""
    orch = _make_orchestrator(tmp_path)
    regions = _seed_regions(orch, [(0, [1.20])])
    _live, plan = orch._plan_wave(n_gpus=3, at_run=10)
    assert set(plan.assignments) == {regions[0].region_id}
    assert len(plan.assignments) == 3


def test_a_cold_campaign_opens_a_bootstrap_region(tmp_path):
    """Otherwise the campaign deadlocks: Agent 4 cannot propose a region
    until a surrogate fits (15 runs), and those runs cannot exist until
    something is dispatched. Agent 1's Sobol cold start IS exploration -- it
    just needs a region to belong to."""
    orch = _make_orchestrator(tmp_path)
    _live, plan = orch._plan_wave(n_gpus=4, at_run=0)
    assert plan.assignments and None not in plan.assignments
    regions = orch.registry.active()
    assert len(regions) == 1
    assert regions[0].origin == "bootstrap"
    assert set(plan.assignments) == {regions[0].region_id}


def test_the_bootstrap_region_is_opened_only_once(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch._plan_wave(n_gpus=4, at_run=0)
    orch._plan_wave(n_gpus=4, at_run=4)
    assert len([r for r in orch.registry.regions if r.origin == "bootstrap"]) == 1


def test_a_retired_region_loses_its_gpu_the_same_wave(tmp_path):
    """maintain() runs before allocation, so a region retired this wave does
    not still get a slot from last wave's state."""
    orch = _make_orchestrator(tmp_path)
    good, stuck = _seed_regions(orch, [
        (0, [1.40, 1.36, 1.32, 1.28, 1.24, 1.20]),
        (5, [1.30] + [1.3001] * 16),
    ])
    _live, plan = orch._plan_wave(n_gpus=2, at_run=20)
    assert orch.registry.get(stuck.region_id).flag == LOCAL_OPTIMUM
    assert stuck.region_id not in plan.assignments


def test_a_paused_region_is_resumed_when_there_is_nowhere_better(tmp_path):
    """The "if no better region exists, unflag and continue there" case."""
    orch = _make_orchestrator(tmp_path)
    regions = _seed_regions(orch, [(0, [1.20]), (5, [1.30])])
    regions[1].set_flag(PAUSED, 5)
    orch.registry.save()

    _live, plan = orch._plan_wave(n_gpus=2, at_run=10)
    assert orch.registry.get(regions[1].region_id).flag == ACTIVE
    assert regions[1].region_id in plan.assignments


# === slot independence =====================================================


def test_slots_in_one_wave_decide_from_their_own_regions_center(tmp_path):
    """The whole point of region scoping. Two slots in the same wave must not
    both propose from whichever center Agent 1 happened to hold last."""
    orch = _make_orchestrator(tmp_path)
    a, b = _seed_regions(orch, [(0, [1.20]), (5, [1.30])])
    a.center = dict(_hyperparams(0)); a.center["n_layer"] = 3
    b.center = dict(_hyperparams(5)); b.center["n_layer"] = 21
    orch.registry.save()

    seen = {}
    for region in (a, b):
        with orch.agent1.search_region(region):
            seen[region.region_id] = orch.agent1.current_hyperparams["n_layer"]
    assert seen[a.region_id] == 3 and seen[b.region_id] == 21


def test_a_decision_carries_its_region_id(tmp_path):
    """results.tsv's region_id column is fed from here; without it a run
    cannot be attributed back to the region that spent the budget."""
    orch = _make_orchestrator(tmp_path)
    region = _seed_regions(orch, [(0, [1.20])])[0]
    hp = orch._decide_next_hyperparams(
        iteration=1, latest_summary=None, recent_evidence=[], recent_results=[],
        latest_val_bpb=None, fresh_summary=False, region=region,
    )
    assert hp["region_id"] == region.region_id


def test_each_region_gets_its_own_planner_state_file(tmp_path):
    orch = _make_orchestrator(tmp_path)
    a, b = _seed_regions(orch, [(0, [1.20]), (5, [1.30])])
    with orch.agent1.search_region(a):
        path_a = orch.agent1._search_planner_state_path
    with orch.agent1.search_region(b):
        path_b = orch.agent1._search_planner_state_path
    assert path_a != path_b


# === result attribution ====================================================


def test_a_result_is_recorded_against_its_own_region(tmp_path):
    orch = _make_orchestrator(tmp_path)
    a, b = _seed_regions(orch, [(0, [1.20]), (5, [1.30])])
    hp = dict(_hyperparams(0)); hp["region_id"] = a.region_id

    orch._process_training_result(
        1, hp, {"val_bpb": 1.15, "training_time": 1.0, "status": "remote_ok"}, [])

    assert orch.registry.get(a.region_id).n_runs == 2
    assert orch.registry.get(b.region_id).n_runs == 1
    assert orch.registry.get(a.region_id).best_val_bpb == pytest.approx(1.15)


def test_a_campaign_record_set_inside_a_region_is_not_lost(tmp_path):
    """Under a region scope Agent 1 only sees the REGION's best (EI needs a
    local reference), so the campaign record has to be maintained by the
    orchestrator or global f_best goes stale for the rest of the run."""
    orch = _make_orchestrator(tmp_path)
    region = _seed_regions(orch, [(0, [1.20])])[0]
    orch.agent1.best_val_bpb = 1.30
    hp = dict(_hyperparams(0)); hp["region_id"] = region.region_id

    orch._process_training_result(
        1, hp, {"val_bpb": 1.10, "training_time": 1.0, "status": "remote_ok"}, [])
    assert orch.agent1.best_val_bpb == pytest.approx(1.10)


def test_a_regions_own_last_result_feeds_its_next_decision(tmp_path):
    """Feeding a region another region's newest run is what made stagnation
    detection compare two different places in the space and call the
    difference a trend."""
    orch = _make_orchestrator(tmp_path)
    a, b = _seed_regions(orch, [(0, [1.20]), (5, [1.30])])
    for region, val in ((a, 1.11), (b, 1.44)):
        hp = dict(region.center); hp["region_id"] = region.region_id
        orch._process_training_result(
            1, hp, {"val_bpb": val, "training_time": 1.0, "status": "remote_ok"}, [])
    assert orch._last_val_bpb_by_region[a.region_id] == pytest.approx(1.11)
    assert orch._last_val_bpb_by_region[b.region_id] == pytest.approx(1.44)


def test_a_run_without_a_region_is_still_logged(tmp_path):
    """The sequential single-GPU path has no region. It must not crash, and
    it must not be silently attributed to one."""
    orch = _make_orchestrator(tmp_path)
    region = _seed_regions(orch, [(0, [1.20])])[0]
    halt, _batch = orch._process_training_result(
        1, dict(_hyperparams(0)),
        {"val_bpb": 1.25, "training_time": 1.0, "status": "remote_ok"}, [])
    assert halt is False
    assert orch.registry.get(region.region_id).n_runs == 1


# === shared store ==========================================================


def test_agent3_and_agent4_read_the_same_registry(tmp_path):
    """The two halves have to agree on one store or the chart silently never
    shows a flag."""
    orch = _make_orchestrator(tmp_path)
    _seed_regions(orch, [(0, [1.20])])
    assert orch.agent4.registry_path == orch.agent3.registry_path
    assert RegionRegistry(str(orch.agent3.registry_path)).flags_snapshot()


def _write_results(tmp_path, n=40):
    from state.results_logger import log_result
    for i in range(n):
        hp = _hyperparams(i)
        log_result(f"run_{i:04d}", hp,
                   {"val_bpb": 1.2 + 0.03 * (i % 11), "training_time": 1.0,
                    "status": "remote_ok"},
                   results_path=str(tmp_path / "results.tsv"))


# === connection discipline =================================================


def test_a_wave_opens_exactly_one_ssh_connection(tmp_path, monkeypatch):
    """The server rate-limits SSH connects (~60-90s block after the first,
    measured). A wave used to open seven -- stale check, GPU discovery, code
    sync, and one per concurrent run -- so the first succeeded and the rest
    were dropped, losing most slots to remote_error every wave."""
    orch = _make_orchestrator(tmp_path)
    _seed_regions(orch, [(0, [1.20])])

    opened = []

    class FakeClient:
        def close(self):
            pass

    def fake_open_client(*a, **k):
        opened.append(1)
        return FakeClient()

    monkeypatch.setattr(remote_runner, "open_client", fake_open_client)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes",
                        lambda **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus",
                        lambda **k: FOUR_GPUS)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda **k: True)

    seen_clients = []

    def fake_train(hyperparams_local_path, gpu_index, hp_remote_name=None,
                   run_label=None, timeout=600, skip_sync=False, display=None, client=None):
        seen_clients.append(client)
        return {"val_bpb": 1.3, "training_time": 1.0, "status": "remote_ok", "device": gpu_index}

    monkeypatch.setattr(remote_runner, "run_training_remote", fake_train)
    orch._run_parallel_wave(0, [], 20)

    assert len(opened) == 1, f"one connection per wave, got {len(opened)}"
    assert seen_clients and all(c is not None for c in seen_clients), \
        "every training slot must reuse the wave's connection"
    assert len(set(id(c) for c in seen_clients)) == 1, "and it must be the same one"


def test_a_wave_that_loses_every_slot_eventually_halts(tmp_path, monkeypatch):
    """results.tsv gains a row per slot whether the run succeeded or not, so
    the iteration counter advances identically -- nothing else in the loop
    would notice a campaign producing only inf."""
    orch = _make_orchestrator(tmp_path)
    _seed_regions(orch, [(0, [1.20])])

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(remote_runner, "open_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: FOUR_GPUS)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda **k: True)
    monkeypatch.setattr(remote_runner, "run_training_remote",
                        lambda **k: {"val_bpb": float("inf"), "status": "remote_error",
                                     "device": k.get("gpu_index"), "training_time": 0.0})

    _it, _batch, halt = orch._run_parallel_wave(0, [], 40)
    assert halt is False, "one bad wave is a blip, not a reason to stop"
    _it, _batch, halt = orch._run_parallel_wave(4, [], 40)
    assert halt is True, "two in a row means every further iteration is wasted"


def test_a_successful_wave_clears_the_all_failed_streak(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    _seed_regions(orch, [(0, [1.20])])
    orch._all_slots_failed_streak = 1

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(remote_runner, "open_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: FOUR_GPUS)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda **k: True)
    monkeypatch.setattr(remote_runner, "run_training_remote",
                        lambda **k: {"val_bpb": 1.3, "status": "remote_ok",
                                     "device": k.get("gpu_index"), "training_time": 1.0})

    orch._run_parallel_wave(0, [], 40)
    assert orch._all_slots_failed_streak == 0


def test_the_sequential_path_still_runs_inside_a_region(tmp_path, monkeypatch):
    """The wave dispatcher returns None whenever fewer than 2 GPUs are free,
    so a busy server drops the campaign onto the sequential path. Without a
    region that silently reverts to single-search mode: blank region_id in
    results.tsv, no region history, and lifecycle thresholds that count "runs
    this region spent" quietly stop counting."""
    orch = _make_orchestrator(tmp_path)
    region = _seed_regions(orch, [(0, [1.20])])[0]
    got = orch._sequential_region(iteration=5)
    assert got is not None and got.region_id == region.region_id


def test_a_dry_run_is_not_attributed_to_any_region(tmp_path):
    """Nothing is trained, so recording the result against a region would put
    fabricated numbers into its history and into every threshold derived from
    it."""
    orch = Orchestrator(
        config_path=str(_config(tmp_path)), state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"), root_dir=str(tmp_path), dry_run=True,
    )
    _seed_regions(orch, [(0, [1.20])])
    assert orch._sequential_region(iteration=5) is None


def test_a_disabled_agent4_leaves_the_sequential_path_alone(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.agent4.enabled = False
    assert orch._sequential_region(iteration=5) is None


def test_reaching_the_server_clears_the_unreachable_streak(tmp_path, monkeypatch):
    """The streak counts an ONGOING outage, not a lifetime total. It used to
    clear only after a successful sync, so a wave that connected fine and then
    returned early (fewer than 2 GPUs free) left earlier failures on the
    counter -- three accumulated over hours would halt a campaign against a
    perfectly reachable server."""
    orch = _make_orchestrator(tmp_path)
    _seed_regions(orch, [(0, [1.20])])
    orch._remote_unreachable_streak = 2

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(remote_runner, "open_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda **k: [])
    # Only one GPU free -- the wave returns early, well before any sync.
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: FOUR_GPUS[:1])

    assert orch._run_parallel_wave(0, [], 20) is None
    assert orch._remote_unreachable_streak == 0


def test_a_persistently_broken_sync_still_halts(tmp_path, monkeypatch):
    """Connect and sync failures need separate counters. Sharing one meant
    each wave's successful connect cleared the accumulated sync failures, so
    a permanently broken sync could never reach the halt threshold and the
    campaign would retry forever."""
    from agents.orchestrator import REMOTE_FAILURE_HALT_STREAK

    orch = _make_orchestrator(tmp_path)
    _seed_regions(orch, [(0, [1.20])])

    class FakeClient:
        def close(self):
            pass

    monkeypatch.setattr(remote_runner, "open_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: FOUR_GPUS)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda **k: False)
    monkeypatch.setattr("agents.orchestrator.time.sleep", lambda s: None)

    halted = False
    for _ in range(REMOTE_FAILURE_HALT_STREAK):
        result = orch._run_parallel_wave(0, [], 20)
        if result is not None:
            _it, _batch, halted = result
    assert halted is True


# === resume ================================================================


def test_the_campaign_best_survives_a_restart(tmp_path):
    """best_val_bpb started at inf and was only advanced by results the
    current process saw, so a restart forgot every record ever set -- while
    the halt messages tell the operator to restart and say "the campaign
    resumes from results.tsv"."""
    _write_results(tmp_path, n=6)
    orch = _make_orchestrator(tmp_path)
    import csv
    best = min(float(r["val_bpb"])
               for r in csv.DictReader(open(tmp_path / "results.tsv"), delimiter="\t"))
    assert orch.agent1.best_val_bpb == pytest.approx(best)


def test_a_fresh_campaign_starts_with_no_record(tmp_path):
    orch = _make_orchestrator(tmp_path)
    assert orch.agent1.best_val_bpb == float("inf")


def test_simulated_runs_never_become_the_campaign_record(tmp_path):
    """dry_run/simulated val_bpb is a hand-tuned formula, not a measurement.
    Letting one become the record would set an unreachable bar for every real
    run that follows."""
    from state.results_logger import log_result

    log_result("run_0000", _hyperparams(0),
               {"val_bpb": 0.1, "training_time": 1.0, "status": "simulated"},
               results_path=str(tmp_path / "results.tsv"))
    log_result("run_0001", _hyperparams(1),
               {"val_bpb": 1.30, "training_time": 1.0, "status": "remote_ok"},
               results_path=str(tmp_path / "results.tsv"))

    orch = _make_orchestrator(tmp_path)
    assert orch.agent1.best_val_bpb == pytest.approx(1.30)


def test_a_capacity_pause_is_not_a_judgement(tmp_path):
    """Losing a GPU to a busier server says nothing about the region. Marking
    it the same as "this area stopped paying" is what made the allocator open
    brand-new regions after a busy wave while partially-explored ones idled."""
    from state.regions import CAPACITY_PAUSED

    orch = _make_orchestrator(tmp_path)
    regions = _seed_regions(orch, [(0, [1.20]), (5, [1.30]), (10, [1.40])])
    orch._plan_wave(n_gpus=1, at_run=10)
    assert orch.registry.get(regions[1].region_id).flag == CAPACITY_PAUSED
    assert orch.registry.get(regions[2].region_id).flag == CAPACITY_PAUSED


def test_capacity_paused_regions_come_back_before_new_ones_are_opened(tmp_path):
    """A region with runs invested and a measured score beats a speculative
    new one for a freed GPU."""
    _write_results(tmp_path, n=40)
    orch = _make_orchestrator(tmp_path)
    regions = _seed_regions(orch, [(0, [1.20]), (5, [1.30]), (10, [1.40])])

    orch._plan_wave(n_gpus=1, at_run=10)     # busy server: two get parked
    before = len(orch.registry.regions)
    _live, plan = orch._plan_wave(n_gpus=3, at_run=14)   # capacity returns

    assert len(orch.registry.regions) == before, "no new region needed"
    assert set(plan.assignments) == {r.region_id for r in regions}


def test_a_lifecycle_pause_is_still_only_revisited_as_a_last_resort(tmp_path):
    """The distinction has to cut both ways: a region judged to have stopped
    paying must not be resumed ahead of exploring somewhere new."""
    from state.regions import PAUSED

    orch = _make_orchestrator(tmp_path)
    stalled, live = _seed_regions(orch, [(0, [1.20]), (5, [1.30])])
    stalled.set_flag(PAUSED, 5)
    orch.registry.save()

    # One GPU, one live region -- no spare slot, so nothing is resumed.
    _live, plan = orch._plan_wave(n_gpus=1, at_run=10)
    assert plan.assignments == [live.region_id]
    assert orch.registry.get(stalled.region_id).flag == PAUSED
