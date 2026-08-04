"""Orchestrator-side tests for Agent 4's exploration window: who decides an
iteration's hyperparameters, how the window opens and closes, and how it
constrains multi-GPU wave size.

Mirrors tests/test_orchestrator_parallel_wave.py -- remote_runner is
monkeypatched throughout, so no real SSH, no real GPU, no real training.
"""

import json

import pytest

from agents import remote_runner
from agents.agent4_landscape_explorer import COMMIT, EXHAUSTED, NEXT_REGION
from agents.orchestrator import Orchestrator
from state.landscape import LANDSCAPE_DEPS_AVAILABLE, load_region_flags

requires_deps = pytest.mark.skipif(
    not LANDSCAPE_DEPS_AVAILABLE, reason="scikit-learn not installed"
)

FOUR_GPUS = [
    {"index": i, "mem_used_mb": 100, "mem_total_mb": 20100, "util_pct": 1, "free_mb": 20000}
    for i in (0, 1, 2, 3)
]


def _config(tmp_path, agent4_enabled=True, check_interval=30, probe_wave_size=3,
            window_iterations=9):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(f"""
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
  enabled: {str(agent4_enabled).lower()}
  check_interval: {check_interval}
  window_iterations: {window_iterations}
  probe_wave_size: {probe_wave_size}
  min_runs_before_commit: 6
  llm_mode: statistics

llm:
  backend: none

orchestrator:
  parallel: true
  max_parallel_runs: 4
""".strip(), encoding="utf-8")
    return config_path


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


def _flat_rows(n=40):
    """A stuck campaign: the best-so-far stops improving after the first
    few runs, which is the condition Agent 4 exists to detect."""
    rows = []
    for i in range(n):
        row = dict(_hyperparams(i))
        row["val_bpb"] = 1.2 + 0.03 * (i % 11)
        row["status"] = "remote_ok"
        rows.append(row)
    return rows


def _open_window(orch, iteration=30):
    return orch.agent4.consider_intervention(
        iteration, _hyperparams(0), best_val_bpb=1.2, rows=_flat_rows()
    )


# --- construction / cadence ------------------------------------------------

def test_agent4_is_inert_when_disabled(tmp_path):
    orch = _make_orchestrator(tmp_path, agent4_enabled=False)
    assert orch.agent4.enabled is False
    assert orch._maybe_open_agent4_window(30) is False
    assert orch.agent4.active is False


def test_window_is_only_considered_on_check_interval_iterations(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, check_interval=30)
    considered = []
    monkeypatch.setattr(orch.agent4, "consider_intervention",
                        lambda *a, **k: considered.append(a[0]) or False)
    for iteration in range(0, 91):
        orch._maybe_open_agent4_window(iteration)
    # Not iteration 0 -- there is no history to judge at campaign start.
    assert considered == [30, 60, 90]


@pytest.mark.parametrize("wave_size", [2, 3, 4])
def test_checks_still_happen_when_the_counter_skips_by_wave_size(tmp_path, monkeypatch, wave_size):
    """Under multi-GPU parallelism the loop counter advances by the wave
    size, not by 1 -- at max_parallel_runs=4 it goes 28 -> 32 and never sees
    30. A modulo-based cadence silently skips those checks (and, since wave
    size varies with GPU availability, does so unpredictably), so a whole
    150-iteration campaign could pass without Agent 4 ever engaging.
    """
    orch = _make_orchestrator(tmp_path, check_interval=30)
    considered = []
    monkeypatch.setattr(orch.agent4, "consider_intervention",
                        lambda *a, **k: considered.append(a[0]) or False)

    iteration = 0
    while iteration < 150:
        orch._maybe_open_agent4_window(iteration)
        iteration += wave_size

    # One check per ~30 iterations regardless of stride, and never at 0.
    # Four over 150 iterations (a fifth would fall past the end).
    assert len(considered) == 4, considered
    assert considered[0] >= 30
    gaps = [b - a for a, b in zip(considered, considered[1:])]
    assert all(30 <= g < 30 + wave_size for g in gaps), gaps


def test_check_clock_is_not_reset_by_a_window_running(tmp_path, monkeypatch):
    """The next check is due check_interval after the last *check*, not
    after the window that check opened -- otherwise a 9-iteration window
    would silently push every later check out of phase."""
    orch = _make_orchestrator(tmp_path, check_interval=30)
    considered = []

    def fake_consider(*a, **k):
        considered.append(a[0])
        return len(considered) == 1  # engage on the first check only

    monkeypatch.setattr(orch.agent4, "consider_intervention", fake_consider)

    orch._maybe_open_agent4_window(30)      # engages
    orch.agent4.active = True
    for it in range(31, 40):
        orch._maybe_open_agent4_window(it)  # window open: no re-check
    orch.agent4.active = False
    for it in range(40, 61):
        orch._maybe_open_agent4_window(it)

    assert considered == [30, 60]


def test_window_is_not_reconsidered_while_one_is_open(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, check_interval=1)
    calls = []
    monkeypatch.setattr(orch.agent4, "consider_intervention",
                        lambda *a, **k: calls.append(a[0]) or False)
    orch.agent4.active = True
    orch._maybe_open_agent4_window(5)
    assert calls == []


# --- decision routing ------------------------------------------------------

def test_decisions_go_to_agent1_outside_a_window(tmp_path):
    orch = _make_orchestrator(tmp_path)
    hp = orch._decide_next_hyperparams(
        iteration=1, latest_summary=None, recent_evidence=[], recent_results=[],
        latest_val_bpb=None, fresh_summary=False,
    )
    assert hp is not None
    assert orch._active_decision_log is orch.agent1.last_decision_log
    assert orch._active_decisions_dir == orch.agent1.decisions_dir


@requires_deps
def test_decisions_go_to_agent4_inside_a_window(tmp_path):
    orch = _make_orchestrator(tmp_path)
    assert _open_window(orch) is True

    hp = orch._decide_next_hyperparams(
        iteration=31, latest_summary=None, recent_evidence=[], recent_results=[],
        latest_val_bpb=None, fresh_summary=False,
    )
    assert hp is not None and "n_layer" in hp
    assert orch._active_decision_log is orch.agent4.last_decision_log
    assert orch._active_decisions_dir == orch.agent4.decisions_dir
    assert orch._active_decision_log["agent"] == "agent4"


@requires_deps
def test_agent1_is_not_consulted_during_a_window(tmp_path, monkeypatch):
    """Agent 4 owning the iteration has to mean Agent 1 doesn't also run --
    otherwise its own state (best-so-far tracking, decision log) advances on
    iterations it didn't decide."""
    orch = _make_orchestrator(tmp_path)
    _open_window(orch)
    monkeypatch.setattr(orch.agent1, "decide_next_hyperparams",
                        lambda *a, **k: pytest.fail("Agent 1 must not decide during a window"))
    orch._decide_next_hyperparams(
        iteration=31, latest_summary=None, recent_evidence=[], recent_results=[],
        latest_val_bpb=None, fresh_summary=False,
    )


@requires_deps
def test_every_agent4_iteration_produces_its_own_decision_log(tmp_path):
    """The validator that runs right after each decision must see this
    iteration's log, never a stale one from when the window opened."""
    orch = _make_orchestrator(tmp_path)
    _open_window(orch)
    for iteration in (31, 32, 33):
        orch._decide_next_hyperparams(
            iteration=iteration, latest_summary=None, recent_evidence=[], recent_results=[],
            latest_val_bpb=None, fresh_summary=False,
        )
        log = json.loads((orch.agent4.decisions_dir / f"decision_{iteration:04d}.json")
                         .read_text(encoding="utf-8"))
        assert log["iteration"] == iteration


# --- wave sizing -----------------------------------------------------------

@requires_deps
def test_wave_size_is_capped_to_the_probe_batch_during_a_window(tmp_path, monkeypatch):
    """4 GPUs are free, but a wave wider than the probe batch would blur the
    abandon test -- so the wave shrinks to 3."""
    orch = _make_orchestrator(tmp_path, probe_wave_size=3)
    _open_window(orch)

    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda: FOUR_GPUS)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda *a, **k: None)

    dispatched = []

    def fake_run(**kwargs):
        dispatched.append(kwargs["gpu_index"])
        return {"val_bpb": 1.30, "training_time": 1.0, "status": "remote_ok",
                "device": kwargs["gpu_index"]}

    monkeypatch.setattr(remote_runner, "run_training_remote", fake_run)
    result = orch._run_parallel_wave(31, [], 100)

    assert result is not None
    assert len(dispatched) == 3


def test_wave_size_is_unconstrained_without_a_window(tmp_path, monkeypatch):
    """Regression guard: outside a window, wave sizing is exactly what it
    was before Agent 4 existed."""
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda: FOUR_GPUS)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda *a, **k: None)

    dispatched = []

    def fake_run(**kwargs):
        dispatched.append(kwargs["gpu_index"])
        return {"val_bpb": 1.30, "training_time": 1.0, "status": "remote_ok",
                "device": kwargs["gpu_index"]}

    monkeypatch.setattr(remote_runner, "run_training_remote", fake_run)
    orch._run_parallel_wave(0, [], 100)

    assert len(dispatched) == 4


# --- verdict handling ------------------------------------------------------

@requires_deps
def test_commit_relocates_agent1s_search_center_and_closes_the_window(tmp_path):
    """The one verdict that changes anything outside Agent 4."""
    orch = _make_orchestrator(tmp_path)
    _open_window(orch)
    committed = dict(orch.agent4._candidate_hyperparams)
    orch.agent4._origin_runs = [1.5] * 8
    best_before = orch.agent1.best_val_bpb
    for slot in range(6):
        orch.agent4.propose_probe(31, slot)
        orch.agent4.record_result(1.0)

    orch._agent4_evaluate(31)

    assert orch.agent4.last_action == "committed"
    assert orch.agent4.active is False
    for key, value in committed.items():
        assert orch.agent1.current_hyperparams[key] == value
    # best_val_bpb is the record of what was actually achieved -- relocating
    # the search must not rewrite it.
    assert orch.agent1.best_val_bpb == best_before


@requires_deps
def test_abandon_returns_control_to_agent1_on_the_next_iteration(tmp_path):
    orch = _make_orchestrator(tmp_path, window_iterations=3)
    _open_window(orch)
    center_before = dict(orch.agent1.current_hyperparams)
    for slot in range(3):
        orch.agent4.propose_probe(31, slot)
        orch.agent4.record_result(1.32)
    orch._agent4_evaluate(31)

    assert orch.agent4.active is False
    assert orch.agent1.current_hyperparams == center_before  # search center untouched

    orch._decide_next_hyperparams(
        iteration=34, latest_summary=None, recent_evidence=[], recent_results=[],
        latest_val_bpb=None, fresh_summary=False,
    )
    assert orch._active_decisions_dir == orch.agent1.decisions_dir


@requires_deps
def test_probe_results_are_fed_back_to_agent4(tmp_path):
    """_process_training_result is the only place a measured val_bpb exists
    -- if it doesn't reach Agent 4, no verdict can ever fire."""
    orch = _make_orchestrator(tmp_path)
    _open_window(orch)
    hp = orch.agent4.propose_probe(31)
    orch._process_training_result(
        31, hp, {"val_bpb": 1.27, "training_time": 1.0, "status": "remote_ok"}, [],
    )
    assert orch.agent4._candidate_runs == [1.27]


def test_results_are_not_fed_back_outside_a_window(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch._process_training_result(
        1, _hyperparams(0), {"val_bpb": 1.27, "training_time": 1.0, "status": "remote_ok"}, [],
    )
    assert orch.agent4._candidate_runs == []


@requires_deps
def test_region_flags_are_written_where_agent3_reads_them(tmp_path):
    """The two halves of this feature have to agree on one path, or the
    chart silently never shows a flag."""
    orch = _make_orchestrator(tmp_path)
    _open_window(orch)
    assert orch.agent4.region_flags_path == orch.agent3.region_flags_path
    assert load_region_flags(orch.agent3.region_flags_path)


def test_a_new_best_found_during_a_window_is_not_lost(tmp_path):
    """Agent 1 records a new best inside decide_next_hyperparams, which it
    never reaches during a window. Without the orchestrator forwarding it,
    a record set by a probe would vanish and EI would keep aiming at a stale
    f_best for the rest of the campaign."""
    orch = _make_orchestrator(tmp_path)
    _open_window(orch)
    orch.agent1.best_val_bpb = 1.2065
    hp = orch.agent4.propose_probe(31)

    orch._process_training_result(
        31, hp, {"val_bpb": 1.1000, "training_time": 1.0, "status": "remote_ok"}, [],
    )
    assert orch.agent1.best_val_bpb == 1.1000


def test_a_worse_probe_does_not_move_the_best(tmp_path):
    orch = _make_orchestrator(tmp_path)
    _open_window(orch)
    orch.agent1.best_val_bpb = 1.2065
    hp = orch.agent4.propose_probe(31)
    orch._process_training_result(
        31, hp, {"val_bpb": 1.5, "training_time": 1.0, "status": "remote_ok"}, [],
    )
    assert orch.agent1.best_val_bpb == 1.2065
