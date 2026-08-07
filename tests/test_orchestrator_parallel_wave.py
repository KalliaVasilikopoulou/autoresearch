"""Synthetic-data tests for agents/orchestrator.py's multi-GPU parallel
wave dispatch (dev/checks.txt item 1). No real SSH is ever attempted --
agents.remote_runner's discovery/sync/training functions are monkeypatched.
Verifies: (1) _run_parallel_wave degrades to None (sequential fallback) in
every case where parallel dispatch shouldn't apply, (2) a real 2-GPU wave
dispatches concurrently, logs both results with distinct device values, and
advances iteration by the wave size, (3) a mid-wave stop signal from
decide_next_hyperparams halts cleanly without training remaining slots,
(4) _kill_stale_remote_training's stale-process cleanup runs both at
campaign start and before every wave's GPU discovery.
"""

from unittest.mock import patch

from agents import remote_runner
from agents.orchestrator import Orchestrator, _format_duration
from state.results_analysis import load_results



class _StubClient:
    """Stands in for the single SSH connection a wave now shares across its
    stale check, GPU discovery, code sync and every training slot."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _stub_wave_connection(monkeypatch):
    """A wave opens exactly one connection (remote_runner.open_client) because
    the server rate-limits connects -- see remote_runner's note by
    CONNECT_BACKOFF_MULTIPLIER. Every test that drives a wave has to stub it
    or it would try to reach the real host."""
    opened = []

    def fake_open_client(*a, **k):
        client = _StubClient()
        opened.append(client)
        return client

    monkeypatch.setattr(remote_runner, "open_client", fake_open_client)
    return opened


def _base_config(tmp_path, parallel=True, max_parallel_runs=4, use_surrogate=False):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(f"""
agent1:
  use_llm: false
  accuracy_threshold: 0.01
  cost_limit_usd: 50.0
  training_budget_seconds: 60
  use_surrogate: {str(use_surrogate).lower()}

agent2:
  xai_method: fast
  use_llm: false
  ablation_k: 3

agent3:
  batch_size: 100
  use_llm: false

orchestrator:
  parallel: {str(parallel).lower()}
  max_parallel_runs: {max_parallel_runs}
""".strip())
    return config_path


def _make_orchestrator(tmp_path, **config_kwargs):
    config_path = _base_config(tmp_path, **config_kwargs)
    return Orchestrator(
        config_path=str(config_path),
        state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"),
        root_dir=str(tmp_path),
        dry_run=False,
    )


TWO_GPUS = [
    {"index": 3, "mem_used_mb": 100, "mem_total_mb": 20100, "util_pct": 1, "free_mb": 20000},
    {"index": 5, "mem_used_mb": 100, "mem_total_mb": 19100, "util_pct": 1, "free_mb": 19000},
]


# --- _run_parallel_wave degrades to None (sequential fallback applies) --

def test_returns_none_when_dry_run(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    orch.dry_run = True
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: (_ for _ in ()).throw(
        AssertionError("discovery must not run when dry_run is True")))
    assert orch._run_parallel_wave(0, [], 10) is None


def test_returns_none_when_parallel_disabled(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path, parallel=False)
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: (_ for _ in ()).throw(
        AssertionError("discovery must not run when parallel is disabled")))
    assert orch._run_parallel_wave(0, [], 10) is None


def test_returns_none_when_remote_not_configured(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: False)
    assert orch._run_parallel_wave(0, [], 10) is None


def test_returns_none_when_zero_gpus_discovered(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    _stub_wave_connection(monkeypatch)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: [])
    assert orch._run_parallel_wave(0, [], 10) is None


def test_returns_none_when_only_one_gpu_discovered(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    _stub_wave_connection(monkeypatch)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: TWO_GPUS[:1])
    assert orch._run_parallel_wave(0, [], 10) is None


def test_run_parallel_wave_cleans_up_stale_processes_before_discovery(tmp_path, monkeypatch):
    """A wave-dispatched run that exceeds its SSH timeout can leave the
    remote train.py process alive even though we logged it locally as
    remote_error -- each new wave must try to reclaim that GPU before
    deciding what's available, not just once at campaign start."""
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    _stub_wave_connection(monkeypatch)

    call_order = []
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes",
                         lambda *a, **k: call_order.append("kill") or [])

    def tracking_discover(*a, **k):
        call_order.append("discover")
        return []

    monkeypatch.setattr(remote_runner, "discover_available_gpus", tracking_discover)

    orch._run_parallel_wave(0, [], 10)

    assert call_order == ["kill", "discover"]


# --- real 2-GPU wave dispatch ---------------------------------------

def test_format_duration_hand_computed():
    assert _format_duration(12.34) == "12.3s"
    assert _format_duration(59.99) == "60.0s"
    assert _format_duration(65) == "1m 05s"
    assert _format_duration(3599) == "59m 59s"
    assert _format_duration(3661) == "1h 01m 01s"
    assert _format_duration(-5) == "0.0s"  # never a negative duration


def test_two_gpu_wave_dispatches_concurrently_and_logs_distinct_devices(tmp_path, monkeypatch, capsys):
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    _stub_wave_connection(monkeypatch)
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: list(TWO_GPUS))
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda *a, **k: True)
    # orch.run() below now also does a startup stale-process check -- must
    # never make a real SSH call in tests (this repo's real .env has real
    # credentials, so an unmocked call here would actually reach the DGX).
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [])

    seen_remote_names = []

    def fake_run_training_remote(hyperparams_local_path, gpu_index, hp_remote_name=None, run_label=None, timeout=600, skip_sync=False, display=None, client=None):
        seen_remote_names.append(hp_remote_name)
        return {
            "val_bpb": 1.0 if gpu_index == 3 else 1.1,
            "training_time": 1.0,
            "status": "remote_ok",
            "device": gpu_index,
        }

    monkeypatch.setattr(remote_runner, "run_training_remote", fake_run_training_remote)

    orch.run(max_iterations=2)

    rows = load_results(str(tmp_path / "results.tsv"))
    assert len(rows) == 2
    devices = sorted(r["device"] for r in rows)
    assert devices == ["3", "5"]
    run_ids = {r["run_id"] for r in rows}
    assert run_ids == {"run_0000", "run_0001"}

    # Regression guard: every concurrent slot must upload to its own remote
    # filename -- reusing the shared default races two SFTP uploads against
    # each other (this exact bug shipped once: paramiko's post-upload size
    # check failed with "size mismatch in put!" when two GPU slots both
    # uploaded to the plain model_hyperparams.yaml at the same time).
    assert len(seen_remote_names) == 2
    assert all(name for name in seen_remote_names)
    assert len(set(seen_remote_names)) == 2

    out = capsys.readouterr().out
    assert "[Orchestrator] Wave complete: 2 run(s) in" in out
    assert "Total run time:" in out


def test_wave_stop_signal_halts_without_training_remaining_slots(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    _stub_wave_connection(monkeypatch)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: list(TWO_GPUS))
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda *a, **k: True)

    dispatched_gpus = []

    def fake_run_training_remote(hyperparams_local_path, gpu_index, hp_remote_name=None, run_label=None, timeout=600, skip_sync=False, display=None, client=None):
        dispatched_gpus.append(gpu_index)
        return {"val_bpb": 1.0, "training_time": 1.0, "status": "remote_ok", "device": gpu_index}

    monkeypatch.setattr(remote_runner, "run_training_remote", fake_run_training_remote)

    real_decide = orch.agent1.decide_next_hyperparams
    call_count = {"n": 0}

    def decide_then_stop(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            return None  # simulate Agent 1's stop signal on the 2nd slot
        return real_decide(*args, **kwargs)

    with patch.object(orch.agent1, "decide_next_hyperparams", side_effect=decide_then_stop):
        result = orch._run_parallel_wave(0, [], max_iterations=10)

    assert result is not None
    _, _, halted = result
    assert halted is True
    assert dispatched_gpus == [3]  # only the first (already-decided) slot was trained


# --- _kill_stale_remote_training (startup cleanup) -----------------------

def test_kill_stale_remote_training_skipped_when_dry_run(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    orch.dry_run = True

    def fail_if_called(*a, **k):
        raise AssertionError("kill_stale_training_processes must not run in dry_run mode")

    monkeypatch.setattr(remote_runner, "is_remote_configured", fail_if_called)
    orch._kill_stale_remote_training()  # must return before ever checking remote config


def test_kill_stale_remote_training_skipped_when_remote_not_configured(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: False)

    def fail_if_called(*a, **k):
        raise AssertionError("kill_stale_training_processes must not run when remote isn't configured")

    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", fail_if_called)
    orch._kill_stale_remote_training()


def test_kill_stale_remote_training_calls_cleanup_when_remote_configured(tmp_path, monkeypatch, capsys):
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    _stub_wave_connection(monkeypatch)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [
        {"pid": "111", "cmd": "python -u train.py", "escalated_to_sigkill": False},
    ])

    orch._kill_stale_remote_training()

    out = capsys.readouterr().out
    assert "Killed stale process PID 111" in out
    assert "stopped cleanly with SIGTERM" in out


def test_kill_stale_remote_training_reports_none_found(tmp_path, monkeypatch, capsys):
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    _stub_wave_connection(monkeypatch)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [])

    orch._kill_stale_remote_training()

    assert "None found." in capsys.readouterr().out


# --- transient/persistent remote failure (real production crash) --------
# A campaign died on one `TimeoutError: [WinError 10060]` raised by
# sync_remote_code inside _run_parallel_wave. Discovery and stale-process
# cleanup -- two other SSH calls to the same host -- had both just succeeded,
# so this was a network blip taking down hours of work.

def _stub_remote(monkeypatch, gpus=None):
    monkeypatch.setattr(remote_runner, "is_remote_configured", lambda: True)
    _stub_wave_connection(monkeypatch)
    monkeypatch.setattr(remote_runner, "kill_stale_training_processes", lambda *a, **k: [])
    monkeypatch.setattr(remote_runner, "discover_available_gpus", lambda **k: gpus or TWO_GPUS)


def test_wave_skips_instead_of_crashing_when_sync_fails(tmp_path, monkeypatch):
    orch = _make_orchestrator(tmp_path)
    _stub_remote(monkeypatch)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda *a, **k: False)
    monkeypatch.setattr(remote_runner, "run_training_remote", lambda **k: (_ for _ in ()).throw(
        AssertionError("must not dispatch training to an unreachable server")))
    monkeypatch.setattr("agents.orchestrator.time.sleep", lambda s: None)

    result = orch._run_parallel_wave(0, [], 10)

    assert result is None          # skipped, campaign continues
    # A sync failure counts on its own streak: the server WAS reachable, so
    # calling it "unreachable" would clear on the next successful connect and
    # a permanently broken sync could never reach the halt threshold.
    assert orch._sync_failure_streak == 1
    assert orch._remote_unreachable_streak == 0


def test_a_successful_sync_resets_the_failure_streak(tmp_path, monkeypatch):
    """The counters track an ongoing outage, not a lifetime total -- an
    occasional blip must never accumulate into a spurious halt."""
    orch = _make_orchestrator(tmp_path)
    orch._remote_unreachable_streak = 2
    orch._sync_failure_streak = 2
    _stub_remote(monkeypatch)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda *a, **k: True)
    monkeypatch.setattr(remote_runner, "run_training_remote", lambda **k: {
        "val_bpb": 1.3, "training_time": 1.0, "status": "remote_ok", "device": k["gpu_index"]})

    orch._run_parallel_wave(0, [], 10)

    assert orch._remote_unreachable_streak == 0
    assert orch._sync_failure_streak == 0


def test_campaign_halts_rather_than_fabricating_results_during_an_outage(tmp_path, monkeypatch):
    """Every path past a dead server fabricates rather than measures --
    Agent1.train_model's last resort is _simulate_training_result, whose
    val_bpb is a hand-tuned formula. Stopping with the iteration budget
    intact beats filling results.tsv with invented numbers."""
    from agents.orchestrator import REMOTE_FAILURE_HALT_STREAK

    orch = _make_orchestrator(tmp_path)
    _stub_remote(monkeypatch)
    monkeypatch.setattr(remote_runner, "sync_remote_code", lambda *a, **k: False)
    monkeypatch.setattr("agents.orchestrator.time.sleep", lambda s: None)

    halted = False
    for _ in range(REMOTE_FAILURE_HALT_STREAK):
        result = orch._run_parallel_wave(0, [], 10)
        if result is not None:
            _iteration, _batch, halted = result

    assert halted is True
    assert load_results(str(tmp_path / "results.tsv")) == []  # nothing invented
