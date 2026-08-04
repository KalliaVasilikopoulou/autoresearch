"""Agent 4 (optimization landscape explorer) unit tests.

Synthetic data throughout -- no GPU, no SSH, no real LLM call. The decisions
under test are deterministic by design (llm_mode "statistics"/"hybrid" never
let the LLM change a verdict), which is exactly why they can be asserted on
directly rather than only smoke-tested.
"""

import json
import math

import pytest

from agents.agent4_landscape_explorer import (
    COMMIT,
    CONTINUE,
    EXHAUSTED,
    NEXT_REGION,
    Agent4LandscapeExplorer,
)
from state.landscape import LANDSCAPE_DEPS_AVAILABLE, load_region_flags

requires_deps = pytest.mark.skipif(
    not LANDSCAPE_DEPS_AVAILABLE, reason="scikit-learn not installed"
)

CONFIG = """
agent4:
  enabled: true
  check_interval: 30
  window_iterations: 9
  probe_wave_size: 3
  bad_tolerance: 0.05
  commit_margin: 0.03
  min_runs_before_commit: 6
  heavy_exploitation_n: 20
  region_radius: 0.15
  stagnation_lookback: 10
  llm_mode: statistics
llm:
  backend: none
"""


def _hyperparams(i=0):
    return {
        "n_layer": 4 + (i % 9), "n_embd": 256 + (i % 6) * 64, "n_head": 4 + (i % 3) * 2,
        "window_s_fraction": 0.2 + (i % 5) * 0.15,
        "embedding_lr": 0.05 * (1 + i % 7), "unembedding_lr": 0.001 * (1 + i % 4),
        "matrix_lr": 0.005 * (1 + i % 6), "scalar_lr": 0.02 * (1 + i % 5),
        "weight_decay": 0.01 * (i % 8), "warmup_ratio": 0.02 * (i % 6),
        "batch_size": 2048 * (1 + i % 4),
    }


def _rows(n=40, improving=False):
    """`improving=True` makes the running best keep dropping (a healthy
    search); otherwise the frontier is flat after the first few runs (the
    stuck case Agent 4 exists for)."""
    rows = []
    for i in range(n):
        row = dict(_hyperparams(i))
        row["val_bpb"] = (1.5 - 0.01 * i) if improving else (1.2 + 0.03 * (i % 11))
        row["status"] = "remote_ok"
        rows.append(row)
    return rows


def _make_agent4(tmp_path, config=CONFIG):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(config.strip(), encoding="utf-8")
    return Agent4LandscapeExplorer(
        config_path=str(config_path), root_dir=str(tmp_path),
        state_dir=str(tmp_path / "state"), reports_dir=str(tmp_path / "reports"),
    )


def _engage(tmp_path, agent4=None, rows=None):
    agent4 = agent4 or _make_agent4(tmp_path)
    rows = rows if rows is not None else _rows()
    engaged = agent4.consider_intervention(30, _hyperparams(0), best_val_bpb=1.2, rows=rows)
    return agent4, engaged


# --- consider_intervention -------------------------------------------------

def test_disabled_agent_never_engages(tmp_path):
    agent4 = _make_agent4(tmp_path, CONFIG.replace("enabled: true", "enabled: false"))
    assert agent4.consider_intervention(30, _hyperparams(0), 1.2, _rows()) is False


def test_no_intervention_when_too_little_history(tmp_path):
    agent4 = _make_agent4(tmp_path)
    assert agent4.consider_intervention(30, _hyperparams(0), 1.2, _rows(n=5)) is False


@requires_deps
def test_no_intervention_while_the_frontier_is_still_improving(tmp_path):
    """The headline behavior: if the search is working, Agent 4 says so and
    spends nothing."""
    agent4 = _make_agent4(tmp_path)
    assert agent4.consider_intervention(30, _hyperparams(0), 1.2, _rows(improving=True)) is False
    assert agent4.active is False
    assert agent4.budget_left == agent4.window_iterations  # not one iteration consumed


@requires_deps
def test_engages_when_the_frontier_is_flat(tmp_path):
    agent4, engaged = _engage(tmp_path)
    assert engaged is True
    assert agent4.active is True
    assert agent4.last_action == "engaged"
    assert agent4._candidate_hyperparams is not None


@requires_deps
def test_does_not_re_engage_while_a_window_is_open(tmp_path):
    agent4, _ = _engage(tmp_path)
    assert agent4.consider_intervention(60, _hyperparams(0), 1.2, _rows()) is False


@requires_deps
def test_engaging_writes_verdict_log_and_region_flags(tmp_path):
    agent4, _ = _engage(tmp_path)
    log = json.loads((agent4.decisions_dir / "verdict_0030.json").read_text(encoding="utf-8"))
    assert log["agent"] == "agent4" and log["action"] == "engaged"
    assert "params" in log and "n_layer" in log["params"]
    assert (agent4.decisions_dir / "verdict_0030.md").exists()

    flags = {r["flag"] for r in load_region_flags(agent4.region_flags_path)}
    assert flags == {"currently_exploiting", "investigating"}


@requires_deps
def test_a_probe_does_not_overwrite_the_judgement_at_the_same_iteration(tmp_path):
    """Opening a window and taking its first probe are both iteration N. If
    they shared a filename the engage record -- the stagnation evidence and
    the region chosen, i.e. everything a later window needs -- would be
    destroyed the instant it was written."""
    agent4, _ = _engage(tmp_path)
    agent4.propose_probe(30, slot=0)

    verdict = json.loads((agent4.decisions_dir / "verdict_0030.json").read_text(encoding="utf-8"))
    probe = json.loads((agent4.decisions_dir / "decision_0030.json").read_text(encoding="utf-8"))
    assert verdict["action"] == "engaged"
    assert probe["action"] == "probe"
    assert [h["action"] for h in agent4._load_history()] == ["engaged"]


# --- probing ---------------------------------------------------------------

@requires_deps
def test_propose_probe_returns_a_full_hyperparams_dict(tmp_path):
    agent4, _ = _engage(tmp_path)
    probe = agent4.propose_probe(31, slot=0)
    for key in _hyperparams(0):
        assert key in probe
        assert math.isfinite(float(probe[key]))


@requires_deps
def test_probes_in_one_wave_are_distinct_draws(tmp_path):
    """Each slot samples the region independently, so "all N came back bad"
    is evidence about the region and not about one point."""
    agent4, _ = _engage(tmp_path)
    probes = [agent4.propose_probe(31, slot=s) for s in range(3)]
    assert len({tuple(sorted(p.items())) for p in probes}) > 1


@requires_deps
def test_propose_probe_consumes_window_budget(tmp_path):
    agent4, _ = _engage(tmp_path)
    before = agent4.budget_left
    agent4.propose_probe(31)
    assert agent4.budget_left == before - 1


@requires_deps
def test_record_result_ignores_non_finite_val_bpb(tmp_path):
    """A crashed run is an absence of evidence about the region, not
    evidence against it."""
    agent4, _ = _engage(tmp_path)
    agent4.record_result(float("nan"))
    agent4.record_result(None)
    assert agent4._candidate_runs == []


# --- evaluate_batch verdicts -----------------------------------------------

@requires_deps
def test_continue_until_the_wave_is_full(tmp_path):
    agent4, _ = _engage(tmp_path)
    agent4.propose_probe(31); agent4.record_result(1.25)
    assert agent4.evaluate_batch(31) == CONTINUE


@requires_deps
def test_all_bad_wave_flags_no_optimum_and_moves_to_another_region(tmp_path):
    agent4, _ = _engage(tmp_path)
    first_region = dict(agent4._candidate_hyperparams)
    for slot in range(3):
        agent4.propose_probe(31, slot)
        agent4.record_result(99.0)  # unambiguously worse than any elite reference
    assert agent4.evaluate_batch(31) == NEXT_REGION

    flags = {r["flag"] for r in load_region_flags(agent4.region_flags_path)}
    assert "no_optimum" in flags
    assert agent4._candidate_hyperparams != first_region
    assert agent4._candidate_runs == []  # the new region starts with a clean sample
    assert agent4.active is True


@requires_deps
def test_a_single_good_probe_prevents_abandoning_the_region(tmp_path):
    agent4, _ = _engage(tmp_path)
    for slot, val in enumerate([99.0, 99.0, 0.5]):
        agent4.propose_probe(31, slot)
        agent4.record_result(val)
    assert agent4.evaluate_batch(31) == CONTINUE


@requires_deps
def test_commit_when_the_region_is_decisively_better(tmp_path):
    agent4, _ = _engage(tmp_path)
    agent4._origin_runs = [1.40, 1.42, 1.45, 1.50, 1.55, 1.60, 1.62, 1.70]
    candidate = dict(agent4._candidate_hyperparams)
    for slot in range(6):
        agent4.propose_probe(31, slot)
        agent4.record_result(1.10)  # whole distribution clearly better
    assert agent4.evaluate_batch(31) == COMMIT

    assert agent4.committed_hyperparams == candidate
    assert agent4.last_action == "committed"
    assert agent4.active is False  # window closes on commit


@requires_deps
def test_commit_does_not_fire_on_one_lucky_run(tmp_path):
    """The distributional guard: top_quartile_by_val_bpb collapses to a
    single entry at these sample sizes, so without it one outlier would be
    enough to permanently relocate the whole campaign."""
    agent4, _ = _engage(tmp_path)
    agent4._origin_runs = [1.40, 1.42, 1.45, 1.50, 1.55, 1.60, 1.62, 1.70]
    for slot, val in enumerate([0.9, 1.75, 1.80, 1.78, 1.76, 1.79]):
        agent4.propose_probe(31, slot)
        agent4.record_result(val)
    verdict = agent4.evaluate_batch(31)

    assert verdict != COMMIT
    assert agent4.committed_hyperparams is None


@requires_deps
def test_commit_does_not_fire_below_min_runs(tmp_path):
    agent4, _ = _engage(tmp_path)
    agent4._origin_runs = [1.40, 1.45, 1.50, 1.60]
    for slot in range(3):  # 3 runs, min_runs_before_commit is 6
        agent4.propose_probe(31, slot)
        agent4.record_result(0.5)
    assert agent4.evaluate_batch(31) != COMMIT


@requires_deps
def test_heavily_exploited_origin_is_flagged_local_optimum_on_commit(tmp_path):
    agent4, _ = _engage(tmp_path)
    agent4._origin_runs = [1.5] * 25  # >= heavy_exploitation_n
    for slot in range(6):
        agent4.propose_probe(31, slot)
        agent4.record_result(1.0)
    assert agent4.evaluate_batch(31) == COMMIT
    flags = {r["flag"] for r in load_region_flags(agent4.region_flags_path)}
    assert "local_optimum" in flags


@requires_deps
def test_lightly_exploited_origin_is_only_paused_on_commit(tmp_path):
    """"We didn't finish looking here" and "this is a local optimum" are
    different claims -- a short window only supports the first."""
    agent4, _ = _engage(tmp_path)
    agent4._origin_runs = [1.5] * 4  # well below heavy_exploitation_n
    for slot in range(6):
        agent4.propose_probe(31, slot)
        agent4.record_result(1.0)
    assert agent4.evaluate_batch(31) == COMMIT
    flags = {r["flag"] for r in load_region_flags(agent4.region_flags_path)}
    assert "exploitation_paused" in flags and "local_optimum" not in flags


@requires_deps
def test_budget_exhaustion_closes_the_window_without_committing(tmp_path):
    agent4, _ = _engage(tmp_path)
    agent4._origin_runs = [1.30, 1.31, 1.32, 1.33]
    for i in range(agent4.window_iterations):
        agent4.propose_probe(31 + i)
        agent4.record_result(1.32)  # neither clearly good nor clearly bad
        agent4.evaluate_batch(31 + i)

    assert agent4.active is False
    assert agent4.last_action == "abandoned"
    assert agent4.committed_hyperparams is None
    flags = {r["flag"] for r in load_region_flags(agent4.region_flags_path)}
    assert "exploitation_paused" in flags


@requires_deps
def test_evaluate_batch_on_a_closed_window_is_a_noop(tmp_path):
    agent4 = _make_agent4(tmp_path)
    assert agent4.evaluate_batch(31) == EXHAUSTED


# --- history ---------------------------------------------------------------

@requires_deps
def test_load_history_reads_back_prior_decisions_in_order(tmp_path):
    agent4, _ = _engage(tmp_path)
    for slot in range(3):
        agent4.propose_probe(31, slot)
        agent4.record_result(99.0)
    agent4.evaluate_batch(31)

    history = agent4._load_history()
    assert [h["action"] for h in history] == ["engaged", "next_region"]
    assert [h["iteration"] for h in history] == [30, 31]


def test_load_history_empty_when_no_decisions_yet(tmp_path):
    assert _make_agent4(tmp_path)._load_history() == []


def test_load_history_skips_corrupt_files(tmp_path):
    agent4 = _make_agent4(tmp_path)
    agent4.decisions_dir.mkdir(parents=True, exist_ok=True)
    (agent4.decisions_dir / "verdict_0001.json").write_text("{{{ not json", encoding="utf-8")
    (agent4.decisions_dir / "verdict_0002.json").write_text(
        json.dumps({"iteration": 2, "action": "engaged"}), encoding="utf-8")
    assert [h["iteration"] for h in agent4._load_history()] == [2]


def test_load_history_ignores_routine_probe_logs(tmp_path):
    """History is for judgements. Dozens of individual draws would crowd out
    the handful of decisions a later window can actually act on."""
    agent4 = _make_agent4(tmp_path)
    agent4.decisions_dir.mkdir(parents=True, exist_ok=True)
    (agent4.decisions_dir / "decision_0005.json").write_text(
        json.dumps({"iteration": 5, "action": "probe"}), encoding="utf-8")
    (agent4.decisions_dir / "verdict_0006.json").write_text(
        json.dumps({"iteration": 6, "action": "committed"}), encoding="utf-8")
    assert [h["action"] for h in agent4._load_history()] == ["committed"]


# --- llm_mode --------------------------------------------------------------

@requires_deps
def test_statistics_mode_makes_no_llm_calls(tmp_path, monkeypatch):
    import agents.claude_cli as claude_cli_module
    calls = []
    monkeypatch.setattr(claude_cli_module, "call_with_budget",
                        lambda *a, **k: calls.append(k.get("call_site")) or "x")
    agent4, _ = _engage(tmp_path)
    assert calls == []


@requires_deps
def test_hybrid_mode_narrates_without_changing_the_verdict(tmp_path, monkeypatch):
    """Same decisions as statistics mode, only prose added."""
    import agents.agent4_landscape_explorer as module
    monkeypatch.setattr(module.claude_cli, "call_with_budget",
                        lambda *a, **k: "Probed an under-explored region; nothing conclusive yet.")

    stats_agent, _ = _engage(tmp_path / "a")
    hybrid_agent, _ = _engage(tmp_path / "b", _make_agent4(
        tmp_path / "b", CONFIG.replace("llm_mode: statistics", "llm_mode: hybrid")))

    for agent in (stats_agent, hybrid_agent):
        agent._origin_runs = [1.5] * 8
        for slot in range(6):
            agent.propose_probe(31, slot)
            agent.record_result(1.0)

    assert stats_agent.evaluate_batch(31) == hybrid_agent.evaluate_batch(31) == COMMIT
    assert "narrative" not in stats_agent.last_decision_log
    assert hybrid_agent.last_decision_log["narrative"].startswith("Probed an")


@requires_deps
def test_llm_mode_can_veto_a_commit(tmp_path, monkeypatch):
    import agents.agent4_landscape_explorer as module
    monkeypatch.setattr(module.claude_cli, "call_with_budget", lambda *a, **k: "VETO")
    agent4, _ = _engage(tmp_path, _make_agent4(
        tmp_path, CONFIG.replace("llm_mode: statistics", "llm_mode: llm")))
    agent4._origin_runs = [1.5] * 8
    for slot in range(6):
        agent4.propose_probe(31, slot)
        agent4.record_result(1.0)
    assert agent4.evaluate_batch(31) != COMMIT


@requires_deps
def test_llm_mode_falls_back_to_the_deterministic_verdict_when_the_call_fails(tmp_path, monkeypatch):
    """A budget-exhausted or unreachable CLI must not silently change what
    the campaign does."""
    import agents.agent4_landscape_explorer as module
    monkeypatch.setattr(module.claude_cli, "call_with_budget", lambda *a, **k: None)
    agent4, _ = _engage(tmp_path, _make_agent4(
        tmp_path, CONFIG.replace("llm_mode: statistics", "llm_mode: llm")))
    agent4._origin_runs = [1.5] * 8
    for slot in range(6):
        agent4.propose_probe(31, slot)
        agent4.record_result(1.0)
    assert agent4.evaluate_batch(31) == COMMIT


def test_unknown_llm_mode_falls_back_to_statistics(tmp_path):
    agent4 = _make_agent4(tmp_path, CONFIG.replace("llm_mode: statistics", "llm_mode: telepathy"))
    assert agent4.llm_mode == "statistics"


# --- pipeline_validator compatibility --------------------------------------

@requires_deps
def test_decision_log_validates_through_agent1s_validator(tmp_path):
    """The whole reason Agent 4's log mirrors Agent 1's params shape: the
    orchestrator validates both with the same call."""
    from agents import pipeline_validator
    agent4, _ = _engage(tmp_path)
    issues = pipeline_validator.validate_agent1_decision(
        agent4.last_decision_log, evidence=[{"x": 1}], latest_summary="s",
        decisions_dir=agent4.decisions_dir,
    )
    assert not [i for i in issues if i.severity in (pipeline_validator.FATAL, pipeline_validator.ERROR)]


# --- sparse / missing origin reference -----------------------------------
# Found by an end-to-end dry run, not by the unit tests: at region_radius
# 0.05 the starting center had ZERO historical runs near it, which crashed a
# format string and -- worse -- made every probe register as "bad" against a
# reference that didn't exist, condemning regions on no evidence.

@requires_deps
def test_sparse_origin_region_falls_back_to_the_whole_campaign(tmp_path, capsys):
    agent4, _ = _engage(tmp_path)
    agent4.region_radius = 0.0  # no run can be within zero distance
    vals = agent4._region_val_bpbs(_hyperparams(0))

    assert len(vals) == len(agent4._rows)  # fell back rather than returning []
    assert "too sparse for a local reference" in capsys.readouterr().out


@requires_deps
def test_no_reference_means_no_probe_is_condemned(tmp_path):
    """Without a bar, "worse than the bar" is not a claim that can be made.
    A region must not be flagged no_optimum on a missing comparison."""
    agent4, _ = _engage(tmp_path)
    agent4._origin_runs = []  # no reference available at all
    for slot in range(3):
        agent4.propose_probe(31, slot)
        agent4.record_result(99.0)  # would be "bad" against any real bar

    assert agent4.evaluate_batch(31) == CONTINUE
    assert "no_optimum" not in {r["flag"] for r in load_region_flags(agent4.region_flags_path)}


@requires_deps
def test_abandon_message_survives_a_missing_reference(tmp_path):
    """Regression guard on the actual crash: an f-string formatted
    elite_ref with :.6f without a None check."""
    agent4, _ = _engage(tmp_path)
    agent4._origin_runs = []
    agent4._batch_results = [99.0, 99.0, 99.0]
    agent4._iterations_used = agent4.window_iterations  # force the exhaust path
    agent4.evaluate_batch(31)  # must not raise
    assert agent4.active is False


@requires_deps
def test_stale_investigating_flag_is_cleared_when_a_new_window_opens(tmp_path):
    """Window state is in-memory only, so a campaign that ends mid-window
    leaves an "investigating" flag that nothing would ever resolve -- the
    chart would show a region under active investigation forever."""
    from state.landscape import save_region_flags
    agent4 = _make_agent4(tmp_path)
    save_region_flags(agent4.region_flags_path, [
        {"hyperparams": _hyperparams(3), "flag": "investigating",
         "since_iteration": 5, "n_runs": 0},
        {"hyperparams": _hyperparams(4), "flag": "local_optimum",
         "since_iteration": 6, "n_runs": 30},
    ])

    _engage(tmp_path, agent4)

    flags = load_region_flags(agent4.region_flags_path)
    investigating = [r for r in flags if r["flag"] == "investigating"]
    assert len(investigating) == 1                     # only the new one
    assert investigating[0]["since_iteration"] == 30
    # Durable flags are untouched -- they are conclusions, not transient state.
    assert any(r["flag"] == "local_optimum" for r in flags)


def test_code_defaults_match_the_shipped_config(tmp_path):
    """The calibrated thresholds must not diverge between agents_config.yaml
    and the in-code fallbacks: a caller without an agent4: block would
    otherwise silently get commit_margin=0.03 (a bar no run in 514 has ever
    cleared, i.e. relocation impossible) and region_radius=0.15 (63% of the
    campaign as one "region")."""
    import yaml
    from pathlib import Path

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "bare.yaml").write_text("agent1:\n  use_llm: false\n", encoding="utf-8")
    defaults = Agent4LandscapeExplorer(
        config_path=str(tmp_path / "bare.yaml"), root_dir=str(tmp_path),
        state_dir=str(tmp_path / "state"), reports_dir=str(tmp_path / "reports"),
    )
    repo_root = Path(__file__).resolve().parent.parent
    shipped = yaml.safe_load((repo_root / "agents_config.yaml").read_text(encoding="utf-8"))["agent4"]

    for key in ("commit_margin", "region_radius", "bad_tolerance", "probe_wave_size",
                "window_iterations", "check_interval", "min_runs_before_commit"):
        assert getattr(defaults, key) == shipped[key], (
            f"{key}: code default {getattr(defaults, key)} != shipped {shipped[key]}")
