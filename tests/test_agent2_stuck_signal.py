"""Agent 2's val_bpb-based stuck-signal check now compares against an
adaptive elite reference (median of the best 25% of real historical runs,
see Agent2XAISpecialist._elite_val_bpb_reference) instead of a hardcoded
val_bpb > 1.32 -- that fixed number turned out to flag ~95% of all real
runs ever recorded, since real val_bpb for this project's actual training
regime sits well above 1.32 most of the time.
"""

import pytest

from agents.agent2_xai_specialist import Agent2XAISpecialist
from state.results_logger import log_result


def _base_hyperparams():
    return {
        "n_layer": 12, "n_head": 4, "n_embd": 512,
        "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04, "scalar_lr": 0.5,
        "batch_size": 8192, "warmup_ratio": 0.1, "weight_decay": 0.1,
    }


def _make_agent2(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("""
agent2:
  xai_method: fast
  use_llm: false
  generate_charts: false
""".strip())
    return Agent2XAISpecialist(config_path=str(config_path), root_dir=str(tmp_path), reports_dir=str(tmp_path / "reports"))


def _seed_real_history(results_path):
    # Elite (best 25% of 8 -> 2 rows): 1.30, 1.35 -> median 1.325.
    # Stuck threshold = 1.325 * 1.15 = 1.52375.
    for i, val_bpb in enumerate([1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.8, 2.0]):
        log_result(f"run_{i:04d}", _base_hyperparams(), {"val_bpb": val_bpb, "status": "remote_ok"},
                   results_path=str(results_path))


def _analyze(agent2, val_bpb, status="remote_ok"):
    evidence = agent2.analyze_result({
        "run_id": f"probe_{val_bpb}", "val_bpb": val_bpb, "status": status,
        "hyperparams": _base_hyperparams(), "metadata": {},
    })
    return evidence.stuck_signal


def test_elite_val_bpb_reference_none_with_no_history(tmp_path):
    agent2 = _make_agent2(tmp_path)
    assert agent2._elite_val_bpb_reference() is None


def test_no_history_never_fabricates_a_stuck_flag_from_val_bpb_alone(tmp_path):
    agent2 = _make_agent2(tmp_path)
    # No results.tsv at all -- elite_reference is None, so the val_bpb
    # clause must not fire (a genuinely bad but merely-finite val_bpb here
    # would have tripped the old hardcoded > 1.32 check).
    assert _analyze(agent2, val_bpb=5.0) is False


def test_stuck_signal_below_elite_threshold_is_not_stuck(tmp_path):
    agent2 = _make_agent2(tmp_path)
    _seed_real_history(agent2.results_path)
    # 1.5 < 1.52375 threshold -- not stuck. The old hardcoded 1.32 would
    # have flagged this (1.5 > 1.32).
    assert _analyze(agent2, val_bpb=1.5) is False


def test_stuck_signal_above_elite_threshold_is_stuck(tmp_path):
    agent2 = _make_agent2(tmp_path)
    _seed_real_history(agent2.results_path)
    # 1.6 > 1.52375 threshold -- stuck.
    assert _analyze(agent2, val_bpb=1.6) is True


def test_elite_reference_excludes_dry_run_rows(tmp_path):
    agent2 = _make_agent2(tmp_path)
    # A dry_run row with a suspiciously low val_bpb must not drag the elite
    # reference down -- load_results() already excludes it (see
    # state/results_analysis.py::SYNTHETIC_STATUSES).
    log_result("run_dry", _base_hyperparams(), {"val_bpb": 0.001, "status": "dry_run"},
               results_path=str(agent2.results_path))
    _seed_real_history(agent2.results_path)

    assert agent2._elite_val_bpb_reference() == pytest.approx(1.325)
    # Same as the pure-real-history case: 1.5 stays under threshold.
    assert _analyze(agent2, val_bpb=1.5) is False
