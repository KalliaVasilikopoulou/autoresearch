import json
import tempfile
from pathlib import Path
import math

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from agents.orchestrator import Orchestrator


def test_orchestrator_dry_run_builds_reports_and_summary(tmp_path):
    state_dir = tmp_path / "state"
    reports_dir = tmp_path / "reports"
    config_path = tmp_path / "agents_config.yaml"

    config_path.write_text(
        """
agent1:
  use_llm: false
  accuracy_threshold: 0.95
  cost_limit_usd: 50.0
  training_budget_seconds: 300

agent2:
  xai_method: fast
  use_llm: false
  ablation_k: 3
  incremental: true

agent3:
  batch_size: 2
  use_llm: false
  preserve_history: true
""".strip()
    )

    orchestrator = Orchestrator(
        config_path=str(config_path),
        state_dir=str(state_dir),
        reports_dir=str(reports_dir),
        dry_run=True,
    )

    summary = orchestrator.run(max_iterations=2)

    assert summary is not None
    assert summary.summary_id.startswith("summary_")
    assert len(orchestrator.state_mgr.get_all_results()) >= 2
    assert orchestrator.state_mgr.get_latest_summary() is not None

    metadata_path = state_dir / "metadata.json"
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text())
    assert payload["latest_summary"] is not None


def test_training_falls_back_to_simulated_run_when_real_training_is_unavailable(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(
        """
agent1:
  use_llm: false
  accuracy_threshold: 0.95
  cost_limit_usd: 50.0
  training_budget_seconds: 300
""".strip()
    )

    specialist = Agent1TrainingSpecialist(config_path=str(config_path))
    result = specialist.train_model(
        {"learning_rate": 0.001, "n_layer": 10, "n_embd": 256},
        dry_run=False,
        iteration=1,
    )

    assert result["status"] == "simulated"
    assert result["val_bpb"] < float("inf")
    assert result["training_time"] >= 0.0


def test_stagnation_signal_triggers_radical_fallback_adjustment(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(
        """
agent1:
  use_llm: false
  accuracy_threshold: 0.95
  cost_limit_usd: 50.0
  training_budget_seconds: 300
""".strip()
    )

    specialist = Agent1TrainingSpecialist(config_path=str(config_path))
    specialist.current_hyperparams = {
        "n_layer": 12,
        "n_head": 8,
        "n_embd": 512,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
    }

    new_hyperparams = specialist.decide_next_hyperparams(
        latest_summary=None,
        evidence=None,
        stuck_signal=False,
        latest_val_bpb=0.995,
        iteration=10,
        recent_results=[{"val_bpb": 1.0}, {"val_bpb": 0.995}, {"val_bpb": 0.995}],
    )

    assert new_hyperparams is not None
    assert new_hyperparams["n_layer"] != 12
    assert new_hyperparams["n_embd"] != 512


def test_importance_weight_changes_are_scaled_by_score(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("agent1:\n  use_llm: false")

    specialist = Agent1TrainingSpecialist(config_path=str(config_path))
    specialist.current_hyperparams = {
        "n_layer": 12,
        "n_head": 8,
        "n_embd": 512,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
    }

    weak = specialist._evidence_adjustment(
        latest_summary=None,
        evidence=[{"hyperparameter_importance": {"n_layer": 0.55}}],
        stuck_signal=False,
        iteration=0,
    )
    specialist.current_hyperparams["n_layer"] = 12
    strong = specialist._evidence_adjustment(
        latest_summary=None,
        evidence=[{"hyperparameter_importance": {"n_layer": 0.95}}],
        stuck_signal=False,
        iteration=0,
    )

    weak_delta = abs(weak["n_layer"] - 12)
    strong_delta = abs(strong["n_layer"] - 12)
    assert strong_delta > weak_delta


def test_summary_adjustment_is_stronger_than_report_only(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("agent1:\n  use_llm: false\n  summary_strength: 2.0")

    specialist = Agent1TrainingSpecialist(config_path=str(config_path))
    specialist.current_hyperparams = {
        "n_layer": 12,
        "n_head": 8,
        "n_embd": 512,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
    }

    report_only = specialist._evidence_adjustment(
        latest_summary=None,
        evidence=[{"hyperparameter_importance": {"learning_rate": 1.0}}],
        stuck_signal=False,
        iteration=0,
    )

    specialist.current_hyperparams["learning_rate"] = 1e-3
    summary_text = "learning_rate importance is stable and important"
    summary_plus_report = specialist._evidence_adjustment(
        latest_summary=summary_text,
        evidence=[{"hyperparameter_importance": {"learning_rate": 1.0}}],
        stuck_signal=False,
        iteration=0,
    )

    report_change = abs(math.log(report_only["learning_rate"] / 1e-3))
    summary_change = abs(math.log(summary_plus_report["learning_rate"] / 1e-3))
    assert summary_change > report_change


def test_learning_rate_is_always_clamped_to_safe_range(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(
        "agent1:\n  use_llm: false\n  learning_rate_min: 1.0e-5\n  learning_rate_max: 5.0e-3"
    )

    specialist = Agent1TrainingSpecialist(config_path=str(config_path))
    specialist.current_hyperparams = {
        "n_layer": 12,
        "n_head": 8,
        "n_embd": 512,
        "learning_rate": 1.0,
        "batch_size": 128,
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
    }

    high_case = specialist._evidence_adjustment(
        latest_summary="learning_rate important",
        evidence=[{"hyperparameter_importance": {"learning_rate": 1.0}}],
        stuck_signal=False,
        iteration=0,
    )
    assert 1e-5 <= high_case["learning_rate"] <= 5e-3

    specialist.current_hyperparams["learning_rate"] = 1.0e-9
    low_case = specialist._evidence_adjustment(
        latest_summary=None,
        evidence=[{"hyperparameter_importance": {"learning_rate": 0.0}}],
        stuck_signal=False,
        iteration=0,
    )
    assert 1e-5 <= low_case["learning_rate"] <= 5e-3
