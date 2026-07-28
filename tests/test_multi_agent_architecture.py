import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import math

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from agents.orchestrator import Orchestrator


def _base_hyperparams():
    """Hyperparameter dict matching train.py's real schema (4 LR groups)."""
    return {
        "n_layer": 12,
        "n_head": 4,
        "n_embd": 512,
        "embedding_lr": 0.6,
        "unembedding_lr": 0.004,
        "matrix_lr": 0.04,
        "scalar_lr": 0.5,
        "batch_size": 8192,
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
    }


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
        root_dir=str(tmp_path),
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


def test_orchestrator_token_xai_cadence_interval_and_new_best(tmp_path):
    """token_xai_enabled should be True on: iteration 0 (interval floor),
    any iteration whose incoming val_bpb just set a new best, and every
    Nth iteration thereafter -- and False otherwise. Isolates the two
    trigger conditions from each other via a controlled val_bpb sequence
    (dry_run's own formula always improves every iteration, which would
    make the "new best" trigger fire on literally every run and mask the
    interval-only and neither-condition cases).
    """
    state_dir = tmp_path / "state"
    reports_dir = tmp_path / "reports"
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(
        """
agent1:
  use_llm: false
  accuracy_threshold: 0.01
  cost_limit_usd: 50.0
  training_budget_seconds: 300
  token_xai_interval: 4

agent2:
  xai_method: fast
  use_llm: false
  ablation_k: 3

agent3:
  batch_size: 100
  use_llm: false
""".strip()
    )

    orchestrator = Orchestrator(
        config_path=str(config_path),
        state_dir=str(state_dir),
        reports_dir=str(reports_dir),
        root_dir=str(tmp_path),
        dry_run=True,
    )
    assert orchestrator.token_xai_interval == 4

    # val_bpb per iteration: iter0=1.0 (best->1.0), iter1=1.2 (worse),
    # iter2=1.3 (worse), iter3=1.4 (worse), iter4=0.9 (new best), iter5=1.5 (worse)
    val_bpb_sequence = [1.0, 1.2, 1.3, 1.4, 0.9, 1.5]
    observed_token_xai: list = []

    def fake_train_model(hyperparams, dry_run=False, iteration=0):
        observed_token_xai.append(hyperparams.get("token_xai_enabled"))
        return {
            "val_bpb": val_bpb_sequence[iteration],
            "training_time": 0.0,
            "checkpoint_path": None,
            "status": "dry_run",
        }

    with patch.object(orchestrator.agent1, "train_model", side_effect=fake_train_model):
        orchestrator.run(max_iterations=len(val_bpb_sequence))

    # iter0: interval (0%4==0) -> True
    # iter1: latest=1.0 sets a new best (best started at inf) -> True
    # iter2: latest=1.2, not a new best (best=1.0); 2%4!=0 -> False
    # iter3: latest=1.3, not a new best; 3%4!=0 -> False
    # iter4: latest=1.4, not a new best; 4%4==0 -> True (interval)
    # iter5: latest=0.9, new best (best=1.0); 5%4!=0 -> True (new best)
    assert observed_token_xai == [True, True, False, False, True, True]


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

    specialist = Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path))
    result = specialist.train_model(
        {"matrix_lr": 0.04, "n_layer": 10, "n_embd": 256},
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

    specialist = Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path))
    specialist.current_hyperparams = _base_hyperparams()

    # Assert on the radical-change path actually firing rather than on the
    # random n_layer/n_embd draw landing on a different value — random.choice
    # can coincidentally reproduce the current value, which made the old
    # value-inequality assertion flaky (~20%+ failure rate). decide_next_hyperparams
    # runs both the evidence and heuristic paths when no evidence list is given
    # (the heuristic result wins), so this can legitimately fire more than once.
    with patch.object(specialist, "_radical_change", wraps=specialist._radical_change) as mock_radical:
        new_hyperparams = specialist.decide_next_hyperparams(
            latest_summary=None,
            evidence=None,
            stuck_signal=False,
            latest_val_bpb=0.995,
            iteration=10,
            recent_results=[{"val_bpb": 1.0}, {"val_bpb": 0.995}, {"val_bpb": 0.995}],
        )

    assert new_hyperparams is not None
    assert mock_radical.called


def test_importance_weight_changes_are_scaled_by_score(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("agent1:\n  use_llm: false")

    specialist = Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path))
    specialist.current_hyperparams = _base_hyperparams()

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

    specialist = Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path))
    specialist.current_hyperparams = _base_hyperparams()

    report_only = specialist._evidence_adjustment(
        latest_summary=None,
        evidence=[{"hyperparameter_importance": {"matrix_lr": 1.0}}],
        stuck_signal=False,
        iteration=0,
    )

    specialist.current_hyperparams["matrix_lr"] = 0.04
    summary_text = "matrix_lr importance is stable and important"
    summary_plus_report = specialist._evidence_adjustment(
        latest_summary=summary_text,
        evidence=[{"hyperparameter_importance": {"matrix_lr": 1.0}}],
        stuck_signal=False,
        iteration=0,
    )

    report_change = abs(math.log(report_only["matrix_lr"] / 0.04))
    summary_change = abs(math.log(summary_plus_report["matrix_lr"] / 0.04))
    assert summary_change > report_change


def test_learning_rate_groups_are_always_clamped_to_safe_range(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(
        "agent1:\n  use_llm: false\n  matrix_lr_min: 0.005\n  matrix_lr_max: 0.2"
    )

    specialist = Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path))
    specialist.current_hyperparams = _base_hyperparams()
    specialist.current_hyperparams["matrix_lr"] = 1.0

    high_case = specialist._evidence_adjustment(
        latest_summary="matrix_lr important",
        evidence=[{"hyperparameter_importance": {"matrix_lr": 1.0}}],
        stuck_signal=False,
        iteration=0,
    )
    assert 0.005 <= high_case["matrix_lr"] <= 0.2

    specialist.current_hyperparams["matrix_lr"] = 1.0e-9
    low_case = specialist._evidence_adjustment(
        latest_summary=None,
        evidence=[{"hyperparameter_importance": {"matrix_lr": 0.0}}],
        stuck_signal=False,
        iteration=0,
    )
    assert 0.005 <= low_case["matrix_lr"] <= 0.2
