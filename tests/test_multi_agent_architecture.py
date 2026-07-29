import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import math

import random

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from agents.agent3_report_analyst import Agent3ReportAnalyst
from agents.orchestrator import Orchestrator
from state.clustering import POS_SALIENCY_LEN


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


def _fake_fingerprint(group: str) -> dict:
    """A synthetic token_fingerprint matching Agent 2's real schema (see
    agents/xai_methods/token_methods.py::compute_behavioral_fingerprint),
    with two clearly-separated value profiles depending on `group`."""
    n_layer = random.choice([8, 10, 12])
    if group == "low":
        base_entropy, base_distance, base_dla, base_x0 = 1.0, 5.0, 0.01, 1.0
    else:
        base_entropy, base_distance, base_dla, base_x0 = 4.0, 50.0, 0.5, 10.0
    # pos_saliency/attn_distance_slope/induction_score are NOT
    # group-differentiated here (real fingerprints would vary, but this test
    # only needs entropy/distance/dla/x0 to carry signal) -- kept at a fixed
    # baseline with tight jitter so they don't contribute noise that could
    # dilute the intended 2-group separation (a real, previously-hit failure
    # mode: wide-range independent noise across many feature dimensions lets
    # Ward+silhouette prefer spurious fine-grained splits over the two
    # genuine groups at small n).
    return {
        "attn_entropy": [base_entropy + random.uniform(-0.05, 0.05) for _ in range(n_layer)],
        "attn_distance": [base_distance + random.uniform(-0.5, 0.5) for _ in range(n_layer)],
        "attn_distance_slope": 0.05 + random.uniform(-0.005, 0.005),
        "pos_saliency": [0.05 + random.uniform(-0.005, 0.005) for _ in range(POS_SALIENCY_LEN)],
        "dla": [base_dla + random.uniform(-0.005, 0.005) for _ in range(n_layer)],
        "induction_score": 0.1 + random.uniform(-0.01, 0.01),
        "x0_lambda": [base_x0 + random.uniform(-0.1, 0.1) for _ in range(n_layer)],
    }


def _write_fake_report(reports_dir: Path, report_id: str, val_bpb: float, token_fingerprint: dict) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    structured = {
        "model_id": report_id,
        "report_id": report_id,
        "stuck_signal": False,
        "confidence": 0.9,
        "val_bpb": val_bpb,
        "hyperparams": {"n_layer": 8, "n_embd": 256},
        "hyperparameter_importance": {},
        "ablation_ran": False,
        "head_importance": {},
        "layer_importance_share_pct": {},
        "layer_scalars": {},
        "token_fingerprint": token_fingerprint,
        "metadata": {"status": "ok"},
    }
    content = (
        f"# XAI Analysis Report: {report_id}\n\n"
        f"- Status: ok\n"
        f"- Validation bpb: {val_bpb:.6f}\n\n"
        "## Structured Metrics (for Agent 3)\n"
        "```json\n"
        f"{json.dumps(structured, indent=2, sort_keys=True)}\n"
        "```\n"
    )
    (reports_dir / f"{report_id}.md").write_text(content)


def test_agent3_cluster_section_absent_below_min_observations(tmp_path):
    random.seed(3)
    reports_dir = tmp_path / "reports"
    for i in range(3):  # well below the default min_cluster_observations=8
        report_id = f"report_{i:04d}"
        _write_fake_report(reports_dir / "agent2_reports", report_id, 1.0 + i * 0.01, _fake_fingerprint("low"))

    analyst = Agent3ReportAnalyst(reports_dir=str(reports_dir), state_dir=str(tmp_path / "state"))
    summary = analyst.analyze_and_summarize([f"report_{i:04d}" for i in range(3)])

    assert summary.fingerprint_clusters == {}
    summary_text = (reports_dir / "agent3_summaries" / f"{summary.summary_id}.md").read_text()
    assert "Not enough historical fingerprints yet" in summary_text


def test_agent3_cluster_section_present_with_enough_separable_data(tmp_path):
    random.seed(4)
    reports_dir = tmp_path / "reports"
    report_ids = []
    for i in range(6):
        report_id = f"report_{i:04d}"
        _write_fake_report(reports_dir / "agent2_reports", report_id, 0.90 + random.uniform(-0.01, 0.01), _fake_fingerprint("low"))
        report_ids.append(report_id)
    for i in range(6, 12):
        report_id = f"report_{i:04d}"
        _write_fake_report(reports_dir / "agent2_reports", report_id, 1.30 + random.uniform(-0.01, 0.01), _fake_fingerprint("high"))
        report_ids.append(report_id)

    analyst = Agent3ReportAnalyst(reports_dir=str(reports_dir), state_dir=str(tmp_path / "state"))
    summary = analyst.analyze_and_summarize(report_ids)

    assert summary.fingerprint_clusters != {}
    assert summary.fingerprint_clusters.get("overall") is not None
    assert summary.fingerprint_clusters["overall"]["k"] == 2

    summary_text = (reports_dir / "agent3_summaries" / f"{summary.summary_id}.md").read_text()
    assert "Behavioral Fingerprint Clusters" in summary_text
    assert "Overall fingerprint clusters" in summary_text


def test_agent3_new_sections_absent_without_data_present_with_data(tmp_path):
    """dev/checks.txt visualization-gaps pass: the two new Agent 3 sections
    (Tier 4 fingerprint_adjustments trend, pipeline_validator issue trend)
    must be honestly absent when reports/agent1_decisions and
    reports/pipeline_validation don't exist/have nothing, and show real
    content once they do -- not fabricated either way.
    """
    reports_dir = tmp_path / "reports"
    _write_fake_report(reports_dir / "agent2_reports", "report_0000", 1.0, {})

    # --- Case 1: neither directory has any data yet ---
    analyst = Agent3ReportAnalyst(reports_dir=str(reports_dir), state_dir=str(tmp_path / "state"))
    summary = analyst.analyze_and_summarize(["report_0000"])
    text = (reports_dir / "agent3_summaries" / f"{summary.summary_id}.md").read_text()
    assert "No fingerprint-driven adjustments fired yet" in text
    assert "No pipeline_validator issues recorded for this run" in text

    # --- Case 2: real decision logs (with fingerprint_adjustments) and a
    # real pipeline_validation run dir (with issues) now exist ---
    decisions_dir = reports_dir / "agent1_decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    for i, delta in enumerate([1, -1]):
        (decisions_dir / f"decision_{i:04d}.json").write_text(json.dumps({
            "iteration": i,
            "fingerprint_adjustments": [{"param": "n_layer", "votes": [1 if delta > 0 else -1], "delta": delta, "new_value": 10}],
        }))

    run_dir = reports_dir / "pipeline_validation" / "run_20260101_000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "iteration_0000.json").write_text(json.dumps({
        "iteration": 0,
        "issues": [{"severity": "WARN", "source": "agent2", "message": "test issue", "context": {}}],
        "suspect": True,
    }))

    _write_fake_report(reports_dir / "agent2_reports", "report_0001", 0.95, {})
    analyst2 = Agent3ReportAnalyst(reports_dir=str(reports_dir), state_dir=str(tmp_path / "state"))
    summary2 = analyst2.analyze_and_summarize(["report_0001"])
    text2 = (reports_dir / "agent3_summaries" / f"{summary2.summary_id}.md").read_text()
    assert "2 fingerprint-driven adjustment(s)" in text2
    assert "1 issue(s)" in text2
    assert "WARN=1" in text2
