"""Synthetic-data tests for the LLM/copilot integration's 4 call sites
(dev/checks.txt item 4): agent1's hyperparameter review, agent2's per-run
report interpretation, agent3's strategic narrative + cluster hypotheses.

agents.claude_cli.call_with_budget is monkeypatched throughout -- no real
CLI/subprocess call happens in this suite. Each site is checked for:
(a) use_llm=False stays byte-identical to before this feature existed,
(b) use_llm=True + a mocked successful call flows the result through,
(c) use_llm=True + call_with_budget returning None (unavailable/budget
    exhausted) degrades honestly, never fabricating or crashing.
"""

import json
from unittest.mock import patch

from agents import claude_cli
from agents.agent1_training_specialist import Agent1TrainingSpecialist
from agents.agent2_xai_specialist import Agent2XAISpecialist
from agents.agent3_report_analyst import Agent3ReportAnalyst


def _base_hyperparams():
    return {
        "n_layer": 12, "n_head": 4, "n_embd": 512,
        "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04, "scalar_lr": 0.5,
        "batch_size": 8192, "warmup_ratio": 0.1, "weight_decay": 0.1,
    }


# --- Agent 1: hyperparameter review --------------------------------------

def _make_agent1(tmp_path, use_llm: bool):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(f"""
agent1:
  use_llm: {str(use_llm).lower()}
  use_surrogate: false
llm:
  campaign_budget_usd: 5.0
  max_call_budget_usd: 0.20
""".strip())
    return Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path), state_dir=str(tmp_path / "state"))


def test_agent1_use_llm_false_never_calls_claude_cli(tmp_path):
    """use_llm gating happens at decide_next_hyperparams's call site (not
    inside _get_claude_suggestion itself, which just asks Claude
    unconditionally when called) -- so this exercises the real gate."""
    agent1 = _make_agent1(tmp_path, use_llm=False)
    with patch.object(claude_cli, "call_with_budget") as mock_call:
        agent1.decide_next_hyperparams(latest_summary="some summary", iteration=0)
    mock_call.assert_not_called()


def test_agent1_use_llm_true_applies_claude_json_adjustment(tmp_path):
    agent1 = _make_agent1(tmp_path, use_llm=True)
    with patch.object(claude_cli, "call_with_budget", return_value='{"n_layer": 20, "matrix_lr": 0.09}'):
        result = agent1._get_claude_suggestion(_base_hyperparams(), "some summary")
    assert result["n_layer"] == 20
    assert result["matrix_lr"] == 0.09
    assert result["n_head"] == 4  # untouched fields preserved


def test_agent1_use_llm_true_degrades_to_none_when_call_returns_none(tmp_path):
    agent1 = _make_agent1(tmp_path, use_llm=True)
    with patch.object(claude_cli, "call_with_budget", return_value=None):
        result = agent1._get_claude_suggestion(_base_hyperparams(), "some summary")
    assert result is None


def test_agent1_use_llm_true_degrades_to_none_on_non_json_response(tmp_path):
    agent1 = _make_agent1(tmp_path, use_llm=True)
    with patch.object(claude_cli, "call_with_budget", return_value="not json at all"):
        result = agent1._get_claude_suggestion(_base_hyperparams(), "some summary")
    assert result is None


# --- Agent 2: per-run report interpretation -------------------------------

def _make_agent2(tmp_path, use_llm: bool):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(f"""
agent2:
  xai_method: fast
  use_llm: {str(use_llm).lower()}
  generate_charts: false
llm:
  campaign_budget_usd: 5.0
  max_call_budget_usd: 0.20
""".strip())
    return Agent2XAISpecialist(config_path=str(config_path), root_dir=str(tmp_path), reports_dir=str(tmp_path / "reports"))


def test_agent2_use_llm_false_reports_disabled_and_never_calls_claude_cli(tmp_path):
    agent2 = _make_agent2(tmp_path, use_llm=False)
    with patch.object(claude_cli, "call_with_budget") as mock_call:
        evidence = agent2.analyze_result({
            "run_id": "run_0000", "val_bpb": 1.0, "status": "ok",
            "hyperparams": _base_hyperparams(), "metadata": {},
        })
    mock_call.assert_not_called()
    report_text = (agent2.reports_dir / f"{evidence.report_id}.md").read_text()
    assert "## LLM Interpretation" in report_text
    assert "Disabled (agent2.use_llm is false)" in report_text


def test_agent2_use_llm_true_includes_llm_text_in_report(tmp_path):
    agent2 = _make_agent2(tmp_path, use_llm=True)
    with patch.object(claude_cli, "call_with_budget", return_value="This run trained normally with no anomalies."):
        evidence = agent2.analyze_result({
            "run_id": "run_0000", "val_bpb": 1.0, "status": "ok",
            "hyperparams": _base_hyperparams(), "metadata": {},
        })
    report_text = (agent2.reports_dir / f"{evidence.report_id}.md").read_text()
    assert "This run trained normally with no anomalies." in report_text


def test_agent2_use_llm_true_reports_unavailable_when_call_returns_none(tmp_path):
    agent2 = _make_agent2(tmp_path, use_llm=True)
    with patch.object(claude_cli, "call_with_budget", return_value=None):
        evidence = agent2.analyze_result({
            "run_id": "run_0000", "val_bpb": 1.0, "status": "ok",
            "hyperparams": _base_hyperparams(), "metadata": {},
        })
    report_text = (agent2.reports_dir / f"{evidence.report_id}.md").read_text()
    assert "Unavailable this run (CLI not reachable, or campaign LLM budget exhausted)" in report_text


# --- Agent 3: strategic narrative + cluster hypotheses + usage section ---

def _make_agent3(tmp_path, use_llm: bool):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(f"""
agent3:
  batch_size: 1
  use_llm: {str(use_llm).lower()}
  generate_charts: false
llm:
  campaign_budget_usd: 5.0
  max_call_budget_usd: 0.20
""".strip())
    return Agent3ReportAnalyst(
        config_path=str(config_path), reports_dir=str(tmp_path / "reports"), state_dir=str(tmp_path / "state"),
    )


def _write_fake_report(reports_dir, report_id, val_bpb):
    reports_dir.mkdir(parents=True, exist_ok=True)
    structured = {
        "model_id": report_id, "report_id": report_id, "stuck_signal": False, "confidence": 0.9,
        "val_bpb": val_bpb, "hyperparams": _base_hyperparams(), "hyperparameter_importance": {},
        "ablation_ran": False, "head_importance": {}, "layer_importance_share_pct": {},
        "layer_scalars": {}, "token_fingerprint": {}, "metadata": {"status": "ok"},
    }
    text = f"# XAI Analysis Report: {report_id}\n\n```json\n{json.dumps(structured)}\n```\n"
    (reports_dir / f"{report_id}.md").write_text(text)


def test_agent3_use_llm_false_reports_no_llm_sections_used(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    _write_fake_report(reports_dir, "report_0000", 1.0)
    agent3 = _make_agent3(tmp_path, use_llm=False)
    with patch.object(claude_cli, "call_with_budget") as mock_call:
        summary = agent3.analyze_and_summarize(["report_0000"])
    mock_call.assert_not_called()
    text = (tmp_path / "reports" / "agent3_summaries" / f"{summary.summary_id}.md").read_text()
    assert "## Strategic Narrative" not in text  # section only added when use_llm is True
    assert "No LLM calls made yet this campaign" in text


def test_agent3_use_llm_true_includes_narrative_and_logs_usage(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    _write_fake_report(reports_dir, "report_0000", 1.0)
    agent3 = _make_agent3(tmp_path, use_llm=True)

    def fake_call(prompt, call_site, **kwargs):
        return {"agent3_strategic_narrative": "We are converging steadily."}.get(call_site)

    with patch.object(claude_cli, "call_with_budget", side_effect=fake_call):
        summary = agent3.analyze_and_summarize(["report_0000"])

    text = (tmp_path / "reports" / "agent3_summaries" / f"{summary.summary_id}.md").read_text()
    assert "## Strategic Narrative" in text
    assert "We are converging steadily." in text


def test_agent3_use_llm_true_narrative_unavailable_when_call_returns_none(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    _write_fake_report(reports_dir, "report_0000", 1.0)
    agent3 = _make_agent3(tmp_path, use_llm=True)
    with patch.object(claude_cli, "call_with_budget", return_value=None):
        summary = agent3.analyze_and_summarize(["report_0000"])
    text = (tmp_path / "reports" / "agent3_summaries" / f"{summary.summary_id}.md").read_text()
    assert "## Strategic Narrative" in text
    assert "Unavailable this run (CLI not reachable, or campaign LLM budget exhausted)" in text


def test_agent3_llm_usage_section_reflects_logged_calls(tmp_path):
    from state import llm_usage
    reports_dir = tmp_path / "reports" / "agent2_reports"
    _write_fake_report(reports_dir, "report_0000", 1.0)
    agent3 = _make_agent3(tmp_path, use_llm=True)
    llm_usage.log_call("agent3_strategic_narrative", {"cost_usd": 0.05, "model": "sonnet", "is_error": False}, agent3._llm_usage_path)
    llm_usage.log_call("agent3_cluster_hypotheses", {"cost_usd": 0.03, "model": "sonnet", "is_error": False}, agent3._llm_usage_path)

    with patch.object(claude_cli, "call_with_budget", return_value=None):
        summary = agent3.analyze_and_summarize(["report_0000"])

    text = (tmp_path / "reports" / "agent3_summaries" / f"{summary.summary_id}.md").read_text()
    assert "## LLM Usage This Campaign" in text
    assert "2 call(s) (2 successful)" in text
    assert "$0.0800" in text  # cumulative cost
