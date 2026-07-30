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
        agent1.decide_next_hyperparams(latest_summary="some summary", iteration=0, fresh_summary=True)
    mock_call.assert_not_called()


def test_agent1_use_llm_true_but_summary_not_fresh_never_calls_claude_cli(tmp_path):
    """The budget fix: use_llm=True alone must NOT be enough once a summary
    exists -- only the one call right after a NEW summary (fresh_summary=True)
    should ever reach Claude. fresh_summary defaults to False, matching every
    iteration except the one right after Agent 3 creates a summary."""
    agent1 = _make_agent1(tmp_path, use_llm=True)
    with patch.object(claude_cli, "call_with_budget") as mock_call:
        agent1.decide_next_hyperparams(latest_summary="some summary", iteration=0)
    mock_call.assert_not_called()


def test_agent1_use_llm_true_and_fresh_summary_calls_claude_cli(tmp_path):
    agent1 = _make_agent1(tmp_path, use_llm=True)
    with patch.object(claude_cli, "call_with_budget", return_value='{"n_layer": 20}') as mock_call:
        result = agent1.decide_next_hyperparams(latest_summary="some summary", iteration=0, fresh_summary=True)
    mock_call.assert_called_once()
    assert result["n_layer"] == 20


def test_fresh_summary_flag_gates_agent1_llm_to_one_iteration_after_new_summary(tmp_path):
    """End-to-end budget-saving behavior: agent3 (batch_size=2, use_llm=True)
    creates a summary at the end of iteration 1 (2nd report); agent1
    (use_llm=True) must call Claude exactly once across the whole run --
    on iteration 2, right after that summary landed -- never on iterations
    0-1 (no summary existed yet) and never again on iteration 3+ even
    though the same summary is still "latest" by then.
    """
    from agents.orchestrator import Orchestrator

    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("""
agent1:
  use_llm: true
  use_surrogate: false
  accuracy_threshold: 0.01
agent2:
  xai_method: fast
  use_llm: false
agent3:
  batch_size: 2
  use_llm: true
llm:
  campaign_budget_usd: 5.0
  max_call_budget_usd: 0.20
""".strip())

    orch = Orchestrator(
        config_path=str(config_path), state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"), root_dir=str(tmp_path), dry_run=True,
    )

    call_sites_seen = []

    def fake_call_with_budget(prompt, call_site, **kwargs):
        call_sites_seen.append(call_site)
        return None

    with patch.object(claude_cli, "call_with_budget", side_effect=fake_call_with_budget):
        orch.run(max_iterations=4)

    assert "agent3_strategic_narrative" in call_sites_seen
    assert call_sites_seen.count("agent1_hyperparameter_review") == 1


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


def _write_fake_report_with_fingerprint(reports_dir, report_id, val_bpb, attn_distance):
    """Like _write_fake_report but with a real (non-empty) token_fingerprint
    payload -- needed to exercise fingerprint_clusters / smoothness_correlation
    and, downstream, the agent3_cluster_hypotheses call site and its skip-guard."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    n_layer = len(attn_distance)
    token_fingerprint = {
        "attn_entropy": [1.0] * n_layer,
        "attn_distance": list(attn_distance),
        "attn_distance_slope": 0.1,
        "dla": [0.1] * n_layer,
        "x0_lambda": [1.0] * n_layer,
        "induction_score": 0.1,
        "pos_saliency": [0.0] * 16,
    }
    structured = {
        "model_id": report_id, "report_id": report_id, "stuck_signal": False, "confidence": 0.9,
        "val_bpb": val_bpb, "hyperparams": _base_hyperparams(), "hyperparameter_importance": {},
        "ablation_ran": False, "head_importance": {}, "layer_importance_share_pct": {},
        "layer_scalars": {}, "token_fingerprint": token_fingerprint, "metadata": {"status": "ok"},
    }
    text = f"# XAI Analysis Report: {report_id}\n\n```json\n{json.dumps(structured)}\n```\n"
    (reports_dir / f"{report_id}.md").write_text(text)


def _write_eight_fingerprint_reports(reports_dir):
    """8 reports (MIN_CLUSTER_N default) with a real attn_distance curve
    each -- enough for trajectory_smoothness_correlation (pure math, no
    scipy needed) and, when scipy/sklearn are installed, the cluster
    functions too. Alternates ramp/zig-zag shapes: a constant offset alone
    normalizes away to an identical curve (min-max normalization), which
    would make every row's total-variation value identical and the
    correlation undefined -- shape must actually vary, not just magnitude.
    """
    report_ids = []
    for i in range(8):
        report_id = f"report_{i:04d}"
        curve = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] if i % 2 == 0 else [0.0, 5.0, 1.0, 4.0, 2.0, 3.0]
        _write_fake_report_with_fingerprint(reports_dir, report_id, val_bpb=0.9 + 0.01 * i, attn_distance=curve)
        report_ids.append(report_id)
    return report_ids


def test_agent3_smoothness_correlation_section_and_cluster_hypotheses_prompt_fire_together(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    report_ids = _write_eight_fingerprint_reports(reports_dir)
    agent3 = _make_agent3(tmp_path, use_llm=True)

    def fake_call(prompt, call_site, **kwargs):
        return {
            "agent3_strategic_narrative": "We are converging steadily.",
            "agent3_cluster_hypotheses": "Smoother trajectories look better.",
        }.get(call_site)

    with patch.object(claude_cli, "call_with_budget", side_effect=fake_call) as mock_call:
        summary = agent3.analyze_and_summarize(report_ids)

    text = (tmp_path / "reports" / "agent3_summaries" / f"{summary.summary_id}.md").read_text()
    assert "### Trajectory volatility vs. val_bpb (Tier 3.4" in text
    assert "## Cluster Hypotheses (Claude)" in text
    assert "Smoother trajectories look better." in text
    cluster_calls = [c for c in mock_call.call_args_list if c.kwargs.get("call_site") == "agent3_cluster_hypotheses"]
    assert len(cluster_calls) == 1
    assert "smoothness_correlation" in cluster_calls[0].args[0]  # prompt includes the field name


def test_agent3_cluster_hypotheses_skipped_when_fingerprint_data_unchanged(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    report_ids = _write_eight_fingerprint_reports(reports_dir)
    agent3 = _make_agent3(tmp_path, use_llm=True)

    def fake_call(prompt, call_site, **kwargs):
        return {
            "agent3_strategic_narrative": "narrative 1",
            "agent3_cluster_hypotheses": "hypothesis 1",
        }.get(call_site)

    with patch.object(claude_cli, "call_with_budget", side_effect=fake_call) as mock_call:
        first_summary = agent3.analyze_and_summarize(report_ids)
    first_text = (tmp_path / "reports" / "agent3_summaries" / f"{first_summary.summary_id}.md").read_text()
    assert "hypothesis 1" in first_text

    # No new reports, same underlying fingerprint data -> second summary
    # should skip the cluster-hypotheses LLM call entirely.
    def fake_call_2(prompt, call_site, **kwargs):
        return {
            "agent3_strategic_narrative": "narrative 2",
            "agent3_cluster_hypotheses": "hypothesis 2 (should not appear)",
        }.get(call_site)

    with patch.object(claude_cli, "call_with_budget", side_effect=fake_call_2) as mock_call_2:
        second_summary = agent3.analyze_and_summarize([report_ids[0]])
    second_text = (tmp_path / "reports" / "agent3_summaries" / f"{second_summary.summary_id}.md").read_text()
    assert "Skipped this run -- underlying fingerprint cluster data is unchanged" in second_text
    assert "hypothesis 2" not in second_text
    cluster_calls_2 = [c for c in mock_call_2.call_args_list if c.kwargs.get("call_site") == "agent3_cluster_hypotheses"]
    assert len(cluster_calls_2) == 0
    # The narrative call site has no skip-guard -- it should still fire every time.
    narrative_calls_2 = [c for c in mock_call_2.call_args_list if c.kwargs.get("call_site") == "agent3_strategic_narrative"]
    assert len(narrative_calls_2) == 1


def test_agent3_cluster_hypotheses_recomputed_when_fingerprint_data_changes(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    report_ids = _write_eight_fingerprint_reports(reports_dir)
    agent3 = _make_agent3(tmp_path, use_llm=True)

    def fake_call(prompt, call_site, **kwargs):
        return {
            "agent3_strategic_narrative": "narrative 1",
            "agent3_cluster_hypotheses": "hypothesis 1",
        }.get(call_site)

    with patch.object(claude_cli, "call_with_budget", side_effect=fake_call):
        agent3.analyze_and_summarize(report_ids)

    # A genuinely new report with a very different curve changes the
    # underlying fingerprint data -> the guard must NOT skip this time.
    new_id = "report_0008"
    _write_fake_report_with_fingerprint(reports_dir, new_id, val_bpb=2.0, attn_distance=[9.0, 0.0, 9.0, 0.0, 9.0, 0.0])

    def fake_call_2(prompt, call_site, **kwargs):
        return {
            "agent3_strategic_narrative": "narrative 2",
            "agent3_cluster_hypotheses": "hypothesis 2",
        }.get(call_site)

    with patch.object(claude_cli, "call_with_budget", side_effect=fake_call_2) as mock_call_2:
        second_summary = agent3.analyze_and_summarize([new_id])
    second_text = (tmp_path / "reports" / "agent3_summaries" / f"{second_summary.summary_id}.md").read_text()
    assert "hypothesis 2" in second_text
    cluster_calls_2 = [c for c in mock_call_2.call_args_list if c.kwargs.get("call_site") == "agent3_cluster_hypotheses"]
    assert len(cluster_calls_2) == 1


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
