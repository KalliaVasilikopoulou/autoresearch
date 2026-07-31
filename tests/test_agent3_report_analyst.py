"""Synthetic-data tests confirming Agent 3's own statistics (Best/Worst/
Mean val_bpb, elite-run hyperparameter recommendations) exclude dry_run/
simulated reports. Agent 3 reads reports/agent2_reports/*.md directly
(a separate data source from results.tsv/load_results), so it needed its
own filter -- see Agent3ReportAnalyst._is_synthetic.
"""

import json

from agents.agent3_report_analyst import Agent3ReportAnalyst


def _base_hyperparams(**overrides):
    hp = {
        "n_layer": 12, "n_head": 4, "n_embd": 512,
        "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04, "scalar_lr": 0.5,
        "batch_size": 8192, "warmup_ratio": 0.1, "weight_decay": 0.1,
    }
    hp.update(overrides)
    return hp


def _write_report(reports_dir, report_id, val_bpb, status, hyperparams=None):
    reports_dir.mkdir(parents=True, exist_ok=True)
    structured = {
        "model_id": report_id, "report_id": report_id, "stuck_signal": False, "confidence": 0.9,
        "val_bpb": val_bpb, "hyperparams": hyperparams or _base_hyperparams(), "hyperparameter_importance": {},
        "ablation_ran": False, "head_importance": {}, "layer_importance_share_pct": {},
        "layer_scalars": {}, "token_fingerprint": {}, "metadata": {"status": status},
    }
    text = f"# XAI Analysis Report: {report_id}\n\n```json\n{json.dumps(structured)}\n```\n"
    (reports_dir / f"{report_id}.md").write_text(text)


def _make_agent3(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("""
agent3:
  batch_size: 1
  use_llm: false
  generate_charts: false
""".strip())
    return Agent3ReportAnalyst(
        config_path=str(config_path), reports_dir=str(tmp_path / "reports"), state_dir=str(tmp_path / "state"),
    )


def test_dry_run_excluded_from_best_worst_mean_val_bpb(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    # An unbeatable, fake val_bpb from a dry_run report...
    _write_report(reports_dir, "report_0000", val_bpb=0.001, status="dry_run")
    # ...alongside real runs whose true best/worst is much higher.
    _write_report(reports_dir, "report_0001", val_bpb=1.2, status="remote_ok")
    _write_report(reports_dir, "report_0002", val_bpb=1.5, status="remote_ok")

    agent3 = _make_agent3(tmp_path)
    summary = agent3.analyze_and_summarize(["report_0000", "report_0001", "report_0002"])
    text = (tmp_path / "reports" / "agent3_summaries" / f"{summary.summary_id}.md").read_text()

    assert "Best/Worst finite val_bpb: 1.200000 / 1.500000" in text
    assert "0.001" not in text
    assert "Finite val_bpb runs: 2" in text  # the dry_run report isn't counted either


def test_simulated_excluded_from_best_worst_mean_val_bpb(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    _write_report(reports_dir, "report_0000", val_bpb=0.001, status="simulated")
    _write_report(reports_dir, "report_0001", val_bpb=1.2, status="remote_ok")

    agent3 = _make_agent3(tmp_path)
    summary = agent3.analyze_and_summarize(["report_0000", "report_0001"])
    text = (tmp_path / "reports" / "agent3_summaries" / f"{summary.summary_id}.md").read_text()

    assert "Best/Worst finite val_bpb: 1.200000 / 1.200000" in text
    assert "0.001" not in text


def test_dry_run_excluded_from_elite_recommendations(tmp_path):
    reports_dir = tmp_path / "reports" / "agent2_reports"
    # dry_run's fake val_bpb (0.001) would be the single "elite" candidate
    # if not filtered, dragging its unrealistic scalar_lr into the
    # geometric-mean recommendation Agent 1 reads.
    _write_report(reports_dir, "report_0000", val_bpb=0.001, status="dry_run",
                   hyperparams=_base_hyperparams(scalar_lr=999.0))
    for i in range(1, 5):
        _write_report(reports_dir, f"report_{i:04d}", val_bpb=1.0 + i * 0.1, status="remote_ok",
                       hyperparams=_base_hyperparams(scalar_lr=0.5))

    agent3 = _make_agent3(tmp_path)
    report_ids = [f"report_{i:04d}" for i in range(5)]
    summary = agent3.analyze_and_summarize(report_ids)
    text = (tmp_path / "reports" / "agent3_summaries" / f"{summary.summary_id}.md").read_text()

    assert "999" not in text
    assert "scalar_lr (geometric mean from elite runs): 0.5" in text

    # Status distribution reporting still legitimately counts the dry_run
    # report -- only the numeric aggregates exclude it.
    assert "- dry_run: 1" in text
