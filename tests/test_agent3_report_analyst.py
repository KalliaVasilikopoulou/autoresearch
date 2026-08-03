"""Synthetic-data tests confirming Agent 3's own statistics (Best/Worst/
Mean val_bpb, elite-run hyperparameter recommendations) exclude dry_run/
simulated reports. Agent 3 reads reports/agent2_reports/*.md directly
(a separate data source from results.tsv/load_results), so it needed its
own filter -- see Agent3ReportAnalyst._is_synthetic.
"""

import json

from agents.agent3_report_analyst import Agent3ReportAnalyst, _read_text_tolerant


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


# --- _strip_markdown_section / _build_prompt_summary (prompt-leanness) ---

def test_strip_markdown_section_removes_named_section_only(tmp_path):
    agent3 = _make_agent3(tmp_path)
    text = "## A\nkeep a\n\n## B\ndrop b\n\n## C\nkeep c\n"
    result = agent3._strip_markdown_section(text, "## B")
    assert "keep a" in result
    assert "keep c" in result
    assert "drop b" not in result
    assert "## B" not in result


def test_strip_markdown_section_noop_when_heading_absent(tmp_path):
    agent3 = _make_agent3(tmp_path)
    text = "## A\nkeep a\n"
    assert agent3._strip_markdown_section(text, "## Not Present") == text


def test_strip_markdown_section_handles_last_section(tmp_path):
    agent3 = _make_agent3(tmp_path)
    text = "## A\nkeep a\n\n## B\ndrop b\nmore b\n"
    result = agent3._strip_markdown_section(text, "## B")
    assert result.strip() == "## A\nkeep a"


def test_build_prompt_summary_strips_charts_usage_and_boilerplate(tmp_path):
    agent3 = _make_agent3(tmp_path)
    full = (
        "## Batch Scope\n- New reports: 3\n\n"
        "![val_bpb trend](../visuals/x.png)\n\n"
        "## LLM Usage This Campaign\n- 5 call(s), cumulative cost $0.50\n\n"
        "## Behavioral Fingerprint Clusters (Tier 3)\n"
        "- Volatility = total variation of each run's normalized attn_distance curve (0 = perfectly smooth/monotonic, higher = zig-zags more). Uses every fingerprint-bearing run.\n"
        "- Spearman correlation: +0.1234\n"
    )
    result = agent3._build_prompt_summary(full, sorted_layers=[])
    assert "New reports: 3" in result  # unrelated content preserved
    assert "![" not in result
    assert "LLM Usage This Campaign" not in result
    assert "Volatility = total variation" not in result
    assert "Spearman correlation: +0.1234" in result  # the actual number is kept


def test_build_prompt_summary_condenses_layer_table(tmp_path):
    agent3 = _make_agent3(tmp_path)
    full = "## Batch Scope\n- x\n"
    sorted_layers = [(str(i), [80.0] if i == 0 else [0.1]) for i in range(10)]
    result = agent3._build_prompt_summary(full, sorted_layers)
    assert "L0: 80.00%" in result
    assert "9 other layer(s) at <0.5% share (dead weight)" in result


def test_build_prompt_summary_no_layer_block_when_no_layers(tmp_path):
    agent3 = _make_agent3(tmp_path)
    result = agent3._build_prompt_summary("## Batch Scope\n- x\n", sorted_layers=[])
    assert "Layer-Level Importance" not in result


# --- _load_annotations (campaign-level chart markers) --------------------

def test_load_annotations_missing_file_returns_empty_list(tmp_path):
    agent3 = _make_agent3(tmp_path)
    assert agent3._load_annotations() == []


def test_load_annotations_corrupt_file_returns_empty_list(tmp_path):
    agent3 = _make_agent3(tmp_path)
    agent3.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    agent3.annotations_path.write_text("not valid json{{{")
    assert agent3._load_annotations() == []


def test_load_annotations_malformed_annotations_key_returns_empty_list(tmp_path):
    agent3 = _make_agent3(tmp_path)
    agent3.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    agent3.annotations_path.write_text(json.dumps({"annotations": "not a list"}))
    assert agent3._load_annotations() == []


def test_load_annotations_reads_real_entries(tmp_path):
    agent3 = _make_agent3(tmp_path)
    agent3.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    agent3.annotations_path.write_text(json.dumps({
        "annotations": [{"report_index": 380, "label": "EI-guided search began"}]
    }))
    result = agent3._load_annotations()
    assert result == [{"report_index": 380, "label": "EI-guided search began"}]


def test_analyze_and_summarize_passes_annotations_to_trend_chart(tmp_path, monkeypatch):
    """End-to-end: a real annotations file actually reaches the chart call,
    not just _load_annotations in isolation."""
    reports_dir = tmp_path / "reports" / "agent2_reports"
    _write_report(reports_dir, "report_0000", val_bpb=1.0, status="remote_ok")

    agent3 = _make_agent3(tmp_path)
    agent3.generate_charts = True
    agent3.annotations_path.parent.mkdir(parents=True, exist_ok=True)
    agent3.annotations_path.write_text(json.dumps({
        "annotations": [{"report_index": 0, "label": "marker"}]
    }))

    captured = {}
    import agents.agent3_report_analyst as agent3_module

    def fake_chart_val_bpb_trend(all_metrics, noise_floor_path, path, annotations=None):
        captured["annotations"] = annotations
        return None

    monkeypatch.setattr(agent3_module, "chart_val_bpb_trend", fake_chart_val_bpb_trend)
    agent3.analyze_and_summarize(["report_0000"])

    assert captured["annotations"] == [{"report_index": 0, "label": "marker"}]


# --- _read_text_tolerant (real production crash: reading historical .md --
# files written before the encoding="utf-8" fix landed, which are real
# cp1252 bytes on disk, not UTF-8) -------------------------------------

def test_read_text_tolerant_reads_utf8_files(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("n_layer≈6 — real content", encoding="utf-8")
    assert _read_text_tolerant(path) == "n_layer≈6 — real content"


def test_read_text_tolerant_recovers_legacy_cp1252_files(tmp_path):
    """Files written before the encoding fix are real cp1252 bytes on disk
    (an em-dash is 0x97 -- the exact byte from the actual production
    crash). Must recover the correct original content, not just avoid
    crashing."""
    path = tmp_path / "report.md"
    text = "Converging steadily — no anomalies."
    path.write_bytes(text.encode("cp1252"))
    assert _read_text_tolerant(path) == text


def test_read_text_tolerant_never_raises_on_undefined_cp1252_byte(tmp_path):
    """0x81 is undefined in cp1252 too (the exact byte behind the earlier
    real claude_cli.py crash) -- must degrade via errors="replace" rather
    than raise UnicodeDecodeError a third time."""
    path = tmp_path / "report.md"
    path.write_bytes(b"before \x81 after")
    result = _read_text_tolerant(path)  # must not raise
    assert "before" in result and "after" in result
