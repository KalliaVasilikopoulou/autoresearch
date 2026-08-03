"""Agent 2: XAI Specialist - Analyzes model behavior and generates reports."""

import os
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Any, Optional, List
from agents import claude_cli
from agents.xai_methods.fast_methods import FastXAIMethods
from agents.protocols import AnalysisEvidence
from agents.agent1_training_specialist import LR_DEFAULTS
from state.results_analysis import (
    HYPERPARAM_COLUMNS,
    hyperparameter_correlations,
    importance_from_correlations,
    load_results,
    top_quartile_by_val_bpb,
)
from state.visualize import (
    chart_head_importance_heatmap,
    chart_hyperparameter_importance,
    chart_layer_scalars,
    chart_token_fingerprint,
)
import yaml


# stuck-signal val_bpb check: how much worse than the elite reference (see
# _elite_val_bpb_reference below) counts as "stuck." Replaces a bare
# hardcoded val_bpb > 1.32 that turned out to flag ~95% of real
# (non-dry_run) runs ever recorded -- 1.32 was miscalibrated against real
# training's actual achievable range from the start, not something that
# regressed. 0.15 == "more than 15% worse than the median of the best 25%
# of real runs ever."
STUCK_VAL_BPB_MARGIN = 0.15


class Agent2XAISpecialist:
    """Analyzes trained models using XAI methods and generates interpretability reports."""

    def __init__(
        self,
        config_path: str = "agents_config.yaml",
        root_dir: Optional[str] = None,
        reports_dir: Optional[str] = None,
    ):
        """root_dir/reports_dir let callers (tests, Orchestrator) redirect
        every file this class touches instead of always hitting the repo
        root. Defaults preserve the original cwd-relative behavior exactly.
        """
        self.config = self._load_config(config_path)
        self.agent2_config = self.config.get("agent2", {})
        self.use_llm = self.agent2_config.get("use_llm", False)
        self.xai_method = self.agent2_config.get("xai_method", "fast")
        self.generate_charts = bool(self.agent2_config.get("generate_charts", True))
        _root = Path(root_dir) if root_dir else Path(".")
        _reports = Path(reports_dir) if reports_dir else Path("reports")
        self.reports_dir = _reports / "agent2_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.visuals_dir = _reports / "visuals"
        self.results_path = _root / "results.tsv"

        # Initialize XAI methods
        if self.xai_method == "fast":
            self.xai = FastXAIMethods()
        else:
            raise ValueError(f"Unknown XAI method: {self.xai_method}")

        self.report_counter = self._count_existing_reports()
        self.all_impacts = []  # Real head-ablation impacts, one dict per run that had them (for stuck detection)

        # LLM/copilot integration (dev/checks.txt item 4): shared campaign
        # budget across agent1/2/3 -- see agents/claude_cli.py's docstring.
        llm_config = self.config.get("llm", {})
        self._llm_backend = llm_config.get("backend", "cli")
        self._llm_model = llm_config.get("model", "sonnet")
        self._llm_campaign_budget_usd = float(llm_config.get("campaign_budget_usd", 5.0))
        self._llm_max_call_budget_usd = float(llm_config.get("max_call_budget_usd", 0.20))
        self._llm_usage_path = llm_config.get("usage_log_path", "state/llm_usage.json")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _count_existing_reports(self) -> int:
        """Count existing reports to continue numbering."""
        reports = list(self.reports_dir.glob("report_*.md"))
        return len(reports)

    def _elite_val_bpb_reference(self) -> Optional[float]:
        """Median val_bpb of the best 25% of real historical runs (see
        state/results_analysis.py::top_quartile_by_val_bpb -- the exact same
        elite selection Agent 3's hyperparameter recommendations use, so
        "elite" means one consistent thing across the system). The median
        of several elite runs, not just the single best, so one lucky
        outlier can't single-handedly set (or blow) the bar. None when
        load_results() has no real (dry_run/simulated already excluded)
        finite-val_bpb history yet -- callers must not fabricate a
        threshold from zero data, and fall back to the ablation-pattern
        signal alone.
        """
        rows = load_results(self.results_path)
        candidates = [
            (row["val_bpb"], None) for row in rows
            if isinstance(row.get("val_bpb"), (int, float)) and math.isfinite(row["val_bpb"])
        ]
        elite = top_quartile_by_val_bpb(candidates)
        if not elite:
            return None
        return statistics.median(val_bpb for val_bpb, _ in elite)

    def analyze_result(self, result_payload: Dict[str, Any]) -> Optional[AnalysisEvidence]:
        """Analyze a completed training result and emit structured evidence."""
        run_id = result_payload.get('run_id', 'unknown')
        val_bpb = result_payload.get("val_bpb", float("inf"))
        status = result_payload.get("status", "unknown")
        print(f"[Agent 2] Starting XAI analysis of {run_id}")
        print(f"[Agent 2]   val_bpb={val_bpb:.6f}, status={status}")

        hyperparams = result_payload.get("hyperparams", {})
        run_metrics = result_payload.get("metadata", {}) if isinstance(result_payload.get("metadata", {}), dict) else {}
        print(f"[Agent 2]   Hyperparams: n_layer={hyperparams.get('n_layer')}, n_embd={hyperparams.get('n_embd')}, "
              f"matrix_lr={float(hyperparams.get('matrix_lr', LR_DEFAULTS['matrix_lr'])):.2e}")

        report_id = f"report_{self.report_counter:04d}"

        head_ablation_impacts = run_metrics.get("head_ablation_impacts") or {}
        if head_ablation_impacts:
            self.all_impacts.append(head_ablation_impacts)
        # Ablation-pattern stuck detection (real) complements the val_bpb-threshold
        # heuristic below — either signal is enough to trigger Agent 1's radical change.
        ablation_stuck = self.xai.detect_stuck_signal(self.all_impacts) if self.all_impacts else False
        elite_reference = self._elite_val_bpb_reference()
        stuck_signal = (
            (not math.isfinite(val_bpb))
            or (elite_reference is not None and val_bpb > elite_reference * (1 + STUCK_VAL_BPB_MARGIN))
            or (status in {"remote_error", "simulated"})
            or ablation_stuck
        )
        confidence = 0.9 if status in {"remote_ok", "ok"} and math.isfinite(val_bpb) else 0.72

        # Placeholder evidence.important_heads/hyperparameter_importance below are
        # immediately replaced with real (or explicitly omitted) values inside
        # _render_markdown_report — kept here only so the dataclass is never
        # constructed with unset required fields.
        evidence = AnalysisEvidence(
            report_id=report_id,
            model_id=result_payload.get("run_id", report_id),
            important_heads=[],
            hyperparameter_importance={},
            stuck_signal=stuck_signal,
            confidence=confidence,
            notes=[
                f"Observed validation bpb {val_bpb:.6f}",
                f"Run status: {status}",
                f"Training time: {run_metrics.get('training_time', 0)}s",
            ],
        )

        report_path = self.reports_dir / f"{report_id}.md"
        report_path.write_text(
            self._render_markdown_report(
                evidence,
                hyperparams,
                val_bpb,
                status=status,
                run_metrics=run_metrics,
            ),
            encoding="utf-8",
        )

        self.report_counter += 1
        print(f"[Agent 2] Analysis complete: {report_path} (stuck={stuck_signal})")
        return evidence

    def _real_hyperparameter_importance(self) -> Dict[str, Dict[str, Any]]:
        """Spearman correlation of each hyperparameter against val_bpb across
        every historical run in results.tsv. Returns {} for parameters with
        fewer than 4 comparable runs — never a fabricated number.
        """
        rows = load_results(self.results_path)
        return hyperparameter_correlations(rows)

    def _real_head_importance(self, run_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Real per-head ablation impacts measured by train.py this run, if any
        (see FastXAIMethods.top_k_ablation_study). Empty when ablation didn't
        run (e.g. disabled or it errored) — never backfilled with a guess.
        """
        impacts = run_metrics.get("head_ablation_impacts")
        return dict(impacts) if isinstance(impacts, dict) else {}

    def _get_llm_interpretation(self, structured: Dict[str, Any]) -> Optional[str]:
        """A short, LLM-generated plain-language read of this run's real,
        measured XAI evidence (dev/checks.txt item 4) -- never a substitute
        for the measured data above it in the report, just a summary of it.
        None (not fabricated) when the CLI is unavailable or the shared
        campaign budget is exhausted; see agents/claude_cli.py.
        """
        top_heads = structured.get("head_importance") or {}
        top_heads_str = ", ".join(
            f"{h}={v:.4f}" for h, v in sorted(top_heads.items(), key=lambda kv: -abs(kv[1]))[:5]
        ) or "none measured this run"

        importance = structured.get("hyperparameter_importance") or {}
        importance_str = ", ".join(f"{k}={v:.3f}" for k, v in importance.items()) or "insufficient history yet"

        fingerprint = structured.get("token_fingerprint") or {}
        slope, induction = fingerprint.get("attn_distance_slope"), fingerprint.get("induction_score")
        fingerprint_str = (
            f"attn_distance_slope={slope:.4f}, induction_score={induction:.4f}"
            if isinstance(slope, (int, float)) and isinstance(induction, (int, float))
            else "not measured this run"
        )

        prompt = f"""Here is one training run's real, measured XAI evidence -- interpret it in
2-4 plain-language sentences for a researcher skimming many reports. Do not
invent numbers not given below; if evidence is thin, say so.

val_bpb: {structured.get('val_bpb')}
stuck_signal: {structured.get('stuck_signal')}
top head-ablation impacts: {top_heads_str}
hyperparameter importance (Spearman |r| vs val_bpb): {importance_str}
token-level fingerprint: {fingerprint_str}"""

        return claude_cli.call_with_budget(
            prompt, call_site="agent2_report_interpretation",
            model=self._llm_model,
            campaign_budget_usd=self._llm_campaign_budget_usd,
            max_call_budget_usd=self._llm_max_call_budget_usd,
            usage_path=self._llm_usage_path,
            backend=self._llm_backend,
        )

    def _render_markdown_report(
        self,
        evidence: AnalysisEvidence,
        hyperparams: Dict[str, Any],
        val_bpb: float,
        status: str = "unknown",
        run_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        run_metrics = run_metrics or {}
        n_layer = int(hyperparams.get("n_layer", 12) or 12)
        n_head = int(hyperparams.get("n_head", 8) or 8)

        # Real per-head signal: only present when train.py actually ran the
        # ablation study this run (see FastXAIMethods.top_k_ablation_study).
        # Empty, not fabricated, when it didn't.
        head_impacts = self._real_head_importance(run_metrics)
        ablation_ran = bool(head_impacts)

        # Real cross-run signal: correlation of each hyperparameter against
        # historical val_bpb. Params with too little history are simply
        # absent, not zero-filled.
        hyper_correlations = self._real_hyperparameter_importance()
        hyper_importance = importance_from_correlations(hyper_correlations)

        # Real, zero-extra-cost per-layer signal: already-trained scalar
        # parameters (see train.py). Always available (unless extraction
        # itself failed, in which case it's absent, not invented).
        layer_scalars = run_metrics.get("interpretable_scalars") or {}

        # Real, token-level signal from an analysis-only forward pass (see
        # agents/xai_methods/token_methods.py) -- only present when
        # token_xai_enabled was on for this run (it costs real GPU time,
        # unlike layer_scalars above). Absent, not fabricated, otherwise.
        token_fingerprint = run_metrics.get("token_fingerprint") or {}
        evidence.token_fingerprint = token_fingerprint

        evidence.important_heads = [
            {"head": head, "impact": impact}
            for head, impact in sorted(
                head_impacts.items(), key=lambda item: abs(item[1]), reverse=True
            )[: min(12, len(head_impacts))]
        ]
        evidence.hyperparameter_importance = hyper_importance

        layer_to_values: Dict[int, List[float]] = {idx: [] for idx in range(n_layer)}
        for key, impact in head_impacts.items():
            layer_idx = int(key.split("_")[0][1:])
            layer_to_values.setdefault(layer_idx, []).append(impact)

        abs_layer_mass = {
            idx: sum(abs(v) for v in values) for idx, values in layer_to_values.items()
        }
        total_abs_mass = sum(abs_layer_mass.values())
        layer_share = (
            {idx: (100.0 * abs_layer_mass[idx] / total_abs_mass) for idx in range(n_layer)}
            if total_abs_mass > 0 else {}
        )

        sorted_impacts = sorted(head_impacts.values())

        metadata = {
            "status": status,
            "training_time": run_metrics.get("training_time"),
            "peak_vram_mb": run_metrics.get("peak_vram_mb"),
            "mfu_percent": run_metrics.get("mfu_percent"),
            "num_params_M": run_metrics.get("num_params_M"),
            "num_steps": run_metrics.get("num_steps"),
            # Only present when holdout_eval was requested for this run (see
            # scripts/holdout_eval.py) -- already parsed out of train.py's
            # stdout by both the local and remote paths, just never surfaced
            # into this structured block before, which is the only reason
            # Agent 3's history-based charts never saw it.
            "holdout_val_bpb": run_metrics.get("holdout_val_bpb"),
        }
        structured = {
            "model_id": evidence.model_id,
            "report_id": evidence.report_id,
            "stuck_signal": evidence.stuck_signal,
            "confidence": evidence.confidence,
            "val_bpb": val_bpb,
            "hyperparams": hyperparams,
            "hyperparameter_importance": hyper_importance,
            "hyperparameter_importance_sample_size": {
                param: info["n"] for param, info in hyper_correlations.items()
            },
            "ablation_ran": ablation_ran,
            "head_importance": head_impacts,
            "layer_importance_share_pct": {
                str(layer_idx): round(layer_share[layer_idx], 4) for layer_idx in layer_share
            },
            "layer_scalars": layer_scalars,
            "token_fingerprint": token_fingerprint,
            "metadata": metadata,
        }

        lines = [
            f"# XAI Analysis Report: {evidence.model_id}",
            "",
            "## Model Configuration",
            f"- Model ID: {evidence.model_id}",
            f"- Status: {status}",
            f"- Validation bpb: {val_bpb:.6f}" if math.isfinite(val_bpb) else "- Validation bpb: inf",
            "- Hyperparameters:",
        ]
        for key, value in hyperparams.items():
            lines.append(f"  - {key}: {value}")
        lines.extend([
            "",
            "## Evidence Summary",
            f"- Stuck signal: {'yes' if evidence.stuck_signal else 'no'}",
            f"- Confidence: {evidence.confidence:.2f}",
            f"- Head-level ablation ran this run: {'yes' if ablation_ran else 'no (see model_hyperparams.yaml: ablation_k)'}",
            "- Important heads (measured via ablation):" if ablation_ran else "- Important heads: unavailable (ablation did not run)",
        ])
        for item in evidence.important_heads:
            lines.append(f"  - {item['head']}: {item['impact']:.6f}")
        lines.extend([
            "",
            "## Hyperparameter Importance (Spearman |r| vs val_bpb, from results.tsv history)",
        ])
        if hyper_importance:
            for param, score in hyper_importance.items():
                n = hyper_correlations[param]["n"]
                r = hyper_correlations[param]["correlation"]
                lines.append(f"- {param}: importance={score:.6f} (r={r:+.4f}, n={n})")
        else:
            lines.append("- Insufficient historical runs (need >=4 per parameter) — no importance scores yet")
        missing_params = [p for p in HYPERPARAM_COLUMNS if p not in hyper_importance]
        if missing_params:
            lines.append(f"- Not enough history yet for: {', '.join(missing_params)}")

        if self.generate_charts:
            try:
                chart_path = chart_hyperparameter_importance(
                    hyper_importance,
                    {p: info["n"] for p, info in hyper_correlations.items()},
                    self.visuals_dir / f"{evidence.report_id}_importance.png",
                )
                if chart_path:
                    lines.extend(["", f"![Hyperparameter importance](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 2] Chart generation (importance) failed: {_e}")

        lines.extend([
            "",
            "## Per-Layer Interpretable Scalars (measured, zero extra GPU cost)",
        ])
        if layer_scalars:
            resid_lambdas = layer_scalars.get("resid_lambdas", [])
            x0_lambdas = layer_scalars.get("x0_lambdas", [])
            ve_gate_norm = layer_scalars.get("ve_gate_norm", {})
            lines.append("| Layer | resid_lambda | x0_lambda | ve_gate_norm |")
            lines.append("|------:|-------------:|----------:|-------------:|")
            for layer_idx in range(n_layer):
                resid = resid_lambdas[layer_idx] if layer_idx < len(resid_lambdas) else float("nan")
                x0 = x0_lambdas[layer_idx] if layer_idx < len(x0_lambdas) else float("nan")
                ve = ve_gate_norm.get(str(layer_idx))
                ve_str = f"{ve:.6f}" if isinstance(ve, (int, float)) else "n/a"
                lines.append(f"| L{layer_idx} | {resid:.6f} | {x0:.6f} | {ve_str} |")
        else:
            lines.append("- Unavailable (extraction failed or run predates this feature)")

        if self.generate_charts and layer_scalars:
            try:
                chart_path = chart_layer_scalars(
                    layer_scalars, n_layer, self.visuals_dir / f"{evidence.report_id}_layers.png",
                )
                if chart_path:
                    lines.extend(["", f"![Per-layer interpretable scalars](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 2] Chart generation (layers) failed: {_e}")

        lines.extend([
            "",
            "## Token-Level Behavioral Fingerprint (Tier 2, analysis-only forward pass)",
        ])
        if token_fingerprint:
            entropy = token_fingerprint.get("attn_entropy", [])
            distance = token_fingerprint.get("attn_distance", [])
            dla = token_fingerprint.get("dla", [])
            lines.append(f"- attn_distance_slope: {token_fingerprint.get('attn_distance_slope', float('nan')):.6f} "
                         "(does attention reach grow with depth?)")
            lines.append(f"- induction_score: {token_fingerprint.get('induction_score', float('nan')):.6f} "
                         "(near-zero is expected on short runs -- induction heads emerge late in training)")
            lines.append("| Layer | attn_entropy | attn_distance | dla |")
            lines.append("|------:|-------------:|--------------:|----:|")
            for layer_idx in range(n_layer):
                e = entropy[layer_idx] if layer_idx < len(entropy) else float("nan")
                d = distance[layer_idx] if layer_idx < len(distance) else float("nan")
                a = dla[layer_idx] if layer_idx < len(dla) else float("nan")
                lines.append(f"| L{layer_idx} | {e:.6f} | {d:.6f} | {a:.6f} |")
            pos_saliency = token_fingerprint.get("pos_saliency", [])
            if pos_saliency:
                lines.append("")
                lines.append("pos_saliency (mean |grad x input|, by distance-back from the predicted position): "
                             + ", ".join(f"{v:.6f}" for v in pos_saliency))
        else:
            lines.append("- Unavailable (token_xai_enabled was off for this run, or extraction failed)")

        if self.generate_charts and token_fingerprint:
            try:
                chart_path = chart_token_fingerprint(
                    token_fingerprint, n_layer, self.visuals_dir / f"{evidence.report_id}_token_fingerprint.png",
                )
                if chart_path:
                    lines.extend(["", f"![Token-level behavioral fingerprint](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 2] Chart generation (token fingerprint) failed: {_e}")

        if ablation_ran:
            lines.extend([
                "",
                "## Attention Importance Distribution (Per Layer, from ablation)",
                "| Layer | Mean | Min | Max | Layer Share (%) |",
                "|------:|-----:|----:|----:|-----------------:|",
            ])
            for layer_idx in sorted(layer_to_values.keys()):
                values = sorted(layer_to_values[layer_idx])
                if not values:
                    continue
                share = layer_share.get(layer_idx, 0.0)
                lines.append(
                    f"| L{layer_idx} | {sum(values)/len(values):.6f} | {values[0]:.6f} | {values[-1]:.6f} | {share:.2f} |"
                )

            lines.extend([
                "",
                "## Ablated Heads (top-K by |c_proj column norm|, ranked by measured impact)",
                "| Head | Impact (bpb drop from baseline) |",
                "|------|----------------------------------:|",
            ])
            for head, impact in sorted(head_impacts.items(), key=lambda item: abs(item[1]), reverse=True):
                lines.append(f"| {head} | {impact:.6f} |")
            lines.extend([
                f"- Mean/Median impact: {sum(sorted_impacts)/len(sorted_impacts):.6f} / {sorted_impacts[len(sorted_impacts)//2]:.6f}",
            ])

            if self.generate_charts:
                try:
                    chart_path = chart_head_importance_heatmap(
                        head_impacts, n_layer, n_head, self.visuals_dir / f"{evidence.report_id}_heads.png",
                    )
                    if chart_path:
                        lines.extend(["", f"![Head importance heatmap](../visuals/{chart_path.name})"])
                except Exception as _e:
                    print(f"[Agent 2] Chart generation (heads) failed: {_e}")

        lines.extend([
            "",
            "## Notes",
        ])
        for note in evidence.notes:
            lines.append(f"- {note}")

        lines.extend([
            "",
            "## LLM Interpretation",
        ])
        if self.use_llm:
            llm_text = self._get_llm_interpretation(structured)
            lines.append(llm_text if llm_text else
                         "- Unavailable this run (CLI not reachable, or campaign LLM budget exhausted)")
        else:
            lines.append("- Disabled (agent2.use_llm is false)")

        lines.extend([
            "",
            "## Structured Metrics (for Agent 3)",
            "```json",
            json.dumps(structured, indent=2, sort_keys=True),
            "```",
        ])
        return "\n".join(lines) + "\n"
