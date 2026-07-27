"""Agent 2: XAI Specialist - Analyzes model behavior and generates reports."""

import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
from agents.xai_methods.fast_methods import FastXAIMethods
from agents.protocols import AnalysisEvidence
from agents.agent1_training_specialist import LR_DEFAULTS
from state.results_analysis import (
    HYPERPARAM_COLUMNS,
    hyperparameter_correlations,
    importance_from_correlations,
    load_results,
)
from state.visualize import (
    chart_head_importance_heatmap,
    chart_hyperparameter_importance,
    chart_layer_scalars,
)
import yaml


class Agent2XAISpecialist:
    """Analyzes trained models using XAI methods and generates interpretability reports."""

    def __init__(self, config_path: str = "agents_config.yaml"):
        self.config = self._load_config(config_path)
        self.agent2_config = self.config.get("agent2", {})
        self.use_llm = self.agent2_config.get("use_llm", False)
        self.xai_method = self.agent2_config.get("xai_method", "fast")
        self.generate_charts = bool(self.agent2_config.get("generate_charts", True))
        self.reports_dir = Path("reports/agent2_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.visuals_dir = Path("reports/visuals")

        # Initialize XAI methods
        if self.xai_method == "fast":
            self.xai = FastXAIMethods()
        else:
            raise ValueError(f"Unknown XAI method: {self.xai_method}")

        self.report_counter = self._count_existing_reports()
        self.results_path = Path("results.tsv")
        self.all_impacts = []  # Real head-ablation impacts, one dict per run that had them (for stuck detection)

        # Claude client (lazy loaded if needed)
        self.claude = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _count_existing_reports(self) -> int:
        """Count existing reports to continue numbering."""
        reports = list(self.reports_dir.glob("report_*.md"))
        return len(reports)

    def _init_claude(self):
        """Lazy-load Claude client."""
        if self.claude is not None:
            return

        try:
            from anthropic import Anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.claude = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed")

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
        stuck_signal = (
            (not math.isfinite(val_bpb))
            or (val_bpb > 1.32)
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
            )
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
            "## Structured Metrics (for Agent 3)",
            "```json",
            json.dumps(structured, indent=2, sort_keys=True),
            "```",
        ])
        return "\n".join(lines) + "\n"
