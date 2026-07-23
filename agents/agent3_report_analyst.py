"""Agent 3: Report Analyst - Aggregates XAI reports and finds trends."""

import os
import json
import math
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None

from agents.protocols import SummaryEvidence


class Agent3ReportAnalyst:
    """Analyzes and aggregates reports from Agent 2 into strategic summaries."""

    def __init__(self, config_path: str = "agents_config.yaml"):
        self.config = self._load_config(config_path)
        self.agent3_config = self.config.get("agent3", {})
        self.use_llm = self.agent3_config.get("use_llm", False)
        self.batch_size = self.agent3_config.get("batch_size", 3)
        self.preserve_history = self.agent3_config.get("preserve_history", True)

        self.summaries_dir = Path("reports/agent3_summaries")
        self.reports_dir = Path("reports/agent2_reports")
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        self.summary_counter = self._count_existing_summaries()
        self.claude = None
        self._latest_summary_evidence: Optional[SummaryEvidence] = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        if yaml is None:
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _count_existing_summaries(self) -> int:
        """Count existing summaries."""
        summaries = list(self.summaries_dir.glob("summary_*.md"))
        return len(summaries)

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

    def should_create_summary(self, report_count: int) -> bool:
        """Check if batch is complete."""
        return report_count % self.batch_size == 0

    def analyze_and_summarize(self, new_report_ids: List[str]) -> SummaryEvidence:
        """
        Create summary from batch of new reports + previous summary.

        Args:
            new_report_ids: List of report IDs to include

        Returns:
            summary_id
        """
        print(f"[Agent 3] Starting summary generation...")
        print(f"[Agent 3]   Reports to include: {new_report_ids}")

        # Load new reports
        new_reports = []
        for report_id in new_report_ids:
            report_path = self.reports_dir / f"{report_id}.md"
            if report_path.exists():
                with open(report_path, "r") as f:
                    new_reports.append((report_id, f.read()))
                print(f"[Agent 3]   Loaded: {report_id}")

        # Load previous summary if exists
        prev_summary = ""
        prev_summary_id = None
        if self.summary_counter > 0:
            prev_summary_path = (
                self.summaries_dir / f"summary_{self.summary_counter - 1:04d}.md"
            )
            if prev_summary_path.exists():
                with open(prev_summary_path, "r") as f:
                    prev_summary = f.read()
                    prev_summary_id = f"summary_{self.summary_counter - 1:04d}"
                print(f"[Agent 3]   Previous summary: {prev_summary_id}")

        summary_id = f"summary_{self.summary_counter:04d}"
        print(f"[Agent 3]   Generating {summary_id}...")
        summary_content, aggregate = self._generate_summary(new_reports, prev_summary)

        summary = SummaryEvidence(
            summary_id=summary_id,
            batch_size=len(new_reports),
            stable_patterns=aggregate.get("stable_patterns", []),
            conflicting_signals=aggregate.get("conflicting_signals", []),
            recommended_hyperparams=aggregate.get("recommended_hyperparams", {}),
            reasoning=aggregate.get("reasoning", []),
        )
        self._latest_summary_evidence = summary

        summary_path = self.summaries_dir / f"{summary_id}.md"
        with open(summary_path, "w") as f:
            f.write(summary_content)

        self.summary_counter += 1
        print(f"[Agent 3] Summary complete: {summary_path}")
        print(f"[Agent 3]   Patterns found: {len(summary.stable_patterns)}, Conflicts: {len(summary.conflicting_signals)}")

        return summary

    def _generate_summary(
        self, new_reports: List[tuple], prev_summary: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate statistical summary (+ optional Claude narrative)."""

        # ALWAYS: Generate statistical summary
        summary, aggregate = self._format_statistical_summary(new_reports, prev_summary)

        # OPTIONAL: If LLM enabled, add Claude narrative
        if self.use_llm:
            try:
                self._init_claude()
                narrative = self._get_claude_narrative(summary)
                summary += f"\n\n## Strategic Narrative\n{narrative}\n"
            except Exception as e:
                summary += f"\n\n## Strategic Narrative\nFailed to generate: {e}\n"

        return summary, aggregate

    def _extract_structured_metrics(self, report_content: str) -> Dict[str, Any]:
        """Extract structured report metrics (JSON block) with fallback parsing."""
        json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", report_content, flags=re.DOTALL)
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict) and "model_id" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

        # Fallback for older report formats.
        extracted: Dict[str, Any] = {
            "model_id": "unknown",
            "stuck_signal": "Stuck signal: yes" in report_content or "Model Stuck" in report_content,
            "confidence": None,
            "val_bpb": None,
            "status": "unknown",
            "hyperparams": {},
            "hyperparameter_importance": {},
            "head_importance": {},
            "layer_importance_share_pct": {},
            "metadata": {},
        }

        model_match = re.search(r"# XAI Analysis Report:\s*(.+)", report_content)
        if model_match:
            extracted["model_id"] = model_match.group(1).strip()

        status_match = re.search(r"- Status:\s*(.+)", report_content)
        if status_match:
            extracted["status"] = status_match.group(1).strip()

        val_match = re.search(r"validation bpb\s+([0-9eE+\-.]+)", report_content, flags=re.IGNORECASE)
        if val_match:
            try:
                extracted["val_bpb"] = float(val_match.group(1))
            except ValueError:
                pass

        section = None
        for line in report_content.splitlines():
            stripped = line.strip()
            if stripped == "## Hyperparameter Importance":
                section = "hyper_importance"
                continue
            if stripped.startswith("## ") and stripped != "## Hyperparameter Importance":
                section = None

            if section == "hyper_importance" and stripped.startswith("-") and ":" in stripped:
                payload = stripped.lstrip("-").strip()
                key, value = payload.split(":", 1)
                try:
                    extracted["hyperparameter_importance"][key.strip()] = float(value.strip())
                except ValueError:
                    continue

        return extracted

    def _load_all_reports(self) -> List[Tuple[str, str]]:
        """Load all report files currently available for global aggregation."""
        reports: List[Tuple[str, str]] = []
        for path in sorted(self.reports_dir.glob("report_*.md")):
            try:
                reports.append((path.stem, path.read_text()))
            except OSError:
                continue
        return reports

    def _safe_mean(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _safe_std(self, values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = self._safe_mean(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(var)

    def _quantile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = max(0, min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * q))))
        return sorted_vals[idx]

    def _format_statistical_summary(
        self, new_reports: List[tuple], prev_summary: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Format statistical analysis of reports."""

        all_reports = self._load_all_reports()
        all_metrics = [self._extract_structured_metrics(content) for _, content in all_reports]
        batch_metrics = [self._extract_structured_metrics(content) for _, content in new_reports]

        finite_bpbs = [
            float(item.get("val_bpb"))
            for item in all_metrics
            if isinstance(item.get("val_bpb"), (int, float)) and math.isfinite(float(item.get("val_bpb")))
        ]
        statuses = Counter(
            [
                str(item.get("metadata", {}).get("status") or item.get("status") or "unknown")
                for item in all_metrics
            ]
        )
        stuck_count = sum(1 for item in all_metrics if item.get("stuck_signal"))

        all_hyper_importance: Dict[str, List[float]] = {}
        importance_presence: Dict[str, Dict[str, int]] = {}
        for item in all_metrics:
            per_report = item.get("hyperparameter_importance", {}) or {}
            for param, raw_score in per_report.items():
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                all_hyper_importance.setdefault(param, []).append(score)
                slot = importance_presence.setdefault(param, {"important": 0, "non_important": 0})
                if score >= 0.5:
                    slot["important"] += 1
                else:
                    slot["non_important"] += 1

        layer_shares: Dict[str, List[float]] = {}
        head_importance_frequency: Counter = Counter()
        high_impact_rate: List[float] = []

        for item in all_metrics:
            layer_dist = item.get("layer_importance_share_pct", {}) or {}
            for layer_name, share in layer_dist.items():
                try:
                    layer_shares.setdefault(layer_name, []).append(float(share))
                except (TypeError, ValueError):
                    continue

            head_map = item.get("head_importance", {}) or {}
            if isinstance(head_map, dict) and head_map:
                abs_values = [abs(float(v)) for v in head_map.values()]
                threshold = self._quantile(abs_values, 0.9)
                if abs_values:
                    high_hits = sum(1 for v in abs_values if v >= threshold)
                    high_impact_rate.append(100.0 * high_hits / len(abs_values))
                for head, score in head_map.items():
                    try:
                        if abs(float(score)) >= threshold:
                            head_importance_frequency[str(head)] += 1
                    except (TypeError, ValueError):
                        continue

        elite_candidates = []
        for item in all_metrics:
            val = item.get("val_bpb")
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                continue
            hp = item.get("hyperparams", {}) or {}
            if not isinstance(hp, dict):
                continue
            elite_candidates.append((float(val), hp))

        elite_candidates.sort(key=lambda x: x[0])
        elite_count = max(1, len(elite_candidates) // 4) if elite_candidates else 0
        elite = elite_candidates[:elite_count] if elite_count else []

        def _avg_hp(name: str, default: float = 0.0) -> float:
            values = []
            for _, hp in elite:
                raw = hp.get(name)
                if isinstance(raw, (int, float)):
                    values.append(float(raw))
            return self._safe_mean(values) if values else default

        recommended = {}
        if elite:
            lr_values = [float(hp.get("learning_rate")) for _, hp in elite if isinstance(hp.get("learning_rate"), (int, float)) and hp.get("learning_rate") > 0]
            if lr_values:
                geo_lr = math.exp(sum(math.log(v) for v in lr_values) / len(lr_values))
                recommended["learning_rate"] = round(geo_lr, 6)
            layer_avg = _avg_hp("n_layer")
            embd_avg = _avg_hp("n_embd")
            head_avg = _avg_hp("n_head")
            if layer_avg:
                recommended["n_layer"] = int(round(layer_avg))
            if embd_avg:
                recommended["n_embd"] = int(round(embd_avg))
            if head_avg:
                recommended["n_head"] = int(round(head_avg))

        stable_patterns = []
        conflicting_patterns = []
        for param, values in all_hyper_importance.items():
            mean_val = self._safe_mean(values)
            std_val = self._safe_std(values)
            if len(values) >= 3 and mean_val >= 0.55 and std_val <= 0.12:
                stable_patterns.append(
                    f"{param} importance is stable (mean={mean_val:.3f}, std={std_val:.3f})"
                )
            if len(values) >= 3 and std_val >= 0.18:
                conflicting_patterns.append(
                    f"{param} importance is inconsistent (mean={mean_val:.3f}, std={std_val:.3f})"
                )

        if not stable_patterns and all_hyper_importance:
            top_param = max(all_hyper_importance.items(), key=lambda x: self._safe_mean(x[1]))
            stable_patterns.append(
                f"{top_param[0]} is the strongest average signal (mean={self._safe_mean(top_param[1]):.3f})"
            )

        if not conflicting_patterns:
            conflicting_patterns.append("No high-variance hyperparameter importance signal detected")

        summary_lines = [
            f"# Summary Report #{self.summary_counter}",
            "",
            "## Batch Scope",
            f"- New reports in this batch: {len(new_reports)}",
            f"- Total reports analyzed (history): {len(all_reports)}",
            f"- Parsed structured reports: {sum(1 for m in all_metrics if isinstance(m, dict))}",
            "",
            "## Performance and Reliability Statistics",
            f"- Finite val_bpb runs: {len(finite_bpbs)}",
            f"- Stuck signal frequency: {stuck_count}/{len(all_metrics) if all_metrics else 1} ({(100.0 * stuck_count / len(all_metrics)) if all_metrics else 0.0:.1f}%)",
            f"- Mean/Median val_bpb: {self._safe_mean(finite_bpbs):.6f} / {self._quantile(finite_bpbs, 0.5):.6f}" if finite_bpbs else "- Mean/Median val_bpb: N/A",
            f"- Best/Worst finite val_bpb: {min(finite_bpbs):.6f} / {max(finite_bpbs):.6f}" if finite_bpbs else "- Best/Worst finite val_bpb: N/A",
            "- Status distribution:",
        ]
        for status_name, count in statuses.items():
            summary_lines.append(f"  - {status_name}: {count}")

        summary_lines.extend([
            "",
            "## Hyperparameter Importance Statistics",
            "| Hyperparameter | Mean Importance | Std Dev | Important Count (>=0.50) | Non-Important Count (<0.50) |",
            "|----------------|----------------:|--------:|--------------------------:|-----------------------------:|",
        ])
        for param in sorted(all_hyper_importance.keys()):
            values = all_hyper_importance[param]
            presence = importance_presence.get(param, {"important": 0, "non_important": 0})
            summary_lines.append(
                f"| {param} | {self._safe_mean(values):.6f} | {self._safe_std(values):.6f} | {presence['important']} | {presence['non_important']} |"
            )
        if not all_hyper_importance:
            summary_lines.append("| N/A | 0.000000 | 0.000000 | 0 | 0 |")

        summary_lines.extend([
            "",
            "## Attention Importance Statistics",
            f"- Average high-impact head ratio per report (top 10% within report): {self._safe_mean(high_impact_rate):.2f}%" if high_impact_rate else "- Average high-impact head ratio per report: N/A",
            "- Most recurrent high-impact heads (count across reports):",
        ])
        for head, count in head_importance_frequency.most_common(10):
            summary_lines.append(f"  - {head}: {count}")
        if not head_importance_frequency:
            summary_lines.append("  - N/A")

        summary_lines.extend([
            "",
            "## Layer-Level Importance Distribution",
            "| Layer | Mean Share (%) | Std Dev |",
            "|------:|---------------:|--------:|",
        ])
        sorted_layers = sorted(layer_shares.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 999)
        for layer_name, shares in sorted_layers:
            summary_lines.append(
                f"| L{layer_name} | {self._safe_mean(shares):.4f} | {self._safe_std(shares):.4f} |"
            )
        if not sorted_layers:
            summary_lines.append("| N/A | 0.0000 | 0.0000 |")

        summary_lines.extend([
            "",
            "## Recommendations for Agent 1 (Data-Backed)",
            f"- Recommendation sample size (elite runs): {len(elite)}",
        ])
        if recommended:
            if "learning_rate" in recommended:
                summary_lines.append(f"- learning_rate (geometric mean from elite runs): {recommended['learning_rate']}")
            if "n_layer" in recommended:
                summary_lines.append(f"- n_layer (rounded elite average): {recommended['n_layer']}")
            if "n_embd" in recommended:
                summary_lines.append(f"- n_embd (rounded elite average): {recommended['n_embd']}")
            if "n_head" in recommended:
                summary_lines.append(f"- n_head (rounded elite average): {recommended['n_head']}")
        else:
            summary_lines.append("- Insufficient finite historical runs for numeric recommendations")

        summary_lines.extend([
            "",
            "## Strategic Insights",
            "- Stable patterns:",
        ])
        for item in stable_patterns[:5]:
            summary_lines.append(f"  - {item}")
        summary_lines.append("- Conflicting signals:")
        for item in conflicting_patterns[:5]:
            summary_lines.append(f"  - {item}")

        if self.preserve_history and prev_summary:
            summary_lines.extend([
                "",
                "## Previous Summary Continuity",
                "- Preserved history is enabled; this summary supersedes prior generic trends with recalculated statistics.",
            ])

        aggregate = {
            "stable_patterns": stable_patterns[:5],
            "conflicting_signals": conflicting_patterns[:5],
            "recommended_hyperparams": recommended,
            "reasoning": [
                f"Computed from {len(all_reports)} total reports and {len(finite_bpbs)} finite val_bpb runs",
                f"Elite recommendation subset size: {len(elite)}",
                "Importance/non-importance split uses threshold 0.50",
            ],
        }

        return "\n".join(summary_lines) + "\n", aggregate

    def get_latest_summary_object(self) -> Optional[SummaryEvidence]:
        """Return the most recent structured summary object if it exists."""
        if self._latest_summary_evidence is not None:
            return self._latest_summary_evidence
        if self.summary_counter <= 0:
            return None
        summary_id = f"summary_{self.summary_counter - 1:04d}"
        return SummaryEvidence(
            summary_id=summary_id,
            batch_size=self.batch_size,
            stable_patterns=["summary artifact exists on disk"],
            conflicting_signals=[],
            recommended_hyperparams={},
            reasoning=["Load markdown summary for full details"],
        )

    def _get_claude_narrative(self, statistical_summary: str) -> str:
        """Get Claude to create strategic narrative."""
        prompt = f"""Based on this summary, provide a strategic narrative:
1. What patterns emerge across multiple models?
2. What should Agent 1 focus on for next iteration?
3. Are we converging or diverging?

Summary:
{statistical_summary}

Be concise (under 250 words)."""

        try:
            message = self.claude.messages.create(
                model="claude-opus-4-7",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"Error getting Claude narrative: {e}"
