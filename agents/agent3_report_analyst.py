"""Agent 3: Report Analyst - Aggregates XAI reports and finds trends."""

import os
import hashlib
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

from agents import claude_cli
from agents.protocols import SummaryEvidence
from agents.agent1_training_specialist import LR_KEYS
from state import llm_usage
from state.clustering import (
    cluster_attention_trajectories,
    cluster_fingerprints,
    trajectory_smoothness_correlation,
)
from state.results_analysis import SYNTHETIC_STATUSES, top_quartile_by_val_bpb
from state.visualize import (
    chart_attention_trajectory_clusters,
    chart_fingerprint_adjustments_trend,
    chart_fingerprint_clusters,
    chart_hyperparameter_importance_evolution,
    chart_layer_importance_distribution,
    chart_noise_floor_trend,
    chart_pipeline_issues_trend,
    chart_status_distribution,
    chart_token_fingerprint_scalars_evolution,
    chart_val_bpb_trend,
)


def _read_text_tolerant(path: Path) -> str:
    """Reads a report/summary .md file as UTF-8 -- what every write in this
    project uses explicitly now -- falling back to cp1252 (Windows'
    default locale codepage, and what files written before that encoding
    fix landed actually contain on disk) so the historical backlog stays
    readable instead of crashing. A final errors="replace" cp1252 pass is
    the last resort for the handful of byte values cp1252 itself doesn't
    define either (0x81, 0x8D, 0x8F, 0x90, 0x9D) -- never raises
    UnicodeDecodeError.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="cp1252")
        except UnicodeDecodeError:
            return path.read_text(encoding="cp1252", errors="replace")


class Agent3ReportAnalyst:
    """Analyzes and aggregates reports from Agent 2 into strategic summaries."""

    def __init__(
        self,
        config_path: str = "agents_config.yaml",
        state_dir: Optional[str] = None,
        reports_dir: Optional[str] = None,
    ):
        """state_dir/reports_dir let callers (tests, Orchestrator) redirect
        every file this class touches instead of always hitting the repo
        root. Defaults preserve the original cwd-relative behavior exactly.
        """
        self.config = self._load_config(config_path)
        self.agent3_config = self.config.get("agent3", {})
        self.use_llm = self.agent3_config.get("use_llm", False)
        self.batch_size = self.agent3_config.get("batch_size", 3)
        self.preserve_history = self.agent3_config.get("preserve_history", True)
        self.generate_charts = bool(self.agent3_config.get("generate_charts", True))
        self.min_cluster_n = int(self.agent3_config.get("min_cluster_observations", 8))

        _state = Path(state_dir) if state_dir else Path("state")
        _reports = Path(reports_dir) if reports_dir else Path("reports")
        self.summaries_dir = _reports / "agent3_summaries"
        self.reports_dir = _reports / "agent2_reports"
        self.visuals_dir = _reports / "visuals"
        self.noise_floor_path = _state / "noise_floor.json"
        # Sibling directories this class didn't read before (dev/checks.txt
        # visualization-gaps pass): Agent 1's per-iteration decision logs
        # (for the Tier 4 fingerprint_adjustments trend) and
        # agents/pipeline_validator.py's per-run issue logs.
        self.decisions_dir = _reports / "agent1_decisions"
        self.validation_dir = _reports / "pipeline_validation"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        self.summary_counter = self._count_existing_summaries()
        self._latest_summary_evidence: Optional[SummaryEvidence] = None

        # LLM/copilot integration (dev/checks.txt item 4): shared campaign
        # budget across agent1/2/3 -- see agents/claude_cli.py's docstring.
        llm_config = self.config.get("llm", {})
        self._llm_backend = llm_config.get("backend", "cli")
        self._llm_model = llm_config.get("model", "sonnet")
        self._llm_campaign_budget_usd = float(llm_config.get("campaign_budget_usd", 5.0))
        self._llm_max_call_budget_usd = float(llm_config.get("max_call_budget_usd", 0.20))
        self._llm_usage_path = llm_config.get("usage_log_path") or str(_state / "llm_usage.json")

        # Cheap guard (dev/checks.txt follow-up): remembers the fingerprint
        # of the last cluster data actually sent to _get_cluster_hypotheses,
        # so a summary whose fingerprint_clusters are byte-identical to the
        # last analyzed batch skips the LLM call instead of re-paying to
        # restate the same finding (observed in practice across summaries
        # #28-30: 15 fingerprint-bearing runs, unchanged clusters, 3 near-
        # identical hypotheses paid for in a row).
        self._cluster_signature_path = _state / "cluster_hypotheses_signature.json"

        # Campaign-level markers for the val_bpb trend chart (e.g. "the
        # search-strategy bug fix landed here") -- see _load_annotations.
        # A real regime change (like a real dry_run/simulated status) is
        # never removed from the underlying data, but it should stay
        # visible rather than silently blending two different eras.
        self.annotations_path = _state / "campaign_annotations.json"

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        if yaml is None:
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _count_existing_summaries(self) -> int:
        """Count existing summaries."""
        summaries = list(self.summaries_dir.glob("summary_*.md"))
        return len(summaries)

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
                new_reports.append((report_id, _read_text_tolerant(report_path)))
                print(f"[Agent 3]   Loaded: {report_id}")

        # Load previous summary if exists
        prev_summary = ""
        prev_summary_id = None
        if self.summary_counter > 0:
            prev_summary_path = (
                self.summaries_dir / f"summary_{self.summary_counter - 1:04d}.md"
            )
            if prev_summary_path.exists():
                prev_summary = _read_text_tolerant(prev_summary_path)
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
            fingerprint_clusters=aggregate.get("fingerprint_clusters", {}),
        )
        self._latest_summary_evidence = summary

        summary_path = self.summaries_dir / f"{summary_id}.md"
        with open(summary_path, "w", encoding="utf-8") as f:
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

        # OPTIONAL: If LLM enabled, add a Claude-generated strategic narrative
        # (via agents/claude_cli.py -- your subscription, not a separately
        # billed API key). None (never fabricated) when the CLI is
        # unavailable or the shared campaign budget is exhausted.
        if self.use_llm:
            narrative = self._get_claude_narrative(aggregate.get("prompt_summary") or summary)
            summary += (f"\n\n## Strategic Narrative\n{narrative}\n" if narrative else
                        "\n\n## Strategic Narrative\nUnavailable this run (CLI not reachable, or campaign LLM budget exhausted)\n")

        # OPTIONAL: If LLM enabled AND there's real cluster data (Tier 3.3)
        # -- not invoked when clustering hasn't found anything yet, so a
        # missing-data run never costs a call. Also skipped (cheap guard)
        # when the underlying fingerprint cluster data is byte-identical to
        # what was last actually sent to Claude -- see _cluster_signature_path.
        if self.use_llm and aggregate.get("fingerprint_clusters"):
            signature = self._fingerprint_clusters_signature(aggregate["fingerprint_clusters"])
            if signature == self._load_last_cluster_signature():
                print(f"[Agent 3] Fingerprint cluster data unchanged since the last analyzed "
                      f"summary -- skipping the cluster-hypotheses LLM call to avoid spending "
                      f"budget restating the same finding.")
                summary += ("\n\n## Cluster Hypotheses (Claude)\nSkipped this run -- underlying "
                            "fingerprint cluster data is unchanged since the last time Claude "
                            "analyzed it (no new evidence; budget preserved).\n")
            else:
                hypotheses = self._get_cluster_hypotheses(aggregate["fingerprint_clusters"])
                if hypotheses:
                    self._save_last_cluster_signature(signature)
                summary += (f"\n\n## Cluster Hypotheses (Claude)\n{hypotheses}\n" if hypotheses else
                            "\n\n## Cluster Hypotheses (Claude)\nUnavailable this run (CLI not reachable, or campaign LLM budget exhausted)\n")

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
                reports.append((path.stem, _read_text_tolerant(path)))
            except OSError:
                continue
        return reports

    def _load_all_decision_logs(self) -> List[Dict[str, Any]]:
        """All of Agent 1's per-iteration decision logs
        (reports/agent1_decisions/decision_*.json), sorted by iteration --
        the Tier 4 fingerprint_adjustments trend needs the full history,
        same "read everything, not a sliding window" convention
        _load_all_reports already uses.
        """
        if not self.decisions_dir.exists():
            return []
        logs = []
        for path in sorted(self.decisions_dir.glob("decision_*.json")):
            try:
                logs.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(logs, key=lambda log: log.get("iteration", 0))

    def _load_latest_run_issues(self) -> List[Dict[str, Any]]:
        """agents/pipeline_validator.py's per-iteration issue logs for the
        most recent orchestrator run only (reports/pipeline_validation/run_*/
        directory naming is chronological by construction -- new_run_dir
        timestamps it -- so the lexicographically-last run_* dir is the
        current session, matching what "this run" should mean here rather
        than mixing issue counts across restarts with their own iteration
        numbering starting back at 0).
        """
        if not self.validation_dir.exists():
            return []
        run_dirs = sorted(p for p in self.validation_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
        if not run_dirs:
            return []
        latest_run_dir = run_dirs[-1]
        logs = []
        for path in sorted(latest_run_dir.glob("iteration_*.json")):
            try:
                logs.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(logs, key=lambda log: log.get("iteration", 0))

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

    def _status_of(self, item: Dict[str, Any]) -> str:
        """Same precedence used for the Status distribution table below --
        metadata.status when present, else the top-level status field."""
        return str(item.get("metadata", {}).get("status") or item.get("status") or "unknown")

    def _strip_markdown_section(self, text: str, heading: str) -> str:
        """Removes the section starting at an exact "## heading" line up to
        (not including) the next "## " heading or end of text. No-op if the
        heading isn't present. Used to build a leaner LLM-prompt variant of
        the full human-readable summary (see _build_prompt_summary) without
        duplicating how each section gets built.
        """
        lines = text.splitlines()
        start = None
        for i, line in enumerate(lines):
            if line.strip() == heading:
                start = i
                break
        if start is None:
            return text
        end = start + 1
        while end < len(lines) and not lines[end].startswith("## "):
            end += 1
        return "\n".join(lines[:start] + lines[end:])

    def _build_prompt_summary(
        self, full_summary: str, sorted_layers: List[Tuple[str, List[float]]]
    ) -> str:
        """A leaner variant of the full statistical summary, used only for
        the strategic-narrative LLM prompt -- the saved report keeps every
        section at full detail. Strips content that's pure overhead for a
        text-only LLM call and costs real money every call:
          - chart image embeds (an LLM reading text can't see a .png)
          - the LLM Usage/budget section (irrelevant to pattern reasoning)
          - the Tier 3.4 methodology sentence (static, identical every call)
          - the full per-layer table (20+ rows) -> condensed to the top 5
            layers by share plus a one-line "N others near zero" note, the
            same conclusion a human draws from the full table
        """
        lines = [
            line for line in full_summary.splitlines()
            if not line.startswith("![")
            and not line.startswith("- Volatility = total variation")
        ]
        text = "\n".join(lines)
        text = self._strip_markdown_section(text, "## LLM Usage This Campaign")
        text = self._strip_markdown_section(text, "## Layer-Level Importance Distribution")

        if sorted_layers:
            top_layers = sorted(sorted_layers, key=lambda kv: -self._safe_mean(kv[1]))[:5]
            dead_count = sum(1 for _, shares in sorted_layers if self._safe_mean(shares) < 0.5)
            layer_block = ["", "## Layer-Level Importance (condensed, top 5 by share)"]
            layer_block += [f"- L{name}: {self._safe_mean(shares):.2f}%" for name, shares in top_layers]
            if dead_count:
                layer_block.append(f"- {dead_count} other layer(s) at <0.5% share (dead weight)")
            text += "\n" + "\n".join(layer_block)
        return text

    def _is_synthetic(self, item: Dict[str, Any]) -> bool:
        """True for dry_run/simulated reports -- their val_bpb is a fixed
        formula (dry_run: 1.0 - 0.001*iteration; simulated: a hand-tuned
        stand-in for local testing), never a measured result. Kept in
        report counts/status-distribution reporting (that's legitimate --
        it's counting how many runs of each kind happened), but must never
        enter a numeric aggregate that treats val_bpb as comparable across
        runs (Best/Worst/Mean, elite-run hyperparameter recommendations) --
        mirrors state/results_analysis.py's SYNTHETIC_STATUSES filter on
        the results.tsv side of this same problem.
        """
        return self._status_of(item) in SYNTHETIC_STATUSES

    def _format_statistical_summary(
        self, new_reports: List[tuple], prev_summary: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Format statistical analysis of reports."""

        all_reports = self._load_all_reports()
        all_metrics = [self._extract_structured_metrics(content) for _, content in all_reports]
        batch_metrics = [self._extract_structured_metrics(content) for _, content in new_reports]

        # Tier 3: cluster token-level behavioral fingerprints across runs
        # (see state/clustering.py). token_fingerprint is only non-empty
        # when token_xai_enabled was on for that run (Tier 2 cadence
        # trigger), so this is usually a strict subset of all_metrics.
        fingerprint_rows = [
            {**(item.get("token_fingerprint") or {}), "val_bpb": item.get("val_bpb")}
            for item in all_metrics
            if item.get("token_fingerprint")
        ]
        overall_clusters = cluster_fingerprints(fingerprint_rows, min_n=self.min_cluster_n)
        trajectory_clusters = cluster_attention_trajectories(fingerprint_rows, min_n=self.min_cluster_n)
        # Tier 3.4: a continuous correlation over all usable rows, instead of
        # reading only cluster membership -- more robust at small n than the
        # trajectory clusters above (which have been landing at silhouette
        # ~0.25-0.29 with clusters as small as n=2 in practice).
        smoothness_correlation = trajectory_smoothness_correlation(fingerprint_rows, min_n=self.min_cluster_n)
        fingerprint_clusters = (
            {
                "overall": overall_clusters,
                "trajectory": trajectory_clusters,
                "smoothness_correlation": smoothness_correlation,
            }
            if (overall_clusters or trajectory_clusters or smoothness_correlation) else {}
        )

        finite_bpbs = [
            float(item.get("val_bpb"))
            for item in all_metrics
            if isinstance(item.get("val_bpb"), (int, float)) and math.isfinite(float(item.get("val_bpb")))
            and not self._is_synthetic(item)
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
            if self._is_synthetic(item):
                continue
            hp = item.get("hyperparams", {}) or {}
            if not isinstance(hp, dict):
                continue
            elite_candidates.append((float(val), hp))

        # Shared "what counts as elite" selection (state/results_analysis.py)
        # -- Agent 2's stuck-signal reference value uses the exact same
        # definition, just aggregates the val_bpb side instead of the
        # hyperparams side.
        elite = top_quartile_by_val_bpb(elite_candidates)

        def _avg_hp(name: str, default: float = 0.0) -> float:
            values = []
            for _, hp in elite:
                raw = hp.get(name)
                if isinstance(raw, (int, float)):
                    values.append(float(raw))
            return self._safe_mean(values) if values else default

        recommended = {}
        if elite:
            for lr_key in LR_KEYS:
                lr_values = [
                    float(hp.get(lr_key)) for _, hp in elite
                    if isinstance(hp.get(lr_key), (int, float)) and hp.get(lr_key) > 0
                ]
                if lr_values:
                    geo_lr = math.exp(sum(math.log(v) for v in lr_values) / len(lr_values))
                    recommended[lr_key] = round(geo_lr, 6)
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

        if self.generate_charts:
            try:
                chart_path = chart_val_bpb_trend(
                    all_metrics, self.noise_floor_path,
                    self.visuals_dir / f"summary_{self.summary_counter:04d}_trend.png",
                    annotations=self._load_annotations(),
                )
                if chart_path:
                    summary_lines.extend(["", f"![val_bpb trend](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 3] Chart generation (trend) failed: {_e}")
            try:
                noise_floor_history = []
                if self.noise_floor_path.exists():
                    noise_floor_history = (json.loads(self.noise_floor_path.read_text()) or {}).get("history", [])
                chart_path = chart_noise_floor_trend(
                    noise_floor_history, self.visuals_dir / f"summary_{self.summary_counter:04d}_noise_floor.png",
                )
                if chart_path:
                    summary_lines.extend(["", f"![Noise floor over time](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 3] Chart generation (noise floor trend) failed: {_e}")
            try:
                chart_path = chart_status_distribution(
                    statuses, self.visuals_dir / f"summary_{self.summary_counter:04d}_status.png",
                )
                if chart_path:
                    summary_lines.extend(["", f"![Run status distribution](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 3] Chart generation (status) failed: {_e}")

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

        if self.generate_charts:
            try:
                chart_path = chart_hyperparameter_importance_evolution(
                    all_metrics, self.visuals_dir / f"summary_{self.summary_counter:04d}_importance_evolution.png",
                )
                if chart_path:
                    summary_lines.extend(["", f"![Hyperparameter importance evolution](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 3] Chart generation (importance evolution) failed: {_e}")

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

        if self.generate_charts:
            try:
                chart_path = chart_layer_importance_distribution(
                    layer_shares, self.visuals_dir / f"summary_{self.summary_counter:04d}_layers.png",
                )
                if chart_path:
                    summary_lines.extend(["", f"![Layer-level importance distribution](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 3] Chart generation (layer distribution) failed: {_e}")

        summary_lines.extend([
            "",
            "## Behavioral Fingerprint Clusters (Tier 3)",
        ])
        if not fingerprint_rows:
            summary_lines.append("- No token-level fingerprints available yet (token_xai_enabled has not run on any historical run)")
        else:
            summary_lines.append(f"- Fingerprint-bearing runs available: {len(fingerprint_rows)} (need >= {self.min_cluster_n} to cluster)")
            if overall_clusters:
                summary_lines.extend([
                    "",
                    f"### Overall fingerprint clusters (k={overall_clusters['k']}, silhouette={overall_clusters['silhouette']:.3f})",
                    "| Cluster | n | Mean val_bpb |",
                    "|--------:|--:|-------------:|",
                ])
                for c in overall_clusters["clusters"]:
                    mean_str = f"{c['mean_val_bpb']:.6f}" if c["mean_val_bpb"] is not None else "N/A"
                    summary_lines.append(f"| {c['cluster_id']} | {c['n']} | {mean_str} |")
            else:
                summary_lines.append(f"- Not enough historical fingerprints yet to cluster overall (need >= {self.min_cluster_n}, have {len(fingerprint_rows)})")

            if trajectory_clusters:
                summary_lines.extend([
                    "",
                    f"### Attention-reach trajectory clusters (k={trajectory_clusters['k']}, silhouette={trajectory_clusters['silhouette']:.3f})",
                    "| Cluster | n | Mean val_bpb | Shape (normalized depth, first -> last) |",
                    "|--------:|--:|-------------:|:-----------------------------------------|",
                ])
                for c in trajectory_clusters["clusters"]:
                    mean_str = f"{c['mean_val_bpb']:.6f}" if c["mean_val_bpb"] is not None else "N/A"
                    shape_str = " -> ".join(f"{v:.2f}" for v in c["mean_shape"])
                    summary_lines.append(f"| {c['cluster_id']} | {c['n']} | {mean_str} | {shape_str} |")
            else:
                summary_lines.append(f"- Not enough historical fingerprints yet to cluster trajectory shapes (need >= {self.min_cluster_n}, have {len(fingerprint_rows)})")

            if smoothness_correlation:
                sign_note = (
                    "more volatile trajectories tend toward higher (worse) val_bpb"
                    if smoothness_correlation["correlation"] > 0 else
                    "more volatile trajectories tend toward lower (better) val_bpb"
                    if smoothness_correlation["correlation"] < 0 else
                    "no directional signal"
                )
                summary_lines.extend([
                    "",
                    f"### Trajectory volatility vs. val_bpb (Tier 3.4, n={smoothness_correlation['n']})",
                    f"- Spearman correlation: {smoothness_correlation['correlation']:+.4f} ({sign_note})",
                    "- Volatility = total variation of each run's normalized attn_distance curve (0 = perfectly smooth/monotonic, higher = zig-zags more). "
                    "Uses every fingerprint-bearing run individually rather than fragile small clusters, so it's a statistically sturdier read on the same question the trajectory clusters above are asking.",
                ])
            else:
                summary_lines.append(f"- Not enough historical fingerprints yet for a trajectory-volatility correlation (need >= {self.min_cluster_n}, have {len(fingerprint_rows)})")

        if self.generate_charts:
            try:
                chart_path = chart_fingerprint_clusters(
                    overall_clusters, self.visuals_dir / f"summary_{self.summary_counter:04d}_clusters.png",
                )
                if chart_path:
                    summary_lines.extend(["", f"![Fingerprint clusters vs. val_bpb](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 3] Chart generation (fingerprint clusters) failed: {_e}")
            try:
                chart_path = chart_attention_trajectory_clusters(
                    trajectory_clusters, self.visuals_dir / f"summary_{self.summary_counter:04d}_trajectories.png",
                )
                if chart_path:
                    summary_lines.extend(["", f"![Attention-reach trajectory clusters](../visuals/{chart_path.name})"])
            except Exception as _e:
                print(f"[Agent 3] Chart generation (trajectory clusters) failed: {_e}")

        # Tier 2 scalar fields (attn_distance_slope/induction_score) never
        # got a trend-over-history chart before -- chart_token_fingerprint
        # (Agent 2's per-run report) only ever showed one run's snapshot.
        summary_lines.extend(["", "## Tier 2 Scalar Fingerprint Fields Over Time"])
        if self.generate_charts:
            try:
                chart_path = chart_token_fingerprint_scalars_evolution(
                    all_metrics, self.visuals_dir / f"summary_{self.summary_counter:04d}_token_scalars.png",
                )
                if chart_path:
                    summary_lines.append(f"![Tier 2 scalar fingerprint fields](../visuals/{chart_path.name})")
                else:
                    summary_lines.append("- No token_fingerprint data yet (token_xai_enabled has not run on any historical run)")
            except Exception as _e:
                print(f"[Agent 3] Chart generation (Tier 2 scalar evolution) failed: {_e}")
        else:
            summary_lines.append("- Chart generation disabled")

        # Tier 4 fingerprint_adjustments: which architecture rules fired,
        # what deltas, over the campaign -- previously JSON decision logs
        # only (reports/agent1_decisions/), zero visualization.
        decision_logs = self._load_all_decision_logs()
        summary_lines.extend(["", "## Tier 4 Fingerprint-Driven Architecture Adjustments"])
        total_adjustments = sum(len(log.get("fingerprint_adjustments", [])) for log in decision_logs)
        if total_adjustments == 0:
            summary_lines.append("- No fingerprint-driven adjustments fired yet")
        else:
            summary_lines.append(f"- {total_adjustments} fingerprint-driven adjustment(s) across {len(decision_logs)} decision(s) on record")
            if self.generate_charts:
                try:
                    chart_path = chart_fingerprint_adjustments_trend(
                        decision_logs, self.visuals_dir / f"summary_{self.summary_counter:04d}_fingerprint_adjustments.png",
                    )
                    if chart_path:
                        summary_lines.append(f"![Tier 4 fingerprint-driven adjustments](../visuals/{chart_path.name})")
                except Exception as _e:
                    print(f"[Agent 3] Chart generation (fingerprint adjustments) failed: {_e}")

        # pipeline_validator issues for the current run -- previously
        # console text + per-iteration JSON only, no trend visualization.
        issue_logs = self._load_latest_run_issues()
        summary_lines.extend(["", "## Pipeline Validation Issues (This Run)"])
        total_issues = sum(len(log.get("issues", [])) for log in issue_logs)
        if total_issues == 0:
            summary_lines.append("- No pipeline_validator issues recorded for this run")
        else:
            severity_counts = Counter(
                issue.get("severity") for log in issue_logs for issue in log.get("issues", [])
            )
            summary_lines.append(
                f"- {total_issues} issue(s) across {len(issue_logs)} iteration(s): "
                + ", ".join(f"{sev}={count}" for sev, count in sorted(severity_counts.items()))
            )
            if self.generate_charts:
                try:
                    chart_path = chart_pipeline_issues_trend(
                        issue_logs, self.visuals_dir / f"summary_{self.summary_counter:04d}_pipeline_issues.png",
                    )
                    if chart_path:
                        summary_lines.append(f"![Pipeline validation issues](../visuals/{chart_path.name})")
                except Exception as _e:
                    print(f"[Agent 3] Chart generation (pipeline issues) failed: {_e}")

        # LLM usage/budget (dev/checks.txt item 4) -- state/llm_usage.json is
        # shared across agent1/2/3, so this is the campaign total, not just
        # this agent's own calls. Self-tracked against the budget you set in
        # agents_config.yaml's llm.campaign_budget_usd -- not a readout of
        # your actual Claude subscription's remaining quota (the CLI doesn't
        # expose that).
        summary_lines.extend(["", "## LLM Usage This Campaign"])
        usage_records = llm_usage.load_usage_log(self._llm_usage_path)
        if not usage_records:
            summary_lines.append("- No LLM calls made yet this campaign")
        else:
            cumulative = llm_usage.cumulative_cost_usd(self._llm_usage_path)
            remaining = llm_usage.remaining_budget_usd(self._llm_campaign_budget_usd, self._llm_usage_path)
            successes = sum(1 for r in usage_records if not r.get("is_error"))
            summary_lines.append(
                f"- {len(usage_records)} call(s) ({successes} successful), "
                f"cumulative cost ${cumulative:.4f} / ${self._llm_campaign_budget_usd:.2f} budget "
                f"(${remaining:.4f} remaining)"
            )
            by_site = Counter(r.get("call_site") for r in usage_records)
            summary_lines.append(
                "- By call site: " + ", ".join(f"{site}={count}" for site, count in sorted(by_site.items()))
            )

        summary_lines.extend([
            "",
            "## Recommendations for Agent 1 (Data-Backed)",
            f"- Recommendation sample size (elite runs): {len(elite)}",
        ])
        if recommended:
            for lr_key in LR_KEYS:
                if lr_key in recommended:
                    summary_lines.append(f"- {lr_key} (geometric mean from elite runs): {recommended[lr_key]}")
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

        full_summary = "\n".join(summary_lines) + "\n"

        aggregate = {
            "stable_patterns": stable_patterns[:5],
            "conflicting_signals": conflicting_patterns[:5],
            "recommended_hyperparams": recommended,
            "reasoning": [
                f"Computed from {len(all_reports)} total reports and {len(finite_bpbs)} finite val_bpb runs",
                f"Elite recommendation subset size: {len(elite)}",
                "Importance/non-importance split uses threshold 0.50",
            ],
            "fingerprint_clusters": fingerprint_clusters,
            # LLM-prompt-only leaner variant (dev/checks.txt follow-up:
            # reduce prompt noise/duplication) -- _get_claude_narrative uses
            # this instead of full_summary; the saved .md report always gets
            # full_summary, unchanged.
            "prompt_summary": self._build_prompt_summary(full_summary, sorted_layers),
        }

        return full_summary, aggregate

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

    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Campaign-level markers for the val_bpb trend chart -- a small,
        manually-curated JSON file (state/campaign_annotations.json), not
        auto-detected: inferring "when did the search strategy change" from
        file timestamps/gaps is exactly the kind of guess this codebase
        avoids elsewhere (see SYNTHETIC_STATUSES, noise_floor's min-n guard).
        A real, known event gets recorded here once, deliberately, with a
        report_index verified against the actual report history -- not
        reconstructed automatically. Missing/corrupt file -> [] (no
        annotations drawn), never an error.

        Expected shape: {"annotations": [{"report_index": int, "label": str}, ...]}
        """
        if not self.annotations_path.exists():
            return []
        try:
            data = json.loads(self.annotations_path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
        annotations = data.get("annotations")
        return annotations if isinstance(annotations, list) else []

    def _fingerprint_clusters_signature(self, fingerprint_clusters: Dict[str, Any]) -> str:
        """Deterministic hash of the exact payload that would be sent to
        _get_cluster_hypotheses -- a byte-for-byte-stable JSON dump (sorted
        keys) so identical cluster data always hashes the same regardless
        of dict insertion order."""
        payload = json.dumps(fingerprint_clusters, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_last_cluster_signature(self) -> Optional[str]:
        if not self._cluster_signature_path.exists():
            return None
        try:
            return json.loads(self._cluster_signature_path.read_text()).get("signature")
        except (json.JSONDecodeError, OSError):
            return None

    def _save_last_cluster_signature(self, signature: str) -> None:
        try:
            self._cluster_signature_path.parent.mkdir(parents=True, exist_ok=True)
            self._cluster_signature_path.write_text(json.dumps({"signature": signature}))
        except OSError:
            pass

    def _get_claude_narrative(self, statistical_summary: str) -> Optional[str]:
        """Ask the Claude Code CLI (agents/claude_cli.py -- your subscription)
        for a strategic narrative. None (not fabricated) when the CLI is
        unavailable or the shared campaign budget is exhausted.
        """
        prompt = f"""Based on this summary, provide a strategic narrative:
1. What patterns emerge across multiple models?
2. What should Agent 1 focus on for next iteration?
3. Are we converging or diverging?

Summary:
{statistical_summary}

Be concise (under 250 words)."""

        return claude_cli.call_with_budget(
            prompt, call_site="agent3_strategic_narrative",
            model=self._llm_model,
            campaign_budget_usd=self._llm_campaign_budget_usd,
            max_call_budget_usd=self._llm_max_call_budget_usd,
            usage_path=self._llm_usage_path,
            backend=self._llm_backend,
        )

    def _get_cluster_hypotheses(self, fingerprint_clusters: Dict[str, Any]) -> Optional[str]:
        """Tier 3.3: feed the cluster table to Claude for hypotheses only --
        never treated as a finding on its own, just candidate explanations
        for Agent 1 to actually test as real runs. None (not fabricated)
        when the CLI is unavailable or the shared campaign budget is
        exhausted.
        """
        prompt = f"""Below is data on behavioral fingerprints found across recent training runs:
cluster tables (from hierarchical clustering of attention/attribution statistics,
each cluster with its mean validation bpb, lower is better) AND, separately, a
"smoothness_correlation" field -- a Spearman correlation (across every individual
fingerprint-bearing run, not cluster membership) between how volatile each run's
attn_distance trajectory is and its val_bpb. Treat smoothness_correlation as the
more statistically reliable signal when the clusters and the correlation seem to
agree or disagree, since it isn't fragmented into small clusters.

1. What distinguishes the best-performing cluster from the others, and does the
   smoothness_correlation support or undercut that distinction?
2. What is the most testable hypothesis for why that distinction might matter?
3. Frame this as a hypothesis to test with a real run, not a conclusion.

Data:
{json.dumps(fingerprint_clusters)}

Be concise (under 200 words)."""

        return claude_cli.call_with_budget(
            prompt, call_site="agent3_cluster_hypotheses",
            model=self._llm_model,
            campaign_budget_usd=self._llm_campaign_budget_usd,
            max_call_budget_usd=self._llm_max_call_budget_usd,
            usage_path=self._llm_usage_path,
            backend=self._llm_backend,
        )
