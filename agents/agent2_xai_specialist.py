"""Agent 2: XAI Specialist - Analyzes model behavior and generates reports."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from agents.xai_methods.fast_methods import FastXAIMethods
import yaml


class Agent2XAISpecialist:
    """Analyzes trained models using XAI methods and generates interpretability reports."""

    def __init__(self, config_path: str = "agents_config.yaml"):
        self.config = self._load_config(config_path)
        self.agent2_config = self.config.get("agent2", {})
        self.use_llm = self.agent2_config.get("use_llm", False)
        self.xai_method = self.agent2_config.get("xai_method", "fast")
        self.reports_dir = Path("reports/agent2_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize XAI methods
        if self.xai_method == "fast":
            self.xai = FastXAIMethods()
        else:
            raise ValueError(f"Unknown XAI method: {self.xai_method}")

        self.report_counter = self._count_existing_reports()
        self.all_impacts = []  # Track impacts across all models for stuck detection

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

    def analyze(
        self,
        model,
        val_dataloader,
        evaluate_fn,
        model_id: str,
        hyperparams: Dict[str, Any],
    ) -> str:
        """
        Analyze trained model and generate report.

        Returns:
            report_id (filename)
        """
        print(f"\n[Agent 2] Analyzing model {model_id}...")

        # Run XAI analysis
        try:
            # Top-K ablation
            ablation_results = self.xai.top_k_ablation_study(
                model,
                val_dataloader,
                evaluate_fn,
                k=self.agent2_config.get("ablation_k", 10),
            )

            # Partial dependence
            partial_dep = self.xai.partial_dependence_hyperparams(
                model, val_dataloader, evaluate_fn, hyperparams
            )

            # Detect stuck signal
            self.all_impacts.append(ablation_results)
            stuck_signal = self.xai.detect_stuck_signal(self.all_impacts)

        except Exception as e:
            print(f"[Agent 2] Error during XAI analysis: {e}")
            ablation_results = {}
            partial_dep = {}
            stuck_signal = False

        # Generate report
        report_content = self._generate_report(
            model_id, hyperparams, ablation_results, partial_dep, stuck_signal
        )

        # Save report
        report_id = f"report_{self.report_counter:04d}"
        report_path = self.reports_dir / f"{report_id}.md"

        with open(report_path, "w") as f:
            f.write(report_content)

        self.report_counter += 1
        print(f"[Agent 2] Report saved: {report_path}")

        return report_id

    def _generate_report(
        self,
        model_id: str,
        hyperparams: Dict[str, Any],
        ablation_results: Dict[str, float],
        partial_dep: Dict[str, List[tuple]],
        stuck_signal: bool,
    ) -> str:
        """Generate statistical report (+ optional Claude insights)."""

        # ALWAYS: Generate statistical summary
        report = self._format_statistical_report(
            model_id, hyperparams, ablation_results, partial_dep, stuck_signal
        )

        # OPTIONAL: If LLM enabled, add Claude insights
        if self.use_llm:
            try:
                self._init_claude()
                insights = self._get_claude_insights(report)
                report += f"\n\n## Claude Insights\n{insights}\n"
            except Exception as e:
                report += f"\n\n## Claude Insights\nFailed to generate: {e}\n"

        return report

    def _format_statistical_report(
        self,
        model_id: str,
        hyperparams: Dict[str, Any],
        ablation_results: Dict[str, float],
        partial_dep: Dict[str, List[tuple]],
        stuck_signal: bool,
    ) -> str:
        """Format statistical analysis as markdown."""

        report = f"""# XAI Analysis Report: {model_id}

## Model Configuration
- Model ID: {model_id}
- Hyperparameters:
"""
        for key, value in hyperparams.items():
            report += f"  - {key}: {value}\n"

        # Ablation results
        report += "\n## Attention Head Ablation (Top-K)\n"
        report += "| Head | Impact (bpb drop) |\n|------|------------------|\n"

        if ablation_results:
            sorted_heads = sorted(
                ablation_results.items(), key=lambda x: x[1], reverse=True
            )
            for head, impact in sorted_heads[:15]:  # Top 15
                report += f"| {head} | {impact:.6f} |\n"
        else:
            report += "| N/A | No ablation data |\n"

        # Hyperparameter importance
        report += "\n## Hyperparameter Importance\n"
        report += "| Parameter | Estimated Impact |\n|-----------|------------------|\n"

        if partial_dep:
            for param, curve in partial_dep.items():
                if curve:
                    # Estimate importance from variation
                    values = [v for _, v in curve]
                    importance = max(values) - min(values) if values else 0
                    report += f"| {param} | {importance:.6f} |\n"
        else:
            report += "| N/A | No partial dependence data |\n"

        # Signals
        report += "\n## Detected Signals\n"
        if stuck_signal:
            report += "- ⚠️ **Model Stuck**: Similar ablation patterns to previous models\n"
        else:
            report += "- ✓ Model showing new patterns\n"

        # Opportunities
        report += "\n## Flagged Opportunities\n"
        if ablation_results:
            # Find unused heads
            unused = [h for h, v in ablation_results.items() if v < 0.0001]
            if unused:
                report += f"- Potentially unused heads (near-zero impact): {unused}\n"
            report += "- Consider pruning low-impact heads to reduce model size\n"
        report += "- Focus Agent 1 on parameters with highest importance\n"

        return report

    def _get_claude_insights(self, statistical_report: str) -> str:
        """Get Claude to interpret the statistical report."""
        prompt = f"""Please analyze this XAI report and provide:
1. Which components matter most?
2. Strategic recommendations for next training iteration
3. Any surprising findings?

Report:
{statistical_report}

Be concise (under 200 words)."""

        try:
            message = self.claude.messages.create(
                model="claude-opus-4-7",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            return f"Error getting Claude insights: {e}"
