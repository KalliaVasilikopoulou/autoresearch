"""Orchestrator: Coordinates a structured multi-agent optimization loop."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from agents.agent2_xai_specialist import Agent2XAISpecialist
from agents.agent3_report_analyst import Agent3ReportAnalyst
from agents.protocols import AnalysisEvidence, SummaryEvidence, TrainingResult
from state.state_manager import StateManager
from state.results_logger import log_result


class Orchestrator:
    """Coordinates the report-driven multi-agent workflow."""

    def __init__(
        self,
        config_path: str = "agents_config.yaml",
        state_dir: str = "./state",
        reports_dir: str = "./reports",
        root_dir: str = ".",
        dry_run: bool = False,
    ):
        """
        root_dir is where model_hyperparams.yaml and results.tsv live. It
        defaults to "." (repo root) because a real (non-dry-run) train.py
        subprocess always reads model_hyperparams.yaml from its own
        directory -- only override it for dry-run/test runs that never
        invoke train.py. state_dir/reports_dir were already accepted here
        but never forwarded to Agent1/2/3 (they always hit the hardcoded
        repo-root paths regardless) -- that's fixed below.
        """
        print("[Orchestrator] Initializing multi-agent system...")

        self.root_dir = Path(root_dir)
        self.results_path = self.root_dir / "results.tsv"
        self.state_mgr = StateManager(state_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.agent1 = Agent1TrainingSpecialist(config_path, root_dir=root_dir, state_dir=state_dir, reports_dir=reports_dir)
        self.agent2 = Agent2XAISpecialist(config_path, root_dir=root_dir, reports_dir=reports_dir)
        self.agent3 = Agent3ReportAnalyst(config_path, state_dir=state_dir, reports_dir=reports_dir)

        self.config_path = config_path
        self.max_iterations = 100
        self.poll_interval = 5
        self.dry_run = dry_run

        print("[Orchestrator] Initialization complete")

    def run(self, max_iterations: Optional[int] = None):
        """Main orchestration loop with structured evidence flow."""
        print("[Orchestrator] Starting autonomous multi-agent loop...\n")

        iteration = 0
        report_batch: List[str] = []
        max_iterations = max_iterations or self.max_iterations

        while iteration < max_iterations:
            print(f"\n{'='*60}")
            print(f"[Orchestrator] Iteration {iteration + 1}")
            print(f"{'='*60}")

            print("\n[Orchestrator] Phase 1: Agent 1 proposes a new configuration")
            latest_summary = self._load_latest_summary()
            recent_evidence = self.state_mgr.get_recent_evidence(limit=5)
            recent_results = self.state_mgr.get_all_results()[-3:]
            latest_val_bpb = None
            if recent_results:
                latest_result = recent_results[-1]
                latest_val_bpb = latest_result.get("val_bpb")
            new_hyperparams = self.agent1.decide_next_hyperparams(
                latest_summary=latest_summary,
                evidence=recent_evidence,
                iteration=iteration,
                latest_val_bpb=latest_val_bpb,
                recent_results=recent_results,
            )

            if new_hyperparams is None:
                print("\n[Orchestrator] STOPPING: Agent 1 stopped optimizing")
                break

            print("\n[Orchestrator] Phase 2: Training")
            print(f"[Orchestrator] Training with hyperparams: {new_hyperparams}")
            train_result = self.agent1.train_model(
                new_hyperparams,
                dry_run=self.dry_run,
                iteration=iteration,
            )
            print(f"[Orchestrator] Training result: val_bpb={train_result.get('val_bpb', 'N/A')}, status={train_result.get('status', 'unknown')}")
            result_payload = TrainingResult(
                run_id=f"run_{iteration:04d}",
                hyperparams=new_hyperparams,
                val_bpb=train_result.get("val_bpb", float("inf")),
                training_time=train_result.get("training_time", 0.0),
                checkpoint_path=train_result.get("checkpoint_path"),
                status=train_result.get("status", "ok"),
                metadata=train_result,
            )
            self.state_mgr.add_result(result_payload.to_dict())
            self.state_mgr.update_val_bpb(result_payload.run_id, result_payload.val_bpb)
            log_result(result_payload.run_id, new_hyperparams, train_result, results_path=str(self.results_path))
            print(f"[Orchestrator] Result logged: {result_payload.run_id}")

            print("\n[Orchestrator] Phase 3: Analyzing result with Agent 2")
            evidence = self.agent2.analyze_result(result_payload.to_dict())
            if evidence is not None:
                evidence_payload = evidence.to_dict()
                self.state_mgr.add_evidence(evidence_payload)
                report_batch.append(evidence_payload["report_id"])
                self.state_mgr.link_model_to_report(result_payload.run_id, evidence_payload["report_id"])
                print(f"[Orchestrator] Agent 2 analysis complete: {evidence_payload['report_id']}")
            else:
                print(f"[Orchestrator] Agent 2: no analysis available")

            print("\n[Orchestrator] Phase 4: Aggregating evidence with Agent 3")
            print(f"[Orchestrator] Batch size: {len(report_batch)} reports")
            if self.agent3.should_create_summary(len(report_batch)):
                print(f"[Orchestrator] Creating summary from {len(report_batch)} reports...")
                summary = self.agent3.analyze_and_summarize(report_batch)
                self.state_mgr.add_summary(summary.to_dict())
                self.state_mgr.set_latest_summary(summary.summary_id, iteration)
                report_batch = []
                print(f"[Orchestrator] Summary created: {summary.summary_id}")
            else:
                print(f"[Orchestrator] Summary threshold not reached (need {self.agent3.batch_size}, have {len(report_batch)})")

            iteration += 1
            print(f"[Orchestrator] Iteration {iteration} complete")

        summary = self.agent3.get_latest_summary_object()
        print(f"\n{'='*60}")
        print("[Orchestrator] MULTI-AGENT LOOP COMPLETE")
        print(f"Total iterations: {iteration}")
        print(f"Final best val_bpb: {self.agent1.best_val_bpb:.6f}")
        print(f"Total API cost: ${self.agent1.total_api_cost:.2f}")
        print(f"{'='*60}\n")
        return summary

    def _load_latest_summary(self) -> Optional[str]:
        """Load latest summary report for Agent 1 to read."""
        latest_id = self.state_mgr.get_latest_summary()
        if not latest_id:
            return None

        summary_path = self.reports_dir / "agent3_summaries" / f"{latest_id}.md"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                return f.read()
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-agent NN optimization")
    parser.add_argument("--config", default="agents_config.yaml", help="Configuration file path")
    parser.add_argument("--iterations", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--dry-run", action="store_true", help="Run without training")

    args = parser.parse_args()

    orchestrator = Orchestrator(config_path=args.config, dry_run=args.dry_run)
    orchestrator.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
