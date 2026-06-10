"""Orchestrator: Coordinates parallel execution of all agents."""

import time
from typing import Optional
from pathlib import Path
from agents.agent1_training_specialist import Agent1TrainingSpecialist
from agents.agent2_xai_specialist import Agent2XAISpecialist
from agents.agent3_report_analyst import Agent3ReportAnalyst
from state.state_manager import StateManager


class Orchestrator:
    """Orchestrates the multi-agent workflow."""

    def __init__(self, config_path: str = "agents_config.yaml"):
        print("[Orchestrator] Initializing multi-agent system...")

        self.state_mgr = StateManager()
        self.agent1 = Agent1TrainingSpecialist(config_path)
        self.agent2 = Agent2XAISpecialist(config_path)
        self.agent3 = Agent3ReportAnalyst(config_path)

        self.config_path = config_path
        self.max_iterations = 100  # Safety limit
        self.poll_interval = 5  # seconds

        print("[Orchestrator] Initialization complete")

    def run(self):
        """Main orchestration loop."""
        print("[Orchestrator] Starting autonomous multi-agent loop...\n")

        iteration = 0
        report_batch = []

        while iteration < self.max_iterations:
            print(f"\n{'='*60}")
            print(f"[Orchestrator] Iteration {iteration + 1}")
            print(f"{'='*60}")

            # ========== AGENT 1: TRAINING PHASE ==========
            print("\n[Orchestrator] Phase 1: Agent 1 Training")

            # Get latest summary (for Agent 1 to read)
            latest_summary = self._load_latest_summary()

            # Get Agent 1 decision
            new_hyperparams = self.agent1.decide_next_hyperparams(
                latest_summary=latest_summary,
                stuck_signal=False,  # TODO: Track this
                latest_val_bpb=None,  # TODO: Track this
                iteration=iteration,
            )

            if new_hyperparams is None:
                print("\n[Orchestrator] STOPPING: Agent 1 stopped optimizing")
                break

            # Train model
            train_metrics = self.agent1.train_model(new_hyperparams)
            val_bpb = train_metrics.get("val_bpb", float("inf"))

            # Save model and track
            try:
                # Need actual model - placeholder for now
                model = None  # TODO: Get model from train.py
                model_id = self.state_mgr.save_model(model, iteration, new_hyperparams)
                self.state_mgr.update_val_bpb(model_id, val_bpb)
                print(f"[Orchestrator] Saved model: {model_id}, val_bpb: {val_bpb:.6f}")
            except Exception as e:
                print(f"[Orchestrator] Error saving model: {e}")
                model_id = None

            if model_id is None:
                print("[Orchestrator] Skipping this iteration due to save error")
                iteration += 1
                continue

            # ========== AGENT 2: XAI ANALYSIS PHASE ==========
            print("\n[Orchestrator] Phase 2: Agent 2 XAI Analysis")

            try:
                # This is a mock - actual implementation needs real model
                print(f"[Orchestrator] Skipping Agent 2 analysis (requires model object)")
                report_id = f"report_{iteration:04d}"
                report_batch.append(report_id)
                self.state_mgr.link_model_to_report(model_id, report_id)
            except Exception as e:
                print(f"[Orchestrator] Agent 2 failed: {e}")

            # ========== AGENT 3: AGGREGATION PHASE ==========
            print("\n[Orchestrator] Phase 3: Agent 3 Report Aggregation")

            if self.agent3.should_create_summary(len(report_batch)):
                print(
                    f"[Orchestrator] Batch complete ({len(report_batch)} reports), creating summary..."
                )
                try:
                    summary_id = self.agent3.analyze_and_summarize(report_batch)
                    self.state_mgr.set_latest_summary(summary_id, iteration)
                    report_batch = []
                    print(f"[Orchestrator] Summary created: {summary_id}")
                except Exception as e:
                    print(f"[Orchestrator] Agent 3 failed: {e}")

            iteration += 1
            print(f"[Orchestrator] Iteration {iteration} complete")

        print(f"\n{'='*60}")
        print("[Orchestrator] MULTI-AGENT LOOP COMPLETE")
        print(f"Total iterations: {iteration}")
        print(f"Final best val_bpb: {self.agent1.best_val_bpb:.6f}")
        print(f"Total API cost: ${self.agent1.total_api_cost:.2f}")
        print(f"{'='*60}\n")

    def _load_latest_summary(self) -> Optional[str]:
        """Load latest summary report for Agent 1 to read."""
        latest_id = self.state_mgr.get_latest_summary()
        if not latest_id:
            return None

        summary_path = Path(f"reports/agent3_summaries/{latest_id}.md")
        if summary_path.exists():
            with open(summary_path, "r") as f:
                return f.read()
        return None


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-agent NN optimization")
    parser.add_argument(
        "--config",
        default="agents_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum iterations",
    )

    args = parser.parse_args()

    orchestrator = Orchestrator(config_path=args.config)
    orchestrator.run()


if __name__ == "__main__":
    main()
