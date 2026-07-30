"""Orchestrator: Coordinates a structured multi-agent optimization loop."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None

from agents import pipeline_validator
from agents import remote_runner
from agents.live_progress import MultiGpuProgressDisplay
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
        interactive: bool = False,
    ):
        """
        root_dir is where model_hyperparams.yaml and results.tsv live. It
        defaults to "." (repo root) because a real (non-dry-run) train.py
        subprocess always reads model_hyperparams.yaml from its own
        directory -- only override it for dry-run/test runs that never
        invoke train.py. state_dir/reports_dir were already accepted here
        but never forwarded to Agent1/2/3 (they always hit the hardcoded
        repo-root paths regardless) -- that's fixed below.

        interactive: when True, a pipeline_validator ERROR (not just FATAL)
        pauses with a blocking y/n prompt. False (the default) never blocks
        -- see agents/pipeline_validator.py: a blocking prompt by default is
        exactly what kills unattended overnight runs on the first spurious
        warning.
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

        # Set the moment Agent 3 creates a new LLM-backed summary
        # (_process_training_result), consumed by whichever hyperparameter
        # decision comes next (sequential Phase 1, or the first slot of the
        # next parallel wave) and reset immediately -- so Agent 1's LLM
        # review fires once per new summary, not on every iteration
        # afterward that happens to still see it as "the latest."
        self._new_summary_ready = False

        self.config_path = config_path
        self.max_iterations = 100
        self.poll_interval = 5
        self.dry_run = dry_run
        self.interactive = interactive

        # Multi-GPU parallel search (dev/checks.txt item 1): orchestrator.*
        # in agents_config.yaml was previously dead config (never read) --
        # now wired up for real. parallel_enabled gates whether each
        # iteration even attempts GPU discovery; max_parallel_runs caps how
        # many concurrent GPUs/SSH sessions one wave may claim.
        orchestrator_config = self._load_orchestrator_config(config_path)
        self.parallel_enabled = bool(orchestrator_config.get("parallel", True))
        self.max_parallel_runs = int(orchestrator_config.get("max_parallel_runs", 4))
        self.parallel_hp_dir = Path(state_dir) / "parallel_hyperparams"

        # Tier 2 token-level XAI (see agents/xai_methods/token_methods.py)
        # costs real extra GPU time (roughly doubled wall-clock in testing),
        # so it isn't on for every run -- decided here each iteration, not
        # by Agent 1 or train.py, since it's an orchestration-level
        # sampling policy, not a hyperparameter search decision.
        self.token_xai_interval = int(self.agent1.agent1_config.get("token_xai_interval", 5))

        # Deterministic pipeline validation (agents/pipeline_validator.py):
        # timestamped run directories, never cleared on startup -- that
        # history is exactly what catches intermittent bugs -- pruned to the
        # most recent 10 instead.
        self.validation_dir = self.reports_dir / "pipeline_validation"
        pipeline_validator.prune_old_runs(self.validation_dir, keep=10)
        self.current_run_dir = pipeline_validator.new_run_dir(self.validation_dir)

        print("[Orchestrator] Initialization complete")

    def _load_orchestrator_config(self, config_path: str) -> Dict[str, Any]:
        if yaml is None or not Path(config_path).exists():
            return {}
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return config.get("orchestrator", {})

    def _kill_stale_remote_training(self, context: str = "a previous run") -> None:
        """Clean up any leftover train.py process on the remote server --
        see remote_runner.kill_stale_training_processes for the 5-condition
        identification (owned by us, running train.py, from our repo,
        carrying our own marker env var) that keeps this from ever touching
        another user's or another project's process.

        Called at campaign start (context="a previous run") AND at the
        start of every parallel wave (context="an earlier wave in this
        campaign"): a wave-dispatched run that exceeds its SSH timeout
        raises locally (logged as remote_error) but doesn't necessarily
        kill the remote process -- it can keep running and hold the GPU
        for the rest of the campaign unless something reclaims it. Cheap
        when nothing's stale (one SSH round-trip for the GPU-attached PID
        list, no further calls).
        """
        if self.dry_run or not remote_runner.is_remote_configured():
            return
        print(f"[Orchestrator] Checking the remote server for stale training processes from {context}...")
        killed = remote_runner.kill_stale_training_processes()
        if not killed:
            print("[Orchestrator]   None found.")
            return
        for entry in killed:
            escalation = " (SIGTERM didn't stop it in time -- escalated to SIGKILL)" if entry["escalated_to_sigkill"] else " (stopped cleanly with SIGTERM)"
            print(f"[Orchestrator]   Killed stale process PID {entry['pid']}: {entry['cmd']}{escalation}")

    def run(self, max_iterations: Optional[int] = None):
        """Main orchestration loop with structured evidence flow."""
        print("[Orchestrator] Starting autonomous multi-agent loop...\n")

        self._kill_stale_remote_training()

        iteration = 0
        report_batch: List[str] = []
        max_iterations = max_iterations or self.max_iterations

        while iteration < max_iterations:
            print(f"\n{'='*60}")
            print(f"[Orchestrator] Iteration {iteration + 1}")
            print(f"{'='*60}")

            wave_result = self._run_parallel_wave(iteration, report_batch, max_iterations)
            if wave_result is not None:
                iteration, report_batch, halted = wave_result
                if halted:
                    break
                continue

            print("\n[Orchestrator] Phase 1: Agent 1 proposes a new configuration")
            latest_summary = self._load_latest_summary()
            fresh_summary = self._new_summary_ready
            self._new_summary_ready = False
            if fresh_summary:
                print(f"[Orchestrator] *** Fresh summary available -- Agent 1 will use LLM-informed reasoning this iteration ***")
            recent_evidence = self.state_mgr.get_recent_evidence(limit=5)
            recent_results = self.state_mgr.get_all_results()[-3:]
            latest_val_bpb = None
            if recent_results:
                latest_result = recent_results[-1]
                latest_val_bpb = latest_result.get("val_bpb")
            best_before_decision = self.agent1.best_val_bpb
            new_hyperparams = self.agent1.decide_next_hyperparams(
                latest_summary=latest_summary,
                evidence=recent_evidence,
                iteration=iteration,
                latest_val_bpb=latest_val_bpb,
                recent_results=recent_results,
                fresh_summary=fresh_summary,
            )

            if new_hyperparams is None:
                print("\n[Orchestrator] STOPPING: Agent 1 stopped optimizing")
                break

            # token_xai_enabled: fixed interval (a floor, so fingerprint
            # history keeps accumulating even during a losing streak) OR
            # the run that just completed set a new best -- train.py fuses
            # training and fingerprinting into one process (no checkpoint
            # save/reload exists), so "fingerprint the best model" isn't
            # possible; this fingerprints the NEXT run after a new best is
            # found, as the closest available approximation.
            # new_hyperparams is the same dict object as
            # self.agent1.current_hyperparams (decide_next_hyperparams sets
            # that reference directly), so mutating it here is what makes
            # train_model's own _save_hyperparams() call persist this flag —
            # decide_next_hyperparams already wrote the file once, without
            # this key; train_model's save overwrites it with this included.
            new_best_just_set = latest_val_bpb is not None and latest_val_bpb < best_before_decision
            token_xai_due = (iteration % self.token_xai_interval == 0) or new_best_just_set
            new_hyperparams["token_xai_enabled"] = token_xai_due
            print(f"[Orchestrator] token_xai_enabled={token_xai_due} "
                  f"(interval={self.token_xai_interval}, new_best_just_set={new_best_just_set})")

            issues = pipeline_validator.validate_agent1_decision(
                self.agent1.last_decision_log, recent_evidence, latest_summary,
                decisions_dir=self.agent1.decisions_dir,
            )
            if self._handle_issues(iteration, issues):
                break

            print("\n[Orchestrator] Phase 2: Training")
            print(f"[Orchestrator] Training with hyperparams: {new_hyperparams}")
            train_result = self.agent1.train_model(
                new_hyperparams,
                dry_run=self.dry_run,
                iteration=iteration,
            )

            halted, report_batch = self._process_training_result(iteration, new_hyperparams, train_result, report_batch)
            if halted:
                break

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

    def _handle_issues(self, iteration: int, issues: List[pipeline_validator.Issue]) -> bool:
        """Prints + persists validator issues. Returns True when the
        orchestrator loop should halt: any FATAL issue (always, no prompt),
        or the user declining to continue past an ERROR in --interactive
        mode. ERROR/WARN alone never halt -- they tag the iteration
        "suspect" and the loop continues, per pipeline_validator's severity
        model.
        """
        if not issues:
            return False
        print(pipeline_validator.render_issues(issues))
        has_fatal = any(i.severity == pipeline_validator.FATAL for i in issues)
        has_error_or_worse = has_fatal or any(i.severity == pipeline_validator.ERROR for i in issues)
        pipeline_validator.write_iteration_issues(self.current_run_dir, iteration, issues, suspect=has_error_or_worse)

        if has_fatal:
            print("[Orchestrator] FATAL issue detected -- halting immediately.")
            return True

        if self.interactive and has_error_or_worse:
            answer = input("[Orchestrator] ERROR issue(s) detected. Continue past this? [y/N]: ").strip().lower()
            if answer != "y":
                print("[Orchestrator] User chose to stop.")
                return True
        return False

    def _process_training_result(
        self,
        iteration: int,
        hyperparams: Dict[str, Any],
        train_result: Dict[str, Any],
        report_batch: List[str],
    ) -> Tuple[bool, List[str]]:
        """Everything that happens after one training run completes: log to
        results.tsv/state, Agent 2 XAI analysis, Agent 3 batch
        summarization, and pipeline_validator checks after each phase.
        Shared verbatim by the sequential single-run path and the parallel
        wave dispatcher (_run_parallel_wave) so a result is handled
        identically regardless of which GPU/thread produced it -- extracted
        from what used to be inline Phase 2/3/4 body, no behavior change
        for the sequential caller. Returns (halt, updated report_batch);
        halt True means the caller should stop the whole campaign (a FATAL
        issue, or the user declining to continue past an ERROR in
        --interactive mode).
        """
        print(f"[Orchestrator] Training result: val_bpb={train_result.get('val_bpb', 'N/A')}, status={train_result.get('status', 'unknown')}")
        result_payload = TrainingResult(
            run_id=f"run_{iteration:04d}",
            hyperparams=hyperparams,
            val_bpb=train_result.get("val_bpb", float("inf")),
            training_time=train_result.get("training_time", 0.0),
            checkpoint_path=train_result.get("checkpoint_path"),
            status=train_result.get("status", "ok"),
            metadata=train_result,
        )
        self.state_mgr.add_result(result_payload.to_dict())
        self.state_mgr.update_val_bpb(result_payload.run_id, result_payload.val_bpb)
        log_result(result_payload.run_id, hyperparams, train_result, results_path=str(self.results_path))
        print(f"[Orchestrator] Result logged: {result_payload.run_id}")

        issues = pipeline_validator.validate_training_result(train_result, hyperparams)
        if self._handle_issues(iteration, issues):
            return True, report_batch

        print("\n[Orchestrator] Phase 3: Analyzing result with Agent 2")
        evidence = self.agent2.analyze_result(result_payload.to_dict())
        if evidence is not None:
            evidence_payload = evidence.to_dict()
            self.state_mgr.add_evidence(evidence_payload)
            report_batch = report_batch + [evidence_payload["report_id"]]
            self.state_mgr.link_model_to_report(result_payload.run_id, evidence_payload["report_id"])
            print(f"[Orchestrator] Agent 2 analysis complete: {evidence_payload['report_id']}")

            issues = pipeline_validator.validate_agent2_report(evidence_payload)
            if self._handle_issues(iteration, issues):
                return True, report_batch
        else:
            print(f"[Orchestrator] Agent 2: no analysis available")
            # Should not currently happen (analyze_result never returns
            # None in the present implementation) -- a canary in case a
            # future change introduces a real None-return path, same
            # "structurally impossible, so flag it loudly if it ever
            # occurs" pattern used elsewhere in this file.
            issues = [pipeline_validator.Issue(
                pipeline_validator.WARN, "agent2",
                "Agent 2 produced no analysis for this run (evidence is None) -- unexpected given the "
                "current implementation, investigate if this occurs",
                {"run_id": result_payload.run_id},
            )]
            if self._handle_issues(iteration, issues):
                return True, report_batch

        print("\n[Orchestrator] Phase 4: Aggregating evidence with Agent 3")
        print(f"[Orchestrator] Batch size: {len(report_batch)} reports")
        if self.agent3.should_create_summary(len(report_batch)):
            print(f"[Orchestrator] Creating summary from {len(report_batch)} reports...")
            summary = self.agent3.analyze_and_summarize(report_batch)
            self.state_mgr.add_summary(summary.to_dict())
            self.state_mgr.set_latest_summary(summary.summary_id, iteration)
            report_batch = []
            if self.agent3.use_llm:
                self._new_summary_ready = True
                print(f"[Orchestrator] *** NEW SUMMARY CREATED (LLM): {summary.summary_id} -- "
                      f"Agent 1 will read it with LLM reasoning next ***")
            else:
                print(f"[Orchestrator] Summary created: {summary.summary_id}")

            issues = pipeline_validator.validate_agent3_summary(summary.to_dict(), total_reports=self.agent2.report_counter)
            if self._handle_issues(iteration, issues):
                return True, report_batch
        else:
            print(f"[Orchestrator] Summary threshold not reached (need {self.agent3.batch_size}, have {len(report_batch)})")
            issues = pipeline_validator.validate_batch_accumulation(len(report_batch), self.agent3.batch_size)
            if self._handle_issues(iteration, issues):
                return True, report_batch

        return False, report_batch

    def _write_temp_hyperparams(self, hyperparams: Dict[str, Any], path: Path) -> None:
        if yaml is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(hyperparams, f)

    def _run_parallel_wave(
        self, iteration: int, report_batch: List[str], max_iterations: int,
    ) -> Optional[Tuple[int, List[str], bool]]:
        """Multi-GPU parallel search (dev/checks.txt item 1): when the
        remote server currently has 2+ live-free GPUs (discovered fresh
        every call -- no static exclusion list), decide N hyperparameter
        proposals and train them concurrently, one per GPU, instead of the
        default one-run-at-a-time loop. Returns None whenever parallel
        dispatch isn't applicable this round (dry run, parallel disabled,
        remote not configured, or fewer than 2 GPUs currently free) -- the
        caller then falls through to the untouched single-run sequential
        path, so local/dry-run/single-GPU users see zero behavior change.

        Each slot's hyperparameter decision reuses the exact same
        latest_summary/recent_evidence/recent_results/latest_val_bpb inputs
        (no new results exist mid-wave -- they only land in state once
        training completes), so diversity across slots comes from each
        call's own randomized search (Sobol cold start, or a different EI
        random seed per slot) rather than sequential feedback -- a known,
        deliberate simplification of true batch Bayesian optimization.
        """
        if self.dry_run or not self.parallel_enabled:
            return None
        if not remote_runner.is_remote_configured():
            return None

        # Reclaim any GPU still held by one of our own leftover processes
        # (e.g. a previous wave's run that exceeded its SSH timeout and got
        # logged as remote_error locally without actually dying remotely)
        # before this wave's discovery call decides what's available.
        self._kill_stale_remote_training(context="an earlier wave in this campaign")

        candidates = remote_runner.discover_available_gpus()[: self.max_parallel_runs]
        if len(candidates) < 2:
            return None

        wave_size = min(len(candidates), max_iterations - iteration)
        print(f"[Orchestrator] Parallel wave: {len(candidates)} GPU(s) available -- "
              f"dispatching {wave_size} concurrent run(s) on GPUs {[c['index'] for c in candidates[:wave_size]]}")

        latest_summary = self._load_latest_summary()
        recent_evidence = self.state_mgr.get_recent_evidence(limit=5)
        recent_results = self.state_mgr.get_all_results()[-3:]
        latest_val_bpb = None
        if recent_results:
            latest_val_bpb = recent_results[-1].get("val_bpb")
        best_before_decision = self.agent1.best_val_bpb

        slots: List[Tuple[int, Dict[str, Any], int, Path]] = []
        decision_halt = False
        for i in range(wave_size):
            iteration_for_slot = iteration + i
            # Only the first decision in the wave can ever consume a
            # pending fresh-summary flag (it's reset the instant it's
            # read) -- keeps LLM usage to once per new summary even when
            # a whole wave of slots gets decided back-to-back.
            fresh_summary = self._new_summary_ready
            self._new_summary_ready = False
            if fresh_summary:
                print(f"[Orchestrator] *** Fresh summary available -- Agent 1 will use LLM-informed "
                      f"reasoning for iteration {iteration_for_slot} ***")
            new_hyperparams = self.agent1.decide_next_hyperparams(
                latest_summary=latest_summary,
                evidence=recent_evidence,
                iteration=iteration_for_slot,
                latest_val_bpb=latest_val_bpb,
                recent_results=recent_results,
                fresh_summary=fresh_summary,
            )
            if new_hyperparams is None:
                print("\n[Orchestrator] STOPPING: Agent 1 stopped optimizing")
                decision_halt = True
                break

            new_best_just_set = latest_val_bpb is not None and latest_val_bpb < best_before_decision
            token_xai_due = (iteration_for_slot % self.token_xai_interval == 0) or new_best_just_set
            new_hyperparams["token_xai_enabled"] = token_xai_due

            issues = pipeline_validator.validate_agent1_decision(
                self.agent1.last_decision_log, recent_evidence, latest_summary,
                decisions_dir=self.agent1.decisions_dir,
            )
            if self._handle_issues(iteration_for_slot, issues):
                decision_halt = True
                break

            gpu_index = candidates[i]["index"]
            hp_path = self.parallel_hp_dir / f"run_{iteration_for_slot:04d}.yaml"
            self._write_temp_hyperparams(new_hyperparams, hp_path)
            print(f"[Orchestrator] Wave dispatch: GPU {gpu_index} <- iteration {iteration_for_slot} "
                  f"(n_layer={new_hyperparams.get('n_layer')}, matrix_lr={new_hyperparams.get('matrix_lr')})")
            slots.append((gpu_index, new_hyperparams, iteration_for_slot, hp_path))

        if not slots:
            return (iteration, report_batch, True) if decision_halt else None

        remote_runner.sync_remote_code()

        # Each GPU gets its own pinned terminal line for the duration of the
        # wave (see agents/live_progress.py) -- without this, every thread's
        # old \r-based in-place progress update fights the others for the
        # same cursor position and garbles the output once 2+ GPUs print
        # concurrently.
        gpu_labels = [f"GPU{gpu_index}" for gpu_index, _hp, _it, _hp_path in slots]
        results_by_iteration: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        with MultiGpuProgressDisplay(gpu_labels) as display:
            with ThreadPoolExecutor(max_workers=len(slots)) as executor:
                future_map = {}
                for gpu_index, hp, it, hp_path in slots:
                    future = executor.submit(
                        remote_runner.run_training_remote,
                        hyperparams_local_path=str(hp_path),
                        gpu_index=gpu_index,
                        # Distinct per-run remote filename -- without this, every
                        # concurrent slot in the wave uploads to the same shared
                        # default (model_hyperparams.yaml) and their SFTP writes
                        # race, corrupting each other (paramiko's post-upload
                        # size check then fails with "size mismatch in put!").
                        hp_remote_name=f"model_hyperparams_run{it:04d}.yaml",
                        run_label=f"GPU{gpu_index}",
                        timeout=self.agent1.training_budget + 120,
                        skip_sync=True,
                        display=display,
                    )
                    future_map[future] = (gpu_index, hp, it)

                for future in as_completed(future_map):
                    gpu_index, hp, it = future_map[future]
                    try:
                        train_result = future.result()
                    except Exception as e:
                        display.print_line(f"[Orchestrator] Wave slot GPU {gpu_index} (iteration {it}) failed: {e}")
                        train_result = {
                            "val_bpb": float("inf"), "error": str(e),
                            "status": "remote_error", "device": gpu_index,
                        }
                    display.print_line(
                        f"[Orchestrator] Wave slot GPU {gpu_index} complete: "
                        f"val_bpb={train_result.get('val_bpb', 'N/A')} status={train_result.get('status', 'unknown')} "
                        f"(iteration {it})"
                    )
                    results_by_iteration[it] = (hp, train_result)

        # Process in iteration order (not completion order) so results.tsv,
        # decision logs, and Agent 2/3 report numbering stay monotonically
        # ordered -- every "most recent" lookup elsewhere in this codebase
        # assumes that.
        halt = decision_halt
        for it in sorted(results_by_iteration.keys()):
            hp, train_result = results_by_iteration[it]
            slot_halt, report_batch = self._process_training_result(it, hp, train_result, report_batch)
            halt = halt or slot_halt

        next_iteration = iteration + len(slots)
        return next_iteration, report_batch, halt

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
    parser.add_argument("--interactive", action="store_true",
                         help="Prompt (blocking) before continuing past a pipeline_validator ERROR. "
                              "Off by default so unattended runs never block on a spurious warning.")

    args = parser.parse_args()

    orchestrator = Orchestrator(config_path=args.config, dry_run=args.dry_run, interactive=args.interactive)
    orchestrator.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
