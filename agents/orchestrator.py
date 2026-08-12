"""Orchestrator: Coordinates a structured multi-agent optimization loop."""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None

from agents import allocator
from agents import pipeline_validator
from agents import remote_runner
from agents.live_progress import MultiGpuProgressDisplay
from agents.agent1_training_specialist import Agent1TrainingSpecialist
from agents.agent2_xai_specialist import Agent2XAISpecialist
from agents.agent3_report_analyst import Agent3ReportAnalyst, _read_text_tolerant
from agents.agent4_landscape_explorer import Agent4LandscapeExplorer
from agents.protocols import AnalysisEvidence, SummaryEvidence, TrainingResult
from state.regions import CAPACITY_PAUSED, RegionRegistry
from state.results_analysis import (
    SYNTHETIC_STATUSES, at_current_budget, load_results,
)
from state.state_manager import StateManager
from state.results_logger import log_result


# Consecutive unreachable-remote waves before the campaign stops. A shared
# server on a flaky network is expected to blip (remote_runner retries each
# connect a few times already); a sustained outage is not something to grind
# through, because every path past it fabricates rather than measures --
# Agent1.train_model's last-resort fallback is _simulate_training_result,
# whose val_bpb is a hand-tuned formula, not a measurement. Stopping with the
# iteration budget intact beats filling results.tsv with invented numbers.
REMOTE_FAILURE_HALT_STREAK = 3
REMOTE_RETRY_SLEEP_S = 60

# Consecutive waves in which EVERY slot came back remote_error before the
# campaign stops. Distinct from the streak above, which only counts waves that
# could not be dispatched at all: a wave whose sync succeeded and whose slots
# then all failed used to be indistinguishable from progress, so a campaign
# could burn its entire iteration budget writing val_bpb=inf rows. Observed
# for real -- two consecutive 4-slot waves produced eight inf rows and the
# loop would happily have continued to twenty.
ALL_SLOTS_FAILED_HALT_STREAK = 2


def _format_duration(seconds: float) -> str:
    """Human-readable wall-clock duration for terminal/log output --
    "12.3s", "5m 12s", or "1h 03m 12s" depending on magnitude, instead of a
    bare (and for a multi-hour campaign, hard-to-read) seconds float."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem_seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {rem_seconds:02d}s"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{hours}h {rem_minutes:02d}m {rem_seconds:02d}s"


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
        self.agent3 = Agent3ReportAnalyst(config_path, root_dir=root_dir, state_dir=state_dir, reports_dir=reports_dir)
        # Agent 4 proposes new regions and judges the lifecycle of running
        # ones. It never proposes a training configuration -- that is Agent 1
        # scoped to a region. Constructed even when disabled so
        # `self.agent4.enabled` is the single switch rather than a None check
        # scattered through the loop.
        self.agent4 = Agent4LandscapeExplorer(config_path, root_dir=root_dir, state_dir=state_dir, reports_dir=reports_dir)
        # The live regions, shared with Agent 4 (same file) so there is one
        # store rather than one per reader.
        self.registry = RegionRegistry(str(Path(state_dir) / "regions.json"))
        # region_id -> the val_bpb that region's newest run produced, carried
        # from one wave to the next so a region-scoped decision sees its own
        # last result rather than the campaign's.
        self._last_val_bpb_by_region: Dict[str, float] = {}

        # Recover the campaign record from results.tsv. best_val_bpb starts at
        # inf and is only advanced by results this process sees, so a restart
        # forgot every record ever set -- while the halt messages below tell
        # the operator to "fix connectivity and restart; the campaign resumes
        # from results.tsv". It did resume the runs; it did not resume the one
        # number the whole search is measured against. The consequences were
        # quiet rather than dramatic: no holdout threshold until something beat
        # infinity, a spurious "new best" on the first run back (which turns on
        # token-level XAI for it), and a final report quoting this process's
        # best as the campaign's.
        self._recover_campaign_best()

        # The decision log the pipeline_validator call after each decision
        # should check. Always Agent 1's now -- Agent 4 stopped proposing
        # training configurations when the exploration window was replaced by
        # continuous multi-region search, and writes verdict_*.json rather
        # than the decision_*.json the validator walks. Kept as a pair of
        # attributes rather than inlined so the validator call site does not
        # have to know which agent that is.
        self._active_decision_log: Optional[Dict[str, Any]] = None
        self._active_decisions_dir: Path = self.agent1.decisions_dir
        # Two different remote failures, two counters, because conflating
        # them makes each one wrong. "Cannot open a connection" means the
        # server is unreachable; "connected but git sync failed" means it is
        # reachable and the repo is the problem. Sharing one counter meant a
        # successful connect cleared accumulated sync failures, so a
        # persistently broken sync could never reach the halt threshold.
        # Each counts CONSECUTIVE occurrences of its own failure, not a
        # lifetime total.
        self._remote_unreachable_streak = 0
        self._sync_failure_streak = 0
        # Consecutive waves in which every dispatched slot failed.
        self._all_slots_failed_streak = 0

        # Set the moment Agent 3 creates a new LLM-backed summary
        # (_process_training_result), consumed by whichever hyperparameter
        # decision comes next (sequential Phase 1, or the first slot of the
        # next parallel wave) and reset immediately -- so Agent 1's LLM
        # review fires once per new summary, not on every iteration
        # afterward that happens to still see it as "the latest."
        self._new_summary_ready = False

        # Set when a completed run beats the campaign record, consumed by the
        # next hyperparameter decision (which turns on token-level XAI for it).
        # An explicit flag rather than the old `latest_val_bpb <
        # best_before_decision` comparison: the orchestrator now updates
        # best_val_bpb the moment a result lands -- it has to, because a
        # region-scoped decision only ever sees its own region's best -- so by
        # the time the next decision reads it, the comparison can never be
        # true. Same intent, stated directly instead of inferred from a race.
        self._new_best_just_set = False

        self.config_path = config_path
        orchestrator_config = self._load_orchestrator_config(config_path)
        # Read from config rather than hardcoded. Both keys have been in
        # agents_config.yaml's orchestrator: block all along while the class
        # ignored them -- editing orchestrator.max_iterations did nothing,
        # which is the same "setting that reads as live and isn't" trap as
        # the removed agent4.min_regions. --iterations still overrides.
        self.max_iterations = int(orchestrator_config.get("max_iterations", 100))
        self.poll_interval = int(orchestrator_config.get("poll_interval_seconds", 5))
        self.dry_run = dry_run
        self.interactive = interactive

        # Multi-GPU parallel search (dev/checks.txt item 1): orchestrator.*
        # in agents_config.yaml was previously dead config (never read) --
        # now wired up for real. parallel_enabled gates whether each
        # iteration even attempts GPU discovery; max_parallel_runs caps how
        # many concurrent GPUs/SSH sessions one wave may claim.
        self.parallel_enabled = bool(orchestrator_config.get("parallel", True))
        self.max_parallel_runs = int(orchestrator_config.get("max_parallel_runs", 4))
        self.parallel_hp_dir = Path(state_dir) / "parallel_hyperparams"

        # Tier 2 token-level XAI (see agents/xai_methods/token_methods.py)
        # costs real extra GPU time (roughly doubled wall-clock in testing),
        # so it isn't on for every run -- decided here each iteration, not
        # by Agent 1 or train.py, since it's an orchestration-level
        # sampling policy, not a hyperparameter search decision.
        self.token_xai_interval = int(self.agent1.agent1_config.get("token_xai_interval", 5))

        # Same shape of policy: whether a run that sets a new best should also
        # be scored on the held-out shard, so selection bias against the one
        # pinned validation shard is tracked continuously instead of only by
        # a manual end-of-campaign script. See _set_holdout_threshold.
        self.holdout_on_new_best = bool(
            self.agent1.agent1_config.get("holdout_on_new_best", True))

        # Deterministic pipeline validation (agents/pipeline_validator.py):
        # timestamped run directories, never cleared on startup -- that
        # history is exactly what catches intermittent bugs -- pruned to the
        # most recent 10 instead.
        self.validation_dir = self.reports_dir / "pipeline_validation"
        pipeline_validator.prune_old_runs(self.validation_dir, keep=10)
        self.current_run_dir = pipeline_validator.new_run_dir(self.validation_dir)

        print("[Orchestrator] Initialization complete")

    def _recover_campaign_best(self) -> None:
        """Seed Agent 1's best_val_bpb from results.tsv, ignoring synthetic
        rows (dry_run/simulated), whose val_bpb is a hand-tuned formula rather
        than a measurement and must never become the record."""
        try:
            # Only runs at the budget in force: a "best" carried over from a
            # longer budget is unbeatable here for a reason that has nothing to
            # do with the search (1.2486 at 12.5M against 1.7063 at 4.19M), so
            # every improvement check would fail forever.
            rows = at_current_budget(load_results(str(self.results_path)))
        except Exception as e:  # pragma: no cover - never block startup on this
            print(f"[Orchestrator] Could not read {self.results_path} for the campaign best: {e}")
            return
        finite = [
            float(r["val_bpb"]) for r in rows
            if isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])
            and r.get("status") not in SYNTHETIC_STATUSES
        ]
        if not finite:
            return
        self.agent1.best_val_bpb = min(finite)
        print(f"[Orchestrator] Resuming: campaign best {self.agent1.best_val_bpb:.6f} "
              f"recovered from {len(finite)} previous run(s)")

    def _load_orchestrator_config(self, config_path: str) -> Dict[str, Any]:
        if yaml is None or not Path(config_path).exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config.get("orchestrator", {})

    def _kill_stale_remote_training(self, context: str = "a previous run", client=None) -> None:
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
        killed = remote_runner.kill_stale_training_processes(client=client)
        if not killed:
            print("[Orchestrator]   None found.")
            return
        for entry in killed:
            escalation = " (SIGTERM didn't stop it in time -- escalated to SIGKILL)" if entry["escalated_to_sigkill"] else " (stopped cleanly with SIGTERM)"
            print(f"[Orchestrator]   Killed stale process PID {entry['pid']}: {entry['cmd']}{escalation}")

    def _next_run_index(self) -> int:
        """One past the highest run_NNNN already in results.tsv.

        Reads RESULTS.TSV, deliberately, and not state_manager's metadata.json:
        results.tsv is the file log_result appends to and therefore the only
        place two runs can collide, while metadata.json is per-session
        bookkeeping that a fresh process starts empty. Reading the latter would
        report "0 recorded runs" against a results.tsv holding 32 of them and
        reissue every id -- the exact failure this method exists to prevent.

        Ids that do not parse (an experiment script's `geom_...` or `size_...`)
        are ignored: they live in their own results files, and anything
        unexpected here should not be able to push the numbering somewhere
        strange.
        """
        from state.results_analysis import load_results

        highest = -1
        try:
            for row in load_results(str(self.results_path)):
                run_id = str(row.get("run_id", ""))
                if not run_id.startswith("run_"):
                    continue
                try:
                    highest = max(highest, int(run_id[4:]))
                except ValueError:
                    continue
        except Exception as e:  # a missing/corrupt results file is a fresh start
            print(f"[Orchestrator] Could not read existing results ({e}); starting at 0.")
            return 0
        return highest + 1

    def run(self, max_iterations: Optional[int] = None):
        """Main orchestration loop with structured evidence flow."""
        print("[Orchestrator] Starting autonomous multi-agent loop...\n")
        run_start = time.time()

        self._kill_stale_remote_training()

        # RESUME, never restart the numbering. run_id is f"run_{iteration:04d}",
        # so starting from 0 against an existing results.tsv silently reissues
        # ids that are already taken -- and load_results de-duplicates by
        # run_id, so the collision does not error, it quietly drops one of the
        # two runs. A campaign relaunched against 32 runs of history would have
        # overwritten every one of them.
        iteration = self._next_run_index()
        report_batch: List[str] = []
        # `max_iterations` is therefore how many MORE runs to do this session,
        # which is also the only reading that stays meaningful on a resume:
        # "100 total" against a campaign already past 100 would exit instantly.
        max_iterations = iteration + (max_iterations or self.max_iterations)
        if iteration:
            print(f"[Orchestrator] Resuming after {iteration} recorded run(s); "
                  f"ids continue at run_{iteration:04d}.")

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
            # The sequential path is not only the dry-run path: the wave
            # dispatcher returns None whenever fewer than 2 GPUs are free, so
            # a busy server drops the whole campaign down here. Without a
            # region that silently reverts to single-search mode -- rows land
            # in results.tsv with a blank region_id, no region's history
            # records them, and the lifecycle thresholds that count "runs
            # this region spent" quietly stop counting. Run it as a region so
            # behaviour does not depend on how busy the server happens to be.
            sequential_region = self._sequential_region(iteration)
            new_hyperparams = self._decide_next_hyperparams(
                iteration=iteration,
                latest_summary=latest_summary,
                recent_evidence=recent_evidence,
                recent_results=recent_results,
                latest_val_bpb=latest_val_bpb,
                fresh_summary=fresh_summary,
                region=sequential_region,
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
            new_best_just_set = self._new_best_just_set
            self._new_best_just_set = False
            token_xai_due = (iteration % self.token_xai_interval == 0) or new_best_just_set
            new_hyperparams["token_xai_enabled"] = token_xai_due
            print(f"[Orchestrator] token_xai_enabled={token_xai_due} "
                  f"(interval={self.token_xai_interval}, new_best_just_set={new_best_just_set})")
            self._set_holdout_threshold(new_hyperparams)

            issues = pipeline_validator.validate_agent1_decision(
                self._active_decision_log, recent_evidence, latest_summary,
                decisions_dir=self._active_decisions_dir,
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

        total_elapsed = time.time() - run_start
        summary = self.agent3.get_latest_summary_object()
        print(f"\n{'='*60}")
        print("[Orchestrator] MULTI-AGENT LOOP COMPLETE")
        print(f"Total iterations: {iteration}")
        print(f"Total run time: {_format_duration(total_elapsed)}")
        print(f"Final best val_bpb: {self.agent1.best_val_bpb:.6f}")
        print(f"Total API cost: ${self.agent1.total_api_cost:.2f}")
        print(f"{'='*60}\n")
        return summary

    def _set_holdout_threshold(self, hyperparams: Dict[str, Any]) -> None:
        """Ask train.py to also score the held-out shard if this run turns out
        to beat the campaign's best.

        The search compares every run against ONE pinned validation shard, and
        this campaign has now made 500+ accept/reject decisions against it --
        a multiple-comparisons problem prepare.py's HOLDOUT_SHARD exists
        specifically to detect (a config can look better on that shard than it
        really is). Until now the holdout was only reachable via a manual
        end-of-campaign script, so drift was never actually tracked.

        Passing a threshold rather than a boolean is what makes this exact:
        whether a run is a new best isn't known until its val_bpb exists, and
        by then train.py's process is over and the model is gone (no
        checkpoint save/reload). train.py evaluates val_bpb and this
        comparison in the same process, so the holdout number always comes
        from the very model that set the record.

        Costs one extra eval pass, and only on a genuine new best -- which
        for this campaign means roughly never (the best has not moved in 300+
        runs). No threshold is set until a finite best exists, so a fresh
        campaign doesn't holdout-eval its opening runs against infinity.
        """
        if not self.holdout_on_new_best:
            return
        best = self.agent1.best_val_bpb
        if isinstance(best, (int, float)) and math.isfinite(best):
            hyperparams["holdout_eval_if_below"] = float(best)

    def _sequential_region(self, iteration: int):
        """The single region a one-run-at-a-time iteration belongs to, or
        None when regions do not apply.

        Returns None for a dry run: nothing is trained, so attributing the
        result to a region would put fabricated numbers into that region's
        history and into every lifecycle threshold computed from it.
        """
        if self.dry_run or not self.agent4.enabled:
            return None
        _live, plan = self._plan_wave(1, iteration)
        if not plan.assignments:
            return None
        return self.registry.get(plan.assignments[0])

    def _plan_wave(self, n_gpus: int, at_run: int) -> Tuple[List[Any], allocator.AllocationPlan]:
        """Decide which region each of this wave's GPUs works on.

        Runs Agent 4's maintenance FIRST (merge converged regions, judge every
        live one) so the allocation is made against this wave's lifecycle
        state rather than last wave's -- otherwise a region retired moments
        ago still gets a GPU, and a region that just merged gets two.

        Then, in order: fill any slot the plan wants a NEW region for; if
        Agent 4 can't produce one (too little data to fit a surrogate, or
        every candidate lands somewhere already ruled out) fall back to
        resuming the best PAUSED region, which is exactly the "if nothing
        better exists, unpause and continue there" case; and if that fails
        too, hand the slot to the best live region rather than idling a GPU.
        """
        self.agent4.maintain(self.registry, at_run)

        live = self.registry.active()
        plan = allocator.plan(live, n_gpus, self.agent4.max_regions)

        # Resume BEFORE proposing. A region that already has runs invested and
        # a measured score is a better use of a freed GPU than a speculative
        # new one, and resume_best_paused takes capacity pauses first -- those
        # regions lost their slot to a busy server, not to a judgement. The
        # other order left four partially-explored regions idle while the
        # allocator opened brand-new ones the moment GPUs came back, which is
        # exactly the state the 32-run campaign ended in.
        for _ in range(plan.assignments.count(None)):
            resumed = self.agent4.resume_best_paused(self.registry, at_run)
            if resumed is None:
                break
            plan.assignments[plan.assignments.index(None)] = resumed.region_id

        wanted_new = plan.assignments.count(None)
        if wanted_new:
            # Same rule as the surrogate: a landscape built across budgets maps
            # a 0.45 bpb cliff that no hyperparameter caused, and would send
            # every region toward the side of it that simply trained longer.
            rows = at_current_budget(load_results(str(self.results_path)))
            opened = self.agent4.propose_regions(
                rows, self.registry, wanted_new, at_run, self.agent1.best_val_bpb,
                # Agent 2's interpretability findings reach the search here, and
                # only here: every one of them is about depth, width or heads,
                # which only Agent 4 can change.
                evidence=self.state_mgr.get_recent_evidence(limit=10))
            for region in opened:
                plan.assignments[plan.assignments.index(None)] = region.region_id

        # Campaign cold start. Agent 4 cannot propose anything until a
        # surrogate fits (15 usable runs), and those runs cannot exist until
        # something is dispatched -- so a fresh campaign would deadlock with
        # zero regions and zero results forever.
        #
        # The resolution is that Agent 1's Sobol cold start IS exploration,
        # and always was: it is a space-filling sample over the whole search
        # space. It just needs to belong to a region so its runs are
        # attributed and its planner state has a home. One bootstrap region
        # is opened at the default center; after ~15 runs land in results.tsv
        # the surrogate fits, Agent 4 starts proposing real regions on its
        # three criteria, and this never runs again.
        if None in plan.assignments and not self.registry.active():
            bootstrap = self.registry.open_region(
                self.agent1.current_hyperparams, at_run=at_run, origin="bootstrap")
            self.registry.save()
            print(f"[Orchestrator] Campaign cold start: opened bootstrap region "
                  f"{bootstrap.region_id} -- Agent 1's Sobol cold start runs here until "
                  f"there is enough history for Agent 4 to propose real ones")

        live = self.registry.active()
        if None in plan.assignments:
            fallback = live[0].region_id if live else None
            remaining = plan.assignments.count(None)
            if fallback is None:
                # Structurally unreachable now that the cold start opens a
                # bootstrap region -- kept as a loud guard rather than an
                # assumption, since the failure it catches (dispatching a
                # configuration no criterion chose) would silently put rows in
                # results.tsv that nothing asked for.
                print(f"[Orchestrator] No region available for {remaining} slot(s) -- "
                      f"not enough history to propose one yet, skipping them")
                plan.assignments = [a for a in plan.assignments if a is not None]
            else:
                print(f"[Orchestrator] {remaining} slot(s) had no new region to open -- "
                      f"reinforcing {fallback} instead")
                plan.assignments = [a if a is not None else fallback for a in plan.assignments]

        # CAPACITY_PAUSED, not PAUSED: nothing was learned about these
        # regions, they just lost a GPU to a busier server. The distinction is
        # what lets resume_best_paused bring them back first when capacity
        # returns, instead of treating them like regions judged to have
        # stopped paying.
        for region_id in plan.to_pause:
            region = self.registry.get(region_id)
            if region is not None:
                region.set_flag(CAPACITY_PAUSED, at_run)
        if plan.to_pause:
            self.registry.save()

        if plan.assignments:
            print(allocator.describe(plan, live))
        return live, plan

    def _decide_next_hyperparams(
        self,
        iteration: int,
        latest_summary: Optional[str],
        recent_evidence: List[Dict[str, Any]],
        recent_results: List[Dict[str, Any]],
        latest_val_bpb: Optional[float],
        fresh_summary: bool,
        slot: int = 0,
        region: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Agent 1 decides every training configuration; `region` says which
        region it is deciding for.

        Agent 4 no longer appears here at all. Under the old window model this
        function had to route between two owners, and the one that won took
        over every slot; now a wave's slots belong to different regions at the
        same time, which a single-owner branch cannot express.

        The region scope (Agent1TrainingSpecialist.search_region) is what
        makes concurrent decisions independent: each gets its own center, its
        own planner state file, its own frozen set and block rotation, and its
        own EI reference.
        """
        if region is None:
            new_hyperparams = self.agent1.decide_next_hyperparams(
                latest_summary=latest_summary,
                evidence=recent_evidence,
                iteration=iteration,
                latest_val_bpb=latest_val_bpb,
                recent_results=recent_results,
                fresh_summary=fresh_summary,
            )
            self._active_decision_log = self.agent1.last_decision_log
            self._active_decisions_dir = self.agent1.decisions_dir
            return new_hyperparams

        with self.agent1.search_region(region):
            new_hyperparams = self.agent1.decide_next_hyperparams(
                latest_summary=latest_summary,
                evidence=recent_evidence,
                iteration=iteration,
                # This region's own newest result, not the campaign's newest.
                # Feeding it another region's run is what used to make
                # stagnation detection compare two different places in the
                # space and call the difference a trend.
                latest_val_bpb=self._last_val_bpb_by_region.get(region.region_id),
                recent_results=recent_results,
                fresh_summary=fresh_summary,
            )
        self._active_decision_log = self.agent1.last_decision_log
        self._active_decisions_dir = self.agent1.decisions_dir
        if new_hyperparams is not None:
            new_hyperparams["region_id"] = region.region_id
        return new_hyperparams

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

        # Attribute the run to the region that dispatched it, so that region's
        # own history -- which every lifecycle threshold counts in -- is the
        # record of what IT spent, never what happened to land near its
        # anchor. See RegionRegistry.assign_run.
        region_id = hyperparams.get("region_id")
        if region_id:
            self.registry.assign_run(region_id, result_payload.run_id,
                                     result_payload.val_bpb, center=hyperparams)
            self.registry.save()
            if isinstance(result_payload.val_bpb, (int, float)) and math.isfinite(result_payload.val_bpb):
                self._last_val_bpb_by_region[region_id] = result_payload.val_bpb

        # Agent 1 records a new best inside decide_next_hyperparams, but under
        # a region scope that call sees only the REGION's best (EI needs a
        # local reference, see search_region). So the campaign record has to
        # be maintained here, or a record set inside a region would leave the
        # global f_best stale for the rest of the campaign.
        # A FABRICATED SCORE MUST NEVER BECOME THE RECORD. _simulate_training_result
        # returns a hand-tuned formula of the iteration index, not a measurement,
        # and _recover_campaign_best already refuses those on restart -- but the
        # in-process update did not, so a campaign whose remote was broken set
        # its best from an invented number on the first iteration. That happened:
        # a malformed launch command made every run fall back to simulation, and
        # 1.251122 was recorded as the campaign best. Everything downstream
        # follows from the record -- the holdout trigger, "is this an
        # improvement", the final report.
        if (isinstance(result_payload.val_bpb, (int, float))
                and result_payload.status not in SYNTHETIC_STATUSES
                and result_payload.val_bpb < self.agent1.best_val_bpb):
            print(f"[Orchestrator] New campaign best: {result_payload.val_bpb:.6f} "
                  f"(was {self.agent1.best_val_bpb:.6f}, region {region_id or '-'})")
            self.agent1.best_val_bpb = result_payload.val_bpb
            self._new_best_just_set = True

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
        with open(path, "w", encoding="utf-8") as f:
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
        # ONE connection for this entire wave. The server rate-limits SSH
        # connects (~60-90s block after the first, measured -- see
        # remote_runner.CONNECT_BACKOFF_MULTIPLIER), and this wave used to open
        # seven: stale check, GPU discovery, code sync, and one per concurrent
        # training run. The first succeeded and the rest were dropped, so every
        # wave lost most of its slots to remote_error. paramiko carries many
        # channels over one TCP connection, so all of it now shares this.
        try:
            wave_client = remote_runner.open_client()
        except Exception as e:
            print(f"[Orchestrator] Could not reach the remote server: {e} -- skipping this wave")
            self._remote_unreachable_streak += 1
            if self._remote_unreachable_streak >= REMOTE_FAILURE_HALT_STREAK:
                print(f"[Orchestrator] HALTING: could not connect to the remote server "
                      f"{self._remote_unreachable_streak} times in a row.")
                return (iteration, report_batch, True)
            time.sleep(REMOTE_RETRY_SLEEP_S)
            return None

        # Reaching the server is exactly what this streak measures the absence
        # of, so a successful connect clears it here rather than only after a
        # successful sync further down. Otherwise a wave that connects fine
        # and then returns early (fewer than 2 GPUs free, say) leaves earlier
        # failures on the counter, and three of those accumulated over hours
        # would halt a campaign against a perfectly reachable server -- a
        # lifetime total masquerading as an ongoing outage.
        self._remote_unreachable_streak = 0
        try:
            return self._dispatch_wave(iteration, report_batch, max_iterations, wave_client)
        finally:
            wave_client.close()

    def _dispatch_wave(
        self, iteration: int, report_batch: List[str], max_iterations: int, wave_client,
    ) -> Optional[Tuple[int, List[str], bool]]:
        """The body of one parallel wave, with every remote call sharing
        `wave_client`. Split out from _run_parallel_wave purely so that
        connection has exactly one open/close site."""
        self._kill_stale_remote_training(context="an earlier wave in this campaign",
                                         client=wave_client)

        candidates = remote_runner.discover_available_gpus(client=wave_client)[: self.max_parallel_runs]
        if len(candidates) < 2:
            return None

        wave_size = min(len(candidates), max_iterations - iteration)

        # Which region does each GPU serve? This is where exploration and
        # exploitation stop being phases: a single wave can hold the champion
        # being exploited, a stalled region getting one more look, and a
        # brand-new region being opened, all at once.
        live_regions, wave_plan = self._plan_wave(wave_size, iteration)
        assignments = wave_plan.assignments[:wave_size]
        if not assignments:
            return None
        wave_size = len(assignments)
        regions_by_id = {r.region_id: r for r in self.registry.regions}

        print(f"[Orchestrator] Parallel wave: {len(candidates)} GPU(s) available -- "
              f"dispatching {wave_size} concurrent run(s) on GPUs {[c['index'] for c in candidates[:wave_size]]}")
        wave_start = time.time()

        latest_summary = self._load_latest_summary()
        recent_evidence = self.state_mgr.get_recent_evidence(limit=5)
        recent_results = self.state_mgr.get_all_results()[-3:]
        latest_val_bpb = None
        if recent_results:
            latest_val_bpb = recent_results[-1].get("val_bpb")

        slots: List[Tuple[int, Dict[str, Any], int, Path]] = []
        decision_halt = False
        for i in range(wave_size):
            iteration_for_slot = iteration + i
            region = regions_by_id.get(assignments[i])
            # Only the first decision in the wave can ever consume a
            # pending fresh-summary flag (it's reset the instant it's
            # read) -- keeps LLM usage to once per new summary even when
            # a whole wave of slots gets decided back-to-back.
            fresh_summary = self._new_summary_ready
            self._new_summary_ready = False
            if fresh_summary:
                print(f"[Orchestrator] *** Fresh summary available -- Agent 1 will use LLM-informed "
                      f"reasoning for iteration {iteration_for_slot} ***")
            new_hyperparams = self._decide_next_hyperparams(
                iteration=iteration_for_slot,
                latest_summary=latest_summary,
                recent_evidence=recent_evidence,
                recent_results=recent_results,
                latest_val_bpb=latest_val_bpb,
                fresh_summary=fresh_summary,
                slot=i,
                region=region,
            )
            if new_hyperparams is None:
                print("\n[Orchestrator] STOPPING: Agent 1 stopped optimizing")
                decision_halt = True
                break

            # Only the first slot of a wave can consume the flag, same rule
            # the fresh-summary flag follows just above -- a record set before
            # the wave justifies fingerprinting one run, not all four.
            new_best_just_set = self._new_best_just_set
            self._new_best_just_set = False
            token_xai_due = (iteration_for_slot % self.token_xai_interval == 0) or new_best_just_set
            new_hyperparams["token_xai_enabled"] = token_xai_due
            self._set_holdout_threshold(new_hyperparams)

            issues = pipeline_validator.validate_agent1_decision(
                self._active_decision_log, recent_evidence, latest_summary,
                decisions_dir=self._active_decisions_dir,
            )
            if self._handle_issues(iteration_for_slot, issues):
                decision_halt = True
                break

            gpu_index = candidates[i]["index"]
            hp_path = self.parallel_hp_dir / f"run_{iteration_for_slot:04d}.yaml"
            self._write_temp_hyperparams(new_hyperparams, hp_path)
            print(f"[Orchestrator] Wave dispatch: GPU {gpu_index} <- iteration {iteration_for_slot} "
                  f"region {new_hyperparams.get('region_id', '-')} "
                  f"(n_layer={new_hyperparams.get('n_layer')}, matrix_lr={new_hyperparams.get('matrix_lr')})")
            slots.append((gpu_index, new_hyperparams, iteration_for_slot, hp_path))

        if not slots:
            return (iteration, report_batch, True) if decision_halt else None

        if not remote_runner.sync_remote_code(client=wave_client):
            # Can't reach the server, so it cannot run anything either.
            # Skipping the wave is the only honest option: dispatching now
            # would just produce a wave of remote_error rows. The consecutive-
            # failure guard in _process_training_result stops the campaign if
            # this keeps happening rather than letting it grind on.
            print("[Orchestrator] Remote code sync failed -- skipping this wave "
                  "(no training dispatched, no iterations consumed)")
            self._sync_failure_streak += 1
            if self._sync_failure_streak >= REMOTE_FAILURE_HALT_STREAK:
                print(f"[Orchestrator] HALTING: remote server unreachable "
                      f"{self._sync_failure_streak} times in a row. Nothing has been "
                      f"trained and no iterations were consumed -- fix connectivity and "
                      f"restart; the campaign resumes from results.tsv.")
                return (iteration, report_batch, True)
            time.sleep(REMOTE_RETRY_SLEEP_S)
            return None
        self._sync_failure_streak = 0

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
                        client=wave_client,
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

        # A wave that dispatched fine and then lost every slot is not
        # progress, and nothing else in this loop would notice: results.tsv
        # gains a row per slot either way, so the iteration counter advances
        # exactly as it does on success.
        statuses = [tr.get("status") for _hp, tr in results_by_iteration.values()]
        if statuses and all(st == "remote_error" for st in statuses):
            self._all_slots_failed_streak += 1
            print(f"[Orchestrator] Every slot in this wave failed "
                  f"({self._all_slots_failed_streak} wave(s) in a row).")
            if self._all_slots_failed_streak >= ALL_SLOTS_FAILED_HALT_STREAK:
                print(f"[Orchestrator] HALTING: {self._all_slots_failed_streak} consecutive "
                      f"waves produced nothing but remote_error. Every further iteration "
                      f"would just add val_bpb=inf rows -- fix connectivity and restart; "
                      f"the campaign resumes from results.tsv.")
                halt = True
        else:
            self._all_slots_failed_streak = 0

        wave_elapsed = time.time() - wave_start
        print(f"[Orchestrator] Wave complete: {len(slots)} run(s) in {_format_duration(wave_elapsed)}")

        next_iteration = iteration + len(slots)
        return next_iteration, report_batch, halt

    def _load_latest_summary(self) -> Optional[str]:
        """Load latest summary report for Agent 1 to read."""
        latest_id = self.state_mgr.get_latest_summary()
        if not latest_id:
            return None

        summary_path = self.reports_dir / "agent3_summaries" / f"{latest_id}.md"
        if summary_path.exists():
            return _read_text_tolerant(summary_path)
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-agent NN optimization")
    parser.add_argument("--config", default="agents_config.yaml", help="Configuration file path")
    parser.add_argument("--iterations", type=int, default=100,
                        help="How many MORE runs to do this session. Run ids resume "
                             "after the highest already in results.tsv rather than "
                             "restarting at run_0000.")
    parser.add_argument("--dry-run", action="store_true", help="Run without training")
    parser.add_argument("--interactive", action="store_true",
                         help="Prompt (blocking) before continuing past a pipeline_validator ERROR. "
                              "Off by default so unattended runs never block on a spurious warning.")

    args = parser.parse_args()

    orchestrator = Orchestrator(config_path=args.config, dry_run=args.dry_run, interactive=args.interactive)
    orchestrator.run(max_iterations=args.iterations)


if __name__ == "__main__":
    main()
