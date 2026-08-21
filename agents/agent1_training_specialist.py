"""Agent 1: Training Specialist - Decides hyperparameters and trains models."""

import contextlib
import math
import os
import random
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional, Tuple
import json

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None

from agents import claude_cli
from agents import train_output
from state.results_analysis import ARCHITECTURE_COLUMNS, at_current_budget


# The 4 learning-rate groups train.py's optimizer actually exposes (matches
# GPT.setup_optimizer's kwargs 1:1 — see train.py). Each has its own default
# and safe range since the groups operate on very different scales.
LR_KEYS = ("embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr")
LR_DEFAULTS = {
    "embedding_lr": 0.6,
    "unembedding_lr": 0.004,
    "matrix_lr": 0.04,
    "scalar_lr": 0.5,
}
LR_SAFE_RANGES = {
    "embedding_lr": (0.05, 3.0),
    "unembedding_lr": (0.0005, 0.02),
    "matrix_lr": (0.005, 0.2),
    "scalar_lr": (0.05, 2.0),
}

# Search space for the Tier 1 surrogate (Sobol cold start + EI acquisition,
# see state/surrogate.py and agents/search_planner.py). n_layer/n_embd/n_head
# ranges below match the bounds already used inline throughout
# _evidence_adjustment/_heuristic_adjustment/_radical_change (not imported
# from there deliberately -- those methods are kept byte-for-byte unchanged
# since 4 existing unit tests call them directly; this is a separate,
# additive constant for the new surrogate-driven code path only).
# weight_decay/warmup_ratio/batch_size were never tuned by Agent 1 before
# Tier 1 even though state/results_analysis.HYPERPARAM_COLUMNS already
# tracks them -- these are new exploration ranges, narrower than train.py's
# own hard safety clamps (which remain the outer safety net regardless).
ARCH_SAFE_RANGES = {
    # n_layer floor widened 4 -> 1 (dev/checks.txt follow-up): elite n_layer
    # averages trended 11-12 -> 8 -> 6 across successive EI-guided batches,
    # and the single best-ever run (val_bpb=1.206506) hit exactly 4, the old
    # floor -- pipeline_validator was also flagging it pinned there for
    # several consecutive iterations. train.py's own hard clamp already
    # allows n_layer down to 1 (_clamp("n_layer", ..., 1, 48)) and every
    # depth-dependent code path (has_ve's modulo arithmetic,
    # _build_window_pattern) is already safe at n_layer=1, so this is
    # purely widening the search's exploration range to match what the
    # model can actually run, not a new safety boundary.
    # RAISED to 28/1280 and PUT BACK to 24/1024, both on 2026-08-11, and the
    # round trip is the point: HOW BIG A MODEL IS WORTH BUILDING DEPENDS ON HOW
    # MUCH YOU TRAIN IT, so this ceiling is not a property of the search, it is
    # a property of the budget.
    #
    # At TOKEN_BUDGET=12.5M the size ladder fell at every one of five steps
    # across 189x and was still falling at 232M params by -0.0135 (6.8 sigma) --
    # the box, not the search, was the limit, so it was raised.
    #
    # At TOKEN_BUDGET=4.19M the same ladder is still monotone but the returns
    # have collapsed: the same final step is -0.0036, inside the noise, and the
    # whole ladder spans 0.091 bpb instead of 0.235. Size stops paying somewhere
    # around 138-232M, which 24 x 1024 = 302M already clears. Room the search
    # cannot profit from is not free -- it costs wall clock on every oversized
    # model it tries.
    #
    # If the budget moves again, re-run scripts/size_sweep.py before touching
    # these. Do not reason about them from first principles.
    "n_layer": (1, 24), "n_embd": (128, 1024), "n_head": (1, 16),
    # Tier 4: fraction of layers using the short attention window (see
    # train.py's _build_window_pattern, which turns this into an actual S/L
    # pattern string). Continuous so the Sobol/EI surrogate can search it
    # like any other dimension. Replaces the old hardcoded
    # WINDOW_PATTERN = "SSSL" constant, which train.py never read from
    # model_hyperparams.yaml at all.
    "window_s_fraction": (0.0, 1.0),
}
OTHER_SAFE_RANGES = {"weight_decay": (0.0, 0.5), "warmup_ratio": (0.0, 0.2), "batch_size": (2048, 32768)}
SEARCH_SPACE = {**LR_SAFE_RANGES, **ARCH_SAFE_RANGES, **OTHER_SAFE_RANGES}

# The ceilings the NON-surrogate decision paths clamp to. Derived here rather
# than repeated as literals in each path: the same numbers used to appear as
# hardcoded 24/1024/16 in eight places across _evidence_adjustment,
# _heuristic_adjustment and _radical_change, so raising the box in one place
# would have left every one of those paths quietly dragging the model back
# down to the old ceiling whenever it ran -- which is most of a campaign's
# opening iterations, before the surrogate has enough data to fit.
# The FLOORS below are deliberately left as they are: they are each path's own
# conservatism (n_layer >= 4 on the evidence path, for instance), not the
# search space's boundary, and nothing measured says to move them.
MAX_N_LAYER = int(ARCH_SAFE_RANGES["n_layer"][1])
MAX_N_EMBD = int(ARCH_SAFE_RANGES["n_embd"][1])
MAX_N_HEAD = int(ARCH_SAFE_RANGES["n_head"][1])

# The campaign's default initial-weight seed (train.py's own SEED default, kept
# in sync here because Agent 1 is what writes it into model_hyperparams.yaml).
# Deliberately NOT in SEARCH_SPACE: see _default_hyperparams.
DEFAULT_SEED = 42

# Tier 4 (see dev/INNOVATION_PLAN.md): thresholds for turning a token-level
# behavioral fingerprint (agents/xai_methods/token_methods.py) into
# directional nudges on the architecture search. Starting points, not
# calibrated against real data yet -- there's essentially no real fingerprint
# history at the time these were written. All comparisons are relative to
# the fingerprint's own scale (a fraction of its own peak), never an
# absolute magic number tied to one model's magnitude.
FINGERPRINT_LATE_LAYER_FRACTION = 0.25      # "late layers" = last 25% of the run's own n_layer (>=1)
FINGERPRINT_DEAD_LAYER_RATIO = 0.10         # late-layer peak < 10% of the array's own overall peak -> "~=0"
FINGERPRINT_LOW_ENTROPY_NATS = math.log(4)  # attention effectively focused on <=4 positions on average
FINGERPRINT_HIGH_INDUCTION = 0.5            # a real, mature induction signal (baseline observed on one real run: ~0.12)
FINGERPRINT_SATURATION_LAYER_IDX = 2        # "by layer 3" (1-indexed) = index 2
FINGERPRINT_SATURATION_RATIO = 0.85         # attn_distance already >=85% of its own peak by that layer
FINGERPRINT_MAX_STEP = 2                    # cap on |sum of votes| applied to n_layer/n_head per iteration


def _late_slice(values: List[float]) -> List[float]:
    """Last FINGERPRINT_LATE_LAYER_FRACTION of a per-layer array (>=1
    element), used by the dla/x0_lambda rules below."""
    n = len(values)
    k = max(1, int(round(n * FINGERPRINT_LATE_LAYER_FRACTION)))
    return values[-k:]

def fingerprint_votes(fingerprint: Dict[str, Any]) -> Dict[str, List[int]]:
    """The 5 Tier 4 rules (see dev/INNOVATION_PLAN.md), each casting a
    simple +-1 vote on one param when its threshold trips. Every
    comparison is relative to the fingerprint's own peak for that
    array -- never an absolute number tied to one run's scale, since
    model magnitude varies a lot across the search.
    """
    votes: Dict[str, List[int]] = {}

    def vote(param: str, direction: int) -> None:
        votes.setdefault(param, []).append(direction)

    dla = [float(v) for v in (fingerprint.get("dla") or [])]
    if dla:
        peak = max(abs(v) for v in dla)
        if peak > 1e-9 and max(abs(v) for v in _late_slice(dla)) < FINGERPRINT_DEAD_LAYER_RATIO * peak:
            vote("n_layer", -1)  # late layers aren't writing to the output

    x0_lambda = [float(v) for v in (fingerprint.get("x0_lambda") or [])]
    if x0_lambda:
        peak = max(x0_lambda)
        if peak > 1e-9 and max(_late_slice(x0_lambda)) < FINGERPRINT_DEAD_LAYER_RATIO * peak:
            vote("n_layer", 1)  # depth is being used, not just echoing the embedding shortcut

    attn_entropy = [float(v) for v in (fingerprint.get("attn_entropy") or [])]
    if attn_entropy:
        mean_entropy = sum(attn_entropy) / len(attn_entropy)
        if mean_entropy < FINGERPRINT_LOW_ENTROPY_NATS:
            vote("n_head", -1)
            vote("n_embd", 1)

    induction_score = fingerprint.get("induction_score")
    if isinstance(induction_score, (int, float)) and induction_score > FINGERPRINT_HIGH_INDUCTION:
        vote("n_layer", 1)  # found the easy structure fast -> raise the difficulty

    attn_distance = [float(v) for v in (fingerprint.get("attn_distance") or [])]
    if attn_distance:
        peak = max(attn_distance)
        idx = min(FINGERPRINT_SATURATION_LAYER_IDX, len(attn_distance) - 1)
        if peak > 1e-9 and attn_distance[idx] >= FINGERPRINT_SATURATION_RATIO * peak:
            vote("window_s_fraction", 1)  # reach stopped growing early -> more short-window layers are safe

    return votes


class Agent1TrainingSpecialist:
    """Trains models and adjusts hyperparameters based on agent feedback."""

    def __init__(
        self,
        config_path: str = "agents_config.yaml",
        root_dir: Optional[str] = None,
        state_dir: Optional[str] = None,
        reports_dir: Optional[str] = None,
    ):
        """
        root_dir/state_dir/reports_dir let callers (tests, or any future
        parallel campaign) redirect every file this class touches instead of
        always hitting the repo root -- e.g. Orchestrator forwards its own
        state_dir/reports_dir here. Defaults preserve the original
        cwd-relative behavior exactly, so no existing caller needs to change.
        model_hyperparams.yaml lives under root_dir (not state_dir): a real
        (non-dry-run) train.py always reads it from its own directory, so
        root_dir must stay "." for any run that actually needs train.py to
        find it -- only dry-run/test callers should ever override it.
        """
        self.config = self._load_config(config_path)
        self.agent1_config = self.config.get("agent1", {})
        self.use_llm = self.agent1_config.get("use_llm", False)
        self.accuracy_threshold = self.agent1_config.get("accuracy_threshold", 0.95)
        self.cost_limit_usd = self.agent1_config.get("cost_limit_usd", 50.0)
        self.training_budget = self.agent1_config.get("training_budget_seconds", 300)
        self.min_improvement = self.agent1_config.get("min_improvement", 0.005)
        self.max_stalled_iterations = self.agent1_config.get("max_stalled_iterations", 3)
        # "Improved" has to be defined against the measured noise floor, not
        # as a round number. _detect_stagnation used a hardcoded 0.01, which
        # at the current sigma (0.00919, state/noise_floor.json) is 1.09
        # sigma -- so a run that was purely luckier than the last one cleared
        # it, and a genuine 1-sigma gain did not. Expressed as a multiple of
        # sigma it keeps meaning the same thing when the noise floor is
        # re-measured (it has already moved ~3x once, when the campaign
        # switched from a time budget to a token budget).
        self.stagnation_sigma_multiple = float(
            self.agent1_config.get("stagnation_sigma_multiple", 2.0))
        self.summary_strength = float(self.agent1_config.get("summary_strength", 2.0))
        self.lr_bounds: Dict[str, tuple] = {
            key: (
                float(self.agent1_config.get(f"{key}_min", LR_SAFE_RANGES[key][0])),
                float(self.agent1_config.get(f"{key}_max", LR_SAFE_RANGES[key][1])),
            )
            for key in LR_KEYS
        }
        # agents_config.yaml's agent2.ablation_k previously had no effect at all
        # (train.py never ran ablation and never read it). It's forwarded through
        # model_hyperparams.yaml here since that's the only file train.py reads.
        self.ablation_k = int(self.config.get("agent2", {}).get("ablation_k", 3))

        # Tier 1 surrogate (see agents/search_planner.py, state/surrogate.py).
        # use_surrogate defaults on; _surrogate_adjustment degrades to None
        # (triggering the existing evidence/heuristic fallback unchanged)
        # whenever scipy/scikit-learn aren't installed or there isn't yet
        # enough data to fit -- this flag just lets it be disabled outright.
        # How far a proposal may sit from its region's anchor, in normalized
        # 8-D tunable distance (state/regions.py::distance).
        #
        # PROVISIONAL -- 0.05 is carried over from the old 11-D geometry and is
        # NOT the right number here. The distance divides by sqrt(n_dims), so
        # the same 0.05 covers a different amount of space in 8-D than it did
        # in 11-D. Step 4 of the redesign re-derives it from measured
        # within-region spread; until then this is a placeholder chosen to be
        # roughly the previous behaviour rather than a calibrated value.
        # MEASURED, not provisional: scripts/region_geometry.py sets it from
        # B(r) against A-within. 0.02 at TOKEN_BUDGET = 4.19M; it was 0.05 at
        # 12.5M, where 0.02 was saturated on arrival. Re-derive it whenever the
        # budget moves -- the surface gets steeper as training gets shorter.
        self.fence_radius = float(self.agent1_config.get("region_fence_radius", 0.02))

        self.use_surrogate = bool(self.agent1_config.get("use_surrogate", True))
        self.surrogate_cold_start_n = int(self.agent1_config.get("surrogate_min_observations", 15))
        self.surrogate_cycle_runs = int(self.agent1_config.get("surrogate_cycle_runs", 10))
        self.surrogate_interaction_threshold = float(self.agent1_config.get("interaction_threshold", 0.15))

        # Search-plan diagnostic charts (predicted-vs-actual, sensitivity,
        # interaction matrix, EI candidates, Sobol coverage) -- same config
        # key name/convention as agent2/agent3's generate_charts.
        self.generate_charts = bool(self.agent1_config.get("generate_charts", True))

        _root = Path(root_dir) if root_dir else Path(".")
        _state = Path(state_dir) if state_dir else Path("state")
        _reports = Path(reports_dir) if reports_dir else Path("reports")
        self.model_config_path = _root / "model_hyperparams.yaml"
        self.results_path = _root / "results.tsv"
        self._search_planner_state_path = str(_state / "search_planner_state.json")
        self._noise_floor_path = str(_state / "noise_floor.json")
        self._search_plan_report_dir = str(_reports / "agent1_search_plan")
        # Root for per-region planner state (state/regions.py). Each region
        # searched concurrently gets its own SearchPlannerState file under
        # here -- see search_region() for why they must not share one.
        self._region_planner_state_dir = str(_state / "search_planner")

        # LLM/copilot integration (dev/checks.txt item 4): shared campaign
        # budget across agent1/2/3, tracked via agents/claude_cli.py and
        # state/llm_usage.json -- see claude_cli.py's docstring for why
        # this calls the Claude Code CLI (your subscription) instead of
        # the separately-billed anthropic SDK.
        llm_config = self.config.get("llm", {})
        self._llm_backend = llm_config.get("backend", "cli")
        self._llm_model = llm_config.get("model", "sonnet")
        self._llm_campaign_budget_usd = float(llm_config.get("campaign_budget_usd", 5.0))
        self._llm_max_call_budget_usd = float(llm_config.get("max_call_budget_usd", 0.20))
        self._llm_usage_path = llm_config.get("usage_log_path") or str(_state / "llm_usage.json")

        self.current_hyperparams = self._init_hyperparams()
        self.total_api_cost = 0.0
        self.best_val_bpb = float("inf")
        #: How much EI's uncertainty term is still worth paying for, 1.0 down
        #: to 0.0. Set by the orchestrator each iteration from the campaign's
        #: REMAINING non-Sobol runs; 1.0 when nobody sets it, so every existing
        #: caller and test behaves exactly as before.
        self.exploration_weight = 1.0
        # Set only inside search_region(): several regions decided in one
        # wave would otherwise each overwrite model_hyperparams.yaml, leaving
        # the file describing whichever region happened to be decided last.
        self._suppress_hyperparams_save = False
        # The region this decision belongs to, or None on the single-search
        # path. Read by the stagnation/stop/stuck checks, which all mean
        # something different inside a region than across the campaign.
        self._active_region = None
        # Set by a region-scoped decision when that region looks stalled or
        # stuck. Reported rather than acted on: inside a region these are
        # LIFECYCLE facts for the allocator to judge (pause it, retire it,
        # open something else), not a licence to jump the search center
        # somewhere random or to stop the whole campaign.
        self.last_region_stalled = False
        self.last_region_stuck = False

        # Decision log (see agents/pipeline_validator.py): a total, recorded
        # disposition for every tunable parameter every iteration, so "was
        # this ignored" and "why is this extreme" are answerable from a file
        # instead of being a mystery. Pure side-channel state -- none of this
        # affects decide_next_hyperparams's return value.
        self.decisions_dir = _reports / "agent1_decisions"
        self.last_decision_log: Optional[Dict[str, Any]] = None
        self._last_lr_clamps: Dict[str, Any] = {}
        self._last_surrogate_phase: Optional[str] = None
        self._last_surrogate_frozen: list = []
        self._last_surrogate_active_block: list = []
        self._last_fingerprint_adjustments: List[Dict[str, Any]] = []

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(config_path):
            return {}
        if yaml is None:
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _init_hyperparams(self) -> Dict[str, Any]:
        """Initialize or load hyperparameters."""
        if self.model_config_path.exists():
            if yaml is not None:
                with open(self.model_config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or self._default_hyperparams()
        return self._default_hyperparams()

    def _default_hyperparams(self) -> Dict[str, Any]:
        """Default hyperparameters for GPT model (mirrors train.py's baseline)."""
        return {
            "n_layer": 8,  # Number of layers
            "n_head": 4,  # Number of attention heads
            "n_embd": 512,  # Embedding dimension
            "window_s_fraction": 0.75,  # matches the old hardcoded WINDOW_PATTERN="SSSL" (3-of-4 = 75% S)
            "embedding_lr": LR_DEFAULTS["embedding_lr"],
            "unembedding_lr": LR_DEFAULTS["unembedding_lr"],
            "matrix_lr": LR_DEFAULTS["matrix_lr"],
            "scalar_lr": LR_DEFAULTS["scalar_lr"],
            "batch_size": 8192,  # tokens per optimizer step (TOTAL_BATCH_SIZE)
            "warmup_ratio": 0.0,
            "weight_decay": 0.2,
            "ablation_k": self.ablation_k,
            # Initial-weight seed. Carried explicitly so every proposal states
            # which initialization it was measured under, instead of that being
            # an unrecorded property of train.py's source.
            #
            # NOT a search dimension -- it is absent from SEARCH_SPACE, so every
            # adjustment path here passes it through untouched and
            # _build_decision_log records it as "pass-through (not a tunable
            # search parameter)". Optimizing over seeds would select the
            # luckiest initialization; the correct treatment is to average over
            # them (see scripts/seed_variance.py).
            "seed": DEFAULT_SEED,
        }

    def _save_hyperparams(self, hyperparams: Optional[Dict[str, Any]] = None):
        """Save hyperparams to YAML -- `hyperparams` when given, otherwise
        this agent's own current_hyperparams.

        The argument exists because train_model writes this file and then
        tells train.py to read it, so whatever lands here IS what gets
        trained. That was safe while Agent 1 was the only decider (its
        decide_next_hyperparams assigns current_hyperparams to the very same
        dict object it returns), but Agent 4 proposes probes from its own
        dict -- so an argument-less save would have trained Agent 1's stale
        center while results.tsv recorded Agent 4's proposal, mislabelling
        every probe and poisoning the surrogate with it.
        """
        if yaml is None:
            return
        # An explicit argument is always honored: train_model passes the exact
        # dict it is about to train, and that must reach disk even when a
        # region-scoped decision is in progress. Only the argument-less
        # "persist my current center" call is suppressed -- see search_region.
        if hyperparams is None and self._suppress_hyperparams_save:
            return
        with open(self.model_config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.current_hyperparams if hyperparams is None else hyperparams, f)

    def relocate_search_center(self, hyperparams: Dict[str, Any]) -> None:
        """Move the search to a different region of the space.

        Called by the Orchestrator when Agent 4
        (agents/agent4_landscape_explorer.py) commits to a new region. This
        is the one concrete mechanism by which that decision changes anything:
        agents/search_planner.py::propose_next pins
        `center = dict(current_best_hyperparams)` on every call and only
        varies the active Gauss-Southwell block around it, so the EI search
        is structurally local and cannot leave a basin on its own -- moving
        what it treats as "here" is the only way out.

        Deliberately does NOT touch best_val_bpb: that stays the true
        best-ever *measurement*, which is still the correct EI f_best target
        and the correct reference for stagnation/stop conditions. Only the
        search center relocates; the record of what has actually been
        achieved is not rewritten.
        """
        self.current_hyperparams = dict(hyperparams)
        self._save_hyperparams()
        print(f"[Agent 1] Search center relocated to: {hyperparams}")

    #: Keys the Orchestrator stamps onto a proposal fresh every iteration --
    #: which region it belongs to, whether to run token-level XAI, and the
    #: threshold below which train.py should also score the holdout shard.
    #: They describe one dispatch, not a configuration, so they are stripped
    #: before a proposal becomes a region's stored center. Left in, they make
    #: regions.json read as a snapshot of orchestration state, and a stale
    #: holdout_eval_if_below could survive into a later run on the one path
    #: that does not overwrite it (no finite best yet).
    ORCHESTRATION_KEYS = ("region_id", "token_xai_enabled", "holdout_eval_if_below")

    @contextlib.contextmanager
    def search_region(self, region, save_hyperparams: bool = False) -> Iterator[None]:
        """Run one decision *as* a given region (state/regions.py::Region),
        then hand the search state back.

        This is what makes several regions searchable at once by a single
        Agent 1. Four pieces of state are region-scoped for the duration:

          - `current_hyperparams` -> the region's own drifting center, so
            every adjustment path (surrogate, evidence, heuristic, radical
            change -- all of which read `self.current_hyperparams`) operates
            on this region and not on some other region's last proposal;
          - the planner state path -> the region's own file, which is the
            entire point: agents/search_planner.py keeps ONE cold-start
            counter, ONE frozen set and ONE active Gauss-Southwell block per
            state file. Shared, four concurrent regions would fight over one
            block rotation and one region's frozen parameters would silently
            freeze them everywhere else;
          - the plan report dir -> likewise per region, since
            _surrogate_adjustment reads back `plan_{iteration}.json` to
            explain its own decision and two regions deciding the same
            iteration number would otherwise overwrite each other's;
          - `best_val_bpb` -> the region's own best, because that value is
            used for exactly one thing downstream (EI's `f_best`, see
            search_planner.propose_next). With the campaign-wide best, EI
            inside any region that isn't the champion sees essentially zero
            improvement probability everywhere and its argmax degenerates
            into noise. A local trust region needs a local reference.

        A genuine campaign record found inside a region still propagates out
        on exit -- the region's search is local, but "the best model anyone
        has trained" is not a per-region fact.

        save_hyperparams is False by default: writing model_hyperparams.yaml
        is meaningless when several regions are deciding back-to-back within
        one wave (the orchestrator writes a per-run YAML per GPU instead, see
        Orchestrator._write_temp_hyperparams). It exists for the sequential
        single-GPU path, which does still train from that file.
        """
        prev_center = self.current_hyperparams
        prev_state_path = self._search_planner_state_path
        prev_report_dir = self._search_plan_report_dir
        prev_best = self.best_val_bpb
        prev_save = self._suppress_hyperparams_save
        prev_region = self._active_region

        self._active_region = region
        self.last_region_stalled = False
        self.last_region_stuck = False
        self.current_hyperparams = dict(region.center)
        self._search_planner_state_path = region.planner_state_path(self._region_planner_state_dir)
        self._search_plan_report_dir = region.report_dir(prev_report_dir)
        region_best = region.best_val_bpb
        self.best_val_bpb = region_best if region_best is not None else float("inf")
        self._suppress_hyperparams_save = not save_hyperparams
        try:
            yield
        finally:
            # The proposal this decision produced becomes the region's new
            # center -- that drift is the region exploiting itself, and it is
            # why identity lives in the (immovable) anchor instead.
            region.center = {k: v for k, v in self.current_hyperparams.items()
                             if k not in self.ORCHESTRATION_KEYS}
            self.current_hyperparams = prev_center
            self._search_planner_state_path = prev_state_path
            self._search_plan_report_dir = prev_report_dir
            self.best_val_bpb = min(prev_best, self.best_val_bpb)
            self._suppress_hyperparams_save = prev_save
            self._active_region = prev_region

    def decide_next_hyperparams(
        self,
        latest_summary: Optional[str] = None,
        evidence: Optional[list] = None,
        stuck_signal: bool = False,
        latest_val_bpb: Optional[float] = None,
        iteration: int = 0,
        recent_results: Optional[list] = None,
        fresh_summary: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Decide next hyperparameters using heuristics (+ optional Claude).

        fresh_summary: True only on the one call right after Agent 3 just
        created a new summary (agents/orchestrator.py tracks this) -- the
        Claude review below is gated on this, not just "a summary exists,"
        so it fires once per new summary instead of every iteration.

        Returns:
            New hyperparams dict, or None to STOP
        """

        # Check stopping conditions
        if latest_val_bpb is not None:
            if latest_val_bpb < self.accuracy_threshold:
                print(f"[Agent 1] Accuracy threshold reached: {latest_val_bpb:.6f}")
                return None

            if latest_val_bpb < self.best_val_bpb:
                self.best_val_bpb = latest_val_bpb

        if self._should_stop_early(recent_results=recent_results, latest_val_bpb=latest_val_bpb):
            print("[Agent 1] Stopping: no meaningful improvement over recent iterations")
            return None

        if self.total_api_cost >= self.cost_limit_usd:
            print(f"[Agent 1] Cost limit exceeded: ${self.total_api_cost:.2f}")
            return None

        print(f"\n[Agent 1] Deciding next hyperparameters (iteration {iteration})...")

        # Snapshot for the decision log (Part 1 of dev/inpsect_workflow_ideas.txt's
        # follow-up) -- a total, recorded before/after/reason for every tunable
        # parameter, built as an external diff so none of the adjustment
        # methods below need to change.
        before_hyperparams = dict(self.current_hyperparams)
        self._last_lr_clamps = {}
        self._last_fingerprint_adjustments = []

        detected_stagnation = self._detect_stagnation(recent_results, latest_val_bpb)
        effective_stuck_signal = stuck_signal or detected_stagnation

        # A stuck region is a lifecycle fact, not a licence to teleport.
        # _radical_change picks a random n_layer/n_embd, which lands the
        # search outside the very region it was scoped to -- so within a
        # region the signal is recorded for the allocator and the local
        # surrogate search continues. Retiring or pausing the region is a
        # decision made from this flag, one level up, where the alternatives
        # (open a new region, move its GPUs to a better one) actually exist.
        if self._active_region is not None and effective_stuck_signal:
            self.last_region_stuck = True
            print(f"[Agent 1] Region {self._active_region.region_id} looks stuck -- "
                  f"reported to the allocator; continuing its local search rather "
                  f"than jumping the center out of the region")
            effective_stuck_signal = False

        # PRIMARY: the Tier 1 surrogate (Sobol cold start -> EI acquisition,
        # see agents/search_planner.py), unless stuck -- radical_change stays
        # wired exactly as before regardless of the surrogate's availability.
        new_hyperparams = None
        path_taken = "unknown"
        if effective_stuck_signal:
            new_hyperparams = self._evidence_adjustment(
                latest_summary=latest_summary, evidence=evidence, stuck_signal=True, iteration=iteration,
            )
            path_taken = "radical_change"
        elif self.use_surrogate:
            new_hyperparams = self._surrogate_adjustment(iteration)
            if new_hyperparams is not None:
                path_taken = "surrogate"

        # FALLBACK: report-driven evidence, then heuristics, exactly as
        # before Tier 1 existed -- reached whenever the surrogate can't run
        # yet (deps missing, not enough data) or returns nothing useful.
        if new_hyperparams is None:
            new_hyperparams = self._evidence_adjustment(
                latest_summary=latest_summary,
                evidence=evidence,
                stuck_signal=effective_stuck_signal,
                iteration=iteration,
            )
            path_taken = "evidence"
            if not evidence:
                new_hyperparams = self._heuristic_adjustment(
                    latest_summary, effective_stuck_signal, iteration
                )
                path_taken = "heuristic"

        # OPTIONAL: If LLM enabled AND this is a fresh summary -- gating on
        # freshness (not just "a summary exists") is what keeps this from
        # firing every iteration once any summary exists at all, which was
        # burning the shared campaign LLM budget far faster than intended.
        if self.use_llm and latest_summary and fresh_summary:
            print(f"[Agent 1] Fresh summary available -- reading it with LLM reasoning...")
            try:
                claude_suggestion = self._get_claude_suggestion(
                    new_hyperparams, latest_summary
                )
                # Claude can override or confirm heuristic suggestion
                new_hyperparams = claude_suggestion or new_hyperparams
                # Real cumulative cost (state/llm_usage.json), not a fabricated
                # per-call estimate -- cost_limit_usd stays a meaningful,
                # separate campaign-wide safety net above the LLM budget's own.
                from state import llm_usage
                self.total_api_cost = llm_usage.cumulative_cost_usd(self._llm_usage_path)
            except Exception as e:
                print(f"[Agent 1] Claude suggestion failed: {e}")

        # Tier 4: nudge architecture params using the most recent
        # fingerprint-bearing evidence entry, if any -- a final,
        # path-independent pass applied regardless of which branch above
        # produced new_hyperparams (including a Claude override), since the
        # surrogate path dominates once enough data exists and would
        # otherwise make a fingerprint hook inside _evidence_adjustment
        # alone get bypassed most of the time in a mature search.
        new_hyperparams = self._fingerprint_adjustment(new_hyperparams, evidence)

        # A REGION IS AN EXACT ARCHITECTURE (step 3a) plus a neighbourhood of
        # the eight tunables, so its architecture is restored here -- once,
        # after every path above, for the same reason the LR re-clamp and the
        # n_embd snap below sit at this point rather than inside each branch.
        #
        # Closing _fingerprint_adjustment (step 5b) was not enough on its own:
        # _evidence_adjustment and _heuristic_adjustment write n_layer, n_embd
        # and n_head directly, and Claude's free-form JSON can too. Those
        # fallbacks are not a rare path -- they are what runs during a
        # campaign's opening iterations, before the surrogate has enough data
        # to fit, which is exactly when a region is youngest.
        #
        # Changing the architecture mid-region produces a run whose INITIAL
        # WEIGHTS differ from the rest of its own region, which destroys the
        # pairing steps 1-3 exist to establish (verified in step 1: same
        # architecture + same seed gives 46/46 bit-identical tensors, and a
        # width change reshapes 41 of the 46). The run would then be compared
        # against its region-mates as though only the tunables had moved.
        #
        # Read from the ANCHOR, never the drifting center: the anchor is what
        # makes a region a fixed place. Outside a region there is nothing to
        # preserve, so every path keeps its freedom to move architecture.
        if self._active_region is not None:
            anchor = self._active_region.anchor
            for col in ARCHITECTURE_COLUMNS:
                if col in anchor:
                    new_hyperparams[col] = anchor[col]

        # Claude's suggestion is free-form JSON — re-clamp every LR group
        # before it's ever saved, since that path can otherwise bypass the
        # safety bounds the rest of this class enforces everywhere else.
        for key in LR_KEYS:
            if key in new_hyperparams:
                new_hyperparams[key] = self._clamp_lr(key, float(new_hyperparams[key]))

        # Same idea for n_embd: only state/surrogate.py's EI/cold-start path
        # snapped it to a value train.py can actually use unchanged (see
        # snap_n_embd there) -- every other path here (heuristic random
        # pick, Tier 4 fingerprint votes, evidence-based hints, Claude's
        # free-form suggestion) could propose an n_embd that doesn't divide
        # evenly by n_head/doesn't leave an even head_dim, which train.py
        # then silently re-snaps at train time (pipeline_validator flags
        # this as an ERROR: "train.py clamped n_embd"). Applying the exact
        # same snap here, universally, means requested == actually used
        # every time, regardless of which path produced new_hyperparams.
        if "n_embd" in new_hyperparams and "n_head" in new_hyperparams:
            from state.surrogate import snap_n_embd
            new_hyperparams["n_embd"] = snap_n_embd(
                float(new_hyperparams["n_embd"]), float(new_hyperparams["n_head"])
            )

        self.current_hyperparams = new_hyperparams
        self._save_hyperparams()

        self.last_decision_log = self._build_decision_log(
            before_hyperparams, new_hyperparams, iteration, path_taken, evidence, latest_summary,
        )
        self._write_decision_log(self.last_decision_log)

        print(f"[Agent 1] Next hyperparams: {new_hyperparams}")
        return new_hyperparams

    def _surrogate_reason_for(self, key: str) -> str:
        """Best-effort explanation for one parameter's disposition on the
        surrogate-driven path, from the frozen/active_block info
        _surrogate_adjustment already stashed this call (see there)."""
        if self._last_surrogate_phase == "cold_start":
            return "surrogate: Sobol cold-start exploration"
        if key in self._last_surrogate_active_block:
            return "surrogate: EI-tuned this cycle (active block)"
        if key in self._last_surrogate_frozen:
            return "surrogate: frozen (total effect below 2*sigma noise floor)"
        if self._last_surrogate_phase == "ei":
            return "surrogate: kept but not in this cycle's active block"
        return "surrogate-driven"

    def _build_decision_log(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        iteration: int,
        path_taken: str,
        evidence: Optional[list],
        latest_summary: Optional[str],
    ) -> Dict[str, Any]:
        """Every key in SEARCH_SPACE (+ any pass-through key present in
        `after`, e.g. ablation_k) gets an explicit before/after/changed/reason
        entry -- built by diffing, not by instrumenting the adjustment
        methods, so it's structurally impossible for a parameter to be
        silently dropped without this noticing, and none of
        _evidence_adjustment/_heuristic_adjustment/_radical_change's tested
        internals need to change.
        """
        params_log: Dict[str, Any] = {}
        for key in SEARCH_SPACE:
            old, new = before.get(key), after.get(key)
            changed = old != new
            if path_taken == "surrogate":
                reason = self._surrogate_reason_for(key)
            elif path_taken == "radical_change":
                reason = "radical_change: stuck signal fired"
            else:
                reason = f"{path_taken}-driven" if changed else "unchanged: no signal for this parameter"
            params_log[key] = {"before": old, "after": new, "changed": changed, "reason": reason}
        for key in after:
            if key not in params_log:
                params_log[key] = {
                    "before": before.get(key), "after": after.get(key),
                    "changed": before.get(key) != after.get(key),
                    "reason": "pass-through (not a tunable search parameter)",
                }
        return {
            "iteration": iteration,
            "path_taken": path_taken,
            "evidence_considered": len(evidence) if evidence else 0,
            "summary_considered": bool(latest_summary),
            # Which region's search produced this decision, or None on the
            # single-search path. pipeline_validator filters its lookback on
            # this: "pinned at its boundary for 6 consecutive iterations" is a
            # statement about one search, and under multi-region search
            # consecutive iterations belong to different ones.
            "region_id": self._active_region.region_id if self._active_region else None,
            "params": params_log,
            "lr_clamps": dict(self._last_lr_clamps),
            "fingerprint_adjustments": list(self._last_fingerprint_adjustments),
        }

    def _write_decision_log(self, decision_log: Dict[str, Any]) -> None:
        try:
            self.decisions_dir.mkdir(parents=True, exist_ok=True)
            path = self.decisions_dir / f"decision_{decision_log['iteration']:04d}.json"
            path.write_text(json.dumps(decision_log, indent=2, sort_keys=True))
        except OSError as e:
            print(f"[Agent 1] Could not write decision log: {e}")

    def _surrogate_adjustment(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Delegates to agents.search_planner. Returns None (triggering the
        unchanged evidence/heuristic fallback in decide_next_hyperparams)
        whenever scipy/scikit-learn aren't installed, the surrogate can't
        fit yet (too little data), or every dimension is currently frozen.
        """
        # Reset for this call -- _build_decision_log reads these afterward to
        # explain *why* each param did or didn't change on the surrogate path.
        self._last_surrogate_phase = None
        self._last_surrogate_frozen = []
        self._last_surrogate_active_block = []

        try:
            from agents import search_planner
        except ImportError:
            return None
        from state.results_analysis import load_results
        # ONE BUDGET ONLY. A surrogate fitted across budgets is fitted on two
        # different tasks: the same configuration read 1.2486 at 12.5M tokens
        # and 1.7063 at 4.19M, so mixing them puts a 0.45 bpb step into the
        # data that no hyperparameter caused, and the model would spend its
        # capacity explaining the budget instead of the search space.
        #
        # Filtered HERE, where the rows come off disk, rather than inside
        # propose_next: a function handed an explicit list of rows should
        # search them, not silently discard most of them.
        rows = at_current_budget(load_results(str(self.results_path)))
        result = search_planner.propose_next(
            rows=rows,
            current_best_hyperparams=self.current_hyperparams,
            current_best_val_bpb=self.best_val_bpb,
            exploration_weight=self.exploration_weight,
            iteration=iteration,
            cold_start_n=self.surrogate_cold_start_n,
            cycle_runs=self.surrogate_cycle_runs,
            interaction_threshold=self.surrogate_interaction_threshold,
            state_path=self._search_planner_state_path,
            noise_floor_path=self._noise_floor_path,
            report_dir=self._search_plan_report_dir,
            generate_charts=self.generate_charts,
            # Keeps EI's denoised incumbent local to this region, matching what
            # search_region already does for best_val_bpb.
            f_best_region_id=self._active_region.region_id if self._active_region else None,
            # The fence is anchored at the region's ANCHOR, not its drifting
            # center. Fencing around the center would let the region walk
            # anywhere one small step at a time -- the anchor is what makes it
            # a fixed place, and it is also what makes escape pressure mean
            # "this region is anchored in the wrong spot" rather than "the last
            # proposal was unusual".
            fence_center=self._active_region.anchor if self._active_region else None,
            fence_radius=self.fence_radius if self._active_region else None,
        )
        if result is None:
            return None

        # search_planner only writes plan_{iteration}.json on the EI-driven
        # branch (not cold-start) -- reuse it rather than recomputing
        # frozen/active_block here. Read-only, best-effort: a missing/corrupt
        # file just means the decision log falls back to "surrogate-driven"
        # without finer detail, never an error.
        plan_path = Path(self._search_plan_report_dir) / f"plan_{iteration:04d}.json"
        if plan_path.exists():
            try:
                plan = json.loads(plan_path.read_text())
                self._last_surrogate_phase = "ei"
                self._last_surrogate_frozen = plan.get("frozen", [])
                self._last_surrogate_active_block = plan.get("active_block", [])
            except (json.JSONDecodeError, OSError):
                self._last_surrogate_phase = "ei"
        else:
            self._last_surrogate_phase = "cold_start"
        return result

    def _sigma(self) -> float:
        """The measured val_bpb noise floor, via search_planner's loader so
        there is one definition of it (and one warning when it's missing)."""
        from agents.search_planner import _load_sigma

        return _load_sigma(self._noise_floor_path)

    def _region_val_bpbs(self) -> Optional[List[float]]:
        """This region's own measured runs, oldest first -- or None when no
        region scope is in effect.

        Every trend check below has to read this rather than the
        orchestrator's `recent_results`, which is the last few runs of the
        WHOLE campaign. Under a multi-GPU wave those come from different
        regions, so comparing them to each other measures the difference
        between two places in the space and calls it a trend.
        """
        if self._active_region is None:
            return None
        return list(self._active_region.val_bpbs)

    def _should_stop_early(
        self,
        recent_results: Optional[list],
        latest_val_bpb: Optional[float],
    ) -> bool:
        """Stop once the recent trend has stalled for enough iterations.

        Returns False unconditionally inside a region scope, and records
        `last_region_stalled` instead. This function's True stops the ENTIRE
        campaign; one region running out of road is not that. Which region to
        pause or retire is the allocator's decision, made from the flag, and
        the other regions keep running meanwhile.
        """
        region_values = self._region_val_bpbs()
        if region_values is not None:
            self.last_region_stalled = self._stalled(region_values, latest_val_bpb)
            if self.last_region_stalled:
                print(f"[Agent 1] Region {self._active_region.region_id} has stalled "
                      f"({len(region_values)} run(s), no gain over the last "
                      f"{self.max_stalled_iterations}) -- reporting it, not stopping the campaign")
            return False

        if not recent_results:
            return False

        values = []
        for item in recent_results:
            if not isinstance(item, dict):
                continue
            val_bpb = item.get("val_bpb")
            if val_bpb is None:
                continue
            try:
                values.append(float(val_bpb))
            except (TypeError, ValueError):
                continue

        if len(values) < self.max_stalled_iterations + 1:
            return False

        finite_values = [value for value in values if math.isfinite(value)]
        if len(finite_values) < self.max_stalled_iterations + 1:
            return False
        return self._stalled(finite_values, latest_val_bpb)

    def _stalled(self, values: List[float], latest_val_bpb: Optional[float]) -> bool:
        """Has the latest run failed to improve on the best of the runs
        before it? Shared by the campaign-wide and region-scoped paths so the
        two can never drift apart in what "stalled" means.

        The baseline deliberately EXCLUDES the latest value. Comparing the
        window's minimum against the latest when the latest is itself part of
        that window makes `improvement` zero exactly when the newest run set
        a record -- so a steadily improving search reported itself as
        stalled. That never surfaced because this was unreachable on the
        campaign path: it needs max_stalled_iterations + 1 = 4 values and the
        orchestrator only ever passes the last 3 (get_all_results()[-3:]), so
        the length guard returned False every time. Region scoping feeds it a
        region's full history, which is what made the latent bug reachable.
        """
        finite_values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        if len(finite_values) < self.max_stalled_iterations + 1:
            return False

        recent_window = finite_values[-(self.max_stalled_iterations + 1):]
        baseline = min(recent_window[:-1])
        latest_value = recent_window[-1]
        if latest_val_bpb is not None and math.isfinite(latest_val_bpb):
            latest_value = latest_val_bpb

        improvement = baseline - latest_value
        return improvement < self.min_improvement

    def _detect_stagnation(
        self,
        recent_results: Optional[list],
        latest_val_bpb: Optional[float],
    ) -> bool:
        """Return True when the recent validation trend shows little or no
        improvement.

        Region-scoped when a region is active, campaign-wide otherwise. That
        distinction is not cosmetic: this signal's True routes into
        _evidence_adjustment(stuck_signal=True) -> _radical_change, which
        picks a random n_layer and n_embd. Fed the campaign's last few runs
        while four regions train concurrently, it compares one region's run
        against another's, concludes "no progress", and jumps the center out
        of the region it was supposed to be exploiting -- destroying the
        region's identity on essentially every wave. See _region_val_bpbs.
        """
        region_values = self._region_val_bpbs()
        if region_values is not None:
            values = list(region_values)
        else:
            if not recent_results:
                return False
            values = []
            for item in recent_results:
                if not isinstance(item, dict):
                    continue
                val_bpb = item.get("val_bpb")
                if val_bpb is None:
                    continue
                try:
                    values.append(float(val_bpb))
                except (TypeError, ValueError):
                    continue

        if len(values) < 3:
            return False

        finite_values = [value for value in values if math.isfinite(value)]
        if len(finite_values) < 3:
            return False

        window = finite_values[-3:]

        # Sized against the measured noise floor rather than the old
        # hardcoded 0.01 (= 1.09 sigma today, i.e. "one lucky run counts as
        # progress"). See stagnation_sigma_multiple.
        margin = self._sigma() * self.stagnation_sigma_multiple
        improved = window[-1] < window[-2] - margin or window[-2] < window[-3] - margin
        if latest_val_bpb is not None and math.isfinite(latest_val_bpb):
            return not improved and latest_val_bpb >= min(window[-2], window[-3]) - margin
        return not improved

    def _radical_change(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Large random architecture jump used when the model looks stuck."""
        print("[Agent 1] Model stuck - trying radical changes")
        new_params["n_layer"] = random.randint(8, 20)
        new_params["n_embd"] = random.choice([256, 384, 512, 768, 1024, MAX_N_EMBD])
        for key in LR_KEYS:
            new_params[key] = self._clamp_lr(key, float(new_params.get(key, LR_DEFAULTS[key])))
        return new_params

    def _nudge_lr(self, new_params: Dict[str, Any], key: str, direction: int, magnitude: float, scale: float = 0.20) -> None:
        """Multiplicative nudge for a single LR group, in-place."""
        lr = float(new_params.get(key, LR_DEFAULTS[key]))
        lr *= math.exp(direction * scale * magnitude)
        new_params[key] = self._clamp_lr(key, lr)

    def _pull_lr_toward(self, new_params: Dict[str, Any], key: str, target: float, pull: float) -> None:
        """Blend a single LR group toward a target value, in-place."""
        current = float(new_params.get(key, LR_DEFAULTS[key]))
        target = self._clamp_lr(key, target)
        new_params[key] = self._clamp_lr(key, current * (1.0 - pull) + target * pull)

    # Thin delegates. The logic is module-level (see fingerprint_votes below)
    # because Agent 4 needs the ARCHITECTURE half of these votes -- every one
    # of these signals is a statement about depth, width or heads, which are
    # Agent 4's parameters now -- and importing an instance method to get them
    # would be worse than sharing a function.
    def _fingerprint_late_slice(self, values: List[float]) -> List[float]:
        return _late_slice(values)

    def _fingerprint_votes(self, fingerprint: Dict[str, Any]) -> Dict[str, List[int]]:
        return fingerprint_votes(fingerprint)

    def _fingerprint_adjustment(self, new_params: Dict[str, Any], evidence: Optional[list]) -> Dict[str, Any]:
        """Tier 4: nudge new_params using the most recent fingerprint-bearing
        evidence entry, if any -- a final, path-independent pass (see
        decide_next_hyperparams) applied on top of whichever path
        (surrogate/evidence/heuristic/radical_change) already proposed
        new_params. No-op, returning new_params unchanged, when no evidence
        entry has a fingerprint yet (this is the common case early in a
        search, or whenever token_xai_enabled hasn't fired recently).
        """
        fingerprint = None
        for item in reversed(evidence or []):
            candidate = item.get("token_fingerprint") if isinstance(item, dict) else None
            if candidate:
                fingerprint = candidate
                break
        if not fingerprint:
            return new_params

        votes = self._fingerprint_votes(fingerprint)
        applied: List[Dict[str, Any]] = []

        # ARCHITECTURE VOTES ARE NOT APPLIED HERE ANY MORE. n_layer, n_embd and
        # n_head belong to Agent 4 and define the region; nudging them here
        # would produce a run whose initial weights differ from the rest of its
        # own region, destroying the pairing that makes a within-region
        # comparison readable -- the very thing steps 1-3 established. This pass
        # ran on EVERY decision path, including the surrogate one, so it
        # bypassed the fence 3b put on the EI search.
        #
        # The signals are not discarded: agents/xai_direction.py hands the
        # architecture half to Agent 4, which is the agent that can act on them.
        # Only window_s_fraction remains, and it belongs here because it does
        # not touch a single weight (verified in step 1: 0.05 vs 0.95 gave
        # 46/46 identical tensors).
        if "window_s_fraction" in votes:
            vote_sum = max(-FINGERPRINT_MAX_STEP, min(FINGERPRINT_MAX_STEP, sum(votes["window_s_fraction"])))
            delta = vote_sum * 0.1
            if delta != 0:
                lo, hi = ARCH_SAFE_RANGES["window_s_fraction"]
                current = float(new_params.get("window_s_fraction", 0.75))
                new_value = max(lo, min(current + delta, hi))
                new_params["window_s_fraction"] = new_value
                applied.append({"param": "window_s_fraction", "votes": votes["window_s_fraction"], "delta": delta, "new_value": new_value})

        self._last_fingerprint_adjustments = applied
        return new_params

    def _evidence_adjustment(
        self,
        latest_summary: Optional[str],
        evidence: Optional[list],
        stuck_signal: bool,
        iteration: int,
    ) -> Dict[str, Any]:
        """Translate structured evidence from Agents 2 and 3 into new hyperparameters."""
        new_params = self.current_hyperparams.copy()

        if stuck_signal:
            return self._radical_change(new_params)

        if not evidence:
            return new_params

        importance_by_param: Dict[str, float] = {}
        evidence_count_by_param: Dict[str, int] = {}
        for item in evidence:
            if not isinstance(item, dict):
                continue
            for param, score in item.get("hyperparameter_importance", {}).items():
                importance_by_param[param] = importance_by_param.get(param, 0.0) + float(score)
                evidence_count_by_param[param] = evidence_count_by_param.get(param, 0) + 1
            if item.get("stuck_signal"):
                print("[Agent 1] Evidence indicates a stuck pattern")

        avg_importance: Dict[str, float] = {}
        for param, total_score in importance_by_param.items():
            count = max(1, evidence_count_by_param.get(param, 1))
            avg_importance[param] = max(0.0, min(1.0, total_score / count))

        # Report-level weighted nudges (single-run evidence): smaller base step.
        for param, score in avg_importance.items():
            magnitude = self._importance_magnitude(score)
            direction = 1 if score >= 0.5 else -1

            if param in LR_KEYS:
                self._nudge_lr(new_params, param, direction, magnitude, scale=0.20)

            elif param == "n_layer":
                current = int(new_params.get("n_layer", 12))
                delta = int(round(direction * (1.0 + magnitude)))
                new_params["n_layer"] = max(4, min(current + delta, MAX_N_LAYER))

            elif param == "n_embd":
                current = int(new_params.get("n_embd", 512))
                factor = 1.0 + direction * (0.08 * magnitude)
                new_params["n_embd"] = min(int(current * factor), MAX_N_EMBD)

            elif param == "n_head":
                current = int(new_params.get("n_head", 8))
                delta = int(round(direction * magnitude))
                new_params["n_head"] = max(1, min(current + delta, MAX_N_HEAD))

        # Summary-level updates: stronger than report-level (2x by default).
        if latest_summary:
            recs = self._extract_summary_recommendations(latest_summary)
            summary_hints = self._summary_importance_hints(latest_summary)
            summary_multiplier = max(1.0, self.summary_strength)

            for param in [*LR_KEYS, "n_layer", "n_embd", "n_head"]:
                if not summary_hints.get(param) and param not in recs:
                    continue

                score = avg_importance.get(param, 0.75)
                magnitude = self._importance_magnitude(score)
                direction = 1 if score >= 0.5 else -1

                if param in LR_KEYS:
                    if param in recs:
                        pull = min(0.8, 0.25 * summary_multiplier)
                        self._pull_lr_toward(new_params, param, float(recs[param]), pull)
                    else:
                        self._nudge_lr(new_params, param, direction, magnitude, scale=0.20 * summary_multiplier)

                elif param == "n_layer":
                    current = int(new_params.get("n_layer", 12))
                    if param in recs:
                        target = int(round(float(recs[param])))
                        step = int(round((target - current) * min(1.0, 0.35 * summary_multiplier)))
                        if step == 0 and target != current:
                            step = 1 if target > current else -1
                        current += step
                    else:
                        current += int(round(direction * summary_multiplier * (1.0 + magnitude)))
                    new_params["n_layer"] = max(4, min(current, MAX_N_LAYER))

                elif param == "n_embd":
                    current = int(new_params.get("n_embd", 512))
                    if param in recs:
                        target = max(128, int(round(float(recs[param]))))
                        pull = min(0.8, 0.25 * summary_multiplier)
                        current = int(current * (1.0 - pull) + target * pull)
                    else:
                        factor = 1.0 + direction * (0.08 * summary_multiplier * magnitude)
                        current = int(current * factor)
                    new_params["n_embd"] = min(max(128, current), MAX_N_EMBD)

                elif param == "n_head":
                    current = int(new_params.get("n_head", 8))
                    if param in recs:
                        target = int(round(float(recs[param])))
                        step = int(round((target - current) * min(1.0, 0.35 * summary_multiplier)))
                        if step == 0 and target != current:
                            step = 1 if target > current else -1
                        current += step
                    else:
                        current += int(round(direction * summary_multiplier * magnitude))
                    new_params["n_head"] = max(1, min(current, MAX_N_HEAD))

        # Hard safety bound for LR groups after all evidence/summary adjustments.
        for key in LR_KEYS:
            new_params[key] = self._clamp_lr(key, float(new_params.get(key, LR_DEFAULTS[key])))

        return new_params

    def _heuristic_adjustment(
        self, summary: Optional[str], stuck: bool, iteration: int
    ) -> Dict[str, Any]:
        """
        Heuristic-based hyperparameter adjustment.

        Rules:
        - If stuck: Try radical changes (different depth, architecture)
        - Else: Make incremental adjustments based on importance
        """

        new_params = self.current_hyperparams.copy()

        if stuck:
            return self._radical_change(new_params)

        # Extract insights from summary
        if summary:
            recs = self._extract_summary_recommendations(summary)
            hints = self._summary_importance_hints(summary)
            strength = max(1.0, self.summary_strength)

            for lr_key in LR_KEYS:
                if lr_key in recs:
                    pull = min(0.8, 0.25 * strength)
                    self._pull_lr_toward(new_params, lr_key, float(recs[lr_key]), pull)
                    print(f"[Agent 1] Summary-guided {lr_key}: {new_params[lr_key]:.2e}")
                elif hints.get(lr_key):
                    self._nudge_lr(new_params, lr_key, direction=1, magnitude=1.0, scale=0.25 * strength)
                    print(f"[Agent 1] Adjusted {lr_key}: {new_params[lr_key]:.2e}")

            if "n_layer" in recs:
                current_layer = int(new_params.get("n_layer", 12))
                target_layer = int(round(float(recs["n_layer"])))
                delta = int(round((target_layer - current_layer) * min(1.0, 0.35 * strength)))
                if delta == 0 and target_layer != current_layer:
                    delta = 1 if target_layer > current_layer else -1
                new_params["n_layer"] = max(4, min(current_layer + delta, MAX_N_LAYER))
            elif hints.get("n_layer"):
                layer_delta = int(round(2 * strength))
                new_params["n_layer"] = max(4, min(int(new_params["n_layer"]) + layer_delta, MAX_N_LAYER))
            if hints.get("n_layer") or "n_layer" in recs:
                print(f"[Agent 1] Adjusted layers: {new_params['n_layer']}")

            if "n_embd" in recs:
                current_embd = int(new_params.get("n_embd", 512))
                target_embd = max(128, int(round(float(recs["n_embd"]))))
                pull = min(0.8, 0.25 * strength)
                current_embd = int(current_embd * (1.0 - pull) + target_embd * pull)
                new_params["n_embd"] = min(max(128, current_embd), MAX_N_EMBD)
            elif hints.get("n_embd"):
                emb_factor = 1.0 + 0.12 * strength
                new_params["n_embd"] = min(int(new_params["n_embd"] * emb_factor), MAX_N_EMBD)
            if hints.get("n_embd") or "n_embd" in recs:
                print(f"[Agent 1] Adjusted embedding: {new_params['n_embd']}")
        else:
            # No summary yet - early iterations, try random exploration
            if iteration < 5:
                print("[Agent 1] Early iteration - random exploration")
                new_params["n_layer"] = random.randint(6, 18)
                new_params["n_embd"] = random.choice([256, 384, 512, 768, 1024, MAX_N_EMBD])
                for lr_key in LR_KEYS:
                    new_params[lr_key] = self._random_lr(lr_key)

        for key in LR_KEYS:
            new_params[key] = self._clamp_lr(key, float(new_params.get(key, LR_DEFAULTS[key])))

        return new_params

    def _clamp_lr(self, key: str, lr: float) -> float:
        """Keep a learning-rate group inside its safe operating range. Loud:
        prints and records whenever it actually changes the value -- this
        was previously silent, which is a real reason "extreme" proposals
        were hard to explain after the fact (see dev/inpsect_workflow_ideas.txt).
        """
        lo, hi = self.lr_bounds.get(key, LR_SAFE_RANGES.get(key, (1e-5, 5.0)))
        original = lr
        clamped = lo if not math.isfinite(lr) else max(lo, min(lr, hi))
        if clamped != original:
            print(f"[Agent 1] CLAMP {key}: requested={original} -> clamped={clamped} (bounds [{lo}, {hi}])")
            self._last_lr_clamps[key] = {"requested": original, "clamped": clamped, "bounds": [lo, hi]}
        return clamped

    def _random_lr(self, key: str) -> float:
        """Log-uniform random sample within the safe range for one LR group."""
        lo, hi = self.lr_bounds.get(key, LR_SAFE_RANGES[key])
        return 10 ** random.uniform(math.log10(lo), math.log10(hi))

    def _importance_magnitude(self, score: float) -> float:
        """Map [0,1] score to distance from neutral 0.5 in [0,1]."""
        bounded = max(0.0, min(1.0, float(score)))
        return abs(bounded - 0.5) * 2.0

    def _summary_importance_hints(self, summary: str) -> Dict[str, bool]:
        """Extract lightweight importance hints from summary text."""
        if not summary:
            return {}
        text = summary.lower()
        important_tokens = ["important", "stable", "strong", "matters"]

        def _has_signal(*tokens: str) -> bool:
            return any(token in text for token in tokens) and any(t in text for t in important_tokens)

        return {
            "embedding_lr": _has_signal("embedding_lr", "embedding lr"),
            "unembedding_lr": _has_signal("unembedding_lr", "unembedding lr"),
            "matrix_lr": _has_signal("matrix_lr", "matrix lr", "learning rate", "learning_rate", "lr"),
            "scalar_lr": _has_signal("scalar_lr", "scalar lr"),
            "n_layer": _has_signal("n_layer", "layer", "depth"),
            "n_embd": _has_signal("n_embd", "embedding"),
            "n_head": _has_signal("n_head", "head"),
        }

    def _extract_summary_recommendations(self, summary: str) -> Dict[str, float]:
        """Parse numeric recommendation lines from Agent 3 markdown summary."""
        if not summary:
            return {}

        patterns = {
            # Negative lookbehind on "embedding_lr" keeps it from also matching
            # inside "unembedding_lr" (which would otherwise double-match).
            "embedding_lr": r"(?<!un)embedding_lr[^\n:]*:\s*([0-9eE+\-.]+)",
            "unembedding_lr": r"unembedding_lr[^\n:]*:\s*([0-9eE+\-.]+)",
            "matrix_lr": r"matrix_lr[^\n:]*:\s*([0-9eE+\-.]+)",
            "scalar_lr": r"scalar_lr[^\n:]*:\s*([0-9eE+\-.]+)",
            "n_layer": r"n_layer[^\n:]*:\s*([0-9eE+\-.]+)",
            "n_embd": r"n_embd[^\n:]*:\s*([0-9eE+\-.]+)",
            "n_head": r"n_head[^\n:]*:\s*([0-9eE+\-.]+)",
        }

        recommendations: Dict[str, float] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, summary, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                recommendations[key] = float(match.group(1))
            except (TypeError, ValueError):
                continue
        return recommendations

    def train_model(
        self, hyperparams: Dict[str, Any], dry_run: bool = False, iteration: int = 0,
        gpu_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Train model and return metrics.

        Uses a real subprocess when available, but gracefully falls back to a
        lightweight simulated run so the multi-agent loop still produces
        meaningful artifacts in local or constrained environments.
        """
        # Persist exactly what we were asked to train -- see _save_hyperparams.
        self._save_hyperparams(hyperparams)

        print(f"[Agent 1] Starting training with: {hyperparams}")

        if dry_run:
            print("[Agent 1] Dry run enabled; skipping actual training")
            return {
                "val_bpb": 1.0 - 0.001 * (iteration + 1),
                "training_time": 0.0,
                "checkpoint_path": None,
                "status": "dry_run",
            }

        # --- Priority 1: remote GPU server (configured via .env) ---
        try:
            from agents.remote_runner import is_remote_configured, run_training_remote
            if is_remote_configured():
                print("[Agent 1] Remote GPU server configured — running training remotely")
                metrics = run_training_remote(
                    hyperparams_local_path=str(self.model_config_path),
                    # None (the default) keeps the existing auto-discovery.
                    # An explicit index matters for measurement rather than
                    # search: scripts/noise_floor.py has to spread its repeats
                    # across GPUs deliberately, because auto-discovery sorts by
                    # free memory and so parks every sequential repeat on the
                    # same device -- which is how the campaign ended up with a
                    # sigma that captured only within-GPU noise.
                    gpu_index=gpu_index,
                    # +120s slack: covers the head-ablation study train.py now
                    # runs after its official eval, plus SSH/data overhead.
                    timeout=self.training_budget + 120,
                )
                print(f"[Agent 1] Remote training complete. Metrics: {metrics}")
                return metrics
        except Exception as e:
            # The TYPE as well as the message. A bare {e} prints "()" for any
            # exception carrying no message -- paramiko's socket.timeout is
            # exactly that -- so the one line explaining why a campaign stopped
            # training said nothing at all. That is how a malformed launch
            # command went unnoticed while every run quietly became a
            # fabricated result.
            print(f"[Agent 1] Remote training failed ({type(e).__name__}: {e or 'no message'}) "
                  f"— falling back to local")

        # --- Priority 2: local uv/train.py subprocess ---
        try:
            if not self._can_run_training_command():
                raise RuntimeError("training command unavailable")

            result = subprocess.run(
                ["uv", "run", "train.py"],
                capture_output=True,
                text=True,
                # +90s slack (not +30s): train.py now also runs a real head-ablation
                # study after the official eval (see train.py), which costs a few
                # extra cheap forward passes on top of the training budget.
                timeout=self.training_budget + 90,
            )
            metrics = self._parse_training_output(result.stdout)
            print(f"[Agent 1] Training complete. Metrics: {metrics}")
            metrics.setdefault("status", "ok")
            return metrics

        except subprocess.TimeoutExpired:
            print("[Agent 1] Training timeout")
            return {"val_bpb": float("inf"), "error": "timeout", "status": "simulated"}
        except Exception as e:
            print(f"[Agent 1] Training error: {e}")
            return self._simulate_training_result(hyperparams, iteration, str(e))

    def _can_run_training_command(self) -> bool:
        """Return True when the training subprocess can be executed."""
        try:
            result = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _simulate_training_result(
        self, hyperparams: Dict[str, Any], iteration: int, error: str
    ) -> Dict[str, Any]:
        """Generate a deterministic surrogate metric for local testing."""
        matrix_lr = float(hyperparams.get("matrix_lr", LR_DEFAULTS["matrix_lr"]))
        depth = int(hyperparams.get("n_layer", 12))
        width = int(hyperparams.get("n_embd", 512))
        base = 1.25 - (0.002 * min(depth, 20)) - (0.000001 * width) + (0.15 * min(matrix_lr / LR_DEFAULTS["matrix_lr"], 2.0))
        val_bpb = max(0.65, base - 0.001 * iteration)
        return {
            "val_bpb": round(val_bpb, 6),
            "training_time": round(0.2 + 0.01 * iteration, 3),
            "checkpoint_path": None,
            "status": "simulated",
            "error": error,
        }

    def _parse_training_output(self, stdout: str) -> Dict[str, Any]:
        """Parse all metrics from train.py's final summary block.

        The field map and the scanning itself live in agents/train_output.py,
        shared with agents/remote_runner.py::_parse_output -- see that module
        for why they must not be two hand-synced copies.

        Only the default below is specific to this path: this is the local
        subprocess fallback, whose caller (train_model) sets the status itself,
        so unlike the remote parser this one seeds no `status`.
        """
        metrics: Dict[str, Any] = {"val_bpb": float("inf")}
        metrics.update(train_output.parse(stdout))
        return metrics

    def _extract_summary_sections(self, summary: str, headings: List[str]) -> str:
        """Pulls out just the named "## Heading" sections (in the given
        order) from a full Agent 3 summary markdown, dropping everything
        else. Agent 3 already digests the raw statistical tables into
        Recommendations/Strategic Insights/Strategic Narrative/Cluster
        Hypotheses -- sending those raw tables to this call too would be
        genuine duplication (the same signal expressed twice, once as
        numbers and once as Claude's own prior paraphrase of them). This
        also sidesteps a blind summary[:N] truncation cutting off the
        narrative/cluster-hypotheses sections, which sit at the very end of
        the file and can be larger than N for a big summary. Returns ""
        (caller falls back to summary[:6000]) if none of `headings` are
        present -- e.g. a plain string with no "## " markup at all.
        """
        lines = summary.splitlines()
        bounds: Dict[str, Tuple[int, int]] = {}
        current: Optional[str] = None
        start = 0
        for i, line in enumerate(lines):
            if line.startswith("## "):
                if current is not None:
                    bounds[current] = (start, i)
                current = line.strip()
                start = i
        if current is not None:
            bounds[current] = (start, len(lines))

        parts = [
            "\n".join(lines[bounds[heading][0]:bounds[heading][1]]).strip()
            for heading in headings if heading in bounds
        ]
        return "\n\n".join(parts)

    def _get_claude_suggestion(
        self, heuristic_params: Dict[str, Any], summary: str
    ) -> Optional[Dict[str, Any]]:
        """Ask the Claude Code CLI (agents/claude_cli.py -- your subscription,
        not a separately-billed API key) to review the heuristic suggestion.
        Only ever called with a *fresh* summary (see decide_next_hyperparams's
        fresh_summary gate) -- reads just the already-distilled sections of
        the summary (Recommendations, Strategic Insights, Strategic
        Narrative, Cluster Hypotheses -- see _extract_summary_sections)
        rather than the raw statistical tables Agent 3 already turned into
        those sections, since this fires once per new summary rather than
        every iteration. Returns None (falling back to heuristic_params
        unchanged, exactly as before this feature existed) whenever the CLI
        is unavailable, the campaign budget is exhausted, or the response
        isn't usable JSON -- never fabricated, never raises.
        """
        distilled = self._extract_summary_sections(summary, [
            "## Recommendations for Agent 1 (Data-Backed)",
            "## Strategic Insights",
            "## Strategic Narrative",
            "## Cluster Hypotheses (Claude)",
        ])
        prompt = f"""You are reviewing a strategic summary from a neural network
hyperparameter search campaign (its data-backed recommendations, strategic
narrative, and cluster hypotheses). Think through what it implies, then
decide whether our heuristic next-step suggestion below should change.

Summary:
{distilled or summary[:6000]}

Our heuristic suggestion for the next run:
{heuristic_params}

Reason about the summary's stable patterns, conflicting signals, and any
strategic narrative/cluster hypotheses -- then provide JSON with
adjustments (or empty {{}} if the heuristic suggestion already looks right).
Example: {{"n_layer": 14, "matrix_lr": 0.03}}"""

        response_text = claude_cli.call_with_budget(
            prompt, call_site="agent1_hyperparameter_review",
            model=self._llm_model,
            campaign_budget_usd=self._llm_campaign_budget_usd,
            max_call_budget_usd=self._llm_max_call_budget_usd,
            usage_path=self._llm_usage_path,
            backend=self._llm_backend,
        )
        if not response_text:
            return None

        try:
            if "{" in response_text and "}" in response_text:
                json_str = response_text[response_text.find("{"): response_text.rfind("}") + 1]
                adjustments = json.loads(json_str)
                if adjustments:
                    updated = heuristic_params.copy()
                    updated.update(adjustments)
                    print(f"[Agent 1] Claude suggested adjustments: {adjustments}")
                    return updated
            return None
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Agent 1] Claude suggestion error: {e}")
            return None
