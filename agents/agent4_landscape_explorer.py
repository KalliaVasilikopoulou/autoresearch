"""Agent 4: the optimization landscape explorer.

Two jobs, and nothing else:

  1. PROPOSE REGIONS -- decide where a newly-freed GPU should start looking.
  2. JUDGE REGIONS   -- decide when a region has earned a lifecycle change
                        (keep exploiting / pause / it's a dead end / it's a
                        local optimum we've finished with).

It no longer proposes training configurations. That belongs to Agent 1,
scoped to a region (Agent1TrainingSpecialist.search_region), and the
orchestrator's allocator decides which region each GPU serves this wave.

What this replaced, and why
---------------------------
Agent 4 used to be a temporary override of a single search: it watched for
stagnation, seized every GPU for a bounded "window", probed one candidate
region at a time, and handed control back. Three things were wrong with that,
all of them measured rather than argued:

  - The trigger was reactive. Exploration only started after the frontier had
    already stalled, so on a short campaign the better region got found with
    no budget left to exploit it.
  - The stagnation trigger does not discriminate. Against 582 real runs it
    reads "stagnant" at 89-92% of iterations for every threshold from 0.001
    to 0.005, because the frontier genuinely does not move in most 10-run
    windows.
  - The abandon rule did not measure what it claimed. "Probe is worse than
    the elite reference by more than bad_tolerance" compares against a
    CAMPAIGN-wide elite, so it asks "is this run not elite" -- true for most
    runs by construction. Sweeping the tolerance 17x moved the fraction of
    runs called bad only from 55% to 85%.

So exploration is scheduled continuously by the allocator instead of
triggered, and a region is retired by RELATIVE comparison against the other
live regions rather than against an absolute bar. See the CALIBRATION
REFERENCE at the top of agents_config.yaml for the full derivation.

Region diversity
----------------
propose_regions deliberately uses three different criteria rather than three
draws from one. Ranking purely by surrogate uncertainty -- which is what the
old _propose_region did -- concentrates candidates at the *edges* of the
sampled space, because that is where a random forest is least confident. Three
regions chosen that way are three versions of the same idea. The three used
here answer three different questions: where has nobody looked, where does the
acquisition function want to go, and where is there a rival optimum.
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal environments
    yaml = None

from agents import claude_cli
from agents.agent1_training_specialist import SEARCH_SPACE
from state import surrogate
from state.landscape import build_landscape
from state.regions import (
    ACTIVE,
    CAPACITY_PAUSED,
    LOCAL_OPTIMUM,
    MIGRATED,
    NO_OPTIMUM,
    PAUSED,
    SATURATED,
    Region,
    RegionRegistry,
    distance,
    same_architecture,
)
from state.results_analysis import HYPERPARAM_COLUMNS
from state.surrogate import fit_surrogate

# Lifecycle verdicts from judge(). Strings rather than an enum to match how
# every other status/flag vocabulary in this codebase is expressed
# (pipeline_validator severities, run statuses, region flags).
KEEP = "keep"

#: Why a region was proposed. Recorded on the Region so a later summary can
#: say which kind of bet paid off -- the three criteria are not
#: interchangeable and lumping their outcomes together would hide that.
ORIGIN_UNEXPLORED = "unexplored"
ORIGIN_HIGH_EI = "high_ei"
ORIGIN_RIVAL_OPTIMUM = "rival_optimum"
PROPOSAL_CRITERIA = (ORIGIN_UNEXPLORED, ORIGIN_HIGH_EI, ORIGIN_RIVAL_OPTIMUM)

LLM_MODES = ("statistics", "hybrid", "llm")


class Agent4LandscapeExplorer:
    """Proposes new regions to search, and judges the ones already running."""

    def __init__(
        self,
        config_path: str = "agents_config.yaml",
        root_dir: Optional[str] = None,
        state_dir: Optional[str] = None,
        reports_dir: Optional[str] = None,
    ):
        self.config = self._load_config(config_path)
        cfg = self.config.get("agent4", {})

        self.enabled = bool(cfg.get("enabled", True))
        # Geometry. Both in normalized 11-D distance (state/regions.py), NOT
        # in the 2-D PCA units these used to carry -- that projection
        # explained 0.296 of the variance and could not answer "are these the
        # same place", which is the question a multi-region search asks
        # constantly. See agents_config.yaml for how 0.05 was re-derived.
        self.region_radius = float(cfg.get("region_radius", 0.05))
        self.merge_radius = float(cfg.get("merge_radius", 0.025))
        self.grid_resolution = int(cfg.get("grid_resolution", 24))

        # How many regions to keep alive. The allocator opens a new one
        # whenever it has a spare GPU and fewer than this are live, and never
        # fragments past it. There is deliberately no min_regions companion:
        # a new region can only be opened when a GPU is free, so a floor
        # would be unenforceable exactly when it mattered.
        self.max_regions = int(cfg.get("max_regions", 4))

        # --- lifecycle thresholds, all counted in RUNS THIS REGION SPENT ---
        # Not in orchestrator iterations: a region holding two GPUs
        # accumulates twice as fast as one holding a single GPU, so an
        # iteration count would judge the two on different amounts of
        # evidence while appearing to use one rule.
        self.min_runs_before_judgement = int(cfg.get("min_runs_before_judgement", 5))
        self.stuck_runs_pause = int(cfg.get("stuck_runs_pause", 5))
        self.stuck_runs_retire = int(cfg.get("stuck_runs_retire", 15))
        # A region still working through Agent 1's Sobol cold start is exempt
        # from the stuck rules -- see judge(). Read from agent1's block
        # because it is agent1's cold start; duplicating the number under
        # agent4: would be two settings for one thing.
        self.cold_start_n = int(
            self.config.get("agent1", {}).get("surrogate_min_observations", 15))

        # sigma_region (0.0028): the spread of val_bpb among DIFFERENT configs
        # inside one region. Every threshold below compares one configuration
        # against another, so this is the right yardstick -- the noise floor's
        # sigma (0.000797) measures repeatability of a single config and is
        # ~4x too sensitive for these. See the CALIBRATION REFERENCE.
        # --- escape pressure (step 5b) ---
        #: How many recent proposals to look at.
        self.escape_window = int(cfg.get("escape_window", 6))
        #: How many of them must have wanted to leave.
        self.escape_runs_to_migrate = int(cfg.get("escape_runs_to_migrate", 3))
        #: How much they must AGREE on a direction, 0-1. This is the real test:
        #: escapes pointing every which way average to nothing and mean the
        #: region is being explored, not mis-anchored. 0.6 demands the mean
        #: vector keep most of its length through the averaging.
        self.escape_coherence = float(cfg.get("escape_coherence", 0.6))

        #: How hard XAI may push the architecture proposal, before being scaled
        #: down by the surrogate's own accuracy. Deliberately SMALL: the
        #: FINGERPRINT_* thresholds these votes fire on are uncalibrated by the
        #: code's own admission, so a large weight would only amplify a guess.
        #: Raise it once those thresholds have been checked against real
        #: fingerprint history.
        self.xai_weight = float(cfg.get("xai_weight", 0.25))
        self._last_xai_direction: Optional[Dict[str, Any]] = None
        self.sigma_region = float(cfg.get("sigma_region", 0.0028))
        self.retire_margin_sigma = float(cfg.get("retire_margin_sigma", 3.0))
        self.improvement_sigma = float(cfg.get("improvement_sigma", 1.0))

        llm_mode = str(cfg.get("llm_mode", "hybrid"))
        if llm_mode not in LLM_MODES:
            print(f"[Agent 4] Unknown llm_mode {llm_mode!r} -- falling back to 'statistics'")
            llm_mode = "statistics"
        self.llm_mode = llm_mode

        _root = Path(root_dir) if root_dir else Path(".")
        _state = Path(state_dir) if state_dir else Path("state")
        _reports = Path(reports_dir) if reports_dir else Path("reports")
        self.results_path = _root / "results.tsv"
        self.decisions_dir = _reports / "agent4_decisions"
        #: Where Agent 1 writes its per-region plan JSONs, which carry the
        #: escape record. Must match Agent1TrainingSpecialist's report dir.
        self._search_plan_root = str(_reports / "agent1_search_plan")
        self.registry_path = _state / "regions.json"

        llm_config = self.config.get("llm", {})
        self._llm_backend = llm_config.get("backend", "cli")
        self._llm_model = llm_config.get("model", "sonnet")
        self._llm_campaign_budget_usd = float(llm_config.get("campaign_budget_usd", 5.0))
        self._llm_max_call_budget_usd = float(llm_config.get("max_call_budget_usd", 0.20))
        self._llm_usage_path = llm_config.get("usage_log_path", str(_state / "llm_usage.json"))

        self.last_decision_log: Optional[Dict[str, Any]] = None

    # -- config -------------------------------------------------------------

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if yaml is None:
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            print(f"[Agent 4] Could not read {config_path}: {e}")
            return {}

    def _a_within(self) -> Optional[float]:
        """The in-region measurement noise -- or None when it has never been
        measured, in which case the saturation test simply does not run.

        Not `_load_sigma`, which falls back to DEFAULT_SIGMA = 0.01 when
        nothing has been measured. That is ~7.5x the real value (0.001342), and
        at 0.01 almost any region looks saturated -- so a fresh checkout would
        silently retire every region it opened. A verdict that throws work away
        must rest on a measurement or not be reached.
        """
        from agents.search_planner import measured_a_within

        return measured_a_within(str(self.registry_path.parent))

    @property
    def improvement_margin(self) -> float:
        """What counts as a region getting genuinely better, in absolute
        val_bpb. Passed to Region.runs_since_improvement, whose default of 0.0
        would count pure noise as progress and never register a region as
        stuck."""
        return self.sigma_region * self.improvement_sigma

    # -- job 1: propose regions ---------------------------------------------

    def propose_regions(
        self,
        rows: List[Dict[str, Any]],
        registry: RegionRegistry,
        n: int,
        at_run: int,
        best_val_bpb: Optional[float] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Region]:
        """Open up to `n` new regions and return them.

        Returns fewer than `n` -- possibly zero -- whenever the landscape
        can't be built yet (too few runs to fit a surrogate) or every
        candidate the criteria produced lands inside a region that already
        exists or has already been ruled out. Returning nothing is the honest
        answer there; inventing a region to fill a GPU slot would put a run
        somewhere no criterion actually recommended.
        """
        if n <= 0 or not self.enabled:
            return []

        finite = [r["val_bpb"] for r in rows
                  if isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])]
        if len(finite) < surrogate.MIN_SURROGATE_N:
            return []
        sm = fit_surrogate(rows)
        if sm is None:
            return []
        landscape = build_landscape(rows, sm, grid_resolution=self.grid_resolution,
                                    hard_bounds=SEARCH_SPACE)
        if landscape is None:
            return []

        f_best = best_val_bpb if best_val_bpb is not None else min(finite)
        opened: List[Region] = []
        # Rotate through the criteria so a wave that opens two regions gets
        # two DIFFERENT kinds of bet, not the top two candidates of one.
        for i in range(n):
            criterion = PROPOSAL_CRITERIA[(len(registry.regions) + i) % len(PROPOSAL_CRITERIA)]
            candidate = self._candidate_for(criterion, landscape, sm, f_best, at_run + i)
            if candidate is None:
                continue
            candidate = self._apply_xai_direction(candidate, sm, evidence)
            if self._too_close_to_known(candidate, registry, opened):
                continue
            region = registry.open_region(candidate, at_run=at_run, origin=criterion)
            opened.append(region)
            print(f"[Agent 4] Opened region {region.region_id} by '{criterion}' "
                  f"(n_layer={candidate.get('n_layer')}, n_embd={candidate.get('n_embd')})")

        if opened:
            registry.save()
            self._record_decision(at_run, "opened_regions", {
                "opened": [{"region_id": r.region_id, "origin": r.origin} for r in opened],
                "requested": n,
                "live_regions": len(registry.active()),
            })
        return opened

    def _apply_xai_direction(self, candidate: Dict[str, Any], sm: Any,
                             evidence: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Nudge a proposed architecture toward what the trained models are
        actually showing.

        This is where interpretability stops being decorative. Everything Agent
        2 measures -- dead heads, layers contributing nothing to the output,
        attention that stops reaching -- is about depth, width and heads, which
        only Agent 4 can change. The surrogate cannot see any of it: it knows
        settings and scores, so it cannot tell "bad because it is too deep"
        from "bad because the learning rate was wrong".

        A BIAS, NOT AN OVERRIDE, and sized by how much the surrogate deserves
        trust right now (its own out-of-bag accuracy, free from the fit). The
        base weight is small on purpose -- these thresholds are uncalibrated,
        and weighting a guess heavily just produces a confident guess.
        """
        from agents import xai_direction

        votes = xai_direction.architecture_votes(evidence)
        if not votes:
            return candidate
        accuracy = xai_direction.surrogate_accuracy(sm)
        steps = xai_direction.weighted_step(votes, accuracy, self.xai_weight)
        if not steps:
            return candidate

        out = dict(candidate)
        for param, step in steps.items():
            lo, hi = SEARCH_SPACE[param]
            scale = 64.0 if param == "n_embd" else 1.0  # n_embd moves in head-sized chunks
            moved = float(out.get(param, (lo + hi) / 2)) + step * scale
            out[param] = int(round(max(lo, min(moved, hi))))
        # n_head must still divide the width with an even quotient, or train.py
        # silently re-snaps the width and the region is not the one proposed.
        out["n_embd"] = surrogate.snap_n_embd(out["n_embd"], out["n_head"])
        print(f"[Agent 4] XAI direction {votes} (surrogate R2="
              f"{'n/a' if accuracy is None else f'{accuracy:.2f}'}, weight {self.xai_weight}) "
              f"-> {steps}")
        self._last_xai_direction = {"votes": votes, "accuracy": accuracy, "steps": steps}
        return out

    def _candidate_for(self, criterion: str, landscape: Dict[str, Any], sm: Any,
                       f_best: float, seed: int) -> Optional[Dict[str, Any]]:
        """One candidate configuration per criterion.

        All three run a full EI search anchored at the cell they picked, with
        EVERY parameter free -- that combination (a center that isn't the
        current best, and no Gauss-Southwell block restriction) is precisely
        the coverage search_planner.propose_next structurally cannot produce,
        and it is the only reason a new region is anywhere new.
        """
        cells = landscape["grid_hyperparams"]

        if criterion == ORIGIN_UNEXPLORED:
            # Highest predicted std: the honest, already-available "nobody has
            # looked here" signal. On its own it biases toward the edges of
            # the sampled space, which is exactly why it is one of three.
            anchor = self._extreme_cell(landscape["grid_z_std"], cells, largest=True)
        elif criterion == ORIGIN_RIVAL_OPTIMUM:
            # Lowest predicted mean: where the surrogate thinks there is
            # another basin. Distinct from EI, which trades prediction off
            # against uncertainty and so tends to stay near what is already
            # known to be good.
            anchor = self._extreme_cell(landscape["grid_z_mean"], cells, largest=False)
        else:
            anchor = None  # high_ei: no anchor, the EI search roams freely

        if criterion != ORIGIN_HIGH_EI and anchor is None:
            return None
        return surrogate.propose_via_ei(
            sm, f_best=f_best, bounds=sm.bounds,
            free_params=list(HYPERPARAM_COLUMNS),
            fixed_values=anchor or {},
            n_candidates=2000, seed=seed,
        )

    @staticmethod
    def _extreme_cell(grid: List[List[float]], cells: List[List[Dict[str, Any]]],
                      largest: bool) -> Optional[Dict[str, Any]]:
        best = None
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                v = grid[r][c]
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    continue
                if best is None or (v > best[0] if largest else v < best[0]):
                    best = (v, r, c)
        return None if best is None else cells[best[1]][best[2]]

    def _too_close_to_known(self, candidate: Dict[str, Any], registry: RegionRegistry,
                            opened: List[Region]) -> bool:
        """Reject a candidate that lands on top of a region we already have --
        including one already ruled out.

        The terminal flags are the point: a region flagged no_optimum or
        local_optimum is a measurement, and re-opening it would spend GPUs
        re-deriving a conclusion already paid for. `include_terminal=True` is
        what makes those flags do work rather than just decorate a chart.
        """
        region, dist = registry.nearest(candidate, include_terminal=True)
        if region is not None and dist <= self.region_radius:
            return True
        # `nearest` already restricts itself to regions sharing this
        # candidate's architecture; this second check covers the regions opened
        # earlier in THIS wave, which are not in the registry's pool yet, and
        # must apply the same rule. A different architecture is a different
        # region however close its tunables sit, so proximity alone must not
        # veto it -- that would refuse to open, say, a 12-layer region merely
        # because a 20-layer one already uses similar learning rates.
        return any(same_architecture(candidate, r.anchor)
                   and distance(candidate, r.anchor, registry.bounds) <= self.region_radius
                   for r in opened)

    # -- job 2: judge regions -----------------------------------------------

    def judge(self, region: Region, registry: RegionRegistry, at_run: int) -> str:
        """One region's lifecycle verdict. Returns KEEP, PAUSED, NO_OPTIMUM
        or LOCAL_OPTIMUM; anything but KEEP has already been applied to the
        region and saved.

        The order of the tests is the argument. Retirement for being stuck a
        long time is checked before retirement for being bad, because they
        are different claims about different evidence: "we exploited this
        thoroughly and it stopped paying" is a stronger, better-supported
        statement than "this looks worse than its neighbours", and a region
        that qualifies for both deserves the stronger label.
        """
        if region.n_measured < self.min_runs_before_judgement:
            return KEEP

        # THE LAST LIVE REGION IS NEVER TERMINALLY RETIRED -- see _retire.
        # Pausing it is still allowed: that is recoverable, and the
        # orchestrator resumes the best paused region when it has a GPU and
        # nowhere better to put it.
        alone = not [r for r in registry.active() if r.region_id != region.region_id]

        # A region still in its Sobol cold start is exempt from the stuck
        # rules. Cold-start points are a space-filling sample, deliberately
        # NOT a descent -- consecutive draws have no reason to improve on each
        # other, so runs_since_improvement measures nothing there. Observed in
        # a dry run: the bootstrap region was paused at 12 runs, immediately
        # resumed (nothing better existed), and paused again at 16, churning
        # its flag every wave while doing exactly what it was supposed to.
        if self._still_cold_starting(region):
            return KEEP

        # Count "stuck" only over runs this region CHOSE, never over its
        # cold-start draws. Sobol points are space-filling, so the best of
        # them sits wherever it happens to sit and the draws after it do not
        # improve on it -- by construction, not because the region is
        # exhausted. Deferring the check until the cold start ended (above)
        # was not enough on its own: the accumulated no-improvement history
        # was still there, so the very first judgement after the exemption
        # lifted saw runs_since_improvement = 15 and retired the region as a
        # local optimum. That happened on the first real 20-run campaign, at
        # exactly run 16.
        # SATURATION FIRST -- it is the better-evidenced claim. "No improvement
        # in 15 runs" says the region stopped moving, which can be bad luck and
        # can recover. Saturation says the differences still inside it are
        # smaller than we can measure, so no further spending can rank them.
        # It also explains WHY, which a run counter never does.
        a_within = self._a_within()
        if a_within is not None:
            saturated = region.is_saturated(a_within,
                                            min_runs=self.min_runs_before_judgement)
            if saturated and not alone:
                return self._retire(region, registry, SATURATED, at_run, {
                    "real_signal": region.real_signal(
                        a_within, min_runs=self.min_runs_before_judgement),
                    "a_within": a_within,
                    "n_measured": region.n_measured,
                    "best_val_bpb": region.best_val_bpb,
                })

        stuck_for = region.runs_since_improvement(
            self.improvement_margin, skip_first=self._cold_start_runs(region))

        if stuck_for >= self.stuck_runs_retire and not alone:
            return self._retire(region, registry, LOCAL_OPTIMUM, at_run, {
                "runs_since_improvement": stuck_for,
                "threshold": self.stuck_runs_retire,
                "n_measured": region.n_measured,
                "best_val_bpb": region.best_val_bpb,
            })

        worse_by = self._worse_than_field_by(region, registry)
        if (worse_by is not None and not alone
                and worse_by > self.retire_margin_sigma * self.sigma_region):
            return self._retire(region, registry, NO_OPTIMUM, at_run, {
                "worse_than_best_live_by": worse_by,
                "margin": self.retire_margin_sigma * self.sigma_region,
                "retire_margin_sigma": self.retire_margin_sigma,
                "n_measured": region.n_measured,
                "elite_score": region.elite_score(),
            })

        if alone and stuck_for >= self.stuck_runs_retire:
            # Nowhere to go, but this region really has stopped paying.
            # Pausing is honest and recoverable -- the orchestrator resumes it
            # if nothing better turns up, and Agent 4 gets a chance to propose
            # somewhere new in the meantime. Retiring it outright would leave
            # the campaign with no region at all.
            print(f"[Agent 4] Region {region.region_id} looks exhausted but is the only "
                  f"live one -- pausing rather than retiring, so the campaign keeps a "
                  f"place to search")

        if stuck_for >= self.stuck_runs_pause:
            # Deliberately NOT a retirement. "We haven't improved here lately"
            # and "there is nothing here" are different claims and only the
            # first is supported. The allocator pauses it and looks for
            # somewhere better; if nothing better exists it can resume, which
            # a terminal flag would forbid.
            return self._retire(region, registry, PAUSED, at_run, {
                "runs_since_improvement": stuck_for,
                "threshold": self.stuck_runs_pause,
                "n_measured": region.n_measured,
                "resumable": True,
            })

        return KEEP

    def _still_cold_starting(self, region: Region) -> bool:
        """Is this region's own search still drawing Sobol cold-start points?

        Only the bootstrap region can be: every other region is opened after
        a surrogate already fits, and agents/search_planner.py gates the cold
        start on the number of usable rows in results.tsv campaign-wide, not
        per region. Tying the exemption to the origin rather than to a run
        count keeps it from silently exempting a real region that happens to
        be young.
        """
        return region.origin == "bootstrap" and region.n_measured < self.cold_start_n

    def _cold_start_runs(self, region: Region) -> int:
        """How many of this region's measured runs were Sobol cold-start
        draws rather than choices its own search made. Zero for every region
        except the bootstrap one, which is the only one that cold-starts."""
        return min(self.cold_start_n, region.n_measured) if region.origin == "bootstrap" else 0

    def _worse_than_field_by(self, region: Region, registry: RegionRegistry) -> Optional[float]:
        """How far this region's elite score sits behind the best live
        region's, or None when there is nothing to compare against.

        Relative, not absolute, and this is the whole correction to the old
        rule: an absolute bar against the campaign-wide elite measured "is
        this run not elite" (true for most runs by construction) rather than
        "is this region bad". A region is only bad relative to somewhere
        better that we are actually able to run instead.
        """
        mine = region.elite_score()
        if mine is None:
            return None
        rivals = [
            r.elite_score() for r in registry.active()
            if r.region_id != region.region_id
            and r.elite_score() is not None
            and r.n_measured >= self.min_runs_before_judgement
        ]
        if not rivals:
            return None
        return mine - min(rivals)

    def _retire(self, region: Region, registry: RegionRegistry, flag: str,
                at_run: int, detail: Dict[str, Any]) -> str:
        region.set_flag(flag, at_run)
        registry.save()
        print(f"[Agent 4] Region {region.region_id} -> {flag} "
              f"after {region.n_runs} run(s): {detail}")
        self._record_decision(at_run, flag, {"region_id": region.region_id, **detail},
                              region_id=region.region_id)
        return flag

    # -- escape pressure: the region is anchored in the wrong place ---------

    def escape_pressure(self, region: Region) -> Optional[Dict[str, Any]]:
        """Has this region's search repeatedly wanted to leave, in a CONSISTENT
        direction?

        Every proposal already records where the best candidate would have gone
        with the fence ignored (see surrogate.propose_via_ei) -- it costs no
        extra training, since those candidates are generated and scored anyway
        and simply are not eligible to run. This reads that trail back.

        Consistency is the whole test, and it is why the mean of the direction
        VECTORS is used rather than a count of escapes. A search bouncing off
        different walls produces escapes pointing every which way, whose mean
        cancels to nearly nothing -- that is a region being explored, not a
        region in the wrong place. Sustained pressure one way survives the
        averaging. `coherence` below is the ratio of the mean vector's length
        to the mean of the individual lengths: 1.0 is perfect agreement, 0.0 is
        pure cancellation.

        Returns None when there is not enough history, or when the pressure is
        incoherent, or when the place it points to is inside the fence anyway.
        """
        plan_dir = Path(region.report_dir(self._search_plan_root))
        if not plan_dir.exists():
            return None
        escapes: List[Dict[str, Any]] = []
        for path in sorted(plan_dir.glob("plan_*.json"))[-self.escape_window:]:
            try:
                esc = json.loads(path.read_text(encoding="utf-8")).get("escape")
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            if isinstance(esc, dict) and esc.get("escaped") and esc.get("direction"):
                escapes.append(esc)
        if len(escapes) < self.escape_runs_to_migrate:
            return None

        params = sorted({p for e in escapes for p in e["direction"]})
        mean_vec = {p: sum(float(e["direction"].get(p, 0.0)) for e in escapes) / len(escapes)
                    for p in params}
        mean_len = math.sqrt(sum(v * v for v in mean_vec.values()))
        lengths = [math.sqrt(sum(float(v) ** 2 for v in e["direction"].values()))
                   for e in escapes]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        coherence = (mean_len / avg_len) if avg_len > 0 else 0.0
        if coherence < self.escape_coherence:
            return None

        target = self._escape_target(region, mean_vec)
        if target is None:
            return None
        travelled = distance(target, region.anchor, None)
        if travelled <= self.region_radius:
            # It wants to go somewhere the fence already allows, so there is
            # nothing to migrate to -- the search can simply go there.
            return None
        return {"n_escapes": len(escapes), "coherence": coherence,
                "mean_direction": mean_vec, "distance": travelled, "target": target}

    def _escape_target(self, region: Region, mean_vec: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Where the successor should be anchored: the region's anchor shifted
        by the mean escape direction, in normalized space, then mapped back.
        Architecture is copied unchanged -- escape is measured over the
        tunables only, so a successor is the same model in a different corner
        of its settings."""
        from state.surrogate import _denormalize, normalized_value

        bounds = SEARCH_SPACE
        target = dict(region.anchor)
        for p, delta in mean_vec.items():
            if p not in bounds or not isinstance(region.anchor.get(p), (int, float)):
                continue
            here = normalized_value(p, float(region.anchor[p]), bounds)
            target[p] = _denormalize(p, min(1.0, max(0.0, here + delta)), bounds)
        return surrogate._snap_discrete(target)

    def migrate(self, region: Region, registry: RegionRegistry,
                pressure: Dict[str, Any], at_run: int) -> Optional[Region]:
        """Close a region whose anchor is in the wrong place and open its
        successor where the search kept trying to go.

        The anchor is NOT moved. Anchor immutability is what region identity,
        merge detection and the don't-reopen-a-ruled-out-area check all rest
        on; moving it would let two regions silently collide and would rewrite
        history under the runs already attributed to it. A successor plus a
        pointer keeps the trail of a search walking downhill readable.
        """
        if not [r for r in registry.active() if r.region_id != region.region_id]:
            return None  # never leave the campaign with nowhere to search
        successor = registry.open_region(pressure["target"], at_run=at_run,
                                         origin=f"migrated_from_{region.region_id}")
        region.successor_id = successor.region_id
        region.set_flag(MIGRATED, at_run)
        registry.save()
        print(f"[Agent 4] Region {region.region_id} kept trying to leave "
              f"({pressure['n_escapes']} escapes, coherence {pressure['coherence']:.2f}, "
              f"{pressure['distance']:.4f} away) -- opened {successor.region_id} there")
        self._record_decision(at_run, MIGRATED, {
            "region_id": region.region_id, "successor_id": successor.region_id,
            "n_escapes": pressure["n_escapes"], "coherence": pressure["coherence"],
            "distance": pressure["distance"],
        }, region_id=region.region_id)
        return successor

    # -- maintenance: one call per wave -------------------------------------

    def maintain(self, registry: RegionRegistry, at_run: int) -> Dict[str, Any]:
        """Merge converged regions, then judge every live one.

        Merging runs FIRST. Each region's local search walks its own center
        downhill and nothing stops two of them walking into the same basin;
        judging before merging would then compare a region against what is
        effectively itself, and could retire one arm of a duplicate pair as
        "worse than the field" on a difference that is entirely noise.
        """
        merges = registry.merge_overlapping(self.merge_radius, at_run)
        for absorbed, survivor in merges:
            print(f"[Agent 4] Regions converged: {absorbed} merged into {survivor}")

        verdicts: Dict[str, str] = {}
        for region in list(registry.active()):
            verdicts[region.region_id] = self.judge(region, registry, at_run)

        # Migration runs AFTER judging, on regions that survived it. A region
        # already retired should not spawn a successor -- "there is nothing
        # here" and "the anchor is in the wrong place" are different findings,
        # and acting on both would open a region next to somewhere just ruled
        # out.
        migrations: List[Tuple[str, str]] = []
        for region in list(registry.active()):
            pressure = self.escape_pressure(region)
            if not pressure:
                continue
            successor = self.migrate(region, registry, pressure, at_run)
            if successor is not None:
                migrations.append((region.region_id, successor.region_id))

        registry.save()
        return {"merges": merges, "verdicts": verdicts, "migrations": migrations}

    def resume_best_paused(self, registry: RegionRegistry, at_run: int) -> Optional[Region]:
        """Bring back the most promising paused region.

        Called when the allocator has a GPU and nowhere better to put it --
        which is exactly the "if no better region exists, unflag and continue
        there" case. Pausing is only worth distinguishing from retirement if
        something can actually undo it.
        """
        # Capacity pauses first, and unconditionally: nothing was learned
        # about those regions, they simply lost a GPU, and they carry runs
        # already spent. Only when none are waiting does an exploitation pause
        # -- an actual judgement that a region stopped paying -- get revisited.
        for flags in ((CAPACITY_PAUSED,), (PAUSED,)):
            paused = [r for r in registry.regions
                      if r.flag in flags and r.merged_into is None
                      and r.elite_score() is not None]
            if paused:
                break
        if not paused:
            return None
        best = min(paused, key=lambda r: r.elite_score())
        best.set_flag(ACTIVE, at_run)
        registry.save()
        print(f"[Agent 4] Resumed paused region {best.region_id} "
              f"(elite {best.elite_score():.4f}) -- nowhere better to look")
        self._record_decision(at_run, "resumed", {
            "region_id": best.region_id, "elite_score": best.elite_score(),
            "n_runs": best.n_runs,
        }, region_id=best.region_id)
        return best

    # -- decision record ----------------------------------------------------

    def _record_decision(self, at_run: int, action: str, detail: Dict[str, Any],
                         region_id: Optional[str] = None) -> None:
        """A JSON log plus a prose summary for every lifecycle decision.

        Filenames carry the region id when there is one. Under the old window
        model a decision belonged to an iteration and `verdict_NNNN.json` was
        unique; now several regions can be judged at the same run number, and
        a bare iteration name would mean each verdict overwrote the last --
        leaving the record showing only whichever region happened to be
        judged last.
        """
        log = {
            "iteration": at_run,
            "agent": "agent4",
            "action": action,
            "path_taken": f"agent4_{action}",
            "region_id": region_id,
            "llm_mode": self.llm_mode,
            "detail": _jsonable(detail),
        }
        narrative = self._narrate(log)
        if narrative:
            log["narrative"] = narrative
        self.last_decision_log = log

        stem = f"verdict_{at_run:04d}" + (f"_{region_id}" if region_id else "")
        try:
            self.decisions_dir.mkdir(parents=True, exist_ok=True)
            (self.decisions_dir / f"{stem}.json").write_text(
                json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
            (self.decisions_dir / f"{stem}.md").write_text(
                self._render_decision_summary(log), encoding="utf-8")
        except OSError as e:
            print(f"[Agent 4] Could not write decision log: {e}")

    def _render_decision_summary(self, log: Dict[str, Any]) -> str:
        lines = [
            f"# Agent 4 decision — run {log['iteration']}",
            "",
            f"**Action:** {log['action']}",
        ]
        if log.get("region_id"):
            lines += ["", f"**Region:** {log['region_id']}"]
        lines += [
            "",
            "## Evidence",
            "```json",
            json.dumps(log.get("detail", {}), indent=2, sort_keys=True),
            "```",
        ]
        if log.get("narrative"):
            lines += ["", "## Reasoning", log["narrative"]]
        return "\n".join(lines) + "\n"

    def _load_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Agent 4's own past judgements, oldest first -- what it already
        opened, paused, or ruled out."""
        if not self.decisions_dir.exists():
            return []
        logs = []
        for path in sorted(self.decisions_dir.glob("verdict_*.json")):
            try:
                logs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, UnicodeDecodeError):
                continue
        logs.sort(key=lambda l: l.get("iteration", 0))
        return logs[-limit:]

    # -- LLM (llm_mode: statistics | hybrid | llm) --------------------------

    def _narrate(self, log: Dict[str, Any]) -> Optional[str]:
        """hybrid/llm: a short prose account of a decision already made.

        Never changes the decision -- by the time this runs the verdict is
        made, applied and logged. Measured on the archived campaign, the LLM's
        real contribution to this system is variance reduction (it avoids bad
        configurations; it did not find the best one, and its runs had less
        than half the spread of the surrogate's), so it is deliberately kept
        off the decision path and on the record-keeping one.
        """
        if self.llm_mode == "statistics":
            return None
        history = self._load_history(limit=5)
        prompt = f"""You are the region explorer in an automated hyperparameter
search that keeps several regions of the space under investigation at once. A
decision has already been made -- do not second-guess it, just explain it for
the run log.

Decision: {log['action']} (region {log.get('region_id')})
Evidence: {json.dumps(log.get('detail', {}), sort_keys=True)}

Your own previous decisions this campaign:
{json.dumps([{'iteration': h.get('iteration'), 'action': h.get('action'),
              'region_id': h.get('region_id')} for h in history])}

In 3-4 sentences: what was decided, on what evidence, and what a future
decision about this part of the space should keep in mind."""
        return claude_cli.call_with_budget(
            prompt, call_site="agent4_decision_summary",
            model=self._llm_model,
            campaign_budget_usd=self._llm_campaign_budget_usd,
            max_call_budget_usd=self._llm_max_call_budget_usd,
            usage_path=self._llm_usage_path,
            backend=self._llm_backend,
        )


def _jsonable(value: Any) -> Any:
    """Decision details are written to disk verbatim; numpy scalars and the
    like would make json.dumps raise mid-decision, so they're coerced here
    rather than at every producing call site."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
