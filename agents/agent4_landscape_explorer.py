"""Agent 4: the optimization landscape explorer.

Every other part of this system searches *locally*. agents/search_planner.py
::propose_next pins `center = dict(current_best_hyperparams)` on every call
and only varies the currently-active Gauss-Southwell block around it -- it
never measures coverage and never relocates that center. So once the campaign
settles into a basin, nothing in the system is structurally capable of leaving
it, however long it runs.

This agent is that missing capability, and nothing more. Periodically (every
`check_interval` iterations) it asks one question: is the frontier still
moving? If yes it returns control immediately without spending a single
iteration. If no, it takes a bounded budget of iterations, probes the
least-explored region of the landscape, and decides one of three things:

  - this region is dead              -> flag it "no_optimum", try another
  - the budget ran out inconclusively -> flag "exploitation_paused", go home
  - this region is decisively better -> COMMIT: relocate the whole search

The commit is the consequential one -- it moves the entire campaign's focus --
so it is deliberately hard to trigger: a minimum sample size, a top-quartile
comparison against the region being left, AND a whole-sample distributional
check, all three. See _commit_verdict for why the third exists.

Probing happens in fixed batches of `probe_wave_size`, which is pinned to the
abandon rule (all N probes bad -> abandon). That makes one probe wave exactly
one decision boundary, and it means the same state machine serves the parallel
multi-GPU path (one wave = N GPUs) and the sequential path (one wave = N
consecutive iterations) with no separate code path for either.
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
from state.landscape import (
    build_landscape,
    load_region_flags,
    project_point,
    region_members,
    save_region_flags,
)
from state.results_analysis import HYPERPARAM_COLUMNS, load_results, top_quartile_by_val_bpb
from state.surrogate import fit_surrogate

# Verdicts returned by evaluate_batch. Strings rather than an enum to match
# how every other status/flag vocabulary in this codebase is expressed
# (pipeline_validator severities, run statuses, region flags).
CONTINUE = "continue"
NEXT_REGION = "next_region"
COMMIT = "commit"
EXHAUSTED = "exhausted"

LLM_MODES = ("statistics", "hybrid", "llm")

# Historical runs a region needs before it can serve as the "how good is
# where we are now" reference. Below this, the neighbourhood is too sparse to
# characterise and _region_val_bpbs falls back to the whole campaign.
MIN_REGION_REFERENCE_N = 4


class Agent4LandscapeExplorer:
    """Watches the whole landscape and, when the search looks trapped,
    spends a bounded budget checking whether somewhere else is better."""

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
        self.check_interval = int(cfg.get("check_interval", 30))
        self.window_iterations = int(cfg.get("window_iterations", 9))
        self.probe_wave_size = int(cfg.get("probe_wave_size", 3))
        self.bad_tolerance = float(cfg.get("bad_tolerance", 0.05))
        # These two defaults are calibrated, not conventional -- see the
        # agents_config.yaml comments. commit_margin is sized in units of the
        # measured noise floor (sigma = 0.003187 = 0.24% of a typical
        # val_bpb), so 0.006 is ~2.3 sigma; an earlier 0.03 was ~11.6 sigma,
        # a bar no run in 514 had ever cleared, which made relocation
        # impossible. region_radius 0.15 swept 63% of the whole campaign into
        # a single "region", collapsing the new-vs-origin comparison; 0.05
        # captures a real ~15% neighbourhood. Keep the fallbacks equal to the
        # shipped config so a caller without an agent4: block doesn't quietly
        # get the broken pair back.
        self.commit_margin = float(cfg.get("commit_margin", 0.006))
        self.min_runs_before_commit = int(cfg.get("min_runs_before_commit", 6))
        self.heavy_exploitation_n = int(cfg.get("heavy_exploitation_n", 20))
        self.region_radius = float(cfg.get("region_radius", 0.05))
        self.stagnation_lookback = int(cfg.get("stagnation_lookback", 10))
        self.stagnation_min_improvement = float(cfg.get("stagnation_min_improvement", 0.005))
        self.grid_resolution = int(cfg.get("grid_resolution", 24))

        llm_mode = str(cfg.get("llm_mode", "hybrid"))
        if llm_mode not in LLM_MODES:
            print(f"[Agent 4] Unknown llm_mode {llm_mode!r} -- falling back to 'statistics'")
            llm_mode = "statistics"
        self.llm_mode = llm_mode

        _root = Path(root_dir) if root_dir else Path(".")
        _state = Path(state_dir) if state_dir else Path("state")
        _reports = Path(reports_dir) if reports_dir else Path("reports")
        self.results_path = _root / "results.tsv"
        self.region_flags_path = _state / "agent4_region_flags.json"
        self.decisions_dir = _reports / "agent4_decisions"

        llm_config = self.config.get("llm", {})
        self._llm_backend = llm_config.get("backend", "cli")
        self._llm_model = llm_config.get("model", "sonnet")
        self._llm_campaign_budget_usd = float(llm_config.get("campaign_budget_usd", 5.0))
        self._llm_max_call_budget_usd = float(llm_config.get("max_call_budget_usd", 0.20))
        self._llm_usage_path = llm_config.get("usage_log_path", str(_state / "llm_usage.json"))

        # --- window state (all reset by close_window) ---
        self.active = False
        self.committed_hyperparams: Optional[Dict[str, Any]] = None
        self.last_action: Optional[str] = None
        self.last_decision_log: Optional[Dict[str, Any]] = None
        self._reset_window_state()

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

    def _reset_window_state(self) -> None:
        self._origin_hyperparams: Dict[str, Any] = {}
        self._origin_runs: List[float] = []
        self._candidate_hyperparams: Optional[Dict[str, Any]] = None
        self._candidate_runs: List[float] = []
        self._batch_results: List[float] = []
        self._tried_centers: List[Dict[str, Any]] = []
        self._iterations_used = 0
        self._landscape: Optional[Dict[str, Any]] = None
        self._surrogate = None
        self._rows: List[Dict[str, Any]] = []
        self._best_val_bpb: Optional[float] = None
        self._opened_at_iteration = 0

    @property
    def budget_left(self) -> int:
        return max(0, self.window_iterations - self._iterations_used)

    # -- phase 1: should we intervene at all? -------------------------------

    def consider_intervention(
        self,
        iteration: int,
        center_hyperparams: Dict[str, Any],
        best_val_bpb: Optional[float] = None,
        rows: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """True if a window was opened (Agent 4 now decides hyperparameters),
        False if the search looks healthy and control stays with Agent 1.

        A False costs zero iterations by design -- "if it decides we're
        heading the right way it stops there and doesn't use its batch."
        """
        if not self.enabled or self.active:
            return False
        rows = load_results(str(self.results_path)) if rows is None else rows

        finite = [r["val_bpb"] for r in rows
                  if isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])]
        if len(finite) < surrogate.MIN_SURROGATE_N:
            return False

        stagnation = self._stagnation_evidence(finite)
        if not stagnation["stagnant"]:
            self.last_action = None
            print(f"[Agent 4] Frontier still improving "
                  f"({stagnation['relative_improvement']:.2%} over the last "
                  f"{self.stagnation_lookback} runs) -- no intervention, 0 iterations used")
            return False

        sm = fit_surrogate(rows)
        if sm is None:
            return False
        landscape = build_landscape(rows, sm, grid_resolution=self.grid_resolution,
                                    hard_bounds=SEARCH_SPACE)
        if landscape is None:
            return False

        f_best = best_val_bpb if best_val_bpb is not None else min(finite)
        candidate = self._propose_region(landscape, sm, f_best, iteration, exclude=[])
        if candidate is None:
            return False

        self._reset_window_state()
        self.active = True
        self.committed_hyperparams = None
        self._opened_at_iteration = iteration
        self._rows, self._surrogate, self._landscape = rows, sm, landscape
        self._best_val_bpb = f_best
        self._origin_hyperparams = dict(center_hyperparams)
        self._origin_runs = self._region_val_bpbs(center_hyperparams)
        self._candidate_hyperparams = candidate
        self._tried_centers = [candidate]

        # "investigating" is a transient state that only means anything while
        # a window is open. A campaign that ends (or crashes) mid-window
        # leaves one behind, and since window state is in-memory only, nothing
        # would ever resolve it -- the chart would show a region under active
        # investigation forever. Clear stale ones before setting this one.
        self._clear_flag("investigating")
        self._flag_region(center_hyperparams, "currently_exploiting", iteration,
                          n_runs=len(self._origin_runs))
        self._flag_region(candidate, "investigating", iteration, n_runs=0)

        print(f"[Agent 4] Frontier flat over the last {self.stagnation_lookback} runs "
              f"({stagnation['relative_improvement']:.2%} improvement) -- taking control for up to "
              f"{self.window_iterations} iteration(s) to probe an under-explored region")
        self.last_action = "engaged"
        self._record_decision(iteration, "engaged", {
            "stagnation": stagnation,
            "origin_region_runs": len(self._origin_runs),
            "candidate_hyperparams": candidate,
            "landscape_variance_explained": sum(landscape["explained_variance_ratio"]),
        }, before=center_hyperparams, after=candidate)
        return True

    def _stagnation_evidence(self, finite_val_bpbs: List[float]) -> Dict[str, Any]:
        """Has the best-so-far actually moved recently? Compared as a relative
        improvement so the threshold means the same thing at val_bpb 1.8 as at
        1.2 -- an absolute epsilon would silently get stricter as the campaign
        improves.
        """
        lookback = min(self.stagnation_lookback, len(finite_val_bpbs) - 1)
        if lookback < 1:
            return {"stagnant": False, "relative_improvement": 0.0, "reason": "not enough history"}
        before = min(finite_val_bpbs[:-lookback])
        recent_best = min(finite_val_bpbs)
        improvement = (before - recent_best) / before if before > 0 else 0.0
        return {
            "stagnant": improvement < self.stagnation_min_improvement,
            "relative_improvement": improvement,
            "best_before_lookback": before,
            "best_now": recent_best,
            "lookback": lookback,
        }

    # -- phase 2: probing ---------------------------------------------------

    def propose_probe(self, iteration: int, slot: int = 0) -> Dict[str, Any]:
        """One more training configuration inside the region being probed.

        Every probe in a wave is an independent EI draw anchored at the same
        region center (different seed per slot), so a wave samples the region
        broadly rather than refining one point -- which is what makes "all N
        came back bad" real evidence about the region rather than about one
        unlucky corner of it.
        """
        self._iterations_used += 1
        anchor = dict(self._candidate_hyperparams or self._origin_hyperparams)
        full = dict(self._origin_hyperparams)  # keep pass-through keys (ablation_k, ...)
        if self._surrogate is None:
            full.update(anchor)
        else:
            full.update(surrogate.propose_via_ei(
                self._surrogate,
                f_best=self._best_val_bpb if self._best_val_bpb is not None else 0.0,
                bounds=self._surrogate.bounds,
                free_params=list(HYPERPARAM_COLUMNS),
                fixed_values=anchor,
                n_candidates=2000,
                seed=iteration * 100 + slot,
            ))
        # Every iteration Agent 4 owns produces a decision log, same as every
        # iteration Agent 1 owns -- so the orchestrator's existing
        # validate_agent1_decision call always has this iteration's real
        # decision to check rather than a stale one from the window's start.
        # narrate=False: a probe is a routine draw, not a judgement call, and
        # narrating each one would cost an LLM call per iteration in hybrid
        # mode. A verdict written later for the same iteration supersedes
        # this file, which is the more informative record of the two.
        self._record_decision(iteration, "probe", {
            "slot": slot, "region_runs_so_far": len(self._candidate_runs),
            "iterations_used": self._iterations_used, "budget_left": self.budget_left,
        }, before=self._origin_hyperparams, after=full, narrate=False)
        return full

    def record_result(self, val_bpb: Optional[float]) -> None:
        """Feed one completed probe's measured val_bpb back in. A non-finite
        result (crashed/OOM run) is recorded as neither good nor bad -- it is
        an absence of evidence about the region, not evidence against it."""
        if val_bpb is None or not math.isfinite(val_bpb):
            return
        self._batch_results.append(float(val_bpb))
        self._candidate_runs.append(float(val_bpb))

    def evaluate_batch(self, iteration: int) -> str:
        """Called at every probe-wave boundary. Returns CONTINUE, NEXT_REGION,
        COMMIT or EXHAUSTED; the caller acts on the verdict and, on the last
        two, the window is already closed."""
        if not self.active:
            return EXHAUSTED
        if len(self._batch_results) < self.probe_wave_size and self.budget_left > 0:
            return CONTINUE

        batch, self._batch_results = self._batch_results, []

        # Commit is tested first, before the abandon streak: a region that has
        # decisively cleared the commit bar should not be thrown away because
        # this particular wave also happened to read as "bad" against the
        # origin's elite reference. Passing the bar is the stronger signal.
        verdict, stats = self._commit_verdict()
        if verdict:
            return self._commit(iteration, stats)

        elite_ref = self._elite_reference()
        # No reference means no evidence -- a probe cannot be called "bad"
        # against a bar that doesn't exist. Condemning a region on a missing
        # comparison would be the same class of mistake as fabricating a
        # statistic from missing data, which this codebase refuses to do
        # everywhere else.
        bad = ([] if elite_ref is None
               else [v for v in batch if v > elite_ref * (1 + self.bad_tolerance)])
        if batch and bad and len(bad) == len(batch):
            return self._abandon_region(iteration, batch, elite_ref, stats)

        if self.budget_left <= 0:
            return self._exhaust(iteration, stats)
        return CONTINUE

    # -- the three outcomes -------------------------------------------------

    def _commit_verdict(self) -> Tuple[bool, Dict[str, Any]]:
        """Three clauses, all required, for the one decision that moves the
        whole campaign:

          1. enough runs in the candidate region to say anything at all;
          2. its top-quartile median beats the origin's by `commit_margin`;
          3. its WHOLE sample's median also beats the origin's.

        Clause 3 exists because top_quartile_by_val_bpb returns
        max(1, int(n * 0.25)) entries -- at the sample sizes a probe window
        can produce (6-9 runs), "top quartile" collapses to the single best
        run. Clauses 1-2 alone would therefore let one lucky draw permanently
        relocate the search. Requiring the distribution to move too is what
        makes this a decision about the region rather than about its best
        outlier.
        """
        stats: Dict[str, Any] = {
            "candidate_n": len(self._candidate_runs),
            "origin_n": len(self._origin_runs),
            "min_runs_before_commit": self.min_runs_before_commit,
            "commit_margin": self.commit_margin,
        }
        if len(self._candidate_runs) < self.min_runs_before_commit or not self._origin_runs:
            return False, stats

        new_q = statistics.median(v for v, _ in top_quartile_by_val_bpb(
            [(v, None) for v in self._candidate_runs]))
        cur_q = statistics.median(v for v, _ in top_quartile_by_val_bpb(
            [(v, None) for v in self._origin_runs]))
        new_median = statistics.median(self._candidate_runs)
        cur_median = statistics.median(self._origin_runs)
        stats.update({
            "candidate_top_quartile_median": new_q, "origin_top_quartile_median": cur_q,
            "candidate_median": new_median, "origin_median": cur_median,
            "required_top_quartile": cur_q * (1 - self.commit_margin),
        })

        quartile_clears = new_q < cur_q * (1 - self.commit_margin)
        distribution_clears = new_median < cur_median
        stats["quartile_clears"] = quartile_clears
        stats["distribution_clears"] = distribution_clears

        if not (quartile_clears and distribution_clears):
            return False, stats
        if self.llm_mode == "llm":
            approved = self._llm_commit_approval(stats)
            stats["llm_approved"] = approved
            if approved is False:
                return False, stats
        return True, stats

    def _commit(self, iteration: int, stats: Dict[str, Any]) -> str:
        """Relocate the campaign. The region being left is flagged
        'local_optimum' only if it was genuinely exploited hard first --
        otherwise it's 'exploitation_paused', because "we didn't finish
        looking here" and "we proved this is a local optimum" are different
        claims and only one of them is supported by a short window."""
        committed = dict(self._candidate_hyperparams or {})
        self.committed_hyperparams = committed
        heavily_exploited = len(self._origin_runs) >= self.heavy_exploitation_n
        origin_flag = "local_optimum" if heavily_exploited else "exploitation_paused"

        self._flag_region(self._origin_hyperparams, origin_flag, iteration, n_runs=len(self._origin_runs))
        self._flag_region(committed, "currently_exploiting", iteration, n_runs=len(self._candidate_runs))

        print(f"[Agent 4] COMMIT -- relocating the search. Candidate region top-quartile "
              f"{stats.get('candidate_top_quartile_median', float('nan')):.6f} vs origin "
              f"{stats.get('origin_top_quartile_median', float('nan')):.6f} "
              f"(needed <= {stats.get('required_top_quartile', float('nan')):.6f}); "
              f"previous region flagged '{origin_flag}' after {len(self._origin_runs)} run(s)")

        self.last_action = "committed"
        self._record_decision(iteration, "committed", {
            **stats, "origin_flag": origin_flag, "iterations_used": self._iterations_used,
        }, before=self._origin_hyperparams, after=committed)
        self.close_window()
        return COMMIT

    def _abandon_region(self, iteration: int, batch: List[float],
                        elite_ref: Optional[float], stats: Dict[str, Any]) -> str:
        """Every probe in this wave came back worse than the origin's elite
        reference -- flag the region and move on rather than spending the
        rest of the window here."""
        abandoned = dict(self._candidate_hyperparams or {})
        self._flag_region(abandoned, "no_optimum", iteration, n_runs=len(self._candidate_runs))
        detail = {**stats, "batch": batch, "elite_reference": elite_ref,
                  "bad_tolerance": self.bad_tolerance}

        if self.budget_left >= self.probe_wave_size:
            nxt = self._propose_region(self._landscape, self._surrogate,
                                       self._best_val_bpb or 0.0, iteration,
                                       exclude=self._tried_centers)
            if nxt is not None:
                self._candidate_hyperparams = nxt
                self._candidate_runs = []
                self._tried_centers.append(nxt)
                self._flag_region(nxt, "investigating", iteration, n_runs=0)
                ref = f"{elite_ref:.6f}" if elite_ref is not None else "n/a"
                print(f"[Agent 4] All {len(batch)} probe(s) worse than the elite reference "
                      f"({ref}) -- region flagged 'no_optimum', trying another "
                      f"({self.budget_left} iteration(s) left)")
                self.last_action = "next_region"
                self._record_decision(iteration, "next_region", detail,
                                      before=abandoned, after=nxt)
                return NEXT_REGION
        return self._exhaust(iteration, detail)

    def _exhaust(self, iteration: int, stats: Dict[str, Any]) -> str:
        """Budget spent without a decisive result. The candidate region is
        paused, not condemned: an inconclusive probe is not evidence of
        absence."""
        if self._candidate_hyperparams:
            self._flag_region(self._candidate_hyperparams, "exploitation_paused",
                              iteration, n_runs=len(self._candidate_runs))
        print(f"[Agent 4] Window closed after {self._iterations_used} iteration(s) with no "
              f"commit -- returning control to Agent 1, search center unchanged")
        self.last_action = "abandoned"
        self._record_decision(iteration, "abandoned",
                              {**stats, "iterations_used": self._iterations_used},
                              before=self._origin_hyperparams,
                              after=self._origin_hyperparams)
        self.close_window()
        return EXHAUSTED

    def close_window(self) -> None:
        self.active = False
        self._reset_window_state()

    # -- region helpers -----------------------------------------------------

    def _elite_reference(self) -> Optional[float]:
        """The bar a probe has to clear to not count as "bad": the median of
        the top quartile of the region being left. Reuses
        top_quartile_by_val_bpb -- the same definition of "elite" Agent 2's
        stuck signal and Agent 3's recommendations already share, so all
        three agents mean the same thing by "a good run"."""
        if not self._origin_runs:
            return None
        return statistics.median(v for v, _ in top_quartile_by_val_bpb(
            [(v, None) for v in self._origin_runs]))

    def _region_val_bpbs(self, center: Dict[str, Any]) -> List[float]:
        """val_bpb of the historical runs near `center`.

        Falls back to the whole campaign when the neighbourhood is too sparse
        to characterise. Both halves of that matter: every comparison Agent 4
        makes is against "how good is where we are now", and a center that
        happens to sit in a thinly-sampled part of the space would otherwise
        yield an empty reference -- which in turn would make every probe
        register as bad against a bar that doesn't exist, condemning regions
        on no evidence. A campaign-wide elite is a weaker reference than a
        local one, but it is a real measurement rather than an absent one,
        and the substitution is printed rather than silent.
        """
        members = region_members(self._rows, center, self._landscape, self.region_radius)
        vals = [float(r["val_bpb"]) for r in members
                if isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])]
        if len(vals) >= MIN_REGION_REFERENCE_N:
            return vals
        allv = [float(r["val_bpb"]) for r in self._rows
                if isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])]
        print(f"[Agent 4] Only {len(vals)} historical run(s) within region_radius="
              f"{self.region_radius} of the current center -- too sparse for a local "
              f"reference; comparing against all {len(allv)} runs instead")
        return allv

    def _propose_region(
        self,
        landscape: Optional[Dict[str, Any]],
        sm: Any,
        f_best: float,
        iteration: int,
        exclude: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Pick the least-confident (highest predicted-std) grid cell that
        isn't already too close to a region we've tried, then run a full EI
        search anchored there.

        Highest surrogate uncertainty is the honest, already-available
        "nobody has looked here" signal -- no new heuristic invented for it.
        The EI search is state/surrogate.py::propose_via_ei used verbatim,
        but with a *different center* and *every* parameter free instead of
        one Gauss-Southwell block: that combination is precisely the coverage
        search_planner.propose_next structurally cannot produce.
        """
        if landscape is None or sm is None:
            return None
        stds = landscape["grid_z_std"]
        cells = landscape["grid_hyperparams"]
        excluded_points = [p for p in (project_point(c, landscape) for c in exclude) if p is not None]
        grid_x, grid_y = landscape["grid_x"], landscape["grid_y"]
        extent = math.hypot(grid_x[-1] - grid_x[0], grid_y[-1] - grid_y[0])
        cutoff = self.region_radius * extent

        ranked = sorted(
            ((stds[r][c], r, c) for r in range(len(stds)) for c in range(len(stds[r]))),
            key=lambda t: -t[0],
        )
        for _std, r, c in ranked:
            point = (grid_x[c], grid_y[r])
            if any(math.hypot(point[0] - ex, point[1] - ey) <= cutoff
                   for ex, ey in excluded_points):
                continue
            anchor = cells[r][c]
            return surrogate.propose_via_ei(
                sm, f_best=f_best, bounds=sm.bounds,
                free_params=list(HYPERPARAM_COLUMNS), fixed_values=anchor,
                n_candidates=2000, seed=iteration,
            )
        return None

    def _flag_region(self, hyperparams: Dict[str, Any], flag: str,
                     iteration: int, n_runs: int) -> None:
        """Persist a region flag for the landscape chart. Matching is by raw
        hyperparameters, never by PCA coordinates -- build_landscape refits
        its basis every call as the campaign grows, so a stored (x, y) goes
        stale while the configuration that defines a region does not."""
        if not hyperparams:
            return
        regions = load_region_flags(self.region_flags_path)
        key = self._region_key(hyperparams)
        regions = [r for r in regions if self._region_key(r.get("hyperparams", {})) != key]
        regions.append({
            "hyperparams": {k: hyperparams[k] for k in HYPERPARAM_COLUMNS if k in hyperparams},
            "flag": flag, "since_iteration": iteration, "n_runs": n_runs,
        })
        save_region_flags(self.region_flags_path, regions)

    def _clear_flag(self, flag: str) -> None:
        """Drop every region currently carrying `flag`."""
        regions = load_region_flags(self.region_flags_path)
        kept = [r for r in regions if r.get("flag") != flag]
        if len(kept) != len(regions):
            save_region_flags(self.region_flags_path, kept)

    @staticmethod
    def _region_key(hyperparams: Dict[str, Any]) -> Tuple:
        return tuple(round(float(hyperparams[k]), 6) if k in hyperparams else None
                     for k in HYPERPARAM_COLUMNS)

    # -- decision record ----------------------------------------------------

    def _record_decision(self, iteration: int, action: str, detail: Dict[str, Any],
                         before: Dict[str, Any], after: Dict[str, Any],
                         narrate: bool = True) -> None:
        """A JSON log plus a prose summary, written after every decision.

        The `params` block mirrors Agent 1's before/after/changed/reason shape
        on purpose: agents/pipeline_validator.py::validate_agent1_decision
        operates generically on that shape, so the orchestrator can validate
        an Agent 4 iteration with the exact same call it already makes for an
        Agent 1 one -- no parallel validator to keep in sync.

        The prose half is what the *next* Agent 4 window reads back
        (_load_history) for context on what has already been tried and ruled
        out.
        """
        params_log = {}
        for key in SEARCH_SPACE:
            old, new = before.get(key), after.get(key)
            params_log[key] = {
                "before": old, "after": new, "changed": old != new,
                "reason": f"agent4:{action}",
            }
        log = {
            "iteration": iteration, "agent": "agent4", "action": action,
            "path_taken": f"agent4_{action}", "params": params_log,
            "llm_mode": self.llm_mode, "detail": _jsonable(detail),
        }
        narrative = self._narrate(log) if narrate else None
        if narrative:
            log["narrative"] = narrative
        self.last_decision_log = log

        # Two filename spaces on purpose. Probes are per-iteration and must
        # be named decision_NNNN.json, because pipeline_validator looks up
        # that exact name when walking back through recent decisions.
        # Judgements (engage/abandon/commit/next_region) happen on the same
        # iteration number as a probe -- opening a window and taking its
        # first probe are both iteration N -- so writing them to the same
        # file means every judgement is immediately overwritten by the probe
        # beside it, and _load_history sees nothing but probes. They get
        # verdict_NNNN.json, which is also what _load_history reads: what a
        # later window needs is the record of what was decided and why, not
        # a list of individual draws.
        stem = "decision" if action == "probe" else "verdict"
        try:
            self.decisions_dir.mkdir(parents=True, exist_ok=True)
            (self.decisions_dir / f"{stem}_{iteration:04d}.json").write_text(
                json.dumps(log, indent=2, sort_keys=True), encoding="utf-8")
            (self.decisions_dir / f"{stem}_{iteration:04d}.md").write_text(
                self._render_decision_summary(log), encoding="utf-8")
        except OSError as e:
            print(f"[Agent 4] Could not write decision log: {e}")

    def _render_decision_summary(self, log: Dict[str, Any]) -> str:
        lines = [
            f"# Agent 4 decision — iteration {log['iteration']}",
            "",
            f"**Action:** {log['action']}",
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
        explored, committed to, or ruled out. Read back at the start of each
        window so a later window has the context an earlier one produced.

        Reads verdict_*.json only: routine probes are draws, not decisions,
        and a history diluted with dozens of them tells a later window
        nothing it can act on."""
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
        """hybrid/llm: a short prose account of this decision for the record
        and for the next window to read. Never changes the decision -- by the
        time this runs, the verdict is already made and logged."""
        if self.llm_mode == "statistics":
            return None
        history = self._load_history(limit=5)
        prompt = f"""You are the optimization-landscape explorer in an automated
hyperparameter search. A decision has already been made -- do not second-guess
it, just explain it for the run log.

Decision: {log['action']}
Evidence: {json.dumps(log.get('detail', {}), sort_keys=True)}

Your own previous decisions this campaign:
{json.dumps([{'iteration': h.get('iteration'), 'action': h.get('action')} for h in history])}

In 3-4 sentences: what was decided, on what evidence, and what a future
exploration window should keep in mind about this region."""
        return claude_cli.call_with_budget(
            prompt, call_site="agent4_decision_summary",
            model=self._llm_model,
            campaign_budget_usd=self._llm_campaign_budget_usd,
            max_call_budget_usd=self._llm_max_call_budget_usd,
            usage_path=self._llm_usage_path,
            backend=self._llm_backend,
        )

    def _llm_commit_approval(self, stats: Dict[str, Any]) -> Optional[bool]:
        """llm_mode == "llm" only: the statistical bar has already been
        cleared; this is a second opinion that can veto, never one that can
        approve on its own. Returns None when the call fails or the budget is
        gone, and the caller treats that as "no veto" -- a budget-exhausted
        campaign falls back to the deterministic verdict rather than silently
        changing behavior.
        """
        history = self._load_history(limit=5)
        prompt = f"""An automated hyperparameter search is deciding whether to
permanently relocate its entire search focus to a newly-probed region. This is
expensive to undo. The statistical bar has already been cleared:

{json.dumps(stats, indent=2, sort_keys=True, default=str)}

Previous exploration decisions this campaign:
{json.dumps([{'iteration': h.get('iteration'), 'action': h.get('action')} for h in history])}

Answer with exactly one word, VETO or PROCEED. VETO only if the evidence looks
like a small-sample artifact rather than a genuinely better region."""
        answer = claude_cli.call_with_budget(
            prompt, call_site="agent4_commit_approval",
            model=self._llm_model,
            campaign_budget_usd=self._llm_campaign_budget_usd,
            max_call_budget_usd=self._llm_max_call_budget_usd,
            usage_path=self._llm_usage_path,
            backend=self._llm_backend,
        )
        if not answer:
            return None
        return False if "VETO" in answer.upper() else True


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
