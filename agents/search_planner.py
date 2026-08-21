"""Persisted search loop that Agent 1 calls to get its next hyperparameter
proposal from the Tier 1 surrogate (see dev/INNOVATION_PLAN.md, state/surrogate.py).

Cold start (Sobol) until enough data exists to fit a surrogate. Then, every
call: prune parameters below the noise floor (frozen, not just
low-priority -- their total effect is unmeasurable at this budget), detect
interactions among the surviving parameters (cheap fANOVA) and group
interacting ones into blocks, rank blocks by summed sensitivity (S_perf),
and run Expected Improvement over whichever block is "active" this cycle
(others pinned at the current best). This is re-ranked from scratch on
every call rather than cached (see the plan's rationale: the cheap
interaction fit is milliseconds at this data scale, so recomputing is
strictly truer Gauss-Southwell and not meaningfully more expensive) --
persisted state only tracks which block/budget we're mid-cycle on.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from state import surrogate
from state.results_analysis import (
    HYPERPARAM_COLUMNS, TUNABLE_COLUMNS, report_at_budget,
)

STATE_PATH_DEFAULT = "state/search_planner_state.json"
NOISE_FLOOR_PATH_DEFAULT = "state/noise_floor.json"
#: Looked for beside whatever noise_floor.json a caller passes, so redirecting
#: a state directory redirects both. See _load_sigma.
SEED_VARIANCE_FILENAME = "seed_variance.json"
#: scripts/noise_audit.py -- every repeat ever measured at this budget,
#: pooled. Preferred over the single-experiment files below because those
#: are medians of two or three configurations of a quantity that varies
#: 17x, which is why successive estimates of it disagreed by 3x.
NOISE_AUDIT_FILENAME = "noise_audit.json"
#: scripts/region_geometry.py's in-region measurement. Preferred over the
#: seed-variance one because it is the RIGHT quantity for a within-region
#: decision: measured with the architecture frozen, across configurations that
#: all sit inside one fence. See _load_sigma.
REGION_GEOMETRY_FILENAME = "region_geometry.json"
REPORT_DIR_DEFAULT = "reports/agent1_search_plan"
DEFAULT_SIGMA = 0.01  # conservative fallback if noise_floor.json is missing


@dataclass
class SearchPlannerState:
    cold_start_points: List[Dict[str, Any]] = field(default_factory=list)
    cold_start_used: int = 0
    frozen: Dict[str, int] = field(default_factory=dict)  # param -> iteration frozen since
    active_block: List[str] = field(default_factory=list)
    budget_used_in_block: int = 0

    @classmethod
    def load(cls, path: str = STATE_PATH_DEFAULT) -> "SearchPlannerState":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls(**json.loads(p.read_text()))
        except Exception:
            # Tolerant of a missing/corrupt file -- a fresh state is always
            # safe (worst case: cold start restarts / blocks re-rank once).
            return cls()

    def save(self, path: str = STATE_PATH_DEFAULT) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _seed_sigma(seed_variance_path: str) -> Optional[float]:
    """The measured seed-to-seed spread, from scripts/seed_variance.py.

    The MEDIAN of the per-configuration spreads, not the pooled figure the
    report also prints. Spread is strongly config-dependent -- measured
    0.00154 / 0.00197 / 0.00887 across three configurations, a ~6x range that
    tracks step count -- so the pooled number is dragged to 0.00532 by the
    single noisiest config and describes nowhere in particular. The median is
    robust to that one outlier and lands near the frontier, which is where
    every decision this sigma feeds actually gets made.

    Returns None if the file is absent or unusable, so the caller falls back to
    the noise floor rather than to a guess.
    """
    p = Path(seed_variance_path)
    if not p.exists():
        return None
    try:
        report = json.loads(p.read_text())
        stds = sorted(
            entry["std"] for entry in report.get("per_config", {}).values()
            if isinstance(entry.get("std"), (int, float)) and entry["std"] > 0
        )
    except (json.JSONDecodeError, OSError, AttributeError, TypeError):
        return None
    if not stds:
        return None
    mid = len(stds) // 2
    return stds[mid] if len(stds) % 2 else (stds[mid - 1] + stds[mid]) / 2


def measured_resolvable_gap(state_dir: str) -> Optional[float]:
    """The smallest val_bpb difference a SINGLE-seed comparison can actually
    support, from scripts/seed_variance.py. 0.0093 at TOKEN_BUDGET=4.19M.

    Below this, "A is better than B" is a coin flip however many DIFFERENT
    configurations are run: extra runs buy coverage, not precision, and only
    repeating the SAME configuration shrinks the wobble (and slowly -- 4x the
    runs to halve it). Measured gaps between the campaign's top regions were
    0.0027 and 0.0065 against this 0.0093, so the ranking printed between them
    was partly noise.

    None when nothing has been measured at the budget in force -- a rule that
    stops work must never fire on a guessed or stale constant.
    """
    from prepare import TOKEN_BUDGET

    # Pooled audit first, for the same reason as measured_a_within: the
    # seed_variance figure is a median over three configurations, and this
    # quantity varies 17x between them.
    audit = report_at_budget(Path(state_dir) / NOISE_AUDIT_FILENAME, TOKEN_BUDGET)
    if audit:
        gap = (audit.get("resolvable_gap_at_k") or {}).get("1")
        if isinstance(gap, (int, float)) and gap > 0:
            return float(gap)

    report = report_at_budget(Path(state_dir) / SEED_VARIANCE_FILENAME, TOKEN_BUDGET)
    if not report:
        return None
    gap = (report.get("resolvable_gap_at_k_seeds") or {}).get("1")
    return float(gap) if isinstance(gap, (int, float)) and gap > 0 else None


def measured_a_within(state_dir: str) -> Optional[float]:
    """The in-region noise, ONLY if it has actually been measured.

    Returns None rather than a fallback, and that distinction is load-bearing.
    `_load_sigma` degrades to DEFAULT_SIGMA = 0.01 when nothing has been
    measured, which is ~7.5x the real value (0.001342) -- and any rule that
    ABANDONS a region on the strength of that number would retire essentially
    every region on a fresh checkout, silently. A decision that throws work away
    has to be backed by a measurement or not made at all.

    Deliberately reads only region_geometry.json. seed_variance.json measures
    across DIFFERENT architectures and noise_floor.json holds the seed fixed;
    neither is the within-region quantity a saturation test needs.

    A MEASUREMENT FROM ANOTHER TOKEN BUDGET IS NOT A MEASUREMENT HERE, and is
    refused for the same reason as an absent one. The in-region noise is a
    property of how much training a run gets: sigma_seed went 0.00197 ->
    0.003215 when the budget went 12.5M -> 4.19M. A stale a_within understates
    it, so regions would look further from saturation than they are and keep
    being searched after they had stopped paying.
    """
    from prepare import TOKEN_BUDGET

    # The pooled audit first: 13 configurations and 53 runs, against
    # region_geometry's ONE configuration measured 3 times. Its sd_median is
    # what a TYPICAL configuration's repeat spread is, which is the quantity a
    # saturation test wants as its global fallback. (sd_pooled, 0.004886, is
    # larger because two noisy configurations dominate a mean of variances --
    # using it as the fallback would make a typical region look saturated on
    # arrival, so the median is the right one of the two here.)
    audit = report_at_budget(Path(state_dir) / NOISE_AUDIT_FILENAME, TOKEN_BUDGET)
    if audit:
        value = audit.get("sd_median")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)

    data = report_at_budget(Path(state_dir) / REGION_GEOMETRY_FILENAME, TOKEN_BUDGET)
    if data is None:
        return None
    value = data.get("a_within")
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _load_sigma(noise_floor_path: str,
                seed_variance_path: Optional[str] = None) -> float:
    """The measurement noise every "is this difference real" test is sized
    against.

    Three sources, most specific first, all looked for BESIDE noise_floor_path
    so redirecting a state directory redirects all of them (a fixed repo-root
    default would let the real repo's measurement leak into an isolated run):

      1. region_geometry.json's `a_within` -- one configuration repeated across
         seeds with the ARCHITECTURE FROZEN, i.e. exactly the noise a
         within-region decision faces. Measured 0.001342.
      2. seed_variance.json -- the same idea but measured across three
         DIFFERENT architectures, so it is a campaign-wide figure (0.00197) and
         too broad for a within-region question.
      3. noise_floor.json -- seed, data and budget all fixed, so it captures
         only bf16/kernel nondeterminism (0.000797) and no statistical
         variation whatsoever.

    Prefers the SEED-inclusive spread when it has been measured. This ordering
    is the whole correction: state/noise_floor.json repeats one configuration
    with the seed, the data and the token budget all held fixed, so it captures
    bf16/kernel nondeterminism and no statistical variation whatsoever. Using
    it to decide whether a hyperparameter's effect is real understates the
    noise by ~2.5x at the frontier (measured 0.000797 vs 0.00197), which means
    parameters whose true effect is smaller than the run-to-run bounce were
    being kept and tuned.
    """
    from prepare import TOKEN_BUDGET

    state_dir = Path(noise_floor_path).parent

    # EVERY LEVEL IS ALSO GATED ON THE TOKEN BUDGET. Noise is a property of how
    # much training a run gets -- it grew 1.6x when the budget was cut 3x -- so
    # a measurement from another budget is not a less-specific answer, it is
    # the wrong one, and wrong in the direction that costs runs: understating
    # the noise keeps sub-noise parameters in the search and calls unresolvable
    # differences real. A report with no stamp is treated as stale, because
    # every report on disk when the stamp was added was measured at 12.5M.
    # 0. noise_audit.json, ahead of all three: it pools every repeat measured
    #    at this budget (13 configurations, 53 runs) instead of one experiment's
    #    two or three. The spread varies 17x between configurations, so a
    #    three-config median is a small sample of a wide distribution -- which
    #    is exactly how successive estimates of this number came to disagree by
    #    3x, each disagreement moving a threshold.
    audit_sigma = measured_a_within(str(state_dir))
    if audit_sigma is not None:
        return audit_sigma

    geometry = report_at_budget(state_dir / REGION_GEOMETRY_FILENAME, TOKEN_BUDGET)
    if geometry:
        a_within = geometry.get("a_within")
        if isinstance(a_within, (int, float)) and a_within > 0:
            return float(a_within)

    if seed_variance_path is None:
        seed_variance_path = str(state_dir / SEED_VARIANCE_FILENAME)
    if report_at_budget(seed_variance_path, TOKEN_BUDGET) is not None:
        seed_sigma = _seed_sigma(seed_variance_path)
        if seed_sigma is not None:
            return seed_sigma

    p = Path(noise_floor_path)
    if not p.exists():
        print(f"[search_planner] WARNING: neither {seed_variance_path} nor "
              f"{noise_floor_path} found -- using conservative fallback "
              f"sigma={DEFAULT_SIGMA}. Run scripts/seed_variance.py to measure "
              f"it for real.")
        return DEFAULT_SIGMA
    print(f"[search_planner] WARNING: using {noise_floor_path}'s sigma, which was "
          f"measured with the seed FIXED and so contains no seed variance -- it "
          f"understates the real noise. Run scripts/seed_variance.py.")
    return float(json.loads(p.read_text())["std"])


def render_report(
    iteration: int,
    surrogate_model,
    main_effect: Dict[str, float],
    blocks: List[List[str]],
    variance_share: Dict[str, float],
    frozen: List[str],
    active_block: List[str],
    proposal: Dict[str, Any],
) -> str:
    """Markdown: per-param S_perf-ranked table (S_behav column marked 'not
    available -- Tier 2'), frozen-param list with reason, block membership +
    variance share, and which block is active this cycle."""
    lines = [
        f"# Search Plan — iteration {iteration}",
        "",
        f"Surrogate fit on {surrogate_model.n_train} historical runs.",
        "",
        "## Sensitivity (near current best)",
        "| parameter | S_perf | S_behav | status |",
        "|---|---:|---|---|",
    ]
    for param, score in sorted(main_effect.items(), key=lambda kv: -kv[1]):
        status = "frozen (< 2σ)" if param in frozen else "active"
        lines.append(f"| {param} | {score:.6f} | not available — Tier 2 | {status} |")

    lines += ["", "## Blocks (interacting parameters tuned jointly, ranked by combined S_perf)"]
    for block in blocks:
        share = variance_share.get(tuple(sorted(block)), 0.0)
        marker = " **(active this cycle)**" if set(block) == set(active_block) else ""
        lines.append(f"- `{block}` — variance share {share:.2%}{marker}")

    lines += [
        "",
        "## This iteration's proposal",
        "```json",
        json.dumps(proposal, indent=2, sort_keys=True),
        "```",
    ]
    return "\n".join(lines) + "\n"


def propose_next(
    rows: List[Dict[str, Any]],
    current_best_hyperparams: Dict[str, Any],
    current_best_val_bpb: float,
    iteration: int,
    cold_start_n: int = 15,
    cycle_runs: int = 10,
    reprobe_every: int = 20,
    interaction_threshold: float = 0.15,
    state_path: str = STATE_PATH_DEFAULT,
    noise_floor_path: str = NOISE_FLOOR_PATH_DEFAULT,
    report_dir: str = REPORT_DIR_DEFAULT,
    generate_charts: bool = True,
    f_best_region_id: Optional[str] = None,
    search_columns: Optional[Sequence[str]] = None,
    fence_center: Optional[Dict[str, Any]] = None,
    fence_radius: Optional[float] = None,
    #: 1.0 = textbook EI. Falls toward 0 as the campaign's non-Sobol runs
    #: deplete, annealing EI's uncertainty term -- exploration is only worth
    #: paying for while runs remain to exploit what it finds.
    exploration_weight: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Returns a full hyperparams dict (pass-through keys like `ablation_k`
    from current_best_hyperparams are preserved), or None when scipy/
    scikit-learn aren't installed, the surrogate can't fit yet, or every
    dimension is currently frozen -- in all of these cases the caller
    (Agent1TrainingSpecialist._surrogate_adjustment) falls back to the
    existing evidence/heuristic path unchanged.
    """
    if not surrogate.SURROGATE_DEPS_AVAILABLE:
        return None

    from agents.agent1_training_specialist import SEARCH_SPACE

    # TWO parameter sets, and the distinction is the point.
    #
    # `params` (all 11) are the surrogate's FEATURES. Architecture must stay a
    # feature: it is the strongest predictor of val_bpb there is (size alone
    # correlates -0.76), and a model fitted without it would blame architecture
    # differences on noise and mis-rank everything else.
    #
    # `search_params` (the 8 tunables) are what Agent 1 may MOVE. Architecture
    # belongs to Agent 4 and defines the region; changing it here would produce
    # a run whose weights differ from the rest of its own region, destroying
    # exactly the pairing that makes a within-region comparison readable.
    params = list(HYPERPARAM_COLUMNS)
    search_params = list(search_columns) if search_columns else list(TUNABLE_COLUMNS)
    state = SearchPlannerState.load(state_path)
    report_dir_path = Path(report_dir)

    # -- Cold start: Sobol until cold_start_n usable rows exist. --
    # A row only counts toward "have we finished cold-starting" if it carries
    # a real measurement. Counting merely-present val_bpb let a crashed run
    # (remote_error logs val_bpb=inf) advance the cold start without
    # contributing any data -- and state/surrogate.py::_rows_to_xy correctly
    # drops those same rows, so 15 failed runs would end the cold start and
    # then fit_surrogate would return None on 0 usable points, silently
    # dropping the campaign onto the heuristic path for good. Observed for
    # real: a network outage failed 8 consecutive dispatches, each logged as
    # val_bpb=inf, and every one of them counted.
    # "Usable" has to mean exactly what fit_surrogate means by it, or the cold
    # start ends while the surrogate still cannot fit and every later call
    # silently returns None. fit_surrogate drops compute-starved runs (a
    # contended GPU gave them far fewer steps than their config predicts), so
    # this must too.
    #
    # Seen for real: a 16-run campaign ended its cold start at 15 raw rows,
    # then fit_surrogate excluded 5 as compute-starved, leaving 11 -- below
    # MIN_SURROGATE_N. Agent 4 could propose no regions, the orchestrator saw
    # no live region, opened a second bootstrap region, and restarted the
    # whole Sobol cold start. Same class of bug as counting val_bpb=inf rows:
    # two different definitions of "usable row" in two places.
    def _countable(candidate_rows):
        return sum(
            1 for r in candidate_rows
            if isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])
            and all(c in r for c in params)
        )

    # "Is the cold start over?" and "can the surrogate fit?" are the same
    # question, so ask it once by actually fitting. Counting rows through
    # without_compute_starved separately meant fitting the step-deficit model
    # twice per proposal, since fit_surrogate applies that same filter
    # internally -- it doubled this module's own test time.
    #
    # The raw count is checked first purely to avoid a doomed fit: filtering
    # only ever reduces the count, so below cold_start_n the answer is already
    # "keep cold-starting".
    sm = None
    if _countable(rows) >= cold_start_n:
        sm = surrogate.fit_surrogate(rows, feature_columns=params, min_n=cold_start_n)
    if sm is None:
        if not state.cold_start_points:
            state.cold_start_points = surrogate.sobol_cold_start(SEARCH_SPACE, search_params, cold_start_n, seed=0)
            if generate_charts:
                try:
                    from state.visualize import chart_sobol_coverage
                    report_dir_path.mkdir(parents=True, exist_ok=True)
                    chart_sobol_coverage(state.cold_start_points, search_params, report_dir_path / "cold_start_coverage.png")
                except Exception as _e:
                    print(f"[search_planner] Chart generation (Sobol coverage) failed: {_e}")
        if state.cold_start_used < len(state.cold_start_points):
            point = state.cold_start_points[state.cold_start_used]
            state.cold_start_used += 1
            state.save(state_path)
            proposal = dict(current_best_hyperparams)
            proposal.update(point)
            print(f"[search_planner] Cold start {state.cold_start_used}/{len(state.cold_start_points)}: {point}")
            return proposal
        # Sobol batch exhausted but n_usable still short (some runs failed)
        # -- generate one more ad hoc point rather than stalling forever.
        extra = surrogate.sobol_cold_start(SEARCH_SPACE, search_params, 1, seed=1000 + iteration)
        proposal = dict(current_best_hyperparams)
        if extra:
            proposal.update(extra[0])
        return proposal

    # -- Surrogate-driven: prune -> block -> EI over the active block. --
    # `sm` was fitted above, where "can it fit at all" doubled as the
    # cold-start test. min_n=cold_start_n there (not fit_surrogate's own
    # MIN_SURROGATE_N default) because cold_start_n is the one real,
    # caller-configurable "how much data before we stop cold-starting"
    # threshold -- letting fit_surrogate silently apply a second, independent,
    # hardcoded threshold underneath it means a caller that lowers
    # cold_start_n (agents_config.yaml's agent1.surrogate_min_observations)
    # doesn't actually get what it asked for.

    sigma = _load_sigma(noise_floor_path)
    center = dict(current_best_hyperparams)
    # k stays at 2.0 and is correct at any sigma: the question it asks is "is
    # this parameter's total effect distinguishable from measurement noise",
    # which is definitionally in units of that noise. But note the CONSEQUENCE
    # moved a lot when the token budget collapsed sigma ~11x (0.00919 ->
    # 0.000797, see the CALIBRATION REFERENCE in agents_config.yaml): the
    # freeze bar fell from ~0.018 to ~0.0016 val_bpb of effect, so far fewer
    # parameters are now judged unmeasurable and the Gauss-Southwell blocks
    # run wider. That is the honest answer -- more parameters really are
    # measurable now -- but it means EI varies more dimensions per proposal
    # than it used to, on the same 2000 candidates.
    kept_now, frozen_now = surrogate.prune_by_noise_floor(sm, search_params, center, sm.bounds, sigma, k=2.0)

    # Age-based unfreeze: a param frozen >= reprobe_every iterations ago gets
    # one more chance this round even if prune_by_noise_floor still wants to
    # freeze it -- otherwise a param frozen early could never be re-measured
    # as more data (and a shifting "current best" center) changes its
    # estimated effect.
    reprobe_now = {p for p, since in state.frozen.items() if iteration - since >= reprobe_every}
    kept = kept_now + [p for p in frozen_now if p in reprobe_now]
    frozen = [p for p in frozen_now if p not in reprobe_now]
    for p in frozen:
        state.frozen.setdefault(p, iteration)
    for p in kept:
        state.frozen.pop(p, None)

    if not kept:
        # Everything measured below the noise floor -- nothing left to
        # usefully tune. Fall back rather than propose a no-op; Agent 1's
        # own stuck-detection/radical-change path can take it from here.
        state.save(state_path)
        return None

    main_effect = dict(surrogate.rank_by_sensitivity(sm, kept, center, sm.bounds))
    interactions = surrogate.interaction_matrix(rows, feature_columns=kept) or {}
    blocks = surrogate.blocks_from_interactions(interactions, main_effect, threshold=interaction_threshold)

    total_effect = sum(main_effect.values()) or 1.0
    variance_share = {
        tuple(sorted(block)): sum(main_effect.get(p, 0.0) for p in block) / total_effect
        for block in blocks
    }

    # Gauss-Southwell block rotation: stay on the currently-active block
    # (matched by member set, since blocks are recomputed fresh every call
    # and their composition can shift as data accumulates) until its budget
    # is used up, then advance to the next-ranked block.
    block_by_key = {tuple(sorted(b)): b for b in blocks}
    active_key = tuple(sorted(state.active_block)) if state.active_block else None
    if active_key not in block_by_key:
        active_idx = 0
        state.active_block = list(blocks[0])
        state.budget_used_in_block = 0
    else:
        active_idx = [tuple(sorted(b)) for b in blocks].index(active_key)

    active_block = blocks[active_idx]
    share = variance_share.get(tuple(sorted(active_block)), 0.0)
    budget = max(2, round(cycle_runs * share))
    if state.budget_used_in_block >= budget:
        next_idx = (active_idx + 1) % len(blocks)
        active_block = blocks[next_idx]
        state.active_block = list(active_block)
        state.budget_used_in_block = 0
        print(f"[search_planner] Block budget exhausted ({budget} runs) — rotating to {active_block}")

    if frozen:
        print(f"[search_planner] Frozen (S_perf < 2*sigma={2 * sigma:.6f}): {frozen}")
    print(f"[search_planner] Blocks: {blocks} — active this cycle: {active_block} "
          f"({state.budget_used_in_block + 1}/{budget})")

    # Denoise the EI incumbent (see surrogate.best_predicted_mean). Restricted
    # to this region's own rows when one is active, because the whole point of
    # a region-scoped search is a LOCAL reference -- a campaign-wide incumbent
    # makes EI inside any non-champion region see no improvement anywhere and
    # its argmax degenerate into noise. Falls back to the observed best
    # whenever no row can be scored, so behaviour is unchanged where this
    # cannot be computed.
    f_best_rows = rows
    if f_best_region_id is not None:
        scoped = [r for r in rows if r.get("region_id") == f_best_region_id]
        if scoped:
            f_best_rows = scoped
    f_best = surrogate.best_predicted_mean(sm, f_best_rows, feature_columns=params)
    if f_best is None:
        f_best = current_best_val_bpb
    else:
        print(f"[search_planner] EI incumbent: {f_best:.6f} (denoised predicted mean) "
              f"instead of {current_best_val_bpb:.6f} (best observed)")

    proposal, ei_diagnostics = surrogate.propose_via_ei(
        sm, f_best=f_best, bounds=sm.bounds,
        free_params=active_block, fixed_values=center, n_candidates=2000, seed=iteration,
        return_diagnostics=True,
        # The fence keeps the proposal inside the region. Without it the active
        # block was sampled across each parameter's FULL range, which moved a
        # region's centre 6-10x its own radius on essentially every proposal --
        # so "region" named a starting point rather than a place.
        fence_center=fence_center, fence_radius=fence_radius,
        fence_dims=len(search_params),
        exploration_weight=exploration_weight,
    )
    full = dict(current_best_hyperparams)
    full.update(proposal)

    escape = (ei_diagnostics or {}).get("escape")
    if escape and escape.get("escaped"):
        print(f"[search_planner] Escape pressure: the best candidate ignoring the fence "
              f"sits {escape['distance']:.4f} away (radius {escape['radius']:.4f}), "
              f"EI {escape['ei_outside']:.6g} vs {escape['ei_inside']:.6g} inside. "
              f"Recorded, not run.")

    state.budget_used_in_block += 1
    state.save(state_path)

    report = render_report(iteration, sm, main_effect, blocks, variance_share, frozen, active_block, full)
    report_dir_path.mkdir(parents=True, exist_ok=True)
    (report_dir_path / f"plan_{iteration:04d}.md").write_text(report, encoding="utf-8")
    (report_dir_path / f"plan_{iteration:04d}.json").write_text(json.dumps({
        "iteration": iteration, "main_effect": main_effect, "blocks": blocks,
        "variance_share": {"|".join(k): v for k, v in variance_share.items()},
        "frozen": frozen, "active_block": active_block, "proposal": full,
        "interaction_matrix": {"|".join(k): v for k, v in interactions.items()},
        "ei_diagnostics": ei_diagnostics,
        # Persisted separately from ei_diagnostics so Agent 4 can read a run of
        # these across iterations without loading every candidate's score. One
        # escape is noise; a consistent direction over several is the signal
        # that the region is anchored in the wrong place.
        "escape": escape,
        "oob_actual": list(sm.oob_actual),
        "oob_predicted": list(sm.oob_predicted),
    }, indent=2))

    if generate_charts:
        try:
            from state.visualize import (
                chart_ei_candidates,
                chart_interaction_matrix,
                chart_predicted_vs_actual,
                chart_surrogate_sensitivity,
            )
            chart_predicted_vs_actual(sm.oob_actual, sm.oob_predicted, report_dir_path / f"plan_{iteration:04d}_predicted_vs_actual.png")
            ranked = sorted(main_effect.items(), key=lambda kv: -kv[1])  # main_effect already came from rank_by_sensitivity -- avoid recomputing it
            chart_surrogate_sensitivity(ranked, frozen, report_dir_path / f"plan_{iteration:04d}_sensitivity.png")
            chart_interaction_matrix(interactions, kept, report_dir_path / f"plan_{iteration:04d}_interactions.png")
            chart_ei_candidates(ei_diagnostics, report_dir_path / f"plan_{iteration:04d}_ei_candidates.png")
        except Exception as _e:
            print(f"[search_planner] Chart generation (surrogate diagnostics) failed: {_e}")

    return full
