"""Region registry: the persistent identity of each concurrently-searched
area of the hyperparameter space.

Everything in this system up to now assumed there was exactly ONE search.
agents/search_planner.py keeps one `active_block`, one `frozen` dict and one
cold-start counter in a single state file; Agent 1 keeps one
`current_hyperparams` center. Agent 4's windows were a temporary override of
that single search, not a second one running beside it.

This module is the missing noun for running several searches at once. A
Region is an area of the space with its own fixed identity, its own drifting
local search center, its own run history, and its own lifecycle flag. The
per-region planner state that agents/search_planner.py needs is addressed by
`Region.planner_state_path` -- propose_next already takes `state_path` as a
parameter, so once regions exist, giving each one an independent Gauss-
Southwell rotation and its own frozen set costs nothing more than passing a
different path.

Three decisions here are load-bearing:

1. **Distance is measured in normalized 11-D, never on the PCA map.**
   state/landscape.py's 2-D projection explained 0.296 of the variance on a
   500-run campaign (reports/agent4_decisions/verdict_0032.json), so "are
   these two regions the same place?" is a question it cannot answer. That
   was tolerable when one candidate region was tested at a time; with
   several live at once the question is asked constantly. The PCA stays
   exactly what its own docstring says it is -- a visualization.

2. **Anchors are normalized against SEARCH_SPACE, not against observed
   data.** state/surrogate.py::fit_surrogate derives its bounds from the
   training rows, so those bounds widen as the campaign explores. An anchor
   normalized against them would silently move every time a new extreme run
   landed -- the same staleness that makes stored PCA coordinates unusable
   (see agent4_landscape_explorer._flag_region). SEARCH_SPACE is fixed for
   the campaign, so an anchor written at run 12 still means the same point
   at run 400.

3. **The anchor never moves; the center does.** A region's local optimizer
   walks its center downhill, which is the whole point of exploiting it. If
   identity moved with it, two regions could converge onto the same basin
   and quietly spend two GPUs doing one job, with nothing able to detect it
   -- "region" would stop naming anything. Identity is the anchor; the
   center is just where that region's search currently is. `merge_overlapping`
   is what handles the convergence case explicitly.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from state.results_analysis import HYPERPARAM_COLUMNS
from state.surrogate import normalized_value

REGISTRY_PATH_DEFAULT = "state/regions.json"
PLANNER_STATE_DIR_DEFAULT = "state/search_planner"

# The lifecycle vocabulary is deliberately the one already written into
# state/agent4_region_flags.json and rendered by state/visualize.py, so a
# chart drawn from the registry means the same thing as one drawn from the
# old flags file. Only the scope changed (many concurrent and persistent
# rather than one at a time).
ACTIVE = "currently_exploiting"
PAUSED = "exploitation_paused"
#: Set aside because there were not enough GPUs this wave, NOT because
#: anything was learned about the region. Kept distinct from PAUSED because
#: the two mean opposite things for what to do next: a capacity pause should
#: be undone the moment capacity returns (the region has runs invested and a
#: known score, which a speculative new region does not), while an
#: exploitation pause is a judgement that this area has stopped paying and
#: should only be revisited when there is nowhere better to look. Sharing one
#: flag made the allocator open brand-new regions after a busy wave while
#: four partially-explored ones sat idle.
CAPACITY_PAUSED = "capacity_paused"
NO_OPTIMUM = "no_optimum"
LOCAL_OPTIMUM = "local_optimum"
MERGED = "merged"

#: Flags that still consume GPU budget. PAUSED regions are kept for a later
#: cycle but are not scheduled; the three terminal flags never are.
SCHEDULABLE = (ACTIVE,)

# A region needs this many runs before its top quartile contains more than a
# single run -- top_quartile_by_val_bpb returns max(1, int(n * 0.25)), so
# below 8 the "median of the top quartile" IS the best run, carrying a full
# +/- sigma of measurement noise. Above it, elite_score is a real quantile;
# below it, plain median is the more honest summary of the same data.
MIN_RUNS_FOR_ELITE_SCORE = 8


def _bounds() -> Dict[str, Tuple[float, float]]:
    """SEARCH_SPACE, imported lazily.

    agents/agent1_training_specialist imports from state.*, so a module-level
    import here would close a cycle. agents/search_planner.py::propose_next
    already defers the same import for the same reason.
    """
    from agents.agent1_training_specialist import SEARCH_SPACE

    return SEARCH_SPACE


def to_vector(hyperparams: Dict[str, Any],
              bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> List[float]:
    """The 11-D normalized coordinate of a configuration.

    Reuses state/surrogate.py::normalized_value rather than normalizing here,
    so the log-scale treatment of the LR groups and batch_size is applied by
    exactly one function in the codebase. A missing key maps to 0.5 (the
    midpoint), matching normalized_value's own degenerate-range behavior --
    an absent parameter is unknown, not zero.
    """
    b = bounds if bounds is not None else _bounds()
    out = []
    for col in HYPERPARAM_COLUMNS:
        v = hyperparams.get(col)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out.append(normalized_value(col, float(v), b))
        else:
            out.append(0.5)
    return out


def distance(a: Dict[str, Any], b: Dict[str, Any],
             bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> float:
    """Normalized Euclidean distance between two configurations, divided by
    sqrt(11) so the result is on a 0-1 scale regardless of how many
    dimensions the search space happens to have.

    That rescaling is what lets a radius keep its meaning if a parameter is
    ever added to or dropped from HYPERPARAM_COLUMNS -- an un-normalized
    Euclidean distance in 11-D and one in 12-D are not the same number, so a
    radius tuned against one would silently change meaning against the other.
    """
    va, vb = to_vector(a, bounds), to_vector(b, bounds)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb))) / math.sqrt(len(va))


def _absorb_history(survivor: "Region", absorbed: "Region") -> None:
    """Fold one region's runs into another's, IN CHRONOLOGICAL ORDER.

    Concatenating the two histories instead would fabricate a chronology:
    the absorbed region's runs would appear to have all happened after the
    survivor's, when in reality the two searches ran concurrently. That
    matters because runs_since_improvement is order-dependent -- a merged
    region whose absorbed arm happened to be worse would read as "no
    improvement for N runs" purely as an artifact of the concatenation, and
    get paused or retired for it. (Found by
    test_maintain_merges_before_judging, which is exactly this scenario.)

    run_id is the sort key because the orchestrator issues them as
    run_{iteration:04d}, so lexicographic order is chronological order. Runs
    that produced no measurement carry no val_bpb and are merged separately;
    val_bpbs is not positionally parallel to run_ids for that reason.
    """
    measured = sorted(
        zip(survivor.measured_run_ids + absorbed.measured_run_ids,
            survivor.val_bpbs + absorbed.val_bpbs),
        key=lambda pair: pair[0],
    )
    survivor.measured_run_ids = [run_id for run_id, _ in measured]
    survivor.val_bpbs = [v for _, v in measured]
    survivor.run_ids = sorted(survivor.run_ids + absorbed.run_ids)


@dataclass
class Region:
    """One area of the space under its own independent search."""

    region_id: str
    #: Fixed at creation. Identity -- see this module's docstring, point 3.
    anchor: Dict[str, float]
    #: Where this region's local search currently is. Drifts every proposal.
    center: Dict[str, Any]
    flag: str = ACTIVE
    created_at_run: int = 0
    flag_since_run: int = 0
    origin: str = "unspecified"
    run_ids: List[str] = field(default_factory=list)
    val_bpbs: List[float] = field(default_factory=list)
    #: run_id of each entry in val_bpbs, positionally. Kept because val_bpbs
    #: is NOT parallel to run_ids -- a crashed run appends to run_ids only --
    #: and because merging two regions has to restore a true chronology
    #: (see RegionRegistry.merge_overlapping).
    measured_run_ids: List[str] = field(default_factory=list)
    #: Set on the losing side of a merge, so history stays readable rather
    #: than being deleted.
    merged_into: Optional[str] = None

    # -- history ----------------------------------------------------------

    def record(self, run_id: str, val_bpb: Optional[float]) -> None:
        """Attribute one completed run to this region.

        A non-finite val_bpb (crashed/OOM) is recorded as a run that
        happened but produced no measurement: it counts toward "how much
        budget has this region consumed" and not toward "how good is it".
        Agent 4 already treats a crashed probe this way
        (agent4_landscape_explorer.record_result) and the two must agree, or
        a region that OOMs repeatedly would look like a region nobody has
        tried yet and be scheduled forever.
        """
        self.run_ids.append(run_id)
        if isinstance(val_bpb, (int, float)) and math.isfinite(float(val_bpb)):
            self.val_bpbs.append(float(val_bpb))
            self.measured_run_ids.append(run_id)

    @property
    def n_runs(self) -> int:
        """Runs dispatched here, including ones that produced no number."""
        return len(self.run_ids)

    @property
    def n_measured(self) -> int:
        return len(self.val_bpbs)

    @property
    def best_val_bpb(self) -> Optional[float]:
        return min(self.val_bpbs) if self.val_bpbs else None

    def runs_since_improvement(self, min_improvement: float = 0.0,
                               skip_first: int = 0) -> int:
        """How many measured runs since this region last got meaningfully
        better.

        `min_improvement` is an ABSOLUTE val_bpb delta and should be passed as
        a multiple of the measured noise floor (state/noise_floor.json). At
        the default 0.0 this counts strict improvements, which on a frontier
        region fires on noise alone -- a stuck-detection threshold built on
        that would trip constantly. The parameter exists so callers are
        forced to say what "better" means.

        `skip_first` drops that many leading runs from the count entirely.
        Used for the bootstrap region's Sobol cold start, whose draws are a
        space-filling sample rather than a descent: the best of them sits
        wherever it happens to sit and later draws do not improve on it by
        construction. Counting them makes a region look stuck for as many
        runs as the cold start was long, which says nothing about whether the
        region is exhausted -- it retired the bootstrap region at exactly run
        16 on the first real campaign.
        """
        values = self.val_bpbs[skip_first:] if skip_first else self.val_bpbs
        if not values:
            return 0
        best = values[0]
        last_improved = 0
        for i, v in enumerate(values[1:], start=1):
            if v < best - min_improvement:
                best, last_improved = v, i
        return len(values) - 1 - last_improved

    def elite_score(self) -> Optional[float]:
        """How good this region is, for ranking it against the others.

        Median of the top quartile once there are enough runs for that to be
        a quantile rather than a single lucky draw (MIN_RUNS_FOR_ELITE_SCORE),
        plain median below it. Both are medians rather than means because
        one crashed-but-finite outlier should not decide which region gets
        the GPUs.
        """
        if not self.val_bpbs:
            return None
        if self.n_measured < MIN_RUNS_FOR_ELITE_SCORE:
            return statistics.median(self.val_bpbs)
        k = max(1, int(self.n_measured * 0.25))
        return statistics.median(sorted(self.val_bpbs)[:k])

    @property
    def schedulable(self) -> bool:
        return self.flag in SCHEDULABLE

    def set_flag(self, flag: str, at_run: int) -> None:
        self.flag = flag
        self.flag_since_run = at_run

    # -- per-region search state -------------------------------------------

    def planner_state_path(self, planner_state_dir: str = PLANNER_STATE_DIR_DEFAULT) -> str:
        """Where this region's own SearchPlannerState lives.

        This is the whole point of the refactor: agents/search_planner.py
        already accepts `state_path`, so pointing each region at its own file
        gives every region an independent cold start, an independent frozen
        set, and an independent Gauss-Southwell block rotation, with no
        change to the planner itself.
        """
        return str(Path(planner_state_dir) / f"{self.region_id}.json")

    def report_dir(self, reports_root: str = "reports/agent1_search_plan") -> str:
        return str(Path(reports_root) / self.region_id)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Region":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class RegionRegistry:
    """Every region this campaign has ever opened, and their current state.

    Persisted as one JSON file so a campaign that stops mid-flight resumes
    with its regions intact -- unlike Agent 4's window state, which was
    in-memory only and left a stale "investigating" flag behind whenever a
    campaign ended mid-window.
    """

    def __init__(self, path: str = REGISTRY_PATH_DEFAULT,
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.path = Path(path)
        self._bounds = bounds
        self.regions: List[Region] = []
        self._next_id = 1
        self.load()

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        if self._bounds is None:
            self._bounds = _bounds()
        return self._bounds

    # -- persistence --------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as e:
            # Same tolerance as SearchPlannerState.load: a corrupt registry
            # is recoverable (regions get re-proposed) but a crash here would
            # take down a campaign that is otherwise fine.
            print(f"[regions] Could not read {self.path}: {e} -- starting with an empty registry")
            return
        self.regions = [Region.from_dict(d) for d in raw.get("regions", [])]
        self._next_id = int(raw.get("next_id", len(self.regions) + 1))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_id": self._next_id,
            "regions": [r.to_dict() for r in self.regions],
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # -- lookup -------------------------------------------------------------

    def get(self, region_id: str) -> Optional[Region]:
        for r in self.regions:
            if r.region_id == region_id:
                return r
        return None

    def active(self) -> List[Region]:
        """Schedulable regions, best first -- the order a GPU allocator wants.

        A region with no measurement yet sorts last rather than first: it has
        no evidence in its favour, and sorting unmeasured regions to the top
        would let a freshly-opened region displace a proven one from the
        allocation on no data at all.
        """
        live = [r for r in self.regions if r.schedulable]
        return sorted(live, key=lambda r: (r.elite_score() is None, r.elite_score() or 0.0))

    def nearest(self, hyperparams: Dict[str, Any],
                include_terminal: bool = False) -> Tuple[Optional[Region], float]:
        """The closest region to a configuration, and how far away it is.

        `include_terminal=True` also considers regions already flagged
        no_optimum / local_optimum. That matters when proposing a NEW region:
        a candidate that lands inside an area already ruled out should be
        rejected, and the record of having ruled it out is the only thing
        that can say so.
        """
        pool = [r for r in self.regions
                if r.merged_into is None and (include_terminal or r.schedulable or r.flag == PAUSED)]
        if not pool:
            return None, float("inf")
        scored = [(distance(hyperparams, r.anchor, self.bounds), r) for r in pool]
        d, r = min(scored, key=lambda t: t[0])
        return r, d

    # -- lifecycle ----------------------------------------------------------

    def open_region(self, center: Dict[str, Any], at_run: int,
                    origin: str = "unspecified") -> Region:
        """Start searching a new area, anchored where it was proposed."""
        region = Region(
            region_id=f"r{self._next_id:04d}",
            anchor={c: float(center[c]) for c in HYPERPARAM_COLUMNS
                    if isinstance(center.get(c), (int, float))},
            center=dict(center),
            flag=ACTIVE,
            created_at_run=at_run,
            flag_since_run=at_run,
            origin=origin,
        )
        self._next_id += 1
        self.regions.append(region)
        return region

    def assign_run(self, region_id: str, run_id: str, val_bpb: Optional[float],
                   center: Optional[Dict[str, Any]] = None) -> Optional[Region]:
        """Record a completed run against the region that dispatched it.

        Attribution is by region_id, NOT by "whichever region this
        configuration is nearest to". A local search legitimately walks its
        center toward the edge of its own region and sometimes past a
        neighbour's anchor; re-attributing that run by proximity would credit
        the neighbour with a run it never spent budget on, and would make
        n_runs -- which every lifecycle threshold counts in -- stop meaning
        "budget this region consumed".
        """
        region = self.get(region_id)
        if region is None:
            return None
        region.record(run_id, val_bpb)
        if center is not None:
            region.center = dict(center)
        return region

    def merge_overlapping(self, radius: float, at_run: int) -> List[Tuple[str, str]]:
        """Fold together regions whose anchors have ended up within `radius`
        of each other, and return the (absorbed, survivor) pairs.

        Two regions can be opened far apart and still converge -- each one's
        local search walks downhill, and there is nothing stopping two of
        them walking into the same basin. Left alone that silently halves the
        parallelism: two GPUs, two planner states, one basin. The survivor is
        the better-scoring region, so the merged history is attributed to the
        anchor that actually earned it; the absorbed region keeps its own
        record and a `merged_into` pointer rather than being deleted, because
        "these two turned out to be the same place" is a finding about the
        landscape worth keeping.

        Only schedulable and paused regions merge. A region already ruled out
        must stay ruled out: absorbing it into a live one would resurrect a
        negative result as if it were evidence in the live region's favour.
        """
        merges: List[Tuple[str, str]] = []
        changed = True
        while changed:
            changed = False
            pool = [r for r in self.regions
                    if r.merged_into is None and (r.schedulable or r.flag == PAUSED)]
            for i, a in enumerate(pool):
                for b in pool[i + 1:]:
                    if distance(a.anchor, b.anchor, self.bounds) > radius:
                        continue
                    survivor, absorbed = self._rank_for_merge(a, b)
                    _absorb_history(survivor, absorbed)
                    absorbed.merged_into = survivor.region_id
                    absorbed.set_flag(MERGED, at_run)
                    merges.append((absorbed.region_id, survivor.region_id))
                    changed = True
                    break
                if changed:
                    break
        return merges

    @staticmethod
    def _rank_for_merge(a: Region, b: Region) -> Tuple[Region, Region]:
        """(survivor, absorbed). Better elite score wins; an unmeasured
        region always loses to a measured one; ties break on the older
        region, so the surviving anchor is the one with the longer record."""
        sa, sb = a.elite_score(), b.elite_score()
        if sa is None and sb is None:
            return (a, b) if a.created_at_run <= b.created_at_run else (b, a)
        if sa is None:
            return b, a
        if sb is None:
            return a, b
        if sa != sb:
            return (a, b) if sa < sb else (b, a)
        return (a, b) if a.created_at_run <= b.created_at_run else (b, a)

    # -- reporting ----------------------------------------------------------

    def flags_snapshot(self) -> List[Dict[str, Any]]:
        """The registry rendered in state/agent4_region_flags.json's shape,
        so state/visualize.py's landscape chart keeps working unchanged."""
        return [
            {
                "hyperparams": dict(r.anchor),
                "flag": r.flag,
                "since_iteration": r.flag_since_run,
                "n_runs": r.n_runs,
                "region_id": r.region_id,
            }
            for r in self.regions
            if r.merged_into is None
        ]

    def summary_rows(self) -> List[Dict[str, Any]]:
        """One row per region -- what Agent 3 needs to write a per-region
        summary, and what a GPU allocator needs to rank them."""
        rows = []
        for r in self.regions:
            if r.merged_into is not None:
                continue
            rows.append({
                "region_id": r.region_id,
                "flag": r.flag,
                "origin": r.origin,
                "n_runs": r.n_runs,
                "n_measured": r.n_measured,
                "best_val_bpb": r.best_val_bpb,
                "elite_score": r.elite_score(),
                "created_at_run": r.created_at_run,
                "flag_since_run": r.flag_since_run,
            })
        return rows
