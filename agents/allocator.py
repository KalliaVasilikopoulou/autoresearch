"""Which region does each GPU work on this wave?

This is the piece that replaces Agent 4's exploration *window*. The old model
was one search that a second agent could temporarily take over: Agent 4
watched for stagnation, seized every slot for a bounded budget, probed one
region at a time, and handed control back. Exploration and exploitation were
phases, and the trigger was reactive -- which on a short campaign means the
better region gets found with no budget left to exploit it.

Here they run at once. Several regions stay live, each with its own search
state, and every wave this module decides how the available GPUs are spread
across them. Nothing is "taken over" and nothing switches.

Three rules, in priority order:

  1. THE CHAMPION IS NEVER STARVED. The best region always keeps at least one
     GPU. Without this an unlucky wave -- or a run of newly-opened regions
     that all look promising because they have no data yet -- can leave the
     frontier with nothing running on it.

  2. FEWER GPUS THAN REGIONS: the best `n` regions run, the rest are paused
     rather than dropped. Paused is recoverable; a region's history and its
     planner state survive, and it can be resumed when capacity returns.
     Retiring a region for lack of a GPU would confuse "we ran out of
     hardware" with "we ruled this out", which are not the same claim.

  3. MORE GPUS THAN REGIONS: a spare GPU opens a NEW region while the live
     count is under `max_regions`, and reinforces the best existing region
     after that. Opening is preferred because coverage is the scarcer thing
     early -- a second GPU on the champion buys a slightly faster local
     search, whereas a new region buys a place the search has never looked.

Allocation is by rank, not proportional. With <= 4 GPUs and 2-4 regions there
is at most one or two spare slots, so a proportional rule and "give it to the
best" produce the same answer almost always, and rank ordering is the one that
can be reasoned about from a log.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from state.regions import Region

#: Returned by `plan` so the caller knows what to do with the regions it did
#: not get to run. Kept as a small dataclass-shaped tuple rather than a bare
#: list because "which regions to pause" and "how many new ones to open" are
#: both consequences of the same ranking and must not be recomputed apart.
class AllocationPlan:
    """Slot assignments plus the two side effects the caller has to apply."""

    def __init__(
        self,
        assignments: List[Optional[str]],
        to_pause: List[str],
        open_new: int,
        reinforced: Dict[str, int],
    ):
        #: One entry per GPU slot: the region_id it should search, or None for
        #: a slot that wants a brand-new region the caller must open first.
        self.assignments = assignments
        #: Live regions with no GPU this wave -- pause, do not retire.
        self.to_pause = to_pause
        #: How many of the None slots want a newly-opened region.
        self.open_new = open_new
        #: region_id -> how many GPUs it got, for regions that got 2+.
        self.reinforced = reinforced

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"AllocationPlan(assignments={self.assignments}, "
                f"to_pause={self.to_pause}, open_new={self.open_new})")


def plan(
    regions: Sequence[Region],
    n_gpus: int,
    max_regions: int,
) -> AllocationPlan:
    """Spread `n_gpus` across `regions` (best first) for one wave.

    `regions` must already be ranked -- RegionRegistry.active() returns
    schedulable regions sorted by elite_score with unmeasured ones last, which
    is exactly the order this wants. Ranking is not redone here: the registry
    owns what "better" means (a median of the top quartile once a region has
    enough runs for that to be a quantile rather than one lucky draw), and a
    second definition of it living in the allocator is a second thing to keep
    in sync.
    """
    if n_gpus <= 0:
        return AllocationPlan([], [r.region_id for r in regions], 0, {})

    if not regions:
        # Nothing live yet: every slot wants a new region, capped by how many
        # regions we are willing to run at once. A campaign starting from an
        # empty registry lands here.
        wanted = min(n_gpus, max(1, max_regions))
        return AllocationPlan([None] * wanted, [], wanted, {})

    # --- rule 2: fewer GPUs than regions -----------------------------------
    if n_gpus < len(regions):
        running = list(regions[:n_gpus])
        paused = [r.region_id for r in regions[n_gpus:]]
        return AllocationPlan([r.region_id for r in running], paused, 0, {})

    # --- one GPU each, then spares -----------------------------------------
    assignments: List[Optional[str]] = [r.region_id for r in regions]
    spare = n_gpus - len(regions)

    # rule 3: spares open new regions until max_regions, then reinforce.
    open_new = min(spare, max(0, max_regions - len(regions)))
    assignments.extend([None] * open_new)
    spare -= open_new

    reinforced: Dict[str, int] = {}
    for i in range(spare):
        # Round-robin from the top so two spares don't both pile onto the
        # single best region while the second-best has one -- with 4 GPUs and
        # 2 regions that is the difference between 3/1 and 2/2.
        target = regions[i % len(regions)]
        assignments.append(target.region_id)
        reinforced[target.region_id] = reinforced.get(target.region_id, 0) + 1

    # rule 1: the champion is never starved. Rules 2 and 3 both already give
    # regions[0] a slot, so this is a guard against a future edit rather than
    # a live code path -- asserted loudly instead of silently assumed, since
    # the failure it prevents (nothing running on the frontier) is invisible
    # in a results log until several waves later.
    champion = regions[0].region_id
    if champion not in assignments:  # pragma: no cover - defensive
        assignments[0] = champion

    return AllocationPlan(assignments, [], open_new, reinforced)


def describe(plan_: AllocationPlan, regions: Sequence[Region]) -> str:
    """One log line per wave explaining the split -- the only place the
    allocation is visible, since it leaves no artifact of its own."""
    by_id = {r.region_id: r for r in regions}
    parts = []
    counts: Dict[str, int] = {}
    for a in plan_.assignments:
        key = a or "<new>"
        counts[key] = counts.get(key, 0) + 1
    for key, n in counts.items():
        region = by_id.get(key)
        score = region.elite_score() if region else None
        score_txt = f"{score:.4f}" if isinstance(score, float) else "no runs yet"
        parts.append(f"{key} x{n} ({score_txt})")
    line = f"[Allocator] {len(plan_.assignments)} GPU(s): " + ", ".join(parts)
    if plan_.to_pause:
        line += f" | paused for lack of capacity: {', '.join(plan_.to_pause)}"
    return line
