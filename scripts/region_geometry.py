"""Step 4 of the region redesign: measure the geometry, then set the radius.

FOUR QUESTIONS, ONE GRID. Every number the remaining design depends on is
currently either unmeasured or carried over from a space that no longer exists.

  A-within   how much ONE configuration moves with the seed, inside a region.
             The resolution limit: the smallest difference a region can read.
  B(r)       how much DIFFERENT configurations differ inside a fence of
             radius r. The remaining opportunity.
  A-between  how much a comparison between two REGIONS moves. Split by kind,
             because step 1 showed depth-neighbours share most of their initial
             weights while width-neighbours share none.
  radius     the fence. Currently 0.05, carried over from the old 11-D
             geometry; the distance divides by sqrt(n_dims), so that number
             means something different in 8-D and must be re-derived.

THE RULE IT FEEDS. Observed spread already contains the noise:
`real signal(r) = sqrt(max(0, B(r)^2 - A_within^2))`. A region is worth
searching while real signal > A-within, and SATURATED when they meet. So the
fence radius should be the smallest r whose real signal comfortably clears
A-within -- small enough that interactions stay negligible inside it, large
enough that there is something left to find.

DESIGN NOTES.

Configurations at each radius are drawn with the SAME sampler the fence itself
uses (surrogate._sample_in_ball), uniformly by volume. Measuring a shell
instead would answer a different question than the one the fence asks.

The three configurations chosen for seed repeats deliberately span batch_size
from low to high. batch_size is the only tunable that changes HOW MUCH training
happens (num_steps = budget / batch_size, exact), so it is the leading suspect
for why A-within varies within a region. Architecture is frozen here, which is
what finally decontaminates that question -- in the 3-config seed experiment
step count and architecture moved together and could not be separated.

Usage:
    uv run python scripts/region_geometry.py --dry-run     # print the plan, dispatch nothing
    uv run python scripts/region_geometry.py
    uv run python scripts/region_geometry.py --analyze-only
"""

import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from state import surrogate
from state.results_analysis import (
    ARCHITECTURE_COLUMNS,
    HYPERPARAM_COLUMNS,
    TUNABLE_COLUMNS,
    load_results,
)
from state.results_logger import log_result

RUN_ID_PREFIX = "geom"
DEFAULT_RESULTS_PATH = "state/region_geometry.tsv"
REPORT_JSON = Path("state/region_geometry.json")
REPORT_MD = Path("reports/region_geometry.md")

#: Seed 42 first -- the campaign's historical seed, so the first cell of every
#: configuration is comparable to whatever history already recorded.
SEEDS = (42, 1, 2)

#: Fence radii to characterise. 0.05 is the current provisional value; 0.02 and
#: 0.10 bracket it so B(r) can be seen rising rather than read at one point.
RADII = (0.02, 0.05, 0.10)
CONFIGS_PER_RADIUS = 5
#: How many configurations get the full seed treatment (the rest run once).
SEED_REPEAT_CONFIGS = 3

OK_STATUSES = {"remote_ok", "ok"}


# ---------------------------------------------------------------------------
# Building the grid
# ---------------------------------------------------------------------------

def _search_space() -> Dict[str, Tuple[float, float]]:
    from agents.agent1_training_specialist import SEARCH_SPACE
    return SEARCH_SPACE


def pick_anchor(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The campaign's best complete run. The frontier is where the radius
    actually matters -- a radius calibrated in a bad part of the space would
    describe a neighbourhood the search never visits."""
    usable = [r for r in rows
              if r.get("status") in OK_STATUSES
              and isinstance(r.get("val_bpb"), (int, float)) and math.isfinite(r["val_bpb"])
              and not (isinstance(r.get("budget_shortfall_pct"), (int, float))
                       and r["budget_shortfall_pct"] > 0)
              and all(c in r for c in HYPERPARAM_COLUMNS)]
    if not usable:
        raise SystemExit("[region_geometry] No complete runs to anchor on.")
    best = min(usable, key=lambda r: r["val_bpb"])
    hp = {c: float(best[c]) for c in HYPERPARAM_COLUMNS}
    for c in ("n_layer", "n_embd", "n_head", "batch_size"):
        hp[c] = int(round(hp[c]))
    hp["_from_run"] = best.get("run_id")
    hp["_historical_val_bpb"] = best["val_bpb"]
    return hp


def configs_in_ball(anchor: Dict[str, Any], radius: float, n: int, seed: int) -> List[Dict[str, Any]]:
    """`n` configurations uniformly inside `radius` of the anchor, varying only
    the tunables. Uses the fence's own sampler so what is measured here is what
    the fence will actually enclose."""
    import numpy as np

    bounds = _search_space()
    center_norm = [surrogate.normalized_value(p, float(anchor[p]), bounds) for p in TUNABLE_COLUMNS]
    max_euclid = radius * math.sqrt(len(TUNABLE_COLUMNS))
    pts = surrogate._sample_in_ball(center_norm, max_euclid, n, np.random.default_rng(seed))

    out = []
    for row in pts:
        hp = dict(anchor)
        for p, t in zip(TUNABLE_COLUMNS, row):
            hp[p] = surrogate._denormalize(p, float(t), bounds)
        out.append(surrogate._snap_discrete(hp))
    return out


def neighbour_architectures(anchor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """One depth-neighbour and one width-neighbour, for A-between.

    Kept separate because step 1 proved they are not the same kind of move:
    +1 layer leaves every earlier layer's weights bit-identical, while a width
    change reshapes 41 of 46 tensors. So the noise in comparing two regions
    depends on WHICH way they differ, and one number cannot serve both.
    """
    n_head = int(anchor["n_head"])
    out: Dict[str, Dict[str, Any]] = {}

    depth = dict(anchor)
    depth["n_layer"] = int(anchor["n_layer"]) + 1
    out["depth_neighbour"] = depth

    # Step down in width to the nearest value that keeps head_dim an even
    # integer, or train.py silently re-snaps it and the region is not the one
    # we asked for.
    target = int(anchor["n_embd"]) * 0.8
    width_val = surrogate.snap_n_embd(target, n_head)
    if width_val != int(anchor["n_embd"]):
        width = dict(anchor)
        width["n_embd"] = width_val
        out["width_neighbour"] = width
    return out


def build_plan(anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every cell to run: {kind, label, seed, hyperparams}."""
    cells: List[Dict[str, Any]] = []

    # The anchor itself, at every seed. Required for A-between, which is the
    # spread of the PAIRED DIFFERENCE between a neighbour and the anchor across
    # seeds -- not the neighbour's own spread, which is just A-within for that
    # architecture. The first version of this grid omitted the anchor and the
    # analysis silently reported the wrong quantity.
    for s in SEEDS:
        cells.append({"kind": "anchor", "radius": None, "config_idx": 0,
                      "label": "anchor", "seed": s, "hyperparams": dict(anchor)})

    for radius in RADII:
        configs = configs_in_ball(anchor, radius, CONFIGS_PER_RADIUS, seed=int(radius * 1000))
        # Spread the seed-repeat picks across batch_size, so A-within can be
        # tested against the one tunable that changes the amount of training.
        order = sorted(range(len(configs)), key=lambda i: configs[i]["batch_size"])
        repeat_idx = {order[0], order[len(order) // 2], order[-1]} if len(order) >= SEED_REPEAT_CONFIGS else set(order)
        for i, hp in enumerate(configs):
            seeds = SEEDS if i in repeat_idx else SEEDS[:1]
            for s in seeds:
                cells.append({"kind": "ball", "radius": radius, "config_idx": i,
                              "label": f"r{radius:.2f}_c{i}", "seed": s, "hyperparams": hp})

    for name, hp in neighbour_architectures(anchor).items():
        for s in SEEDS:
            cells.append({"kind": name, "radius": None, "config_idx": 0,
                          "label": name, "seed": s, "hyperparams": hp})
    return cells


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _hyperparams_for(cell: Dict[str, Any]) -> Dict[str, Any]:
    hp = {k: v for k, v in cell["hyperparams"].items() if not k.startswith("_")}
    hp["seed"] = cell["seed"]
    # Pure measurement runs: post-training analysis costs GPU time and cannot
    # change val_bpb.
    hp["ablation_k"] = 0
    hp["token_xai_enabled"] = False
    return hp


def already_done(results_path: str) -> set:
    """run_ids already recorded with a real measurement, so a re-run does only
    what is missing. The first attempt lost 13 of 39 cells to a dropped SSH
    session; redoing the 26 that succeeded would have cost ~5 GPU-hours for
    nothing."""
    done = set()
    for row in load_results(results_path):
        if row.get("status") in OK_STATUSES and isinstance(row.get("val_bpb"), (int, float)) \
                and math.isfinite(row["val_bpb"]):
            done.add(str(row.get("run_id")))
    return done


def run_all(cells: List[Dict[str, Any]], results_path: str, timeout: int) -> None:
    from agents import remote_runner
    from agents.live_progress import MultiGpuProgressDisplay

    if not remote_runner.is_remote_configured():
        raise SystemExit("[region_geometry] No remote configured.")

    done = already_done(results_path)
    pending = [c for c in cells
               if f"{RUN_ID_PREFIX}_{c['label']}_s{c['seed']}" not in done]
    if len(pending) < len(cells):
        print(f"[region_geometry] Resuming: {len(cells) - len(pending)} cell(s) already "
              f"measured, {len(pending)} to run.")
    if not pending:
        print("[region_geometry] Nothing to run.")
        return

    hp_dir = Path("state/region_geometry_hyperparams")
    hp_dir.mkdir(parents=True, exist_ok=True)

    # ONE CONNECTION PER WAVE, not one for the whole experiment. The first
    # attempt held a single SSH session across all 10 waves (~2.5 hours); it
    # dropped partway and every remaining run failed instantly with "SSH
    # session not active" -- 13 of 39 cells lost, including both neighbour
    # architectures and most of the widest radius. scripts/seed_variance.py
    # survived 88 minutes precisely because it reconnects per block, and that
    # is the pattern agents/orchestrator.py uses per wave too.
    wave_no = 0
    start = 0
    while start < len(pending):
        wave_no += 1
        try:
            client = remote_runner.open_client()
        except Exception as e:
            print(f"[region_geometry] Could not reach the server for wave {wave_no}: {e}")
            break
        try:
            if not remote_runner.sync_remote_code(client=client):
                print(f"[region_geometry] Sync failed on wave {wave_no} -- stopping so the "
                      f"remaining cells stay re-runnable rather than becoming inf rows.")
                break
            gpus = [g["index"] for g in remote_runner.discover_available_gpus(client=client)]
            if not gpus:
                print("[region_geometry] No free GPUs -- stopping; re-run to continue.")
                break

            wave = pending[start:start + len(gpus)]
            wave_statuses: List[Optional[str]] = []
            labels = [f"GPU{gpus[i]}" for i in range(len(wave))]
            print(f"\n[region_geometry] === wave {wave_no} ({len(wave)} run(s), "
                  f"{start}/{len(pending)} done) ===")
            with MultiGpuProgressDisplay(labels) as display:
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    futures = {}
                    for i, cell in enumerate(wave):
                        run_id = f"{RUN_ID_PREFIX}_{cell['label']}_s{cell['seed']}"
                        hp = _hyperparams_for(cell)
                        path = hp_dir / f"{run_id}.yaml"
                        with open(path, "w", encoding="utf-8") as f:
                            yaml.dump(hp, f)
                        fut = pool.submit(
                            remote_runner.run_training_remote,
                            hyperparams_local_path=str(path), gpu_index=gpus[i],
                            hp_remote_name=f"model_hyperparams_{run_id}.yaml",
                            run_label=f"GPU{gpus[i]}", timeout=timeout,
                            skip_sync=True, display=display, client=client)
                        futures[fut] = (run_id, hp, cell)
                    for fut in as_completed(futures):
                        run_id, hp, cell = futures[fut]
                        try:
                            metrics = fut.result()
                        except Exception as e:
                            display.print_line(f"[region_geometry] {run_id} failed: {e}")
                            metrics = {"val_bpb": float("inf"), "status": "remote_error", "error": str(e)}
                        display.print_line(f"[region_geometry] {run_id}: "
                                           f"val_bpb={metrics.get('val_bpb')} "
                                           f"status={metrics.get('status')}")
                        log_result(run_id, hp, metrics, results_path=results_path)
                        wave_statuses.append(metrics.get("status"))
            start += len(wave)

            # A whole wave failing is not bad luck, it is a broken link -- when
            # the SSH session drops, train.py dies with it and EVERY concurrent
            # run in that wave dies at the same moment. Carrying on just spends
            # the remaining cells against the same broken connection; that is
            # how 13 of 13 were lost on the previous attempt. Resume makes
            # stopping free.
            if wave_statuses and all(s == "remote_error" for s in wave_statuses):
                print(f"\n[region_geometry] Every run in wave {wave_no} failed. Stopping "
                      f"rather than spending the remaining {len(pending) - start} cell(s) "
                      f"against the same fault -- re-run to resume where this left off.")
                break
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _parse(run_id: str) -> Optional[Tuple[str, int]]:
    """geom_<label>_s<seed> -> (label, seed)."""
    if not run_id.startswith(f"{RUN_ID_PREFIX}_"):
        return None
    body = run_id[len(RUN_ID_PREFIX) + 1:]
    if "_s" not in body:
        return None
    label, _, seed = body.rpartition("_s")
    try:
        return label, int(seed)
    except ValueError:
        return None


def analyze(results_path: str) -> Dict[str, Any]:
    by_label: Dict[str, Dict[int, float]] = {}
    for row in load_results(results_path):
        parsed = _parse(str(row.get("run_id", "")))
        if not parsed or row.get("status") not in OK_STATUSES:
            continue
        val = row.get("val_bpb")
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        label, seed = parsed
        by_label.setdefault(label, {})[seed] = float(val)
        by_label[label]["_batch"] = row.get("batch_size")  # type: ignore[index]

    # --- A-within: spread of one configuration across seeds ---
    # Ball configurations ONLY. A-within is defined inside one region, i.e. at a
    # fixed architecture; pooling in the neighbour architectures (a different
    # n_layer, a different n_embd) mixes three regions into one number. The
    # first version of this analysis did exactly that and reported 0.001117
    # where the region's own value is 0.001342.
    per_config = {}
    for label, cells in by_label.items():
        vals = [v for k, v in cells.items() if isinstance(k, int)]
        if len(vals) > 1:
            per_config[label] = {
                "n_seeds": len(vals), "mean": statistics.mean(vals),
                "std": statistics.stdev(vals), "batch_size": cells.get("_batch"),
                "in_region": label.startswith("r0."),
            }
    in_region = [c["std"] for c in per_config.values() if c["in_region"]]
    a_within = statistics.median(in_region) if in_region else None

    # --- B(r): spread across configurations inside each radius ---
    b_by_radius = {}
    for radius in RADII:
        prefix = f"r{radius:.2f}_c"
        vals = [cells[SEEDS[0]] for label, cells in by_label.items()
                if label.startswith(prefix) and SEEDS[0] in cells]
        if len(vals) > 1:
            b_obs = statistics.stdev(vals)
            real = math.sqrt(max(0.0, b_obs ** 2 - (a_within or 0.0) ** 2))
            b_by_radius[f"{radius:.2f}"] = {
                "n_configs": len(vals), "b_observed": b_obs, "real_signal": real,
                "ratio_to_a_within": (real / a_within) if a_within else None,
            }

    # --- A-between: how much the COMPARISON between two regions moves ---
    # The spread of the PAIRED DIFFERENCE (neighbour - anchor) across seeds, not
    # the neighbour's own spread. Its own spread is A-within for that
    # architecture and says nothing about comparing two regions; the first
    # version of this analysis reported that by mistake.
    a_between = {}
    anchor_cells = {k: v for k, v in by_label.get("anchor", {}).items() if isinstance(k, int)}
    for kind in ("depth_neighbour", "width_neighbour"):
        cells = by_label.get(kind, {})
        shared = sorted(s for s in cells if isinstance(s, int) and s in anchor_cells)
        if len(shared) < 2:
            continue
        diffs = [cells[s] - anchor_cells[s] for s in shared]
        # If one side of the pair is far noisier than the other, the difference
        # is essentially that side's wobble and the comparison says nothing
        # about the OTHER. Measured here: the anchor's own std was 0.00197
        # while both neighbours sat near 0.00023, so std_of_gap came out
        # identical (0.001988) for depth and width -- arithmetically forced,
        # not a finding about either. Flagged rather than reported bare.
        own = statistics.stdev([cells[s] for s in shared])
        anchor_own = statistics.stdev([anchor_cells[s] for s in shared])
        a_between[kind] = {
            "n_seeds": len(shared),
            "mean_gap": statistics.mean(diffs),
            "std_of_gap": statistics.stdev(diffs),
            "own_std": own,
            "anchor_std": anchor_own,
            "anchor_dominated": anchor_own > 3 * own,
            "per_seed_gap": {str(s): d for s, d in zip(shared, diffs)},
            # Above ~2 the two regions are genuinely different; below it the
            # gap is unreadable however large the means look.
            "separation": (abs(statistics.mean(diffs))
                           / (statistics.stdev(diffs) / len(diffs) ** 0.5))
            if statistics.stdev(diffs) > 0 else None,
        }

    # --- the recommendation ---
    recommended = None
    for radius in RADII:
        entry = b_by_radius.get(f"{radius:.2f}")
        if entry and entry["ratio_to_a_within"] and entry["ratio_to_a_within"] >= 3.0:
            recommended = radius
            break

    return {
        "a_within": a_within, "per_config": per_config,
        "b_by_radius": b_by_radius, "a_between": a_between,
        "recommended_fence_radius": recommended,
        "n_labels": len(by_label),
    }


def render(report: Dict[str, Any]) -> str:
    a = report["a_within"]
    lines = ["# Region geometry", "",
             f"A-within (median across configurations): "
             f"**{a:.6f}**" if a else "A-within: not measurable yet", ""]

    if report["per_config"]:
        lines += ["## Does A-within track batch_size?", "",
                  "batch_size is the only tunable that changes how much training happens "
                  "(`num_steps = budget / batch_size`), so it is the suspect for why noise "
                  "varies inside a region. Architecture is frozen here, which is what makes "
                  "this separable at all.", "",
                  "| configuration | batch_size | seeds | std |", "|---|---:|---:|---:|"]
        for label, c in sorted(report["per_config"].items(),
                               key=lambda kv: kv[1]["batch_size"] or 0):
            lines.append(f"| {label} | {c['batch_size']} | {c['n_seeds']} | {c['std']:.6f} |")

    if report["b_by_radius"]:
        lines += ["", "## B(r): is there anything left to find inside a fence of radius r?", "",
                  "`real signal = sqrt(B_observed^2 - A_within^2)` -- the observed spread "
                  "already contains the noise. A region is saturated when real signal "
                  "falls to A-within.", "",
                  "| radius | configs | B observed | real signal | real / A-within |",
                  "|---:|---:|---:|---:|---:|"]
        for r, e in sorted(report["b_by_radius"].items()):
            ratio = f"{e['ratio_to_a_within']:.2f}" if e["ratio_to_a_within"] else "n/a"
            lines.append(f"| {r} | {e['n_configs']} | {e['b_observed']:.6f} | "
                         f"{e['real_signal']:.6f} | {ratio} |")

    if report["a_between"]:
        lines += ["", "## A-between, by kind of architecture change", "",
                  "The spread of the PAIRED DIFFERENCE (neighbour minus anchor) across "
                  "seeds -- how much the comparison between two regions moves, not how "
                  "much either one moves on its own.", "",
                  "Step 1 predicted these would differ: +1 layer leaves every earlier "
                  "layer's weights bit-identical, while a width change reshapes 41 of 46 "
                  "tensors. If they come out equal, sharing an initialization does not "
                  "survive training, and nesting weights across architectures buys "
                  "nothing.", "",
                  "| neighbour | seeds | mean gap | A-between | its own std | anchor std | separation |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for kind, e in sorted(report["a_between"].items()):
            sep = f"{e['separation']:.1f}" if e["separation"] else "n/a"
            lines.append(f"| {kind} | {e['n_seeds']} | {e['mean_gap']:+.6f} | "
                         f"{e['std_of_gap']:.6f} | {e['own_std']:.6f} | "
                         f"{e['anchor_std']:.6f} | {sep} |")
        if any(e["anchor_dominated"] for e in report["a_between"].values()):
            lines += ["",
                      "> **A-between here is ANCHOR-DOMINATED and must not be read as a "
                      "property of the neighbours.** The anchor's own seed spread is more "
                      "than 3x each neighbour's, so `neighbour - anchor` is essentially "
                      "the anchor's wobble -- the same anchor in every row, which is why "
                      "the values come out equal. This design cannot compare depth "
                      "against width; that needs a quieter anchor or many more seeds.",
                      "",
                      "> The MEAN GAP is still usable where `separation` is large: it is a "
                      "difference of means, which the anchor's noise widens but does not "
                      "bias."]

    rec = report["recommended_fence_radius"]
    lines += ["", "## Recommended fence radius", "",
              f"**{rec}** -- the smallest tested radius whose real signal is at least 3x "
              f"A-within." if rec else
              "No tested radius reached 3x A-within. Either widen the range, or lower "
              "A-within with seed replicates before tightening the fence.", ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default="results.tsv")
    ap.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    ap.add_argument("--timeout", type=int, default=2200)
    ap.add_argument("--dry-run", action="store_true", help="Print the plan; dispatch nothing.")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    if not args.analyze_only:
        anchor = pick_anchor(load_results(args.history))
        print(f"[region_geometry] Anchor: {anchor.get('_from_run')} "
              f"(val_bpb {anchor.get('_historical_val_bpb'):.6f})")
        print(f"[region_geometry]   architecture: "
              + ", ".join(f"{c}={anchor[c]}" for c in ARCHITECTURE_COLUMNS))
        cells = build_plan(anchor)
        by_kind: Dict[str, int] = {}
        for c in cells:
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
        print(f"[region_geometry] {len(cells)} runs: " +
              ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
        for r in RADII:
            n = sum(1 for c in cells if c["radius"] == r)
            print(f"[region_geometry]   radius {r:.2f}: {n} run(s) "
                  f"({CONFIGS_PER_RADIUS} configs, {SEED_REPEAT_CONFIGS} of them re-seeded)")
        if args.dry_run:
            print("\n[region_geometry] --dry-run: nothing dispatched.")
            return
        run_all(cells, args.results_path, args.timeout)

    report = analyze(args.results_path)
    # STAMP THE BUDGET. a_within is what the saturation rule and _load_sigma
    # both read, and it is a property of how much training a run gets -- it is
    # not portable to another TOKEN_BUDGET. Unstamped reads as stale.
    from prepare import TOKEN_BUDGET
    report["token_budget"] = int(TOKEN_BUDGET)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text = render(report)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"[region_geometry] Written to {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
