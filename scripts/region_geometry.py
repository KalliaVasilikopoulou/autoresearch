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


def run_all(cells: List[Dict[str, Any]], results_path: str, timeout: int) -> None:
    from agents import remote_runner
    from agents.live_progress import MultiGpuProgressDisplay

    if not remote_runner.is_remote_configured():
        raise SystemExit("[region_geometry] No remote configured.")

    client = remote_runner.open_client()
    try:
        remote_runner.kill_stale_training_processes(client=client)
        if not remote_runner.sync_remote_code(client=client):
            raise SystemExit("[region_geometry] Remote code sync failed -- nothing dispatched.")
        gpus = [g["index"] for g in remote_runner.discover_available_gpus(client=client)]
        if not gpus:
            raise SystemExit("[region_geometry] No free GPUs.")
        print(f"[region_geometry] {len(cells)} run(s) across GPUs {gpus}")

        hp_dir = Path("state/region_geometry_hyperparams")
        hp_dir.mkdir(parents=True, exist_ok=True)

        for start in range(0, len(cells), len(gpus)):
            wave = cells[start:start + len(gpus)]
            labels = [f"GPU{gpus[i]}" for i in range(len(wave))]
            print(f"\n[region_geometry] === wave {start // len(gpus) + 1} "
                  f"({len(wave)} run(s)) ===")
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
    per_config = {}
    for label, cells in by_label.items():
        vals = [v for k, v in cells.items() if isinstance(k, int)]
        if len(vals) > 1:
            per_config[label] = {
                "n_seeds": len(vals), "mean": statistics.mean(vals),
                "std": statistics.stdev(vals), "batch_size": cells.get("_batch"),
            }
    a_within = (statistics.median([c["std"] for c in per_config.values()])
                if per_config else None)

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

    # --- A-between: same configuration, neighbouring architecture ---
    a_between = {}
    base_labels = [l for l in by_label if l.startswith(f"r{RADII[1]:.2f}_c")]
    for kind in ("depth_neighbour", "width_neighbour"):
        cells = by_label.get(kind)
        if not cells:
            continue
        vals = [v for k, v in cells.items() if isinstance(k, int)]
        if len(vals) > 1:
            a_between[kind] = {"n_seeds": len(vals), "mean": statistics.mean(vals),
                               "std": statistics.stdev(vals)}

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
                  "Step 1 showed +1 layer leaves every earlier layer bit-identical while a "
                  "width change reshapes 41 of 46 tensors -- so these should NOT be equal, "
                  "and one constant cannot serve both.", "",
                  "| neighbour | seeds | mean | std |", "|---|---:|---:|---:|"]
        for kind, e in sorted(report["a_between"].items()):
            lines.append(f"| {kind} | {e['n_seeds']} | {e['mean']:.6f} | {e['std']:.6f} |")

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
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    text = render(report)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"[region_geometry] Written to {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
