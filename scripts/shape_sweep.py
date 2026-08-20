"""Does SHAPE matter once SIZE is held fixed?

scripts/size_sweep.py answered "is architecture a smooth hill" for SIZE only:
it climbs a ladder where depth, width and heads all grow together, so every
rung is a different-sized model of the same shape. That leaves the other half
of the question untouched -- given a fixed parameter budget, does it matter
whether you spend it on a DEEP NARROW model or a SHALLOW WIDE one?

This is the sweep the campaign has been assuming an answer to. Agent 4 treats
(n_layer, n_embd, n_head) as a region's identity, and Agent 2's findings are
all about depth and width -- but nothing has ever measured whether the search
should care about shape at all, or only about size.

    non-embedding params  N = 12 * n_layer * n_embd^2

Hold N fixed and n_embd falls as depth rises: n_embd = sqrt(N / (12*n_layer)).
Every rung is therefore the same model size wearing a different shape, and any
difference in val_bpb is attributable to shape alone.

WHAT THE ANSWER CHANGES
  flat   -> shape is a free parameter. Regions should stop keying identity on
            it, and the search should spend its architecture budget on size.
  peaked -> there is a right aspect ratio, and every region anchored away from
            it is paying a tax no hyperparameter can refund.

The verdict is FORM-FREE, for the reason recorded in size_sweep: a poor fit to
some assumed curve means a rough surface OR the wrong curve, and it cannot
tell you which. Count direction changes among steps that clear the noise
instead -- that is what hill-climbing actually needs to know.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import size_sweep
from scripts.size_sweep import (
    OK_STATUSES,
    REAL_STEP_SIGMA,
    SEED,
    _same_config,
    _search_space,
    architecture_noise,
    already_done,
    current_token_budget,
    non_embedding_params,
    pick_anchor,
    run_all,
)
from state import surrogate
from state.results_analysis import load_results

RUN_ID_PREFIX = "shape"
DEFAULT_RESULTS_PATH = "state/shape_sweep.tsv"
REPORT_JSON = Path("state/shape_sweep.json")
REPORT_MD = Path("reports/shape_sweep.md")

#: Depths to try. The ladder is bounded by the n_embd ceiling rather than by
#: choice: at the anchor's ~123M non-embedding parameters, n_embd hits its 1024
#: cap at about 10 layers, so anything shallower cannot be built at this size.
DEPTHS = (10, 12, 15, 18, 21, 24)

#: A rung is only comparable if its actual parameter count lands near the
#: target. n_embd must be an integer multiple of n_head, so exact equality is
#: not available -- but a rung more than this far off is measuring size, which
#: is the one thing this sweep exists to hold still.
PARAM_TOLERANCE = 0.03


def build_shape_ladder(anchor: Dict[str, Any],
                       depths: Tuple[int, ...] = DEPTHS) -> List[Dict[str, Any]]:
    """Rungs of equal SIZE and differing SHAPE, everything else the anchor's.

    n_head is held at the anchor's. It is a third shape axis and it deserves
    its own sweep, but varying it here would confound two effects in one
    ladder and neither would be readable.
    """
    bounds = _search_space()
    embd_lo, embd_hi = bounds["n_embd"]
    layer_lo, layer_hi = bounds["n_layer"]

    n_head = int(anchor["n_head"])
    target = non_embedding_params(int(anchor["n_layer"]), int(anchor["n_embd"]))

    rungs: List[Dict[str, Any]] = []
    for depth in depths:
        if not (layer_lo <= depth <= layer_hi):
            continue
        exact = math.sqrt(target / (12.0 * depth))
        n_embd = surrogate.snap_n_embd(int(round(exact)), n_head)
        if not (embd_lo <= n_embd <= embd_hi):
            continue
        params = non_embedding_params(depth, n_embd)
        drift = abs(params - target) / target
        if drift > PARAM_TOLERANCE:
            continue
        hp = dict(anchor)
        hp["n_layer"], hp["n_embd"], hp["n_head"] = depth, n_embd, n_head
        rungs.append({
            "label": f"L{depth}",
            "n_layer": depth,
            "n_embd": n_embd,
            "n_head": n_head,
            "params": params,
            "param_drift": drift,
            "aspect": n_embd / depth,
            "hyperparams": hp,
        })
    return rungs


def import_existing(rungs: List[Dict[str, Any]], results_path: str,
                    history_path: str = "results.tsv") -> int:
    """Reuse any run in the campaign history that already IS one of these
    rungs -- same config, same seed, same token budget. Costs nothing and can
    only reduce what has to be trained."""
    have = already_done(results_path)
    history = load_results(history_path)
    imported = 0
    for rung in rungs:
        run_id = f"{RUN_ID_PREFIX}_{rung['label']}"
        if run_id in have:
            continue
        hp = dict(rung["hyperparams"])
        hp["seed"] = SEED
        for row in history:
            if row.get("status") in OK_STATUSES and _same_config(row, hp):
                rung["imported_from"] = row.get("run_id")
                rung["val_bpb"] = float(row["val_bpb"])
                imported += 1
                break
    return imported


def _measurements(rungs: List[Dict[str, Any]], results_path: str) -> List[Dict[str, Any]]:
    """Each rung with its val_bpb attached, in depth order, skipping any that
    never produced one."""
    by_id = {}
    for row in load_results(results_path):
        if row.get("status") in OK_STATUSES and isinstance(row.get("val_bpb"), (int, float)) \
                and math.isfinite(row["val_bpb"]):
            by_id[str(row.get("run_id"))] = float(row["val_bpb"])
    out = []
    for rung in sorted(rungs, key=lambda r: r["n_layer"]):
        val = by_id.get(f"{RUN_ID_PREFIX}_{rung['label']}", rung.get("val_bpb"))
        if val is None:
            continue
        out.append({**rung, "val_bpb": val})
    return out


def analyze(results_path: str, rungs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Is the shape surface flat, a single hill, or rough?"""
    points = _measurements(rungs, results_path)
    sigma, sigma_source = architecture_noise()
    real = REAL_STEP_SIGMA * sigma

    report: Dict[str, Any] = {
        "token_budget": current_token_budget(),
        "sigma": sigma,
        "sigma_source": sigma_source,
        "real_step": real,
        "n_points": len(points),
        "points": [
            {k: p[k] for k in ("label", "n_layer", "n_embd", "n_head",
                               "params", "param_drift", "aspect", "val_bpb")}
            for p in points
        ],
    }
    if len(points) < 3:
        report["verdict"] = "NOT ENOUGH DATA -- need at least 3 rungs measured."
        return report

    vals = [p["val_bpb"] for p in points]
    spread = max(vals) - min(vals)
    best = min(points, key=lambda p: p["val_bpb"])
    worst = max(points, key=lambda p: p["val_bpb"])

    # Steps that clear the noise, and how often they change direction. A flat
    # surface has NO readable steps; a single hill has readable steps and no
    # sign change; anything else is rough.
    steps = []
    for a, b in zip(points, points[1:]):
        delta = b["val_bpb"] - a["val_bpb"]
        steps.append({"from": a["label"], "to": b["label"], "delta": delta,
                      "readable": abs(delta) > real})
    readable = [s for s in steps if s["readable"]]
    signs = [1 if s["delta"] > 0 else -1 for s in readable]
    sign_changes = sum(1 for x, y in zip(signs, signs[1:]) if x != y)

    report.update({
        "spread": spread,
        "spread_in_sigma": spread / sigma if sigma > 0 else None,
        "best": {"label": best["label"], "n_layer": best["n_layer"],
                 "n_embd": best["n_embd"], "aspect": best["aspect"],
                 "val_bpb": best["val_bpb"]},
        "worst": {"label": worst["label"], "val_bpb": worst["val_bpb"]},
        "steps": steps,
        "n_readable_steps": len(readable),
        "sign_changes": sign_changes,
        "max_param_drift": max(p["param_drift"] for p in points),
    })

    if not readable:
        report["verdict"] = (
            f"FLAT -- no step between adjacent depths clears {real:.6f} "
            f"({REAL_STEP_SIGMA:.0f} sigma). At this budget shape is a free "
            f"parameter: the whole ladder spans {spread:.6f}, which is "
            f"{spread / sigma:.1f} sigma. Spend the architecture budget on SIZE, "
            f"and treat depth/width as interchangeable at fixed parameters.")
    elif sign_changes == 0:
        report["verdict"] = (
            f"MONOTONE -- {len(readable)} readable step(s), no change of "
            f"direction. Shape matters and the surface is climbable: best is "
            f"{best['label']} (aspect {best['aspect']:.1f}) at {best['val_bpb']:.6f}, "
            f"worst {worst['label']} at {worst['val_bpb']:.6f}. The optimum is at "
            f"the END of the ladder, so the real one may lie beyond it -- extend "
            f"before trusting the best rung as an optimum.")
    elif sign_changes == 1:
        report["verdict"] = (
            f"SINGLE HILL -- {len(readable)} readable step(s), one change of "
            f"direction. There IS a right aspect ratio: {best['aspect']:.1f} "
            f"(n_layer={best['n_layer']}, n_embd={best['n_embd']}) at "
            f"{best['val_bpb']:.6f}, against {worst['val_bpb']:.6f} at the worst "
            f"shape -- a {spread / sigma:.1f} sigma penalty for getting it wrong. "
            f"Regions anchored away from this ratio pay it.")
    else:
        report["verdict"] = (
            f"ROUGH -- {sign_changes} direction changes among {len(readable)} "
            f"readable steps. Shape is not a hill at this budget, so a search "
            f"that hill-climbs on depth/width will land wherever it started.")
    return report


def render(report: Dict[str, Any]) -> str:
    lines = [
        "# Shape at fixed size",
        "",
        f"**{report['verdict']}**",
        "",
        f"- token budget: {report['token_budget']:,}",
        f"- noise (sigma): {report['sigma']:.6f} ({report['sigma_source']})",
        f"- a step counts as real above: {report.get('real_step', 0):.6f}",
    ]
    if "max_param_drift" in report:
        lines.append(f"- largest parameter drift across rungs: "
                     f"{report['max_param_drift'] * 100:.2f}% "
                     f"(tolerance {PARAM_TOLERANCE * 100:.0f}%)")
    lines += ["", "| rung | n_layer | n_embd | params (M) | aspect | val_bpb |",
              "|---|---|---|---|---|---|"]
    for p in report.get("points", []):
        lines.append(f"| {p['label']} | {p['n_layer']} | {p['n_embd']} | "
                     f"{p['params'] / 1e6:.1f} | {p['aspect']:.1f} | {p['val_bpb']:.6f} |")
    if report.get("steps"):
        lines += ["", "| step | delta | clears noise? |", "|---|---|---|"]
        for s in report["steps"]:
            lines.append(f"| {s['from']} -> {s['to']} | {s['delta']:+.6f} | "
                         f"{'yes' if s['readable'] else 'no'} |")
    lines += ["", "Every rung holds non-embedding parameters, n_head, the seed and the "
              "token budget fixed, so any difference above is shape and nothing else.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    ap.add_argument("--history", default="results.tsv")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--analyze-only", action="store_true",
                    help="Report on whatever is already measured; train nothing.")
    args = ap.parse_args()

    history = load_results(args.history)
    anchor = pick_anchor(history)
    rungs = build_shape_ladder(anchor)
    if not rungs:
        print("[shape_sweep] No rung fits inside the architecture box at this size.")
        return

    print(f"[shape_sweep] anchor: n_layer={anchor['n_layer']} n_embd={anchor['n_embd']} "
          f"n_head={anchor['n_head']} "
          f"-> {non_embedding_params(int(anchor['n_layer']), int(anchor['n_embd'])) / 1e6:.1f}M "
          f"non-embedding params")
    for r in rungs:
        print(f"    {r['label']}: n_layer={r['n_layer']:>2} n_embd={r['n_embd']:>4} "
              f"params={r['params'] / 1e6:.1f}M drift={r['param_drift'] * 100:.2f}% "
              f"aspect={r['aspect']:.1f}")

    imported = import_existing(rungs, args.results, args.history)
    if imported:
        print(f"[shape_sweep] reused {imported} run(s) already in {args.history}")

    if not args.analyze_only:
        # The runner is size_sweep's, unchanged -- only the run-id prefix
        # differs, so the two sweeps cannot collide in results files or in
        # already_done().
        size_sweep.RUN_ID_PREFIX = RUN_ID_PREFIX
        run_all(rungs, args.results, args.timeout)

    report = analyze(args.results, rungs)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render(report), encoding="utf-8")
    print()
    print(render(report))
    print(f"[shape_sweep] wrote {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
