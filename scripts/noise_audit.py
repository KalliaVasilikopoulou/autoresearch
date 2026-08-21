"""What IS the measurement noise, pooled over every repeat ever paid for?

Every abandon/retire/tie rule in this codebase compares a difference against a
noise figure, and those figures kept being medians of two or three
configurations. That is a small sample of a quantity that turns out to vary
17x, so successive estimates disagreed by 3x and each one moved a threshold.

This reads EVERY results file, groups rows by exact configuration (architecture
+ the 8 tunables), keeps groups with enough repeats to support a spread, and
reports the distribution rather than a single number.

WHY A DISTRIBUTION. There is no such thing as "the" noise here:

    quietest config   0.000724
    median            0.002941
    noisiest config   0.012486

A single global sigma is therefore wrong somewhere by construction -- too loose
for a quiet config (freezing parameters that still matter) and too tight for a
noisy one (calling coin flips real). That is the argument for the region-local
estimators (state.regions.local_noise) rather than for picking a better
constant.

Note repeats here are NOT restricted to differing seeds. Three runs of one
config at the SAME seed on the SAME GPU spread 0.0070, so training is not
reproducible given the seed and a repeat is a repeat however it was obtained.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state.results_analysis import (
    ARCHITECTURE_COLUMNS, TUNABLE_COLUMNS, at_current_budget, load_results,
)

REPORT_JSON = Path("state/noise_audit.json")
REPORT_MD = Path("reports/noise_audit.md")

#: Every file that has ever recorded a real run at some budget.
SOURCES = (
    "results.tsv",
    "state/shape_sweep.tsv",
    "state/shape_confirm.tsv",
    "state/seed_variance_4M.tsv",
    "state/region_geometry_4M.tsv",
    "state/size_sweep.tsv",
)

OK_STATUSES = {"remote_ok", "ok"}

#: Below this a standard deviation is mostly noise about the noise.
MIN_REPEATS = 3

CONFIG_COLUMNS = list(ARCHITECTURE_COLUMNS) + list(TUNABLE_COLUMNS)


def _config_key(row: Dict[str, Any]):
    try:
        return tuple(round(float(row[c]), 9) for c in CONFIG_COLUMNS)
    except (KeyError, TypeError, ValueError):
        return None


def collect(sources=SOURCES, min_repeats: int = MIN_REPEATS) -> List[Dict[str, Any]]:
    """Every configuration measured at least `min_repeats` times at the budget
    in force, with its spread."""
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for src in sources:
        if not Path(src).exists():
            continue
        for row in at_current_budget(load_results(src)):
            if row.get("status") not in OK_STATUSES:
                continue
            val = row.get("val_bpb")
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                continue
            key = _config_key(row)
            if key is not None:
                groups.setdefault(key, []).append({"val_bpb": float(val), "source": src,
                                                   "run_id": str(row.get("run_id")),
                                                   "n_layer": row.get("n_layer"),
                                                   "n_embd": row.get("n_embd")})
    out = []
    for key, rows in groups.items():
        if len(rows) < min_repeats:
            continue
        vals = [r["val_bpb"] for r in rows]
        out.append({
            "n_repeats": len(vals),
            "n_layer": rows[0]["n_layer"],
            "n_embd": rows[0]["n_embd"],
            "mean": statistics.mean(vals),
            "sd": statistics.stdev(vals),
            "range": max(vals) - min(vals),
            "run_ids": sorted(r["run_id"] for r in rows),
        })
    return sorted(out, key=lambda d: -d["n_repeats"])


def summarise(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from prepare import TOKEN_BUDGET

    sds = sorted(c["sd"] for c in configs)
    if not sds:
        return {"token_budget": TOKEN_BUDGET, "n_configs": 0}
    # Pooled: the sqrt of the MEAN VARIANCE, which is what a "typical" spread
    # is when spreads differ. The median is what a typical CONFIG has. They
    # answer different questions and both are reported, because quoting one as
    # if it were the other is how the 3x disagreement happened.
    pooled = math.sqrt(sum(s * s for s in sds) / len(sds))
    return {
        "token_budget": TOKEN_BUDGET,
        "n_configs": len(sds),
        "n_runs": sum(c["n_repeats"] for c in configs),
        "sd_min": sds[0],
        "sd_median": statistics.median(sds),
        "sd_max": sds[-1],
        "sd_ratio": sds[-1] / sds[0] if sds[0] > 0 else None,
        "sd_pooled": pooled,
        # What a comparison can actually support, ~2 standard errors on a
        # difference of two k-repeat means: 2 * sqrt(2) * sigma / sqrt(k).
        "resolvable_gap_at_k": {
            str(k): 2.0 * math.sqrt(2.0) * pooled / math.sqrt(k) for k in (1, 2, 3, 5, 10)
        },
    }


def render(summary: Dict[str, Any], configs: List[Dict[str, Any]]) -> str:
    if not configs:
        return "# Noise audit\n\nNo configuration has enough repeats to measure a spread.\n"
    lines = [
        "# Noise audit -- pooled over every repeat measured at this budget",
        "",
        f"- token budget: {summary['token_budget']:,}",
        f"- configurations with >= {MIN_REPEATS} repeats: {summary['n_configs']}"
        f" ({summary['n_runs']} runs)",
        "",
        "## There is no single noise figure",
        "",
        f"| quietest config | {summary['sd_min']:.6f} |",
        "|---|---|",
        f"| median config | **{summary['sd_median']:.6f}** |",
        f"| noisiest config | {summary['sd_max']:.6f} |",
        f"| ratio | **{summary['sd_ratio']:.1f}x** |",
        f"| pooled (sqrt mean variance) | **{summary['sd_pooled']:.6f}** |",
        "",
        "A global threshold is therefore wrong somewhere by construction: too",
        "loose for a quiet configuration, too tight for a noisy one. This is the",
        "argument for region-local estimators, not for a better constant.",
        "",
        "## Smallest difference a comparison can support",
        "",
        "| repeats of EACH config | resolvable gap |",
        "|---|---|",
    ]
    for k, gap in summary["resolvable_gap_at_k"].items():
        lines.append(f"| {k} | {gap:.6f} |")
    lines += ["", "## Per configuration", "",
              "| repeats | n_layer | n_embd | mean | sd | range |", "|---|---|---|---|---|---|"]
    for c in configs:
        lines.append(f"| {c['n_repeats']} | {c['n_layer']} | {c['n_embd']} | "
                     f"{c['mean']:.6f} | {c['sd']:.6f} | {c['range']:.6f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-repeats", type=int, default=MIN_REPEATS)
    args = parser.parse_args()

    configs = collect(min_repeats=args.min_repeats)
    summary = summarise(configs)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps({**summary, "configs": configs}, indent=2),
                           encoding="utf-8")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(render(summary, configs), encoding="utf-8")
    print(render(summary, configs))
    print(f"[noise_audit] wrote {REPORT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
