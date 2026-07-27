"""Read-only report: fit the Tier 1 surrogate over results.tsv and print an
S_perf-ranked, noise-floor-pruned coordinate-view table.

This does NOT feed into Agent 1's decisions yet (that's a later stage) --
it's a standalone diagnostic so the surrogate's output can be sanity-checked
against real (or synthetic) data before anything is wired into the search
loop itself.

Usage:
    uv run python scripts/surrogate_report.py
    uv run python scripts/surrogate_report.py --results-path path/to/results.tsv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from state.results_analysis import HYPERPARAM_COLUMNS, load_results
from state.surrogate import (
    MIN_SURROGATE_N,
    SURROGATE_DEPS_AVAILABLE,
    fit_surrogate,
    prune_by_noise_floor,
    rank_by_sensitivity,
)

DEFAULT_NOISE_FLOOR_PATH = Path("state/noise_floor.json")
DEFAULT_SIGMA = 0.01  # conservative fallback if state/noise_floor.json is absent


def _load_sigma(path: Path) -> float:
    if not path.exists():
        print(f"[surrogate_report] WARNING: {path} not found — using conservative "
              f"fallback sigma={DEFAULT_SIGMA}. Run scripts/noise_floor.py to measure it for real.")
        return DEFAULT_SIGMA
    data = json.loads(path.read_text())
    return float(data["std"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("--noise-floor-path", default=str(DEFAULT_NOISE_FLOOR_PATH))
    parser.add_argument("--prune-k", type=float, default=2.0, help="Freeze params below k*sigma total effect")
    args = parser.parse_args()

    if not SURROGATE_DEPS_AVAILABLE:
        print("[surrogate_report] scipy/scikit-learn not installed — cannot fit a surrogate.")
        sys.exit(1)

    rows = load_results(args.results_path)
    surrogate = fit_surrogate(rows)
    if surrogate is None:
        n_usable = sum(
            1 for r in rows
            if "val_bpb" in r and all(c in r for c in HYPERPARAM_COLUMNS)
        )
        print(f"[surrogate_report] Not enough comparable data yet: {n_usable} usable rows "
              f"in {args.results_path}, need >= {MIN_SURROGATE_N}.")
        print("[surrogate_report] Nothing fabricated — this is the honest 'don't know yet' state.")
        sys.exit(0)

    print(f"[surrogate_report] Fit on {surrogate.n_train} rows from {args.results_path}")

    sigma = _load_sigma(Path(args.noise_floor_path))
    print(f"[surrogate_report] Using noise floor sigma={sigma:.6f}, prune threshold={args.prune_k}*sigma="
          f"{args.prune_k * sigma:.6f}")

    center = {name: sum(b) / 2.0 for name, b in surrogate.bounds.items()}
    # Center the slice at the best observed point, not the range midpoint,
    # when we have one -- sensitivity "near current best" is what the plan
    # asks for, and the midpoint is meaningless once data is asymmetric.
    finite_rows = [r for r in rows if "val_bpb" in r]
    if finite_rows:
        best_row = min(finite_rows, key=lambda r: r["val_bpb"])
        for name in surrogate.feature_names:
            if name in best_row:
                center[name] = best_row[name]

    ranked = rank_by_sensitivity(surrogate, list(surrogate.feature_names), center, surrogate.bounds)
    kept, frozen = prune_by_noise_floor(
        surrogate, list(surrogate.feature_names), center, surrogate.bounds, sigma, k=args.prune_k
    )
    frozen_set = set(frozen)

    print()
    print(f"{'parameter':<16} {'S_perf':>10}  {'status':<10} {'bounds'}")
    print("-" * 70)
    for param, score in ranked:
        status = "frozen" if param in frozen_set else "active"
        lo, hi = surrogate.bounds[param]
        print(f"{param:<16} {score:>10.6f}  {status:<10} [{lo:.4g}, {hi:.4g}]")

    print()
    print(f"[surrogate_report] {len(kept)} active, {len(frozen)} frozen (S_perf < {args.prune_k}*sigma)")
    print("[surrogate_report] S_behav column not available yet (needs Tier 2's behavioral fingerprint).")


if __name__ == "__main__":
    main()
