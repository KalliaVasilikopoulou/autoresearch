"""Measure the empirical noise floor: run one fixed config N times and
report mean/std of val_bpb (and other metrics) across the repeats.

Without this, there's no way to tell whether a delta between two rows in
results.tsv reflects a real hyperparameter effect or just run-to-run noise
(dataloader position, bf16 nondeterminism, contention on the shared GPU).
Any future delta smaller than roughly this std is not a signal Agent 1
should act on.

Usage:
    uv run python scripts/noise_floor.py --repeats 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from state.results_analysis import load_results, noise_floor
from state.results_logger import log_result

RUN_ID_PREFIX = "noise_floor"
NOISE_FLOOR_JSON_PATH = Path("state/noise_floor.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated runs")
    parser.add_argument("--config", default="agents_config.yaml", help="Path to agents_config.yaml")
    parser.add_argument("--results-path", default="results.tsv", help="Path to results.tsv")
    args = parser.parse_args()

    agent1 = Agent1TrainingSpecialist(config_path=args.config)
    fixed_hyperparams = agent1._default_hyperparams()

    print(f"[noise_floor] Fixed hyperparams: {fixed_hyperparams}")
    print(f"[noise_floor] Running {args.repeats} repeats...")

    for i in range(args.repeats):
        run_id = f"{RUN_ID_PREFIX}_{i:04d}"
        print(f"\n[noise_floor] --- Run {i + 1}/{args.repeats} ({run_id}) ---")
        # train_model saves self.current_hyperparams (not its argument) to
        # model_hyperparams.yaml, so this must be set explicitly for every
        # repeat — see Agent1TrainingSpecialist.train_model.
        agent1.current_hyperparams = dict(fixed_hyperparams)
        metrics = agent1.train_model(fixed_hyperparams, dry_run=False, iteration=i)
        log_result(run_id, fixed_hyperparams, metrics, results_path=args.results_path)

    rows = load_results(args.results_path)
    stats = noise_floor(rows, run_id_prefix=RUN_ID_PREFIX)

    print()
    if stats is None:
        print("[noise_floor] Fewer than 2 finite val_bpb runs completed — cannot compute sigma.")
        sys.exit(1)

    print(f"[noise_floor] val_bpb: mean={stats['mean']:.6f} std={stats['std']:.6f} (n={stats['n']})")
    print("[noise_floor] Treat any future results.tsv delta smaller than ~std as noise, not signal.")

    # Top-level mean/std/n/hyperparams stay the LATEST measurement (unchanged
    # shape -- _load_sigma in agents/search_planner.py and
    # scripts/surrogate_report.py both read state["std"] directly and must
    # keep working untouched). "history" is additive: every measurement ever
    # taken, oldest first, for the noise-floor trend chart -- previously this
    # file just got silently overwritten each run, discarding that history.
    history = []
    if NOISE_FLOOR_JSON_PATH.exists():
        try:
            history = json.loads(NOISE_FLOOR_JSON_PATH.read_text()).get("history", [])
        except (json.JSONDecodeError, OSError):
            history = []
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mean": stats["mean"],
        "std": stats["std"],
        "n": stats["n"],
        "hyperparams": fixed_hyperparams,
    }
    history.append(entry)

    NOISE_FLOOR_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOISE_FLOOR_JSON_PATH.write_text(json.dumps({**entry, "history": history}, indent=2))
    print(f"[noise_floor] Persisted to {NOISE_FLOOR_JSON_PATH} for programmatic use (e.g. surrogate pruning). "
          f"History now has {len(history)} measurement(s).")


if __name__ == "__main__":
    main()
