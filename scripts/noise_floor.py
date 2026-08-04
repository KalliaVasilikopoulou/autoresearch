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
    parser.add_argument("--across-gpus", action="store_true", default=True,
                        help="Round-robin the repeats across all available GPUs (default). "
                             "This is the sigma the search actually needs, since a parallel "
                             "wave compares configs trained on different devices.")
    parser.add_argument("--single-gpu", dest="across_gpus", action="store_false",
                        help="Keep every repeat on one auto-selected GPU (measures only "
                             "within-device nondeterminism -- how the original n=3 sigma "
                             "was obtained).")
    parser.add_argument("--config", default="agents_config.yaml", help="Path to agents_config.yaml")
    parser.add_argument("--results-path", default="results.tsv", help="Path to results.tsv")
    args = parser.parse_args()

    agent1 = Agent1TrainingSpecialist(config_path=args.config)
    fixed_hyperparams = agent1._default_hyperparams()

    # Spread the repeats across GPUs on purpose. Auto-discovery sorts by free
    # memory, so sequential repeats otherwise all land on the same device and
    # the resulting sigma measures only bf16/kernel nondeterminism -- missing
    # the larger effect that actually contaminates the search: GPUs differ in
    # throughput (18-26% MFU spread across this server), a slower GPU fits
    # fewer steps into the fixed 300s budget, and fewer steps means a worse
    # val_bpb for an identical config. Since a parallel wave compares configs
    # trained on *different* GPUs, that is the variance the search's thresholds
    # need to be calibrated against, not the within-GPU one.
    gpus = []
    if args.across_gpus:
        try:
            from agents.remote_runner import discover_available_gpus, is_remote_configured
            if is_remote_configured():
                gpus = [g["index"] for g in discover_available_gpus()]
        except Exception as e:
            print(f"[noise_floor] Could not discover GPUs ({e}) -- falling back to auto-placement")

    print(f"[noise_floor] Fixed hyperparams: {fixed_hyperparams}")
    print(f"[noise_floor] Running {args.repeats} repeats"
          + (f" round-robin across GPUs {gpus}..." if gpus else " (auto GPU placement)..."))

    placements = {}
    for i in range(args.repeats):
        run_id = f"{RUN_ID_PREFIX}_{i:04d}"
        gpu = gpus[i % len(gpus)] if gpus else None
        placements[run_id] = gpu
        print(f"\n[noise_floor] --- Run {i + 1}/{args.repeats} ({run_id})"
              + (f" on GPU {gpu}" if gpu is not None else "") + " ---")
        metrics = agent1.train_model(fixed_hyperparams, dry_run=False, iteration=i, gpu_index=gpu)
        log_result(run_id, fixed_hyperparams, metrics, results_path=args.results_path)

    rows = load_results(args.results_path)
    stats = noise_floor(rows, run_id_prefix=RUN_ID_PREFIX)

    print()
    if stats is None:
        print("[noise_floor] Fewer than 2 finite val_bpb runs completed — cannot compute sigma.")
        sys.exit(1)

    print(f"[noise_floor] val_bpb: mean={stats['mean']:.6f} std={stats['std']:.6f} (n={stats['n']})")
    print("[noise_floor] Treat any future results.tsv delta smaller than ~std as noise, not signal.")

    # Decomposition: how much of that sigma is the device rather than the run?
    # Reported rather than folded into one number, because the two mean
    # different things -- within-GPU sigma is the floor for repeating a run on
    # the same device, across-GPU sigma is the floor for comparing two configs
    # in a parallel wave, which is what every search threshold actually does.
    if placements and any(v is not None for v in placements.values()):
        import statistics as _st
        from collections import defaultdict
        session = {r["run_id"]: r for r in rows if r.get("run_id") in placements}
        by_gpu = defaultdict(list)
        for run_id, gpu in placements.items():
            row = session.get(run_id)
            if row and isinstance(row.get("val_bpb"), (int, float)):
                by_gpu[gpu].append((row["val_bpb"], row.get("num_steps")))
        print("\n[noise_floor] Per-GPU breakdown (this session):")
        gpu_means, within = [], []
        for gpu, vals in sorted(by_gpu.items()):
            bpbs = [v for v, _ in vals]
            steps = [s for _, s in vals if isinstance(s, (int, float))]
            gpu_means.append(_st.mean(bpbs))
            if len(bpbs) > 1:
                within.append(_st.stdev(bpbs))
            step_txt = f"  mean_steps={_st.mean(steps):.0f}" if steps else ""
            print(f"    GPU {gpu}: n={len(bpbs)} mean_val_bpb={_st.mean(bpbs):.6f}{step_txt}")
        if len(gpu_means) > 1:
            across = _st.stdev(gpu_means)
            print(f"\n[noise_floor]   within-GPU sigma  ~= {(_st.mean(within) if within else float('nan')):.6f}"
                  f"   (repeating a run on the same device)")
            print(f"[noise_floor]   across-GPU sigma  ~= {across:.6f}"
                  f"   (comparing configs trained on different devices)")
            print(f"[noise_floor]   -> calibrate search thresholds against the LARGER of these; "
                  f"a parallel wave is always the across-GPU case.")

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
