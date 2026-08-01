"""Append every completed training run to results.tsv for easy comparison."""

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict

# Columns written for every run
COLUMNS = [
    "timestamp",
    "run_id",
    "n_layer",
    "n_embd",
    "n_head",
    "embedding_lr",
    "unembedding_lr",
    "matrix_lr",
    "scalar_lr",
    "weight_decay",
    "warmup_ratio",
    "batch_size",
    "val_bpb",
    "training_time",
    "peak_vram_mb",
    "mfu_percent",
    "num_params_M",
    "num_steps",
    "depth",
    "status",
    "holdout_val_bpb",
    "device",
    # Tier 4 window-pattern tunable (see agents/agent1_training_specialist.py's
    # SEARCH_SPACE/ARCH_SAFE_RANGES) -- was part of HYPERPARAM_COLUMNS
    # (state/results_analysis.py) and proposed/tuned as a real search
    # dimension the whole time, but never actually logged here. That silent
    # gap meant search_planner.propose_next()'s n_usable count (rows with
    # every HYPERPARAM_COLUMNS field present) was 0 for every historical
    # row, forever -- the cold-start check (n_usable < cold_start_n) never
    # passed, so the EI-guided surrogate search never activated even once;
    # every proposal was Sobol random sampling mislabeled "surrogate".
    "window_s_fraction",
]

# Karpathy's published baseline for DEPTH=8 on the same dataset/budget.
# Update this if you find a more precise figure.
KARPATHY_BASELINE_VAL_BPB = None  # fill in once you have it


def _ensure_current_schema(path: Path) -> None:
    """Guard against silently appending rows under a stale/mismatched header
    (this happened for real: a prior schema change left 59 rows under one
    header and 23 rows under a column count the header didn't match, making
    the file unparseable as a table). If the file's first line doesn't match
    COLUMNS, move it aside — nothing is lost — so future appends start clean.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, newline="") as f:
        # csv.writer terminates lines with \r\n regardless of platform —
        # strip both, not just \n, or this never matches even a header this
        # same code just wrote (and re-triggers the rename below every call).
        first_line = f.readline().rstrip("\r\n")
    expected = "\t".join(COLUMNS)
    if first_line == expected:
        return
    backup = path.with_name(f"{path.name}.legacy-{time.strftime('%Y%m%d%H%M%S')}")
    if backup.exists():
        backup = backup.with_name(f"{backup.name}-{os.getpid()}")
    path.rename(backup)
    print(f"[ResultsLogger] {path} had a stale/mismatched header — moved to {backup}, starting fresh")


def log_result(
    run_id: str,
    hyperparams: Dict[str, Any],
    metrics: Dict[str, Any],
    results_path: str = "results.tsv",
) -> None:
    """Append one row to results.tsv."""
    path = Path(results_path)
    _ensure_current_schema(path)
    write_header = not path.exists() or path.stat().st_size == 0

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "n_layer": hyperparams.get("n_layer", ""),
        "n_embd": hyperparams.get("n_embd", ""),
        "n_head": hyperparams.get("n_head", ""),
        "embedding_lr": hyperparams.get("embedding_lr", ""),
        "unembedding_lr": hyperparams.get("unembedding_lr", ""),
        "matrix_lr": hyperparams.get("matrix_lr", ""),
        "scalar_lr": hyperparams.get("scalar_lr", ""),
        "weight_decay": hyperparams.get("weight_decay", ""),
        "warmup_ratio": hyperparams.get("warmup_ratio", ""),
        "batch_size": hyperparams.get("batch_size", ""),
        "val_bpb": metrics.get("val_bpb", ""),
        "training_time": metrics.get("training_time", ""),
        "peak_vram_mb": metrics.get("peak_vram_mb", ""),
        "mfu_percent": metrics.get("mfu_percent", ""),
        "num_params_M": metrics.get("num_params_M", ""),
        "num_steps": metrics.get("num_steps", ""),
        "depth": metrics.get("depth", hyperparams.get("n_layer", "")),
        "status": metrics.get("status", ""),
        "holdout_val_bpb": metrics.get("holdout_val_bpb", ""),
        # Multi-GPU parallel search: which remote GPU index produced this
        # run (see agents/remote_runner.py::run_training_remote). Blank for
        # local/dry-run/simulated runs, which have no such concept. Logged
        # for observability now, not used for filtering -- the DGX's GPUs
        # are homogeneous.
        "device": metrics.get("device", ""),
        "window_s_fraction": hyperparams.get("window_s_fraction", ""),
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    val_bpb = metrics.get("val_bpb", float("inf"))
    status = metrics.get("status", "unknown")
    training_time = metrics.get("training_time", 0)
    print(f"[ResultsLogger] Logged {run_id}: val_bpb={val_bpb:.6f}, status={status}, time={training_time:.1f}s")
    print(f"[ResultsLogger]   Hyperparams: n_layer={hyperparams.get('n_layer')}, n_embd={hyperparams.get('n_embd')}, "
          f"matrix_lr={float(hyperparams.get('matrix_lr', 0) or 0):.2e}")
    if KARPATHY_BASELINE_VAL_BPB is not None:
        delta = val_bpb - KARPATHY_BASELINE_VAL_BPB
        sign = "+" if delta >= 0 else ""
        print(f"[ResultsLogger] vs Karpathy baseline: {sign}{delta:.6f}")
