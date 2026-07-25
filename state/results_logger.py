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
]

# Karpathy's published baseline for DEPTH=8 on the same dataset/budget.
# Update this if you find a more precise figure.
KARPATHY_BASELINE_VAL_BPB = None  # fill in once you have it


def log_result(
    run_id: str,
    hyperparams: Dict[str, Any],
    metrics: Dict[str, Any],
    results_path: str = "results.tsv",
) -> None:
    """Append one row to results.tsv."""
    path = Path(results_path)
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
