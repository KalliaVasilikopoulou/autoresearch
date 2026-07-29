"""Real, non-fabricated hyperparameter-importance signal, computed from
actual historical runs in results.tsv.

This replaces `FastXAIMethods._estimate_metric_for_param` (a mock that
returned invented numbers from hand-picked formulas) and
`Agent2XAISpecialist._estimate_hyperparameter_importance` (a heuristic
"distance from default" formula, also not derived from any measurement).

When there isn't enough historical data for a given hyperparameter, the
functions here omit it rather than guess — callers must treat a missing key
as "unknown", never substitute a fabricated default.
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from state.results_logger import COLUMNS as CURRENT_COLUMNS

# Frozen historical layout used before the 4-LR-group refactor. Its
# `learning_rate` column is known to contain corrupted values for real runs
# (e.g. `3705932325767.106`) and is deliberately NOT loaded — everything
# else in this row shape (n_layer, n_embd, val_bpb, ...) is trustworthy.
LEGACY_COLUMNS = (
    "timestamp", "run_id", "n_layer", "n_embd", "learning_rate",
    "weight_decay", "warmup_ratio", "val_bpb", "training_time",
    "peak_vram_mb", "mfu_percent", "num_params_M", "num_steps", "depth", "status",
)
LEGACY_DROP_FIELDS = {"learning_rate"}

# Hyperparameter columns worth correlating against val_bpb (excludes
# identifiers, timestamps, and outcome/runtime metrics).
HYPERPARAM_COLUMNS = (
    "n_layer", "n_embd", "n_head", "window_s_fraction",
    "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr",
    "weight_decay", "warmup_ratio", "batch_size",
)

_NUMERIC_FIELDS = set(HYPERPARAM_COLUMNS) | {
    "val_bpb", "training_time", "peak_vram_mb", "mfu_percent", "num_params_M", "num_steps",
    "holdout_val_bpb",
}


def _coerce_row(fieldnames: Sequence[str], raw_values: Sequence[str], schema: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"schema": schema}
    for name, value in zip(fieldnames, raw_values):
        if schema == "legacy" and name in LEGACY_DROP_FIELDS:
            continue
        if value == "" or value is None:
            continue
        if name in _NUMERIC_FIELDS:
            try:
                row[name] = float(value)
            except ValueError:
                continue
        else:
            row[name] = value
    return row


def load_results(paths: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Dict[str, Any]]:
    """Load one or more results.tsv-shaped files, tolerant of both the
    current 20-column schema and the frozen legacy 15-column schema (rows
    are told apart by field count, not by the file's header line, since a
    stale/mismatched header is exactly the bug this loader has to survive).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    rows: List[Dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            first = True
            for fields in reader:
                if first:
                    first = False
                    # Skip a header line only if it's actually a header
                    # (first cell isn't a timestamp).
                    if fields and fields[0] == "timestamp":
                        continue
                if len(fields) == len(CURRENT_COLUMNS):
                    rows.append(_coerce_row(CURRENT_COLUMNS, fields, "current"))
                elif len(fields) == len(LEGACY_COLUMNS):
                    rows.append(_coerce_row(LEGACY_COLUMNS, fields, "legacy"))
                # Rows matching neither known shape are skipped rather than
                # guessed at.
    return rows


def _rank(values: Sequence[float]) -> List[float]:
    """Fractional ranks (ties get the average rank), 1-indexed."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def hyperparameter_correlations(
    rows: List[Dict[str, Any]], min_n: int = 4
) -> Dict[str, Dict[str, Any]]:
    """Spearman rank correlation of each hyperparameter against val_bpb,
    across every historical run where both are present and val_bpb is
    finite. A hyperparameter is omitted (not zero-filled) when fewer than
    `min_n` comparable runs exist — the honest "we don't know yet" case.

    Returns {param: {"correlation": float, "n": int}}. Correlation sign:
    negative means higher param value -> lower (better) val_bpb.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for param in HYPERPARAM_COLUMNS:
        xs, ys = [], []
        for row in rows:
            if param not in row or "val_bpb" not in row:
                continue
            val_bpb = row["val_bpb"]
            if not math.isfinite(val_bpb):
                continue
            xs.append(row[param])
            ys.append(val_bpb)
        if len(xs) < min_n or len(set(xs)) < 2:
            continue
        results[param] = {"correlation": round(spearman(xs, ys), 6), "n": len(xs)}
    return results


def importance_from_correlations(
    correlations: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Map correlation strength (|r|) to an importance score in [0, 1] for
    the params we have enough data for. Params without enough historical
    data are simply absent — callers must not backfill them with a guess.
    """
    return {param: min(1.0, abs(info["correlation"])) for param, info in correlations.items()}


def noise_floor(rows: List[Dict[str, Any]], run_id_prefix: Optional[str] = None) -> Optional[Dict[str, float]]:
    """Mean/std of val_bpb across rows sharing identical hyperparameters
    (e.g. repeated runs from scripts/noise_floor.py) — the empirical noise
    floor sigma. Returns None if fewer than 2 finite runs are found.
    """
    finite = [row["val_bpb"] for row in rows if math.isfinite(row.get("val_bpb", float("inf")))]
    if run_id_prefix:
        finite = [
            row["val_bpb"] for row in rows
            if str(row.get("run_id", "")).startswith(run_id_prefix)
            and math.isfinite(row.get("val_bpb", float("inf")))
        ]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    variance = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
    return {"mean": mean, "std": math.sqrt(variance), "n": len(finite)}
