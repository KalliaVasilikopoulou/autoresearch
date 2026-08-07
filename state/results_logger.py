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
    # >0 means the run hit the wall-clock safety cap before consuming its
    # full TOKEN_BUDGET -- an incomplete measurement, not a worse config.
    "budget_shortfall_pct",
    "device",
    # Which region (state/regions.py) dispatched this run. Blank for runs
    # that predate multi-region search, and for the noise-floor script.
    #
    # This has to live HERE, not only in state/regions.json, because
    # results.tsv is the source of truth every consumer reads
    # (load_results -> the surrogate, the landscape, Agent 2's elite
    # reference, Agent 3's recommendations). The registry's own run_ids list
    # is a convenience index, and RegionRegistry.load deliberately tolerates
    # a corrupt file by starting empty -- if attribution existed only there,
    # one bad write would permanently orphan every historical run while
    # results.tsv sat intact beside it. With the column, the registry can
    # always be rebuilt from the results.
    "region_id",
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
    # The initial-weight seed this run actually used (train.py's SEED, echoed
    # back in its summary block and parsed by Agent 1).
    #
    # A RECORDED NUISANCE VARIABLE, not a search dimension -- deliberately
    # absent from state/results_analysis.py's HYPERPARAM_COLUMNS so the
    # surrogate never treats it as something to optimize. It is logged because
    # every run before this column existed shared one hardcoded seed, which
    # made seed-to-seed spread unmeasurable after the fact; that history cannot
    # be recovered retroactively, only recorded from here on.
    "seed",
]

# Frozen superseded layouts, newest first. A file written under one of these is
# MIGRATED IN PLACE (header rewritten, rows padded) rather than parked in
# legacy_results.tsv/, because every schema change so far has only appended
# columns and parking a file loses its history from every consumer -- they all
# read results.tsv, never the archive. That cost is real: 32 runs went into an
# archive this way and the campaign silently restarted its Sobol cold start
# from zero history.
#
# Written out literally, never as a slice of COLUMNS: a slice would re-point at
# a different layout on the next schema change, which is the exact failure this
# exists to prevent.
PRE_SEED_COLUMNS = (
    "timestamp", "run_id", "n_layer", "n_embd", "n_head",
    "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr",
    "weight_decay", "warmup_ratio", "batch_size", "val_bpb", "training_time",
    "peak_vram_mb", "mfu_percent", "num_params_M", "num_steps", "depth",
    "status", "holdout_val_bpb", "budget_shortfall_pct", "device",
    "region_id", "window_s_fraction",
)

SUPERSEDED_SCHEMAS = (PRE_SEED_COLUMNS,)

# Karpathy's published baseline for DEPTH=8 on the same dataset/budget.
# Update this if you find a more precise figure.
KARPATHY_BASELINE_VAL_BPB = None  # fill in once you have it


def _migrate_appended_columns(path: Path, old_columns: tuple) -> None:
    """Rewrite a file written under `old_columns` (a prefix of COLUMNS) with the
    current header, padding every row with empty values for the new trailing
    columns. Blank is the honest value: those runs genuinely have no measurement
    for a column that did not exist, and _coerce_row omits blanks rather than
    fabricating a default -- so a pre-seed run reports no seed at all instead of
    claiming the seed that happened to be the default at the time.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    width = len(COLUMNS)
    body = [row + [""] * (width - len(row)) for row in rows[1:] if row]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(COLUMNS)
        writer.writerows(body)
    added = [c for c in COLUMNS if c not in old_columns]
    print(f"[ResultsLogger] {path} used a superseded schema — migrated {len(body)} row(s) "
          f"in place, adding blank {added}")


def _ensure_current_schema(path: Path) -> None:
    """Guard against silently appending rows under a stale/mismatched header
    (this happened for real: a prior schema change left 59 rows under one
    header and 23 rows under a column count the header didn't match, making
    the file unparseable as a table).

    A header matching a known superseded layout is migrated in place, keeping
    the history. Anything else is moved aside — nothing is lost, but it stops
    being visible to load_results — so future appends start clean.
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

    for old_columns in SUPERSEDED_SCHEMAS:
        # Only an append is migratable. A layout that reordered or removed a
        # column cannot be padded into the current one, and guessing would
        # put values under the wrong headers — those still get parked.
        if first_line == "\t".join(old_columns) and tuple(COLUMNS[:len(old_columns)]) == old_columns:
            _migrate_appended_columns(path, old_columns)
            return
    # Into legacy_results.tsv/ beside results.tsv, which is where these have
    # always been kept by hand and is already gitignored. Written next to
    # results.tsv itself, the file matched no ignore rule (".gitignore" lists
    # "results.tsv", which does not cover "results.tsv.legacy-*") and showed
    # up as untracked junk -- guaranteed for anyone carrying an older file
    # across the schema change that added region_id.
    legacy_dir = path.parent / "legacy_results.tsv"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    backup = legacy_dir / f"{path.name}.legacy-{time.strftime('%Y%m%d%H%M%S')}"
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
        "budget_shortfall_pct": metrics.get("budget_shortfall_pct", ""),
        # Multi-GPU parallel search: which remote GPU index produced this
        # run (see agents/remote_runner.py::run_training_remote). Blank for
        # local/dry-run/simulated runs, which have no such concept. Logged
        # for observability now, not used for filtering -- the DGX's GPUs
        # are homogeneous.
        "device": metrics.get("device", ""),
        # Read from metrics first (the orchestrator knows which region's slot
        # produced this run) and hyperparams second, so a caller that carries
        # it either way still records it. Never inferred from proximity to a
        # region anchor -- see RegionRegistry.assign_run for why attribution
        # must follow the budget, not the geometry.
        "region_id": metrics.get("region_id", hyperparams.get("region_id", "")),
        "window_s_fraction": hyperparams.get("window_s_fraction", ""),
        # metrics first, hyperparams second -- same rule as `depth` above. The
        # metrics value is what train.py reported it actually seeded with; the
        # hyperparams value is only what was requested. They diverge exactly
        # when the hyperparams file failed to load and train.py fell back to
        # its default, which is the case this ordering is here to catch.
        "seed": metrics.get("seed", hyperparams.get("seed", "")),
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
