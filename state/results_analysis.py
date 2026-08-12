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
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from state.results_logger import COLUMNS as CURRENT_COLUMNS
from state.results_logger import PRE_SEED_COLUMNS, PRE_TIME_BREAKDOWN_COLUMNS

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

# PRE_SEED_COLUMNS (25 columns, before the seed addition) and
# PRE_TIME_BREAKDOWN_COLUMNS (26, before startup_seconds/eval_seconds) are the
# frozen superseded layouts. Rows are told apart here by FIELD COUNT, so every
# superseded width needs a frozen tuple or its rows silently stop parsing the
# moment COLUMNS grows -- they match neither the new width nor LEGACY_COLUMNS
# and get skipped, which looks identical to having no history at all.
#
# THAT IS NOT HYPOTHETICAL. Adding the two timing columns without touching this
# dispatch took results.tsv, region_geometry.tsv, seed_variance.tsv and
# size_sweep.tsv to ZERO rows each in one edit -- every measurement this
# project has made, invisible, with no error raised anywhere. In-place header
# migration does not save you: it only runs when a file is WRITTEN, and these
# experiment files are only ever read.
#
# They live in results_logger because that module also needs them to migrate a
# file in place; defining them in both would be two things to keep in sync.

# Statuses whose val_bpb is a synthetic placeholder, not a measured result:
# "dry_run" (agents/agent1_training_specialist.py::train_model) returns
# val_bpb = 1.0 - 0.001*(iteration+1) -- a fixed formula of iteration index
# alone, unrelated to the hyperparameters it's logged against -- and
# "simulated" (._simulate_training_result, the local-fallback-of-last-resort
# path) returns a hand-tuned formula meant to look plausible for local
# testing, not a real measurement either. Both get logged to results.tsv
# like any other run (so status-distribution/count reporting still sees
# them), but every numeric consumer of load_results() below (hyperparameter
# correlations, the Tier 1 surrogate, noise floor, holdout/dashboard
# analysis) must never mix these in with real remote_ok/remote_error rows --
# doing so was silently injecting iteration-correlated noise into every
# hyperparameter's importance score and the surrogate model's fit.
SYNTHETIC_STATUSES = frozenset({"dry_run", "simulated"})

# Hyperparameter columns worth correlating against val_bpb (excludes
# identifiers, timestamps, and outcome/runtime metrics).
#
# `seed` is deliberately NOT here even though results.tsv logs it. This tuple
# is the surrogate's feature set and the search space's coordinate system
# (state/surrogate.py, state/regions.py both key off it), so anything listed
# becomes something the search optimizes. The seed is a nuisance variable to
# average over, not a dimension to maximize along -- searching it would just
# find the luckiest initialization.
HYPERPARAM_COLUMNS = (
    "n_layer", "n_embd", "n_head", "window_s_fraction",
    "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr",
    "weight_decay", "warmup_ratio", "batch_size",
)

# The two halves of HYPERPARAM_COLUMNS, split by WHO OWNS THEM and, underneath
# that, by whether they touch the initial weights.
#
# Verified 2026-08-09 (scripts/verify_shared_init.py): the RNG draw order in
# GPT.init_weights consumes randomness proportional to vocab_size, n_embd and
# n_layer alone. n_head only reshapes `ve_gate`, which is zero-initialized. So
# two configurations agreeing on these three start from BIT-IDENTICAL weights
# (46/46 tensors, across two opposite extreme corners of everything below).
#
# That is what makes a region's comparisons paired: fix the architecture and
# the initialization cancels, so a difference between two configurations inside
# one region is the configuration, not the luck of the draw.
ARCHITECTURE_COLUMNS = ("n_layer", "n_embd", "n_head")

# Everything Agent 1 may vary within a region. None of these touches a single
# weight at initialization -- confirmed by probing the real train.py with the
# learning rates spanning their full ranges, batch_size from 2048 to 32768,
# weight_decay 0 to 0.48, warmup 0 to 0.19 and window_s_fraction 0.05 to 0.95.
TUNABLE_COLUMNS = tuple(c for c in HYPERPARAM_COLUMNS if c not in ARCHITECTURE_COLUMNS)

_NUMERIC_FIELDS = set(HYPERPARAM_COLUMNS) | {
    "val_bpb", "training_time", "peak_vram_mb", "mfu_percent", "num_params_M", "num_steps",
    "holdout_val_bpb",
    # Coerced to a number so analysis can group by it, without being a feature.
    "seed",
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


#: DEVICE_BATCH_SIZE * MAX_SEQ_LEN in train.py -- the granularity a requested
#: batch_size is truncated to, since grad_accum_steps is an integer division.
TOKENS_PER_FWDBWD = 2048


def tokens_seen(row: Dict[str, Any]) -> Optional[float]:
    """How many training tokens this run actually consumed, or None if the row
    cannot say.

    DERIVED from num_steps and batch_size rather than read from a column,
    because `total_tokens_M` is parsed out of train.py's summary and then
    thrown away -- it has never been a results.tsv column. Deriving it works
    RETROACTIVELY on every run ever recorded, which a new column could not:
    the whole point is to tell runs from different token budgets apart, and
    the runs that need telling apart are the ones already on disk.

    The identity is `num_steps * (batch_size // 2048) * 2048`, verified exact
    on 31/31 historical runs: train.py computes grad_accum_steps by integer
    division, so a requested batch_size is truncated to a multiple of
    DEVICE_BATCH_SIZE * MAX_SEQ_LEN before any token is seen.
    """
    steps, batch = row.get("num_steps"), row.get("batch_size")
    if not isinstance(steps, (int, float)) or not isinstance(batch, (int, float)):
        return None
    snapped = (int(batch) // TOKENS_PER_FWDBWD) * TOKENS_PER_FWDBWD
    if snapped <= 0 or steps <= 0:
        return None
    return float(int(steps) * snapped)


def same_token_budget(row: Dict[str, Any], budget: float, rel_tol: float = 0.02) -> bool:
    """Did this run see the same amount of training as `budget` asks for?

    A row that cannot say is NOT a match. Absent is not "probably fine": the
    reason to ask at all is that mixing budgets is invisible in the numbers --
    when TOKEN_BUDGET went 12.5M -> 4.19M the same configuration moved 1.2486
    -> 1.7063, which reads as a spectacular result rather than as an error.
    """
    seen = tokens_seen(row)
    return seen is not None and math.isclose(seen, budget, rel_tol=rel_tol)


def at_current_budget(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`rows` restricted to runs trained under the token budget in force.

    ANY MODEL FITTED ACROSS BUDGETS IS FITTED ON TWO DIFFERENT TASKS. The same
    configuration scored 1.2486 at 12.5M tokens and 1.7063 at 4.19M, so mixing
    them puts a 0.45 bpb step into the data that no hyperparameter caused. A
    surrogate would spend its capacity explaining that step, and every
    comparison drawn from it -- which configuration is better, where the
    frontier is, which parameters matter -- would be about the budget instead.

    Filtering here rather than discarding the old runs keeps them available to
    anything that asks a budget-aware question, and keeps the file as the
    campaign's real history.
    """
    from prepare import TOKEN_BUDGET

    return [r for r in rows if same_token_budget(r, TOKEN_BUDGET)]


def report_at_budget(path: Union[str, Path], budget: float) -> Optional[Dict[str, Any]]:
    """A measurement report's contents, or None if it was not measured under
    `budget` tokens of training.

    A NOISE FLOOR IS NOT PORTABLE ACROSS BUDGETS. sigma_seed went 0.00197 ->
    0.003215 when TOKEN_BUDGET went 12.5M -> 4.19M, because less training
    leaves a run further from convergence and so more dependent on its initial
    weights. Every threshold sized from a stale floor is then wrong in the SAME
    direction -- it understates the noise, so differences that cannot be
    resolved get treated as real, parameters below the noise stay in the
    search, and regions that are finished keep being searched. All three waste
    runs.

    A report with no `token_budget` key is treated as stale, not as current.
    Every report on disk when this stamp was introduced was measured at 12.5M,
    so "unstamped" is positive evidence of age rather than an absence of it.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    stamped = data.get("token_budget")
    if not isinstance(stamped, (int, float)):
        return None
    return data if math.isclose(float(stamped), float(budget), rel_tol=0.02) else None


def holdout_drift(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """How far the pinned validation shard has drifted from the untouched
    holdout shard, over the life of the campaign.

    Every accept/reject decision the search makes is against ONE validation
    shard. Make enough of those decisions and a configuration can look good on
    that shard specifically -- not because it is better, but because it was
    lucky there. The holdout shard exists to detect that, and the statistic is
    the per-run gap `holdout_val_bpb - val_bpb`.

    A CONSTANT gap is harmless: the two shards are different text, so one is
    simply a bit harder, and a fixed offset affects every run equally. What
    matters is whether the gap GROWS -- that is the signature of the search
    progressively fitting the validation shard's quirks.

    Returns None when fewer than two runs carry both numbers, rather than
    reporting a trend from a single point. `growing` is deliberately reported
    as a raw half-split difference with its own noise band, not as a p-value:
    with the handful of holdout-scored runs a campaign accumulates, anything
    fancier would imply precision that isn't there.
    """
    paired = []
    for row in rows:
        val, hold = row.get("val_bpb"), row.get("holdout_val_bpb")
        if not isinstance(val, (int, float)) or not isinstance(hold, (int, float)):
            continue
        if not (math.isfinite(val) and math.isfinite(hold)):
            continue
        paired.append({
            "run_id": row.get("run_id"),
            "timestamp": row.get("timestamp"),
            "val_bpb": float(val),
            "holdout_val_bpb": float(hold),
            "gap": float(hold) - float(val),
        })
    if len(paired) < 2:
        return None

    paired.sort(key=lambda r: (r.get("timestamp") or "", r.get("run_id") or ""))
    gaps = [r["gap"] for r in paired]
    mean_gap = sum(gaps) / len(gaps)
    variance = sum((g - mean_gap) ** 2 for g in gaps) / (len(gaps) - 1)
    std_gap = variance ** 0.5

    half = len(gaps) // 2
    early, late = gaps[:half], gaps[half:]
    early_mean = sum(early) / len(early) if early else None
    late_mean = sum(late) / len(late) if late else None
    growth = (late_mean - early_mean) if (early_mean is not None and late_mean is not None) else None

    return {
        "n": len(paired),
        "runs": paired,
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "early_mean_gap": early_mean,
        "late_mean_gap": late_mean,
        "growth": growth,
        # Only call it drift when the change between halves exceeds the spread
        # of the gaps themselves -- otherwise it is just the same offset
        # measured twice.
        "growing": bool(growth is not None and std_gap > 0 and growth > std_gap),
    }


def load_results(paths: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Dict[str, Any]]:
    """Load one or more results.tsv-shaped files, tolerant of the current
    schema and every frozen superseded one (PRE_SEED_COLUMNS, LEGACY_COLUMNS).
    Rows are told apart by field count, not by the file's header line, since a
    stale/mismatched header is exactly the bug this loader has to survive --
    and it is also why archived files under legacy_results.tsv/ can be passed
    here directly alongside the live results.tsv.

    Rows with a status in SYNTHETIC_STATUSES (dry_run, simulated) are
    dropped -- every caller of this loader treats val_bpb as a real,
    comparable measurement (correlations, surrogate fitting, noise floor,
    elite-run selection), and a synthetic placeholder value mixed into any
    of those silently distorts the result. Callers that specifically need
    run *counts* by status (e.g. Agent 3's status-distribution reporting)
    don't go through this loader -- they read agent2_reports/*.md directly.
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
                    row = _coerce_row(CURRENT_COLUMNS, fields, "current")
                elif len(fields) == len(PRE_TIME_BREAKDOWN_COLUMNS):
                    row = _coerce_row(PRE_TIME_BREAKDOWN_COLUMNS, fields,
                                      "pre_time_breakdown")
                elif len(fields) == len(PRE_SEED_COLUMNS):
                    row = _coerce_row(PRE_SEED_COLUMNS, fields, "pre_seed")
                elif len(fields) == len(LEGACY_COLUMNS):
                    row = _coerce_row(LEGACY_COLUMNS, fields, "legacy")
                else:
                    # Rows matching neither known shape are skipped rather
                    # than guessed at.
                    continue
                if row.get("status") in SYNTHETIC_STATUSES:
                    continue
                rows.append(row)
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


def top_quartile_by_val_bpb(
    candidates: Sequence[Tuple[float, Any]], fraction: float = 0.25
) -> List[Tuple[float, Any]]:
    """The best `fraction` of (val_bpb, payload) pairs, lowest (best)
    val_bpb first -- at least 1 whenever `candidates` is non-empty, matching
    the "at least one elite run to recommend from" behavior this replaces.

    The single shared definition of "elite" used both for Agent 3's
    data-backed hyperparameter recommendations (payload = a hyperparams
    dict) and Agent 2's stuck-signal reference value (payload unused,
    only the val_bpb side matters) -- one consistent notion of "good"
    across the system instead of two independently-computed ones that
    could quietly drift apart. Callers aggregate the returned subset
    differently (geometric mean of hyperparams vs. median of val_bpb),
    but which runs count as elite is decided exactly once, here.
    """
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c[0])
    count = max(1, int(len(ordered) * fraction))
    return ordered[:count]


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
