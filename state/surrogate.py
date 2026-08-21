"""Random-Forest surrogate over results.tsv: (hyperparams -> val_bpb).

Tier 1 "surrogate" work (see dev/INNOVATION_PLAN.md):
  - fit + predict + coordinate-slice sensitivity + noise-floor-based pruning
  - Expected Improvement acquisition + Sobol cold-start sampling
  - interaction detection (cheap fANOVA) + Gauss-Southwell blocking

Optional-dependency-guarded like the `yaml` import used elsewhere in this
repo: importing this module never raises even when scipy/scikit-learn
aren't installed, but every function that needs them returns None/[] rather
than fabricating a result.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from state.results_analysis import HYPERPARAM_COLUMNS

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    SURROGATE_DEPS_AVAILABLE = True
except ImportError:
    SURROGATE_DEPS_AVAILABLE = False


MIN_SURROGATE_N = 15

# A run is "compute-starved" when it completed this fraction fewer training
# steps than its OWN hyperparameters predict it should have. Measured on 581
# real runs: num_steps is 91% predictable from the config alone (OOB R2=0.909),
# so a large shortfall is not the config being slow -- it is the run having
# been robbed of compute by something outside the experiment.
#
# On this shared university DGX that something is other tenants. Watching one
# hour of identical-config repeats, GPUs went from all-idle to four occupied
# and step counts fell 1688 -> 1304 (-23%) on the SAME config, moving val_bpb
# by 0.028 -- roughly the entire elite-to-best gap of the campaign. 29% of
# historical runs are >10% short, 14% are >20% short.
#
# Such a run's val_bpb is a real number, but it answers "how good is this
# config when robbed of a fifth of its training?" -- not the question the
# search is asking. Mixing it into the statistics is the same category of
# error as mixing in a dry_run placeholder, which state/results_analysis.py's
# SYNTHETIC_STATUSES already refuses to do.
STEP_DEFICIT_THRESHOLD = 0.20

# Params whose useful range spans orders of magnitude -- sensitivity/slicing
# treats these on a log scale, everything else linearly.
LOG_SCALE_PARAMS = {"embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "batch_size"}


@dataclass
class SurrogateModel:
    model: Any  # sklearn RandomForestRegressor
    feature_names: Tuple[str, ...]
    bounds: Dict[str, Tuple[float, float]]
    n_train: int
    # Out-of-bag predictions: each row's prediction using only the trees
    # that didn't see it during bootstrap sampling -- a free, real
    # held-out-style accuracy check with no separate train/test split
    # needed. Empty tuples for a SurrogateModel built before this field
    # existed (e.g. hand-constructed in a test), so every prior
    # construction call site keeps working unchanged.
    oob_actual: Tuple[float, ...] = ()
    oob_predicted: Tuple[float, ...] = ()

    def predict(self, hyperparams: Dict[str, Any]) -> Tuple[float, float]:
        """Returns (mean, std-across-trees) for one hyperparameter dict."""
        x = np.array([[float(hyperparams.get(name, 0.0)) for name in self.feature_names]])
        tree_preds = np.array([tree.predict(x)[0] for tree in self.model.estimators_])
        return float(tree_preds.mean()), float(tree_preds.std())


def _rows_to_xy(rows: List[Dict[str, Any]], feature_columns: Sequence[str]):
    xs, ys = [], []
    for row in rows:
        if "val_bpb" not in row or not math.isfinite(row["val_bpb"]):
            continue
        if any(col not in row for col in feature_columns):
            continue
        xs.append([float(row[col]) for col in feature_columns])
        ys.append(float(row["val_bpb"]))
    return np.array(xs), np.array(ys)


def step_deficits(
    rows: List[Dict[str, Any]],
    feature_columns: Sequence[str] = HYPERPARAM_COLUMNS,
    min_n: int = MIN_SURROGATE_N,
    random_state: int = 0,
) -> Optional[List[Optional[float]]]:
    """Per-row fractional shortfall of num_steps against what that row's own
    hyperparameters predict. Positive = fewer steps than deserved.

    Aligned with `rows`; an entry is None where the row can't be judged (no
    num_steps, or a missing feature). Returns None entirely when sklearn is
    absent or too few judgeable rows exist -- never a guess.

    Predicting from the config is the crux: a genuinely slow configuration
    has a correspondingly low prediction and so shows no deficit. Only a run
    slower than its own config warrants is flagged, which is what separates
    "this architecture is expensive" (real signal, keep) from "this run was
    fighting three other tenants for the GPU" (measurement failure, drop).

    Out-of-bag predictions are used, so no row is judged by a model that
    trained on it.
    """
    if not SURROGATE_DEPS_AVAILABLE:
        return None
    judgeable = [
        i for i, r in enumerate(rows)
        if isinstance(r.get("num_steps"), (int, float))
        and math.isfinite(r["num_steps"]) and r["num_steps"] > 0
        and all(c in r for c in feature_columns)
    ]
    if len(judgeable) < min_n:
        return None

    x = np.array([[float(rows[i][c]) for c in feature_columns] for i in judgeable])
    y = np.array([float(rows[i]["num_steps"]) for i in judgeable])
    model = RandomForestRegressor(n_estimators=300, random_state=random_state,
                                  n_jobs=-1, oob_score=True)
    model.fit(x, y)
    predicted = np.asarray(model.oob_prediction_)

    out: List[Optional[float]] = [None] * len(rows)
    for slot, row_idx in enumerate(judgeable):
        p = predicted[slot]
        if not np.isfinite(p) or p <= 0:
            continue
        out[row_idx] = float((p - y[slot]) / p)
    return out


def without_compute_starved(
    rows: List[Dict[str, Any]],
    feature_columns: Sequence[str] = HYPERPARAM_COLUMNS,
    threshold: float = STEP_DEFICIT_THRESHOLD,
    min_n: int = MIN_SURROGATE_N,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """`rows` minus the runs robbed of a `threshold` fraction of their steps.

    Degrades to returning `rows` unchanged whenever the judgement can't be
    made (no sklearn, too little history, no num_steps logged) -- the same
    "omit rather than fabricate" contract the rest of this module follows.
    A row that cannot be judged is kept, never dropped on suspicion.
    """
    # Direct evidence first, and it needs no model: a run reporting
    # budget_shortfall_pct > 0 hit train.py's wall-clock safety cap and
    # never consumed its TOKEN_BUDGET, so it measured a different (smaller)
    # amount of training than every run it would be compared against. Under
    # the token budget this is the ONLY starvation signal that matters --
    # num_steps becomes a deterministic function of batch_size, so the
    # inferred step-deficit below has nothing left to detect. It stays for
    # the wall-clock-budget history, where it is the only signal available.
    def _truncated(row: Dict[str, Any]) -> bool:
        s = row.get("budget_shortfall_pct")
        return isinstance(s, (int, float)) and math.isfinite(s) and s > 0.0

    rows_after_direct = [r for r in rows if not _truncated(r)]

    def _proven_complete(row: Dict[str, Any]) -> bool:
        """The run itself reported consuming its whole token budget.

        NO INFERRED MODEL MAY OVERRULE THAT. budget_shortfall_pct is computed by
        train.py from tokens actually seen against TOKEN_BUDGET, so 0.0 is a
        measurement that the run was complete -- there is nothing for a
        step-deficit estimate to add.

        This is the case the comment above already anticipated: under a token
        budget num_steps is a deterministic function of batch_size, so the
        inferred deficit "has nothing left to detect". It was never actually
        switched off, and it does not fail quietly -- fitting num_steps against
        configurations whose batch sizes span 2048-32768 makes most of them look
        like outliers. Measured on a real campaign: 10 of 16 complete runs
        excluded, leaving 6 against a min_n of 15, so the surrogate could never
        fit and the search cold-started forever while every run succeeded.
        """
        s = row.get("budget_shortfall_pct")
        return isinstance(s, (int, float)) and math.isfinite(s) and s == 0.0

    # The inferred model still judges rows that CANNOT say -- the wall-clock
    # budget history, where it is the only signal there is.
    judged = [r for r in rows_after_direct if not _proven_complete(r)]
    proven = [r for r in rows_after_direct if _proven_complete(r)]
    if not judged:
        if verbose and len(rows_after_direct) != len(rows):
            print(f"[surrogate] Excluded {len(rows) - len(rows_after_direct)}/{len(rows)} run(s) "
                  f"that hit the wall-clock safety cap before finishing their token budget")
        return rows_after_direct

    deficits = step_deficits(judged, feature_columns=feature_columns, min_n=min_n)
    if deficits is None:
        if verbose and len(rows_after_direct) != len(rows):
            print(f"[surrogate] Excluded {len(rows) - len(rows_after_direct)}/{len(rows)} run(s) "
                  f"that hit the wall-clock safety cap before finishing their token budget")
        return rows_after_direct
    # `deficits` lines up with `judged`, not with rows_after_direct -- the
    # proven-complete rows were never handed to the model and are kept as they
    # are. Zipping against the wrong list would drop them by position.
    survived = [r for r, d in zip(judged, deficits) if d is None or d < threshold]
    kept = [r for r in rows_after_direct if r in proven or r in survived]
    if verbose and len(kept) != len(rows):
        print(f"[surrogate] Excluded {len(rows) - len(kept)}/{len(rows)} compute-starved run(s) "
              f"(>{threshold:.0%} fewer training steps than their config predicts -- "
              f"contended GPU, not a property of the hyperparameters)")
    return kept


def fit_surrogate(
    rows: List[Dict[str, Any]],
    feature_columns: Sequence[str] = HYPERPARAM_COLUMNS,
    min_n: int = MIN_SURROGATE_N,
    n_estimators: int = 200,
    random_state: int = 0,
    exclude_compute_starved: bool = True,
) -> Optional[SurrogateModel]:
    """Fits a Random Forest over rows with ALL feature_columns present and a
    finite val_bpb. Returns None (never raises, never fabricates a fit) if
    dependencies are missing or there isn't enough comparable data yet.
    """
    if not SURROGATE_DEPS_AVAILABLE:
        return None
    if exclude_compute_starved:
        # Fit the objective on runs that actually got the compute their
        # config called for. Without this the model learns "this region is
        # bad" from runs that were merely unlucky about server load.
        rows = without_compute_starved(rows, feature_columns=feature_columns,
                                       min_n=min_n, verbose=True)
    x, y = _rows_to_xy(rows, feature_columns)
    if len(y) < min_n:
        return None
    bounds = {
        name: (float(x[:, i].min()), float(x[:, i].max()))
        for i, name in enumerate(feature_columns)
    }
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, oob_score=True)
    model.fit(x, y)

    # oob_prediction_ is NaN for any row that happened to appear in every
    # single tree's bootstrap sample (astronomically rare at n_estimators=200,
    # but not impossible for very small n_train) -- excluded rather than
    # coerced to 0, since that would fabricate a data point.
    oob_pred = np.asarray(model.oob_prediction_)
    valid = ~np.isnan(oob_pred)
    oob_actual = tuple(float(v) for v in y[valid])
    oob_predicted = tuple(float(v) for v in oob_pred[valid])

    return SurrogateModel(
        model=model, feature_names=tuple(feature_columns), bounds=bounds, n_train=len(y),
        oob_actual=oob_actual, oob_predicted=oob_predicted,
    )


def best_predicted_mean(
    surrogate: "SurrogateModel",
    rows: List[Dict[str, Any]],
    feature_columns: Sequence[str] = HYPERPARAM_COLUMNS,
) -> Optional[float]:
    """The lowest value the surrogate PREDICTS at any point that was actually
    measured -- the denoised stand-in for "best so far".

    Expected Improvement needs an incumbent to beat. Using the best OBSERVED
    val_bpb makes that incumbent the luckiest draw rather than the best
    configuration, and this campaign measured exactly how large that gap is:
    the record 1.248540 came from the single kindest of 5 seeds for that
    config, whose honest mean is 1.251022 (scripts/seed_variance.py). Aiming
    EI at a value ~1.6 sigma below anything actually achievable depresses the
    acquisition everywhere and pushes the search toward wherever the noise
    happened to be kindest -- which is where the record already is.

    A random forest's prediction at a training point averages over the trees
    that saw it, so it is already smoothed toward that point's neighbourhood.
    Taking the minimum of those predictions is the cheap, standard "noisy EI"
    correction: it asks "what is the best value we have reason to believe in",
    not "what is the luckiest number we happened to see".

    Returns None when no row can be scored, so the caller keeps its existing
    observed-best behaviour rather than being handed a fabricated number.
    """
    if not SURROGATE_DEPS_AVAILABLE:
        return None
    best = None
    for row in rows:
        if any(col not in row for col in feature_columns):
            continue
        if not isinstance(row.get("val_bpb"), (int, float)) or not math.isfinite(row["val_bpb"]):
            continue
        mu, _ = surrogate.predict({col: float(row[col]) for col in feature_columns})
        if math.isfinite(mu) and (best is None or mu < best):
            best = mu
    return best


def normalized_value(param: str, value: float, bounds: Dict[str, Tuple[float, float]]) -> float:
    """Maps value into [0, 1] within its observed bounds -- log-scale for
    LOG_SCALE_PARAMS (LR groups and batch_size span orders of magnitude and
    a linear normalization would make sensitivity comparisons meaningless
    across parameters), linear otherwise. Returns 0.5 for a degenerate
    (zero-width, i.e. never-varied-yet) range.
    """
    lo, hi = bounds.get(param, (0.0, 1.0))
    if hi <= lo:
        return 0.5
    if param in LOG_SCALE_PARAMS and lo > 0:
        log_lo, log_hi = math.log(lo), math.log(hi)
        v = max(lo, min(hi, value))
        return (math.log(v) - log_lo) / (log_hi - log_lo)
    v = max(lo, min(hi, value))
    return (v - lo) / (hi - lo)


def denormalize(param: str, t: float, bounds: Dict[str, Tuple[float, float]]) -> float:
    """Inverse of normalized_value: t in [0,1] -> a value within bounds.

    Public (like normalized_value, its exact inverse) because state/landscape.py
    needs it to map PCA-inverse-transformed points back into real
    hyperparameter space using this module's own log/linear convention --
    reimplementing that mapping there would be two definitions of the same
    thing, free to drift apart. t outside [0,1] deliberately extrapolates
    past `bounds` rather than clamping; callers that need a legal value
    clamp afterwards against their own hard limits.
    """
    lo, hi = bounds[param]
    if param in LOG_SCALE_PARAMS and lo > 0:
        log_lo, log_hi = math.log(lo), math.log(hi)
        return math.exp(log_lo + t * (log_hi - log_lo))
    return lo + t * (hi - lo)


_denormalize = denormalize  # pre-existing private name, kept for call sites/tests


def coordinate_slice(
    metric_fn: Callable[[Dict[str, Any]], float],
    param: str,
    center: Dict[str, Any],
    bounds: Dict[str, Tuple[float, float]],
    n_points: int = 25,
) -> List[Dict[str, Any]]:
    """Holds every hyperparameter at `center` except `param`, which is swept
    across its observed range (n_points values, evenly spaced on the same
    log/linear scale normalized_value uses). Returns a list of
    {"value": v, "metric": metric_fn(...)}, sorted by value ascending.
    Empty list if `param` has a degenerate (zero-width) or unknown range.
    """
    lo, hi = bounds.get(param, (None, None))
    if lo is None or hi is None or hi <= lo:
        return []
    out = []
    for i in range(n_points):
        t = i / (n_points - 1) if n_points > 1 else 0.5
        v = _denormalize(param, t, bounds)
        point = dict(center)
        point[param] = v
        out.append({"value": v, "metric": metric_fn(point)})
    return out


def sensitivity_perf(
    metric_fn: Callable[[Dict[str, Any]], float],
    param: str,
    center: Dict[str, Any],
    bounds: Dict[str, Tuple[float, float]],
    n_points: int = 9,
) -> float:
    """Coordinate-slice total-variation sensitivity: max(metric) - min(metric)
    across param's full observed range, holding everything else at `center`.

    This is the extension point named in the innovation plan: Tier 1 passes
    `lambda hp: surrogate.predict(hp)[0]` here to get S_perf. A later tier
    will pass a fingerprint-distance function through this exact same
    signature to get S_behav -- neither this function nor a future
    blocks_from_interactions needs to change when that lands.
    """
    points = coordinate_slice(metric_fn, param, center, bounds, n_points=n_points)
    if not points:
        return 0.0
    metrics = [p["metric"] for p in points]
    return max(metrics) - min(metrics)


def rank_by_sensitivity(
    surrogate: SurrogateModel,
    params: Sequence[str],
    center: Dict[str, Any],
    bounds: Dict[str, Tuple[float, float]],
) -> List[Tuple[str, float]]:
    """[(param, S_perf)] for every param, sorted descending by S_perf."""
    metric_fn = lambda hp: surrogate.predict(hp)[0]
    scored = [(p, sensitivity_perf(metric_fn, p, center, bounds)) for p in params]
    return sorted(scored, key=lambda item: -item[1])


def prune_by_noise_floor(
    surrogate: SurrogateModel,
    params: Sequence[str],
    center: Dict[str, Any],
    bounds: Dict[str, Tuple[float, float]],
    sigma: float,
    k: float = 2.0,
) -> Tuple[List[str], List[str]]:
    """Splits params into (kept, frozen), each sorted by S_perf descending.
    A parameter is frozen when its total effect across its full observed
    range is below k*sigma -- below the measurement noise floor at this
    budget, not merely "low priority". `sigma` should come from a real
    measurement (scripts/noise_floor.py), never a guess.
    """
    ranked = rank_by_sensitivity(surrogate, params, center, bounds)
    threshold = k * sigma
    kept = [p for p, s in ranked if s >= threshold]
    frozen = [p for p, s in ranked if s < threshold]
    return kept, frozen


# ---------------------------------------------------------------------------
# Expected Improvement acquisition + Sobol cold start.
# ---------------------------------------------------------------------------

# Params that must be integers, and (for n_embd/n_head) satisfy train.py's
# hard assert `n_embd % n_head == 0` (CausalSelfAttention.__init__) AND land
# on an even head_dim (train.py's apply_rotary_emb splits each head into two
# equal halves -- RoPE rotates 2D pairs, so an odd head_dim crashes; see
# train.py's MODEL_DIM computation for the matching last-mile snap there).
INT_PARAMS = {"n_layer", "n_head", "n_embd", "batch_size"}


def snap_n_embd(n_embd: float, n_head: float) -> int:
    """Round n_embd so that n_embd/n_head (head_dim) is an even integer --
    byte-for-byte the same snap train.py itself applies (see train.py's
    MODEL_DIM computation): RoPE splits each head into two equal rotation
    halves, so head_dim must be even, and a valid multi-head split needs
    n_head to divide n_embd evenly. Any code path that can set n_embd
    (the surrogate/EI proposal below, or any of Agent 1's other decision
    paths in agents/agent1_training_specialist.py) must apply this exact
    snap before the value is ever recorded or trained on -- otherwise
    "requested" and "actually used" diverge, and every downstream consumer
    of the n_embd column (hyperparameter correlations, this very surrogate,
    Tier 3 fingerprint clustering) ends up fitting against the wrong label.
    """
    n_head = int(n_head)
    if n_head <= 0:
        return int(round(n_embd))
    head_dim = max(1, round(n_embd / n_head))
    if head_dim % 2 != 0:
        head_dim += 1
    return head_dim * n_head


def _snap_discrete(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Rounds int-valued params and snaps n_embd so that n_embd/n_head is an
    even integer -- a continuous acquisition/sampling proposal must be
    projected back onto the space train.py can actually run before it's
    ever returned to a caller. Keeping this in sync with train.py's own
    snap matters: if the proposal train.py silently alters at run time
    doesn't match what's recorded in results.tsv, the surrogate ends up
    fitting against the wrong labels.
    """
    out = dict(hyperparams)
    for p in INT_PARAMS:
        if p in out:
            out[p] = int(round(out[p]))
    if "n_embd" in out and "n_head" in out and out["n_head"] > 0:
        out["n_embd"] = snap_n_embd(out["n_embd"], out["n_head"])
    return out


def expected_improvement(mu: float, sigma: float, f_best: float, xi: float = 0.01) -> float:
    """Expected Improvement for MINIMIZING val_bpb (lower is better).
    mu/sigma: surrogate's predicted mean/std-across-trees at a candidate.
    f_best: best (lowest) val_bpb observed so far. xi: small exploration
    margin so points at the observed optimum don't get EI=0 forever.
    """
    if not SURROGATE_DEPS_AVAILABLE:
        return 0.0
    from scipy.stats import norm
    if sigma <= 1e-12:
        return max(0.0, f_best - mu - xi)
    z = (f_best - mu - xi) / sigma
    return float((f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z))


def _sample_in_ball(center_norm, max_euclid: float, n: int, rng) -> Any:
    """`n` points drawn uniformly inside a ball of radius `max_euclid` around
    `center_norm`, in normalized [0,1] space, then clipped to stay legal.

    A random Gaussian direction normalized to unit length is uniform on the
    sphere; scaling by u**(1/d) makes the radius uniform by VOLUME rather than
    by length, which would otherwise pile most candidates near the centre and
    barely probe the edge of the region -- the opposite of what a search that
    is trying to leave wants.

    Clipping to [0,1] does distort the ball for a region whose anchor sits near
    a parameter's limit: candidates bunch against that face. Accepted rather
    than rejection-sampled, because a region ANCHORED at the edge of the search
    space genuinely has less room on that side, and pretending otherwise would
    propose values the clamps would reject anyway.
    """
    d = len(center_norm)
    direction = rng.normal(size=(n, d))
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    direction /= norms
    radius = max_euclid * rng.random(n) ** (1.0 / d)
    return np.clip(np.asarray(center_norm) + direction * radius[:, None], 0.0, 1.0)


def propose_via_ei(
    surrogate: SurrogateModel,
    f_best: float,
    bounds: Dict[str, Tuple[float, float]],
    free_params: Sequence[str],
    fixed_values: Dict[str, Any],
    n_candidates: int = 2000,
    seed: Optional[int] = None,
    return_diagnostics: bool = False,
    fence_center: Optional[Dict[str, Any]] = None,
    fence_radius: Optional[float] = None,
    fence_dims: Optional[int] = None,
    exploration_weight: float = 1.0,
) -> Any:
    """Random-search EI maximization: draws `n_candidates` points uniformly
    (in normalized space) over `free_params`, holds everything else at
    `fixed_values` (frozen params, or params outside the currently-active
    block once blocking exists), and returns the full hyperparams dict for
    whichever candidate has the highest Expected Improvement.

    Random search over EI (rather than a gradient-based optimizer) is
    intentional: this is a mixed discrete/continuous, low-dimensional (<=10)
    space where a few thousand candidate evaluations of a Random Forest cost
    milliseconds -- exact EI optimization would be overkill machinery for no
    measurable benefit here. Candidates are scored as one batched matrix
    (n_estimators tree.predict() calls total), not one SurrogateModel.predict()
    call per candidate -- the latter is n_candidates*n_estimators individual
    calls, which is slow enough at n_candidates=2000 to matter.

    return_diagnostics=False (default, every existing caller): returns just
    the winning candidate dict, unchanged from before this parameter
    existed. return_diagnostics=True: returns (winning_candidate,
    diagnostics), where diagnostics holds free_params, each free param's
    sampled values across all n_candidates, and the mus/sigmas/eis arrays --
    enough to chart what the search actually considered, without persisting
    every candidate's full (mostly-redundant, since only free_params vary)
    hyperparameter dict.
    """
    if not SURROGATE_DEPS_AVAILABLE:
        empty = dict(fixed_values)
        return (empty, {}) if return_diagnostics else empty
    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Any]] = []
    for _ in range(n_candidates):
        candidate = dict(fixed_values)
        for p in free_params:
            t = float(rng.uniform(0.0, 1.0))
            candidate[p] = _denormalize(p, t, bounds)
        candidates.append(_snap_discrete(candidate))
    n_unfenced = len(candidates)

    # The fence. Candidates so far roam each free parameter's FULL range, which
    # is why a region's centre could jump 6-10x its own radius in one proposal.
    # Those are still generated and scored -- they are the escape signal, free
    # of charge -- but when a fence is set they are not eligible to be run.
    fenced = bool(fence_center) and fence_radius is not None and len(free_params) > 0
    if fenced:
        # The region metric divides by sqrt(total tunable dims), and parameters
        # outside the active block sit exactly at the centre and contribute
        # nothing, so the budget for the free ones is radius * sqrt(dims).
        dims = fence_dims if fence_dims else len(free_params)
        max_euclid = float(fence_radius) * math.sqrt(dims)
        center_norm = [normalized_value(p, float(fence_center.get(p, 0.0)), bounds)
                       for p in free_params]
        pts = _sample_in_ball(center_norm, max_euclid, n_candidates, rng)
        for row in pts:
            candidate = dict(fixed_values)
            for p, t in zip(free_params, row):
                candidate[p] = _denormalize(p, float(t), bounds)
            candidates.append(_snap_discrete(candidate))

    x = np.array([[float(c.get(name, 0.0)) for name in surrogate.feature_names] for c in candidates])
    tree_preds = np.array([tree.predict(x) for tree in surrogate.model.estimators_])  # [n_trees, n_candidates]
    mus = tree_preds.mean(axis=0)
    sigmas = tree_preds.std(axis=0)

    # BUDGET-AWARE EXPLORATION. EI = promise + uncertainty, and the second term
    # is only worth paying for while there are runs left to EXPLOIT what it
    # teaches. Scaling sigma by `exploration_weight` anneals that term: at 1.0
    # this is textbook EI, and as it approaches 0 the acquisition collapses to
    # max(f_best - mu, 0), which is monotone in -mu -- i.e. pure exploitation,
    # "take the best predicted point".
    #
    # THE FAILURE THIS FIXES. With a flat weight, EI spent the last third of a
    # 30-run campaign chasing variance it had no budget left to use. Measured
    # at iterations 22/25/28, it chose candidates predicted at 1.5688 / 1.5638 /
    # 1.5701 while candidates predicted at 1.4532 / 1.4809 / 1.4898 sat in the
    # same batch -- because sigma at the chosen points was 0.186 / 0.087 /
    # 0.067, up to 64x the 0.0029 noise floor. The run that had found 1.4463
    # was never revisited.
    w = max(0.0, float(exploration_weight))
    eis = [expected_improvement(float(mu), float(sigma) * w, f_best)
           for mu, sigma in zip(mus, sigmas)]
    # Two winners: the one we run (inside the fence) and the one the search
    # WANTED (ignoring it). Their gap is the escape pressure.
    unfenced_best = int(np.argmax(eis[:n_unfenced]))
    best_idx = (n_unfenced + int(np.argmax(eis[n_unfenced:]))) if fenced else unfenced_best
    winner = candidates[best_idx]
    if not return_diagnostics:
        return winner

    diagnostics = {
        # THE INCUMBENT THE ACQUISITION MEASURED AGAINST. It was not recorded,
        # so every plan on disk says `f_best used = None` and the log cannot
        # show what "improvement" meant for that proposal -- the same blind
        # spot as the migration gain, which hid four bugs.
        "f_best": float(f_best),
        "exploration_weight": w,
        "free_params": list(free_params),
        "candidate_values": {p: [c[p] for c in candidates] for p in free_params},
        "mus": [float(v) for v in mus],
        "sigmas": [float(v) for v in sigmas],
        "eis": [float(v) for v in eis],
        "best_idx": best_idx,
        "fenced": fenced,
    }
    if fenced:
        want = candidates[unfenced_best]
        # Signed, in normalized units, so a direction can be averaged across
        # iterations -- which is what turns one stray proposal into a trend
        # Agent 4 can act on.
        diagnostics["escape"] = {
            "unfenced_best_idx": unfenced_best,
            "ei_inside": float(eis[best_idx]),
            "ei_outside": float(eis[unfenced_best]),
            # THE PREDICTED MEANS, not just the acquisition values. EI rewards
            # uncertainty as well as promise, so a candidate far from any data
            # scores highly for being UNKNOWN. That is right for choosing where
            # to sample next, and wrong for deciding whether a region's anchor
            # is in the wrong place -- the two questions differ, and answering
            # the second with EI produced a runaway: each migration landed
            # further from the campaign's data, where uncertainty and therefore
            # EI were higher still, so the next escape pointed further out
            # again. Measured step sizes 0.074 -> 0.104 -> 0.174 -> 0.206
            # against a 0.02 fence, growing rather than converging.
            "mean_inside": float(mus[best_idx]),
            "mean_outside": float(mus[unfenced_best]),
            "distance": float(math.sqrt(sum(
                (normalized_value(p, float(want[p]), bounds)
                 - normalized_value(p, float(fence_center.get(p, 0.0)), bounds)) ** 2
                for p in free_params)) / math.sqrt(dims)),
            "radius": float(fence_radius),
            "direction": {
                p: float(normalized_value(p, float(want[p]), bounds)
                         - normalized_value(p, float(fence_center.get(p, 0.0)), bounds))
                for p in free_params
            },
        }
        diagnostics["escape"]["escaped"] = diagnostics["escape"]["distance"] > float(fence_radius)
    return winner, diagnostics


def sobol_cold_start(
    bounds: Dict[str, Tuple[float, float]],
    params: Sequence[str],
    n_samples: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Low-discrepancy Sobol samples across `params`' bounds (log-scale for
    LOG_SCALE_PARAMS via the same normalized_value/_denormalize convention
    used everywhere else in this module), snapped to satisfy discrete
    constraints. Used to seed the surrogate with a space-filling initial
    design -- Agent 1's old `random.randint`/`random.choice` exploration
    duplicates poorly across a ~10-dim space, which is exactly what makes
    "can't fit an RF on <15 points" hard to escape quickly.
    """
    if not SURROGATE_DEPS_AVAILABLE or n_samples <= 0:
        return []
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=len(params), scramble=True, seed=seed)
    raw = sampler.random(n_samples)
    points = []
    for row in raw:
        hp = {p: _denormalize(p, float(t), bounds) for p, t in zip(params, row)}
        points.append(_snap_discrete(hp))
    return points


# ---------------------------------------------------------------------------
# Interaction detection ("cheap fANOVA") + Gauss-Southwell blocking.
#
# Real fANOVA needs the fanova/pyrfr package family (old-scikit-learn-pinned,
# C++ extension, known Windows build pain) or a multi-day hand-rolled
# variance decomposition. This is the roadmap's explicitly-sanctioned cheap
# substitute: fit a second RF on normalized features plus their pairwise
# products, and read the product terms' feature_importances_ as an
# interaction-strength signal. Real fANOVA remains a documented future
# upgrade, not built here.
# ---------------------------------------------------------------------------

def interaction_matrix(
    rows: List[Dict[str, Any]],
    feature_columns: Sequence[str] = HYPERPARAM_COLUMNS,
    min_n: int = MIN_SURROGATE_N,
    n_estimators: int = 200,
    random_state: int = 0,
) -> Optional[Dict[Tuple[str, str], float]]:
    """Fits a Random Forest on [normalized features] + [pairwise products
    x_i*x_j] and returns {(param_i, param_j): importance} for the product
    terms only. Features are normalized first (log-scale for LOG_SCALE_PARAMS)
    so an unnormalized product can't let a large-range parameter dominate the
    interaction signal for reasons unrelated to actual interaction strength.
    Returns None (never fabricates) if deps are missing or data is short.
    """
    if not SURROGATE_DEPS_AVAILABLE:
        return None
    x, y = _rows_to_xy(rows, feature_columns)
    if len(y) < min_n:
        return None

    bounds = {
        name: (float(x[:, i].min()), float(x[:, i].max()))
        for i, name in enumerate(feature_columns)
    }
    norm_x = np.array([
        [normalized_value(name, v, bounds) for v in x[:, i]]
        for i, name in enumerate(feature_columns)
    ]).T  # [n_rows, n_features]

    n_features = len(feature_columns)
    pair_names: List[Tuple[str, str]] = []
    pair_cols = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pair_names.append((feature_columns[i], feature_columns[j]))
            pair_cols.append(norm_x[:, i] * norm_x[:, j])
    if not pair_cols:
        return {}
    pair_x = np.array(pair_cols).T

    augmented = np.concatenate([norm_x, pair_x], axis=1)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(augmented, y)

    pair_importances = model.feature_importances_[n_features:]
    return {pair: float(score) for pair, score in zip(pair_names, pair_importances)}


def blocks_from_interactions(
    interaction_scores: Dict[Tuple[str, str], float],
    main_effect: Dict[str, float],
    threshold: float = 0.15,
) -> List[List[str]]:
    """Union-find merge: two params whose interaction score is within
    `threshold` of the strongest observed interaction (a fraction, so this
    stays scale-free across different fits/datasets) end up in the same
    block -- interacting parameters get tuned jointly rather than greedily
    one at a time, which is what makes Gauss-Southwell safe under
    interaction. Params with no strong interactions get their own singleton
    block. Blocks are returned sorted by descending summed main_effect
    (S_perf) -- the currently-steepest block first.
    """
    params = list(main_effect.keys())
    parent = {p: p for p in params}

    def find(p: str) -> str:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    if interaction_scores:
        max_score = max(interaction_scores.values())
        if max_score > 0:
            for (a, b), score in interaction_scores.items():
                if a in parent and b in parent and score / max_score >= threshold:
                    union(a, b)

    groups: Dict[str, List[str]] = {}
    for p in params:
        groups.setdefault(find(p), []).append(p)

    blocks = list(groups.values())
    blocks.sort(key=lambda block: -sum(main_effect.get(p, 0.0) for p in block))
    return blocks
