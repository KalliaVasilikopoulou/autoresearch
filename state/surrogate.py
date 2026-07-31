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


def fit_surrogate(
    rows: List[Dict[str, Any]],
    feature_columns: Sequence[str] = HYPERPARAM_COLUMNS,
    min_n: int = MIN_SURROGATE_N,
    n_estimators: int = 200,
    random_state: int = 0,
) -> Optional[SurrogateModel]:
    """Fits a Random Forest over rows with ALL feature_columns present and a
    finite val_bpb. Returns None (never raises, never fabricates a fit) if
    dependencies are missing or there isn't enough comparable data yet.
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


def _denormalize(param: str, t: float, bounds: Dict[str, Tuple[float, float]]) -> float:
    """Inverse of normalized_value: t in [0,1] -> a value within bounds."""
    lo, hi = bounds[param]
    if param in LOG_SCALE_PARAMS and lo > 0:
        log_lo, log_hi = math.log(lo), math.log(hi)
        return math.exp(log_lo + t * (log_hi - log_lo))
    return lo + t * (hi - lo)


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


def propose_via_ei(
    surrogate: SurrogateModel,
    f_best: float,
    bounds: Dict[str, Tuple[float, float]],
    free_params: Sequence[str],
    fixed_values: Dict[str, Any],
    n_candidates: int = 2000,
    seed: Optional[int] = None,
    return_diagnostics: bool = False,
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

    x = np.array([[float(c.get(name, 0.0)) for name in surrogate.feature_names] for c in candidates])
    tree_preds = np.array([tree.predict(x) for tree in surrogate.model.estimators_])  # [n_trees, n_candidates]
    mus = tree_preds.mean(axis=0)
    sigmas = tree_preds.std(axis=0)

    eis = [expected_improvement(float(mu), float(sigma), f_best) for mu, sigma in zip(mus, sigmas)]
    best_idx = int(np.argmax(eis))
    winner = candidates[best_idx]
    if not return_diagnostics:
        return winner

    diagnostics = {
        "free_params": list(free_params),
        "candidate_values": {p: [c[p] for c in candidates] for p in free_params},
        "mus": [float(v) for v in mus],
        "sigmas": [float(v) for v in sigmas],
        "eis": [float(v) for v in eis],
        "best_idx": best_idx,
    }
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
