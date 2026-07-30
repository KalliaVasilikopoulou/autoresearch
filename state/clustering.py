"""Cluster token-level behavioral fingerprints across runs (Tier 3, see
dev/INNOVATION_PLAN.md). Pure functions: structured data in, cluster
assignments/stats out (or None when there isn't enough data) -- mirrors
state/surrogate.py and state/results_analysis.py's contract in this
codebase: never fabricate a cluster from too little data.

Optional-dependency-guarded like state/surrogate.py: importing this module
never raises even when scipy/scikit-learn aren't installed, but every
function that needs them returns None rather than fabricating a result.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from state.results_analysis import spearman

try:
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score
    CLUSTERING_DEPS_AVAILABLE = True
except ImportError:
    CLUSTERING_DEPS_AVAILABLE = False


MIN_CLUSTER_N = 8    # exposed via agents_config.yaml's agent3.min_cluster_observations
MAX_K = 6             # candidate cluster counts swept, 2..min(MAX_K, n-1)
MIN_CLUSTER_SIZE = 2  # every candidate cluster must have at least this many members
POS_SALIENCY_LEN = 16  # position_saliency's default n_buckets (see agents/xai_methods/token_methods.py)


def _linear_slope(values: List[float]) -> float:
    """OLS slope of values vs. index. A tiny, stable pure-math duplicate of
    agents/xai_methods/token_methods.py's attn_distance_slope -- duplicated
    rather than imported so this module stays torch-free (same rationale
    Tier 2 used for duplicating norm()/apply_rotary_emb locally instead of
    importing train.py: avoid pulling in an unrelated heavy dependency for
    a few lines of pure math)."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def _summary_stats(values: List[float]) -> Tuple[float, float, float]:
    """mean, std, OLS slope -- a fixed-size summary of a variable-length
    per-layer array (n_layer differs across runs, so raw per-layer arrays
    can't be compared directly across runs)."""
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) if len(values) > 1 else 0.0
    return mean, std, _linear_slope(values)


def _build_feature_vector(fingerprint: Dict[str, Any]) -> Optional[List[float]]:
    """Fixed-length feature vector regardless of n_layer: attn_distance_slope
    and induction_score (already scalars), the 16 pos_saliency buckets
    (fixed length by construction) reduced to mean/std/slope like the
    per-layer arrays -- NOT its 16 raw values, since z-score standardization
    gives every column equal weight regardless of how informative it is:
    16 raw dimensions from one field would systematically outvote the ~2-3
    dimensions summarizing each of the other five fields, diluting real
    cross-field signal with what is, per-dimension, mostly saliency-bucket
    noise. Returns None (skip, don't guess) when pos_saliency isn't the
    expected length -- e.g. a stale/partial fingerprint from a schema
    change.
    """
    pos_saliency = fingerprint.get("pos_saliency") or []
    if len(pos_saliency) != POS_SALIENCY_LEN:
        return None

    features: List[float] = [
        float(fingerprint.get("attn_distance_slope", 0.0)),
        float(fingerprint.get("induction_score", 0.0)),
    ]

    for key, values in (
        ("pos_saliency", [float(v) for v in pos_saliency]),
        ("attn_entropy", [float(v) for v in (fingerprint.get("attn_entropy") or [])]),
        ("attn_distance", [float(v) for v in (fingerprint.get("attn_distance") or [])]),
        ("dla", [float(v) for v in (fingerprint.get("dla") or [])]),
        ("x0_lambda", [float(v) for v in (fingerprint.get("x0_lambda") or [])]),
    ):
        mean, std, slope = _summary_stats(values)
        features.extend([mean, std, slope])

    return features


def _standardize(matrix: List[List[float]]) -> List[List[float]]:
    """z-score each column. A constant column (std=0) is left at 0.0 for
    every row rather than dividing by zero."""
    n_rows = len(matrix)
    n_features = len(matrix[0])
    means = [sum(row[j] for row in matrix) / n_rows for j in range(n_features)]
    stds = [
        math.sqrt(sum((row[j] - means[j]) ** 2 for row in matrix) / n_rows)
        for j in range(n_features)
    ]
    return [
        [((row[j] - means[j]) / stds[j]) if stds[j] > 1e-12 else 0.0 for j in range(n_features)]
        for row in matrix
    ]


def _resample_curve(curve: List[float], n_resample: int) -> List[float]:
    """Linear interpolation of `curve` (length = n_layer, varies per run)
    onto n_resample points over normalized depth [0,1], after min-max
    normalizing the VALUES to [0,1] too -- so two curves with the same
    shape but different overall attention-reach scale cluster together,
    and curves of different raw length become directly comparable.
    """
    n = len(curve)
    lo, hi = min(curve), max(curve)
    span = hi - lo
    normalized = [(v - lo) / span if span > 1e-12 else 0.5 for v in curve]

    if n == 1:
        return [normalized[0]] * n_resample

    xs_source = [i / (n - 1) for i in range(n)]
    xs_target = [i / (n_resample - 1) for i in range(n_resample)] if n_resample > 1 else [0.0]

    out: List[float] = []
    for x in xs_target:
        if x <= xs_source[0]:
            out.append(normalized[0])
            continue
        if x >= xs_source[-1]:
            out.append(normalized[-1])
            continue
        for i in range(n - 1):
            if xs_source[i] <= x <= xs_source[i + 1]:
                span_x = xs_source[i + 1] - xs_source[i]
                t = (x - xs_source[i]) / span_x if span_x > 1e-12 else 0.0
                out.append(normalized[i] + t * (normalized[i + 1] - normalized[i]))
                break
    return out


def _total_variation(curve: List[float]) -> float:
    """Sum of absolute step-to-step changes -- how much a curve zig-zags
    rather than moving smoothly in one direction. `curve` is expected to
    already be min-max normalized to [0,1] (see _resample_curve) so runs
    with different overall attention-reach scales are comparable. Low value
    = smooth/monotonic; high value = volatile (repeatedly reverses
    direction)."""
    return sum(abs(curve[i + 1] - curve[i]) for i in range(len(curve) - 1))


def trajectory_smoothness_correlation(
    rows: List[Dict[str, Any]], min_n: int = MIN_CLUSTER_N, n_resample: int = 8,
) -> Optional[Dict[str, Any]]:
    """3.4: Spearman correlation between each run's attn_distance trajectory
    volatility (_total_variation of the same normalized curve
    cluster_attention_trajectories uses) and its val_bpb.

    This is a statistically more robust alternative to reading the
    trajectory *clusters* for the same "does a volatile early trajectory
    predict worse bpb" question: a continuous correlation over n rows
    doesn't fragment the data into 2-4-member clusters (which is what made
    silhouette scores as low as 0.25-0.29 in practice), so it can detect the
    same signal earlier and with a real n instead of ad hoc cluster sizes.
    Positive correlation = more volatile trajectory -> higher (worse)
    val_bpb. Returns None below min_n usable rows or when scipy isn't
    needed here at all (pure math) but there just isn't enough data -- same
    "don't fabricate a signal from too little data" contract as the rest of
    this module.
    """
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        curve = [float(v) for v in (row.get("attn_distance") or [])]
        if len(curve) < 2:
            continue
        val_bpb = row.get("val_bpb")
        if not isinstance(val_bpb, (int, float)) or not math.isfinite(float(val_bpb)):
            continue
        xs.append(_total_variation(_resample_curve(curve, n_resample)))
        ys.append(float(val_bpb))

    if len(xs) < min_n or len(set(xs)) < 2:
        return None

    return {"correlation": round(spearman(xs, ys), 6), "n": len(xs)}


def _ward_cluster_with_best_silhouette(
    matrix: List[List[float]], max_k: int = MAX_K,
) -> Optional[Tuple[List[int], int, float]]:
    """Hierarchical Ward linkage, sweep k=2..max_k, pick the k with the best
    silhouette score. Returns (labels, k, silhouette) or None if
    scipy/scikit-learn aren't installed or there's too little data for any
    valid k."""
    if not CLUSTERING_DEPS_AVAILABLE:
        return None

    arr = np.asarray(matrix)
    n = len(matrix)
    upper_k = min(max_k, n - 1)
    if upper_k < 2:
        return None

    z = linkage(arr, method="ward")
    best: Optional[Tuple[List[int], int, float]] = None
    for k in range(2, upper_k + 1):
        labels = fcluster(z, k, criterion="maxclust")
        labels_list = labels.tolist()
        if len(set(labels_list)) < 2:
            continue
        # Silhouette alone tends to reward isolating a small handful of
        # points as their own cluster (spuriously high score at small n,
        # not genuine structure) -- require every candidate cluster to have
        # at least MIN_CLUSTER_SIZE members before it's even considered,
        # same "don't overstate from too little data" principle as
        # MIN_CLUSTER_N itself.
        if min(labels_list.count(label) for label in set(labels_list)) < MIN_CLUSTER_SIZE:
            continue
        score = silhouette_score(arr, labels)
        if best is None or score > best[2]:
            best = (labels_list, k, float(score))
    return best


def _cluster_val_bpb_stats(val_bpbs: List[float]) -> Tuple[Optional[float], int]:
    finite = [v for v in val_bpbs if isinstance(v, (int, float)) and math.isfinite(v)]
    mean = (sum(finite) / len(finite)) if finite else None
    return mean, len(finite)


def cluster_fingerprints(
    rows: List[Dict[str, Any]], min_n: int = MIN_CLUSTER_N, max_k: int = MAX_K,
) -> Optional[Dict[str, Any]]:
    """3.1: cluster the overall fingerprint. Each row is a token_fingerprint
    dict (attn_entropy, attn_distance, attn_distance_slope, pos_saliency,
    dla, induction_score, x0_lambda) plus a "val_bpb" key. Per-layer arrays
    are reduced to fixed-size summary stats since n_layer varies across
    runs and raw arrays of different lengths can't be clustered directly.
    Standardizes features, hierarchical Ward, picks k by best silhouette,
    then reports per-cluster val_bpb stats -- correlating cluster
    membership with model quality. Returns None when there are fewer than
    min_n usable fingerprints or scipy/scikit-learn aren't installed --
    never a fabricated cluster.
    """
    usable = [(row, _build_feature_vector(row)) for row in rows]
    usable = [(row, vec) for row, vec in usable if vec is not None]
    if len(usable) < min_n:
        return None

    matrix = _standardize([vec for _, vec in usable])
    result = _ward_cluster_with_best_silhouette(matrix, max_k=max_k)
    if result is None:
        return None
    labels, k, silhouette = result

    buckets: Dict[int, Dict[str, Any]] = {}
    for (row, _vec), label in zip(usable, labels):
        bucket = buckets.setdefault(int(label), {"n": 0, "val_bpbs": []})
        bucket["n"] += 1
        val_bpb = row.get("val_bpb")
        if isinstance(val_bpb, (int, float)):
            bucket["val_bpbs"].append(float(val_bpb))

    clusters = []
    for label, bucket in sorted(buckets.items()):
        mean_val_bpb, n_with_val_bpb = _cluster_val_bpb_stats(bucket["val_bpbs"])
        clusters.append({
            "cluster_id": label,
            "n": bucket["n"],
            "mean_val_bpb": mean_val_bpb,
            "n_with_val_bpb": n_with_val_bpb,
        })

    return {"k": k, "silhouette": silhouette, "n_total": len(usable), "clusters": clusters}


def cluster_attention_trajectories(
    rows: List[Dict[str, Any]], min_n: int = MIN_CLUSTER_N, max_k: int = MAX_K, n_resample: int = 8,
) -> Optional[Dict[str, Any]]:
    """3.2: cluster the SHAPE of the attn_distance curve specifically (not
    magnitude) -- resample each run's attn_distance[n_layer] to a fixed
    n_resample points over normalized depth, min-max normalizing each
    curve's own range first (see _resample_curve) so clustering reflects
    trajectory shape (steady ramp vs. early saturation vs. ...) rather than
    overall attention-reach scale. Same Ward+silhouette procedure as
    cluster_fingerprints, but skips standardization -- the curves are
    already normalized to a comparable [0,1] range by _resample_curve, and
    standardizing on top would erase the resampled shape. Returns None
    below min_n or without scipy/scikit-learn.
    """
    usable = []
    for row in rows:
        curve = [float(v) for v in (row.get("attn_distance") or [])]
        if len(curve) < 2:
            continue
        usable.append((row, _resample_curve(curve, n_resample)))
    if len(usable) < min_n:
        return None

    matrix = [vec for _, vec in usable]
    result = _ward_cluster_with_best_silhouette(matrix, max_k=max_k)
    if result is None:
        return None
    labels, k, silhouette = result

    buckets: Dict[int, Dict[str, Any]] = {}
    for (row, vec), label in zip(usable, labels):
        bucket = buckets.setdefault(int(label), {"n": 0, "val_bpbs": [], "curves": []})
        bucket["n"] += 1
        bucket["curves"].append(vec)
        val_bpb = row.get("val_bpb")
        if isinstance(val_bpb, (int, float)):
            bucket["val_bpbs"].append(float(val_bpb))

    clusters = []
    for label, bucket in sorted(buckets.items()):
        mean_val_bpb, n_with_val_bpb = _cluster_val_bpb_stats(bucket["val_bpbs"])
        curves = bucket["curves"]
        mean_shape = [sum(c[i] for c in curves) / len(curves) for i in range(n_resample)]
        clusters.append({
            "cluster_id": label,
            "n": bucket["n"],
            "mean_val_bpb": mean_val_bpb,
            "n_with_val_bpb": n_with_val_bpb,
            "mean_shape": mean_shape,
        })

    return {"k": k, "silhouette": silhouette, "n_total": len(usable), "n_resample": n_resample, "clusters": clusters}
