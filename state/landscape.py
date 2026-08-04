"""PCA-compressed view of the hyperparameter optimization landscape.

Turns the ~11-dimensional search space into a 2D "floor" (two PCA components
over the normalized HYPERPARAM_COLUMNS) with val_bpb as the height axis, from
two sources:

  - real runs (results.tsv), whose height is a measured val_bpb, and
  - a grid of never-tried points, whose height is the Tier 1 surrogate's
    predicted val_bpb with a confidence derived from its across-tree spread.

Honest about what this is: projecting a grid cell back into hyperparameter
space goes through PCA's inverse transform, which is a 2-component
*approximation* of an 11-dimensional point. The surface is a projection of
the surrogate's belief, not a literal slice of the true objective -- every
consumer (the chart title, Agent 4's decision logs) says so explicitly.

Pure computation, no matplotlib. Optional-dependency-guarded exactly like
state/surrogate.py and state/clustering.py: importing this module never
raises when scikit-learn is missing, but every function that needs it
returns None rather than fabricating a landscape.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from state.results_analysis import HYPERPARAM_COLUMNS
from state.surrogate import (
    INT_PARAMS,
    denormalize,
    normalized_value,
    snap_n_embd,
)

try:
    import numpy as np
    from sklearn.decomposition import PCA
    LANDSCAPE_DEPS_AVAILABLE = True
except ImportError:
    LANDSCAPE_DEPS_AVAILABLE = False


MIN_LANDSCAPE_N = 15            # mirrors state/surrogate.py::MIN_SURROGATE_N
DEFAULT_GRID_RESOLUTION = 24    # 24x24 = 576 cells -- milliseconds when batch-predicted
GRID_PADDING_FRACTION = 0.15    # pad the real points' PCA extent before gridding

# The flag vocabulary Agent 4 writes and the landscape chart draws. Kept here
# (not on Agent 4) because both the writer and the reader need it, and a
# typo'd flag string on one side would otherwise fail silently.
REGION_FLAGS = (
    "investigating",         # Agent 4 is probing this region right now
    "currently_exploiting",  # the search center lives here
    "no_optimum",            # probed, every probe came back bad
    "local_optimum",         # heavily exploited, then beaten by a better region
    "exploitation_paused",   # heavily exploited, set aside but not ruled out
)


def _usable_rows(
    rows: Sequence[Dict[str, Any]], feature_columns: Sequence[str]
) -> List[Dict[str, Any]]:
    """Rows with every feature column present and a finite val_bpb -- the same
    filter state/surrogate.py::_rows_to_xy applies, so the landscape is built
    over exactly the run set the surrogate was fitted on."""
    out = []
    for row in rows:
        val = row.get("val_bpb")
        if val is None or not math.isfinite(val):
            continue
        if any(col not in row for col in feature_columns):
            continue
        out.append(row)
    return out


def _observed_bounds(
    usable: Sequence[Dict[str, Any]], feature_columns: Sequence[str]
) -> Dict[str, Tuple[float, float]]:
    return {
        col: (
            min(float(r[col]) for r in usable),
            max(float(r[col]) for r in usable),
        )
        for col in feature_columns
    }


def _snap_to_trainable(
    hyperparams: Dict[str, Any], hard_bounds: Optional[Dict[str, Tuple[float, float]]]
) -> Dict[str, Any]:
    """Clamp to hard_bounds, round INT_PARAMS, and apply state/surrogate.py's
    own snap_n_embd -- the exact same projection back onto "what train.py can
    actually run" that every real proposal path already goes through. A grid
    cell nobody will ever train still has to describe a buildable model, or
    the surrogate is being asked to predict for a config that cannot exist.
    """
    out = dict(hyperparams)
    if hard_bounds:
        for param, (lo, hi) in hard_bounds.items():
            if param in out:
                out[param] = max(lo, min(hi, out[param]))
    for param in INT_PARAMS:
        if param in out:
            out[param] = int(round(out[param]))
    if "n_embd" in out and "n_head" in out and out["n_head"] > 0:
        out["n_embd"] = snap_n_embd(out["n_embd"], out["n_head"])
        if hard_bounds and "n_embd" in hard_bounds:
            lo, hi = hard_bounds["n_embd"]
            out["n_embd"] = int(max(lo, min(hi, out["n_embd"])))
    return out


def build_landscape(
    rows: List[Dict[str, Any]],
    surrogate_model: Optional[Any],
    feature_columns: Sequence[str] = HYPERPARAM_COLUMNS,
    grid_resolution: int = DEFAULT_GRID_RESOLUTION,
    min_n: int = MIN_LANDSCAPE_N,
    hard_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[Dict[str, Any]]:
    """Real runs + a surrogate-predicted grid, both projected onto two PCA
    components of the normalized hyperparameter space.

    `hard_bounds` should be the *legal* range (agents/agent1_training_specialist
    .py::SEARCH_SPACE), not the surrogate's narrower *observed* range -- the
    entire point of the predicted grid is to show combinations nobody has
    tried yet, which by definition sit outside the observed hull.

    Returns None (never a fabricated landscape) when scikit-learn is missing,
    no surrogate was fitted, or fewer than `min_n` usable runs exist.
    """
    if not LANDSCAPE_DEPS_AVAILABLE or surrogate_model is None:
        return None
    feature_columns = tuple(feature_columns)
    usable = _usable_rows(rows, feature_columns)
    if len(usable) < min_n or grid_resolution < 2:
        return None

    bounds = _observed_bounds(usable, feature_columns)
    # normalized_value, not a raw z-score: the LR groups and batch_size span
    # orders of magnitude and are already handled on a log scale everywhere
    # else in this system. A z-score here would make the PCA basis disagree
    # with the space the surrogate and EI search actually reason in.
    norm_matrix = np.array([
        [normalized_value(col, float(row[col]), bounds) for col in feature_columns]
        for row in usable
    ])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(norm_matrix)

    real_points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(row["val_bpb"]),
            "hyperparams": {col: float(row[col]) for col in feature_columns},
        }
        for i, row in enumerate(usable)
    ]

    # Grid extent from the real points' own spread, padded -- NOT the full
    # PCA-legal plane, which would extrapolate a 2-component fit into
    # territory no observation supports.
    x_lo, x_hi = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_lo, y_hi = float(coords[:, 1].min()), float(coords[:, 1].max())
    x_pad = (x_hi - x_lo) * GRID_PADDING_FRACTION or 1.0
    y_pad = (y_hi - y_lo) * GRID_PADDING_FRACTION or 1.0
    grid_x = np.linspace(x_lo - x_pad, x_hi + x_pad, grid_resolution)
    grid_y = np.linspace(y_lo - y_pad, y_hi + y_pad, grid_resolution)

    mesh = np.array([[gx, gy] for gy in grid_y for gx in grid_x])  # row-major: y outer, x inner
    inverse_norm = pca.inverse_transform(mesh)  # [n_cells, n_features], normalized space

    grid_hyperparams_flat: List[Dict[str, Any]] = []
    for norm_vec in inverse_norm:
        raw = {
            col: denormalize(col, float(t), bounds)
            for col, t in zip(feature_columns, norm_vec)
        }
        grid_hyperparams_flat.append(_snap_to_trainable(raw, hard_bounds))

    # One batched matrix, one predict() per tree -- mirrors propose_via_ei.
    # A per-cell surrogate.predict() loop would be n_cells * n_estimators
    # single-row calls (115k at 24x24 and 200 trees).
    x_matrix = np.array([
        [float(cell.get(name, 0.0)) for name in surrogate_model.feature_names]
        for cell in grid_hyperparams_flat
    ])
    tree_preds = np.array([tree.predict(x_matrix) for tree in surrogate_model.model.estimators_])
    mus = tree_preds.mean(axis=0)
    stds = tree_preds.std(axis=0)

    span = float(stds.max() - stds.min())
    if span <= 1e-12:
        confidence_flat = np.full(stds.shape, 0.5)  # uniformly uncertain: say so, don't invent a gradient
    else:
        confidence_flat = np.clip(1.0 - (stds - stds.min()) / span, 0.0, 1.0)

    shape = (grid_resolution, grid_resolution)
    return {
        "real_points": real_points,
        "grid_x": [float(v) for v in grid_x],
        "grid_y": [float(v) for v in grid_y],
        "grid_z_mean": mus.reshape(shape).tolist(),
        "grid_z_std": stds.reshape(shape).tolist(),
        "grid_confidence": confidence_flat.reshape(shape).tolist(),
        # [row=y][col=x], same orientation as grid_z_mean -- Agent 4 reads a
        # cell straight out of here instead of inverse-transforming twice.
        "grid_hyperparams": [
            grid_hyperparams_flat[r * grid_resolution:(r + 1) * grid_resolution]
            for r in range(grid_resolution)
        ],
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "n_real": len(usable),
        # Fit artifacts, so project_point can place an arbitrary point on this
        # exact basis later. The basis is refit on every call as the row set
        # grows, so a stored (x, y) goes stale -- always re-project.
        "feature_columns": list(feature_columns),
        "bounds": {k: [v[0], v[1]] for k, v in bounds.items()},
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
    }


def project_point(
    hyperparams: Dict[str, Any], landscape: Dict[str, Any]
) -> Optional[Tuple[float, float]]:
    """Place an arbitrary hyperparameter dict on an existing landscape's PCA
    basis. Returns None (never a guessed position) if any feature is missing.

    Always call this against the *current* landscape rather than reusing a
    previously-computed (x, y): build_landscape refits PCA on whatever rows
    exist at that moment, so the basis itself rotates over a campaign. A
    region's durable identity is its raw hyperparams, never its coordinates.
    """
    if not LANDSCAPE_DEPS_AVAILABLE or not landscape:
        return None
    feature_columns = landscape["feature_columns"]
    if any(col not in hyperparams for col in feature_columns):
        return None
    bounds = {k: (v[0], v[1]) for k, v in landscape["bounds"].items()}
    vec = np.array([
        normalized_value(col, float(hyperparams[col]), bounds) for col in feature_columns
    ])
    centered = vec - np.array(landscape["pca_mean"])
    coords = np.array(landscape["pca_components"]) @ centered
    return float(coords[0]), float(coords[1])


def region_members(
    rows: Sequence[Dict[str, Any]],
    center_hyperparams: Dict[str, Any],
    landscape: Dict[str, Any],
    radius: float,
) -> List[Dict[str, Any]]:
    """Historical runs lying within `radius` of `center_hyperparams` in the
    landscape's PCA plane, where `radius` is a fraction of the grid's own
    extent (so it means the same thing regardless of how the basis is scaled
    at this point in the campaign).

    Defining a "region" in the same 2D space the chart draws is deliberate:
    it means what Agent 4 calls a region and what you see as a neighborhood
    on the landscape are the same thing.
    """
    if not landscape:
        return []
    center = project_point(center_hyperparams, landscape)
    if center is None:
        return []
    grid_x, grid_y = landscape["grid_x"], landscape["grid_y"]
    extent = math.hypot(grid_x[-1] - grid_x[0], grid_y[-1] - grid_y[0])
    cutoff = radius * extent
    out = []
    for row in rows:
        point = project_point(row, landscape)
        if point is None:
            continue
        if math.hypot(point[0] - center[0], point[1] - center[1]) <= cutoff:
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# Region flags: written by Agent 4, drawn by the landscape chart.
# ---------------------------------------------------------------------------

def load_region_flags(path: Any) -> List[Dict[str, Any]]:
    """Tolerant read -- a missing, unreadable, or malformed file yields [],
    never an exception and never a fabricated flag (same convention as
    Agent3ReportAnalyst._load_annotations)."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return []
    regions = data.get("regions") if isinstance(data, dict) else None
    if not isinstance(regions, list):
        return []
    return [r for r in regions if isinstance(r, dict) and "hyperparams" in r]


def save_region_flags(path: Any, regions: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"regions": regions}, indent=2, sort_keys=True), encoding="utf-8")
