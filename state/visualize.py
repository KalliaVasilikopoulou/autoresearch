"""Chart rendering for Agent 2/Agent 3 analysis results.

Pure functions: structured data in, a PNG file path out (or None when there's
nothing real to draw -- these never fabricate a chart for missing data any
more than state/results_analysis.py fabricates a correlation). Mirrors that
module and state/surrogate.py in being a pure computation layer that
agents/* calls into and embeds the result of.

Style follows the dataviz skill's validated reference palette
(references/palette.md): sequential blue for magnitude, diverging blue<->red
for polarity (ablation impact can be negative), the fixed status palette for
run status, categorical hues in fixed order and capped at 8. Light mode only
-- these are static PNGs embedded in markdown, rendered once, so they target
the light chart surface (reads fine in both light and dark markdown viewers).
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: this runs inside the orchestrator process, never a display
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

# --- Reference palette (see dataviz skill, references/palette.md) ----------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

SEQUENTIAL_BLUE = "#2a78d6"
DIVERGING_NEG = "#2a78d6"   # blue: negative (ablating this head helped)
DIVERGING_POS = "#e34948"   # red: positive (ablating this head hurt)
DIVERGING_MID = "#f0efec"   # neutral gray at zero

# Categorical theme, fixed order (never cycled/reassigned) -- 8-slot ceiling.
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]

STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_MAP = {
    "remote_ok": "good", "ok": "good",
    "simulated": "warning", "dry_run": "warning",
    "remote_error": "critical", "timeout": "critical",
}


def _new_figure(figsize: Tuple[float, float] = (7, 4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    return fig, ax


def _finalize(fig, path, title: Optional[str] = None) -> Path:
    if title:
        fig.suptitle(title, color=INK_PRIMARY, fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Part A -- Agent 2 report charts
# ---------------------------------------------------------------------------

def chart_hyperparameter_importance(
    hyper_importance: Dict[str, float],
    sample_sizes: Dict[str, int],
    path: Path,
) -> Optional[Path]:
    """Horizontal bar, sequential blue, sorted descending. One bar per param
    actually present in hyper_importance -- params without enough historical
    data are simply absent from the input, so there is never a fabricated
    zero-bar. Returns None (writes nothing) when hyper_importance is empty.
    """
    if not hyper_importance:
        return None
    items = sorted(hyper_importance.items(), key=lambda kv: kv[1])
    params = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = _new_figure(figsize=(7, 0.4 * len(params) + 1.2))
    bars = ax.barh(params, values, color=SEQUENTIAL_BLUE, height=0.6, zorder=3)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("importance (|Spearman r| vs val_bpb)")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for bar, param in zip(bars, params):
        n = sample_sizes.get(param)
        label = f"{bar.get_width():.2f}" + (f"  (n={n})" if n else "")
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, label,
                va="center", ha="left", fontsize=8, color=INK_SECONDARY)
    return _finalize(fig, path, title="Hyperparameter importance")


def chart_head_importance_heatmap(
    head_impacts: Dict[str, float],
    n_layer: int,
    n_head: int,
    path: Path,
) -> Optional[Path]:
    """Layer x head grid heatmap. Diverging (not sequential): ablation
    impact (baseline_bpb - ablated_bpb) can be negative -- a head whose
    removal *improved* bpb -- so this is a polarity job. Only meaningful
    when ablation actually ran; returns None otherwise (caller should only
    invoke this when ablation_ran is True and head_impacts is non-empty).
    """
    if not head_impacts:
        return None
    grid = [[float("nan")] * n_head for _ in range(n_layer)]
    for key, impact in head_impacts.items():
        try:
            layer_str, head_str = key.split("_")
            layer_idx, head_idx = int(layer_str[1:]), int(head_str[1:])
        except (ValueError, IndexError):
            continue
        if 0 <= layer_idx < n_layer and 0 <= head_idx < n_head:
            grid[layer_idx][head_idx] = impact

    import numpy as np
    arr = np.array(grid)
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return None
    bound = max(abs(finite.min()), abs(finite.max()), 1e-9)

    cmap = LinearSegmentedColormap.from_list("diverging", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS])
    cmap.set_bad(color=GRIDLINE)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    fig, ax = _new_figure(figsize=(max(4, 0.5 * n_head + 1.5), max(3, 0.4 * n_layer + 1.2)))
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_head))
    ax.set_xticklabels([f"H{h}" for h in range(n_head)])
    ax.set_yticks(range(n_layer))
    ax.set_yticklabels([f"L{l}" for l in range(n_layer)])
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("impact (ablating this head: + hurt, − helped)", color=INK_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=7)
    return _finalize(fig, path, title="Head importance (via ablation)")


def chart_layer_scalars(
    layer_scalars: Dict[str, Any],
    n_layer: int,
    path: Path,
) -> Optional[Path]:
    """Small multiples (not dual-axis -- different units): one panel each
    for resid_lambdas, x0_lambdas, ve_gate_norm vs layer index. Called
    whenever layer_scalars is non-empty (effectively every real run, since
    Tier 0's free-scalar extraction has zero GPU cost).
    """
    if not layer_scalars:
        return None
    resid = layer_scalars.get("resid_lambdas") or []
    x0 = layer_scalars.get("x0_lambdas") or []
    ve_gate = layer_scalars.get("ve_gate_norm") or {}
    if not resid and not x0 and not ve_gate:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    panels = [
        ("resid_lambda", resid, range(len(resid))),
        ("x0_lambda", x0, range(len(x0))),
        ("ve_gate_norm", [ve_gate.get(str(i)) for i in range(n_layer) if str(i) in ve_gate],
         [i for i in range(n_layer) if str(i) in ve_gate]),
    ]
    for ax, (label, values, xs) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        if values:
            ax.plot(list(xs), list(values), color=SEQUENTIAL_BLUE, marker="o", markersize=4,
                    linewidth=1.5, zorder=3)
        ax.set_title(label, color=INK_PRIMARY, fontsize=9, loc="left")
        ax.set_xlabel("layer", fontsize=8, color=INK_SECONDARY)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return _finalize(fig, path, title="Per-layer interpretable scalars")


def chart_token_fingerprint(
    token_fingerprint: Dict[str, Any],
    n_layer: int,
    path: Path,
) -> Optional[Path]:
    """Small multiples for the Tier 2 token-level behavioral fingerprint:
    attn_entropy, attn_distance, dla vs layer index, plus a bar for
    pos_saliency by distance-back bucket. Called whenever token_fingerprint
    is non-empty -- only true when token_xai_enabled was on for that run
    (absent, not fabricated, otherwise, same convention as layer_scalars).
    """
    if not token_fingerprint:
        return None
    entropy = token_fingerprint.get("attn_entropy") or []
    distance = token_fingerprint.get("attn_distance") or []
    dla = token_fingerprint.get("dla") or []
    pos_saliency = token_fingerprint.get("pos_saliency") or []
    if not entropy and not distance and not dla and not pos_saliency:
        return None

    fig, axes = plt.subplots(1, 4, figsize=(14, 3), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    line_panels = [("attn_entropy", entropy), ("attn_distance", distance), ("dla", dla)]
    for ax, (label, values) in zip(axes[:3], line_panels):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        if values:
            ax.plot(list(range(len(values))), values, color=SEQUENTIAL_BLUE, marker="o", markersize=4,
                    linewidth=1.5, zorder=3)
            if label == "dla":
                ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.set_title(label, color=INK_PRIMARY, fontsize=9, loc="left")
        ax.set_xlabel("layer", fontsize=8, color=INK_SECONDARY)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = axes[3]
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    if pos_saliency:
        bucket_labels = list(range(1, len(pos_saliency) + 1))
        ax.bar(bucket_labels, pos_saliency, color=SEQUENTIAL_BLUE, width=0.7, zorder=3)
    ax.set_title("pos_saliency", color=INK_PRIMARY, fontsize=9, loc="left")
    ax.set_xlabel("distance back", fontsize=8, color=INK_SECONDARY)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return _finalize(fig, path, title="Token-level behavioral fingerprint")


# ---------------------------------------------------------------------------
# Part B -- Agent 3 summary charts
# ---------------------------------------------------------------------------

def chart_val_bpb_trend(
    all_metrics: List[Dict[str, Any]],
    noise_floor_path: Path,
    path: Path,
) -> Optional[Path]:
    """The single most important chart: val_bpb per report index as muted
    dots, with a running-minimum "frontier" line emphasized in accent blue
    (emphasis form -- the frontier is the point, individual runs are
    context). Shades a +/-2*sigma band around the frontier's latest value
    when noise_floor_path exists, so a viewer can see at a glance whether a
    drop is inside or outside the measured noise floor. None if there are no
    finite val_bpb entries yet.
    """
    xs, ys = [], []
    for i, item in enumerate(all_metrics):
        val = item.get("val_bpb")
        if isinstance(val, (int, float)) and math.isfinite(val):
            xs.append(i)
            ys.append(float(val))
    if not ys:
        return None

    frontier_x, frontier_y, running_min = [], [], math.inf
    for x, y in zip(xs, ys):
        running_min = min(running_min, y)
        frontier_x.append(x)
        frontier_y.append(running_min)

    fig, ax = _new_figure(figsize=(8, 4))
    ax.scatter(xs, ys, color=INK_MUTED, s=18, alpha=0.6, zorder=2, label="run")
    ax.plot(frontier_x, frontier_y, color=SEQUENTIAL_BLUE, linewidth=2, zorder=4, label="best so far")

    sigma = None
    if Path(noise_floor_path).exists():
        try:
            sigma = float(json.loads(Path(noise_floor_path).read_text())["std"])
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            sigma = None
    if sigma is not None:
        latest_best = frontier_y[-1]
        ax.axhspan(latest_best - 2 * sigma, latest_best + 2 * sigma, color=SEQUENTIAL_BLUE, alpha=0.08, zorder=1,
                   label=f"±2σ noise floor ({sigma:.4f})")

    ax.set_xlabel("report index (chronological)")
    ax.set_ylabel("val_bpb")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finalize(fig, path, title="val_bpb over the campaign")


def chart_hyperparameter_importance_evolution(
    all_metrics: List[Dict[str, Any]],
    path: Path,
) -> Optional[Path]:
    """Multi-line: importance per param across the report history (one
    point per report where that param had enough data, so lines can have
    gaps -- never interpolated over missing data). Categorical hues, capped
    at the palette's 8 slots: keeps the top-6 params by latest importance
    direct-labeled, folds the rest into one muted "Other" line (mean of
    whichever remaining params are present at each report index) -- the
    skill's series-count ladder rule, never a generated 9th/10th hue.
    """
    per_param: Dict[str, List[Tuple[int, float]]] = {}
    for i, item in enumerate(all_metrics):
        importance = item.get("hyperparameter_importance") or {}
        for param, score in importance.items():
            try:
                per_param.setdefault(param, []).append((i, float(score)))
            except (TypeError, ValueError):
                continue
    if not per_param:
        return None

    ranked = sorted(per_param.items(), key=lambda kv: kv[1][-1][1], reverse=True)
    top = ranked[:6]
    rest = ranked[6:]

    fig, ax = _new_figure(figsize=(8, 4.5))
    for (param, points), color in zip(top, CATEGORICAL):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=color, marker="o", markersize=3, linewidth=1.5, label=param, zorder=3)

    if rest:
        by_index: Dict[int, List[float]] = {}
        for _param, points in rest:
            for i, score in points:
                by_index.setdefault(i, []).append(score)
        xs = sorted(by_index.keys())
        ys = [sum(by_index[i]) / len(by_index[i]) for i in xs]
        ax.plot(xs, ys, color=INK_MUTED, linestyle="--", linewidth=1.5,
                label=f"Other ({len(rest)} params)", zorder=2)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("report index (chronological)")
    ax.set_ylabel("importance")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finalize(fig, path, title="Hyperparameter importance evolution")


def chart_status_distribution(statuses: "Counter", path: Path) -> Optional[Path]:
    """Horizontal bar using the fixed status palette semantically
    (STATUS_MAP), not a generic categorical one -- a status color never
    impersonates a series.
    """
    if not statuses:
        return None
    items = sorted(statuses.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    counts = [v for _, v in items]
    colors = [STATUS_COLORS.get(STATUS_MAP.get(label, ""), INK_MUTED) for label in labels]

    fig, ax = _new_figure(figsize=(6, 0.4 * len(labels) + 1.2))
    bars = ax.barh(labels, counts, color=colors, height=0.6, zorder=3)
    ax.set_xlabel("runs")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    for bar in bars:
        ax.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                 str(int(bar.get_width())), va="center", ha="left", fontsize=8, color=INK_SECONDARY)
    return _finalize(fig, path, title="Run status distribution")


def chart_layer_importance_distribution(
    layer_shares: Dict[str, List[float]],
    path: Path,
) -> Optional[Path]:
    """Mean share % per layer with std as an error bar, sequential blue,
    x-axis ordered by layer index.
    """
    if not layer_shares:
        return None
    ordered = sorted(layer_shares.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999)
    labels = [f"L{name}" for name, _ in ordered]
    means = [sum(v) / len(v) for _, v in ordered if v]
    stds = [
        math.sqrt(sum((x - (sum(v) / len(v))) ** 2 for x in v) / len(v)) if len(v) > 1 else 0.0
        for _, v in ordered if v
    ]
    if not means:
        return None

    fig, ax = _new_figure(figsize=(max(6, 0.4 * len(labels) + 1.5), 4))
    ax.bar(labels, means, yerr=stds, color=SEQUENTIAL_BLUE, capsize=3, zorder=3,
           error_kw={"ecolor": INK_SECONDARY, "linewidth": 1})
    ax.set_ylabel("mean layer share (%)")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return _finalize(fig, path, title="Layer-level importance distribution")


def chart_fingerprint_clusters(
    cluster_result: Optional[Dict[str, Any]],
    path: Path,
) -> Optional[Path]:
    """Tier 3.1: mean val_bpb per behavioral-fingerprint cluster, n
    annotated per bar, categorical color per cluster (fixed order, capped
    at 8 -- see CATEGORICAL). None (not a fabricated chart) when there
    isn't yet enough fingerprint history to cluster
    (state/clustering.py::cluster_fingerprints returned None).
    """
    if not cluster_result or not cluster_result.get("clusters"):
        return None
    clusters = [c for c in cluster_result["clusters"] if c.get("mean_val_bpb") is not None]
    if not clusters:
        return None

    labels = [f"Cluster {c['cluster_id']}" for c in clusters]
    means = [c["mean_val_bpb"] for c in clusters]
    ns = [c["n"] for c in clusters]
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(clusters))]

    fig, ax = _new_figure(figsize=(max(6, 1.2 * len(labels) + 1.5), 4))
    bars = ax.bar(labels, means, color=colors, zorder=3)
    ax.set_ylabel("mean val_bpb")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}",
                 va="bottom", ha="center", fontsize=8, color=INK_SECONDARY)
    k = cluster_result.get("k")
    silhouette = cluster_result.get("silhouette")
    subtitle = f"k={k}, silhouette={silhouette:.3f}" if k is not None else None
    return _finalize(fig, path, title=f"Fingerprint clusters vs. val_bpb" + (f" ({subtitle})" if subtitle else ""))


def chart_attention_trajectory_clusters(
    cluster_result: Optional[Dict[str, Any]],
    path: Path,
) -> Optional[Path]:
    """Tier 3.2: each cluster's mean attn_distance shape (resampled onto a
    fixed number of points over normalized depth, min-max normalized per
    curve -- see state/clustering.py::cluster_attention_trajectories) as a
    line, categorical colors, legend labeled by cluster id + n. None when
    there isn't yet enough fingerprint history to cluster.
    """
    if not cluster_result or not cluster_result.get("clusters"):
        return None
    clusters = [c for c in cluster_result["clusters"] if c.get("mean_shape")]
    if not clusters:
        return None

    n_resample = cluster_result.get("n_resample") or len(clusters[0]["mean_shape"])
    xs = [i / (n_resample - 1) for i in range(n_resample)] if n_resample > 1 else [0.0]

    fig, ax = _new_figure(figsize=(7, 4))
    for i, c in enumerate(clusters):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        label = f"Cluster {c['cluster_id']} (n={c['n']})"
        ax.plot(xs, c["mean_shape"], color=color, linewidth=2, marker="o", markersize=4, zorder=3, label=label)
    ax.set_xlabel("normalized depth (0=first layer, 1=last layer)")
    ax.set_ylabel("attn_distance (min-max normalized per run)")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finalize(fig, path, title="Attention-reach trajectory clusters")
