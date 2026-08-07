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
    annotations: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """The single most important chart: val_bpb per report index as muted
    dots, with a running-minimum "frontier" line emphasized in accent blue
    (emphasis form -- the frontier is the point, individual runs are
    context). Shades a +/-2*sigma band around the frontier's latest value
    when noise_floor_path exists, so a viewer can see at a glance whether a
    drop is inside or outside the measured noise floor. Overlays holdout_val_bpb
    as a distinct marker wherever present (sparse -- only computed for a
    handful of final top-K candidates, see scripts/holdout_eval.py -- so a
    few extra points on this chart, not a chart of its own). None if there
    are no finite val_bpb entries yet.

    annotations: optional campaign-level markers (see
    state/campaign_annotations.json / Agent3ReportAnalyst._load_annotations)
    -- e.g. "the search-strategy bug fix landed here" -- drawn as a neutral
    dashed reference line + label, never a data color, so a real regime
    change stays visible on the chart instead of silently mixing two
    different eras of runs with no visual cue. Each needs a "report_index"
    (int) and "label" (str); anything malformed is skipped, never guessed.
    """
    xs, ys = [], []
    holdout_xs, holdout_ys = [], []
    for i, item in enumerate(all_metrics):
        val = item.get("val_bpb")
        if isinstance(val, (int, float)) and math.isfinite(val):
            xs.append(i)
            ys.append(float(val))
        holdout = (item.get("metadata") or {}).get("holdout_val_bpb")
        if isinstance(holdout, (int, float)) and math.isfinite(holdout):
            holdout_xs.append(i)
            holdout_ys.append(float(holdout))
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
    if holdout_xs:
        ax.scatter(holdout_xs, holdout_ys, color=DIVERGING_POS, marker="D", s=36, zorder=5,
                   edgecolors=SURFACE, linewidths=0.8, label="holdout_val_bpb")

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

    for ann in (annotations or []):
        idx = ann.get("report_index")
        label = ann.get("label")
        if not isinstance(idx, int) or not isinstance(label, str) or not label:
            continue
        if idx < min(xs) or idx > max(xs):
            continue  # outside the plotted range -- would render off-axis
        ax.axvline(x=idx, color=INK_SECONDARY, linestyle="--", linewidth=1, alpha=0.6, zorder=3)
        ax.annotate(
            label, xy=(idx, 1.0), xycoords=("data", "axes fraction"),
            xytext=(4, -4), textcoords="offset points",
            fontsize=7, color=INK_SECONDARY, rotation=90, va="top", ha="left",
        )

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


# ---------------------------------------------------------------------------
# Part C -- Tier 1 surrogate diagnostics (agents/search_planner.py)
# ---------------------------------------------------------------------------

def chart_predicted_vs_actual(
    oob_actual: Sequence[float],
    oob_predicted: Sequence[float],
    path: Path,
) -> Optional[Path]:
    """Tier 1 surrogate diagnostic: out-of-bag predicted vs. actual val_bpb
    (state/surrogate.py::fit_surrogate's oob_actual/oob_predicted) -- each
    point's prediction used only trees that didn't see it during training, a
    free held-out-style accuracy check with no separate split needed. Points
    near the y=x reference line mean the surrogate tracks real outcomes, not
    just memorizes them.
    """
    if not oob_actual or not oob_predicted:
        return None
    fig, ax = _new_figure(figsize=(5, 5))
    ax.scatter(oob_actual, oob_predicted, color=SEQUENTIAL_BLUE, s=18, alpha=0.6, zorder=3)
    lo = min(min(oob_actual), min(oob_predicted))
    hi = max(max(oob_actual), max(oob_predicted))
    ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1.5, linestyle="--", zorder=2, label="y = x")
    ax.set_xlabel("actual val_bpb")
    ax.set_ylabel("out-of-bag predicted val_bpb")
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finalize(fig, path, title="Surrogate: predicted vs. actual (out-of-bag)")


def chart_surrogate_sensitivity(
    ranked: List[Tuple[str, float]],
    frozen: List[str],
    path: Path,
) -> Optional[Path]:
    """Tier 1 surrogate diagnostic: S_perf per parameter (coordinate-slice
    sensitivity), frozen/active shown via color -- frozen means measured
    below the noise floor (state/surrogate.py::prune_by_noise_floor), not
    "unimportant."
    """
    if not ranked:
        return None
    frozen_set = set(frozen)
    labels = [p for p, _ in ranked]
    values = [s for _, s in ranked]
    colors = [BASELINE if p in frozen_set else SEQUENTIAL_BLUE for p in labels]

    fig, ax = _new_figure(figsize=(max(6, 0.5 * len(labels) + 1.5), 4))
    ax.bar(labels, values, color=colors, zorder=3)
    ax.set_ylabel("S_perf (coordinate-slice sensitivity)")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    from matplotlib.patches import Patch
    handles = [Patch(color=SEQUENTIAL_BLUE, label="active"), Patch(color=BASELINE, label="frozen (< 2σ)")]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finalize(fig, path, title="Surrogate sensitivity ranking")


def chart_interaction_matrix(
    interaction_scores: Dict[Tuple[str, str], float],
    params: Sequence[str],
    path: Path,
) -> Optional[Path]:
    """Tier 1 surrogate diagnostic: params x params interaction-strength
    heatmap (state/surrogate.py::interaction_matrix's cheap-fANOVA product-
    term importances -- see that function's docstring for what this is a
    substitute for). Sequential, not diverging: these are RF feature
    importances, always >= 0, a magnitude job not a polarity one. Diagonal
    and any missing pair are masked (no self-interaction is computed).
    """
    if not interaction_scores or len(params) < 2:
        return None
    import numpy as np
    n = len(params)
    idx = {p: i for i, p in enumerate(params)}
    grid = np.full((n, n), float("nan"))
    for (a, b), score in interaction_scores.items():
        if a in idx and b in idx:
            grid[idx[a], idx[b]] = score
            grid[idx[b], idx[a]] = score
    finite = grid[~np.isnan(grid)]
    if finite.size == 0:
        return None

    cmap = LinearSegmentedColormap.from_list("sequential", [SURFACE, SEQUENTIAL_BLUE])
    cmap.set_bad(color=GRIDLINE)

    fig, ax = _new_figure(figsize=(max(4, 0.5 * n + 1.5), max(4, 0.5 * n + 1.5)))
    im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=max(float(finite.max()), 1e-9), aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(list(params), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(params), fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("interaction strength (product-term importance)", color=INK_SECONDARY, fontsize=8)
    cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=7)
    return _finalize(fig, path, title="Parameter interaction matrix")


def chart_ei_candidates(diagnostics: Dict[str, Any], path: Path) -> Optional[Path]:
    """Tier 1 surrogate diagnostic: the candidates Expected Improvement
    actually considered this cycle (state/surrogate.py::propose_via_ei's
    return_diagnostics=True output) -- one panel per free parameter, each
    sampled candidate's value on the x-axis vs. its EI score on the y-axis,
    the chosen candidate highlighted. Shows whether EI is concentrating on a
    clear region or scattered (no strong signal yet).
    """
    if not diagnostics or not diagnostics.get("free_params"):
        return None
    free_params = diagnostics["free_params"]
    candidate_values = diagnostics.get("candidate_values") or {}
    eis = diagnostics.get("eis") or []
    best_idx = diagnostics.get("best_idx")
    if not eis or not candidate_values:
        return None

    fig, axes = plt.subplots(1, len(free_params), figsize=(max(5, 4 * len(free_params)), 4), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    axes = axes[0]
    for ax, param in zip(axes, free_params):
        values = candidate_values.get(param) or []
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
        if values and len(values) == len(eis):
            ax.scatter(values, eis, color=INK_MUTED, s=10, alpha=0.5, zorder=2, label="candidate")
            if isinstance(best_idx, int) and 0 <= best_idx < len(values):
                ax.scatter([values[best_idx]], [eis[best_idx]], color=DIVERGING_POS, s=60, zorder=4,
                           marker="*", label="chosen")
        ax.set_title(param, color=INK_PRIMARY, fontsize=9, loc="left")
        ax.set_xlabel(param, fontsize=8, color=INK_SECONDARY)
        if ax is axes[0]:
            ax.set_ylabel("Expected Improvement", fontsize=8, color=INK_SECONDARY)
    axes[-1].legend(loc="best", frameon=False, fontsize=7, labelcolor=INK_SECONDARY)
    return _finalize(fig, path, title="EI acquisition: candidates considered this cycle")


def chart_sobol_coverage(
    cold_start_points: List[Dict[str, Any]],
    params: Sequence[str],
    path: Path,
) -> Optional[Path]:
    """Tier 1 cold-start diagnostic: one 1D rug of sampled values per
    parameter (small multiples, mirrors chart_layer_scalars' layout),
    confirming the Sobol design (state/surrogate.py::sobol_cold_start)
    actually spreads evenly across each dimension rather than clumping.
    """
    if not cold_start_points:
        return None
    n_cols = min(len(params), 6)
    n_rows = math.ceil(len(params) / n_cols) if n_cols else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(6, 2.2 * n_cols), 2.2 * n_rows), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    flat_axes = [ax for row in axes for ax in row]
    for ax, param in zip(flat_axes, params):
        values = [p.get(param) for p in cold_start_points if isinstance(p.get(param), (int, float))]
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7)
        ax.set_yticks([])
        if values:
            ax.scatter(values, [0.0] * len(values), color=SEQUENTIAL_BLUE, s=20, alpha=0.7, zorder=3)
        ax.set_title(param, color=INK_PRIMARY, fontsize=8, loc="left")
    for ax in flat_axes[len(params):]:
        ax.set_visible(False)
    return _finalize(fig, path, title="Sobol cold-start design coverage")


def chart_fingerprint_adjustments_trend(decision_logs: List[Dict[str, Any]], path: Path) -> Optional[Path]:
    """Tier 4 diagnostic: which fingerprint-driven rule fired on which
    param, and what delta, over the campaign (agents/agent1_training_specialist.py's
    _fingerprint_adjustment, recorded per decision log). One line per param
    that ever received a fingerprint-driven vote -- most params will never
    appear here, which is itself informative (the rules aren't touching
    everything).
    """
    if not decision_logs:
        return None
    series: Dict[str, List[Tuple[int, float]]] = {}
    for log in decision_logs:
        iteration = log.get("iteration")
        if not isinstance(iteration, int):
            continue
        for entry in log.get("fingerprint_adjustments") or []:
            param, delta = entry.get("param"), entry.get("delta")
            if param and isinstance(delta, (int, float)):
                series.setdefault(param, []).append((iteration, float(delta)))
    if not series:
        return None

    fig, ax = _new_figure(figsize=(8, 4))
    for i, (param, points) in enumerate(sorted(series.items())):
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(xs, ys, color=color, marker="o", markersize=4, linewidth=1.5, zorder=3, label=param)
    ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("fingerprint-driven delta")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finalize(fig, path, title="Tier 4: fingerprint-driven architecture adjustments")


def chart_pipeline_issues_trend(issue_logs: List[Dict[str, Any]], path: Path) -> Optional[Path]:
    """agents/pipeline_validator.py diagnostic: FATAL/ERROR/WARN issue
    counts per iteration for the current run (agents/agent3_report_analyst.py's
    _load_latest_run_issues), stacked bar, fixed status-style severity
    colors -- a spike or a rising baseline is a workflow-health signal on
    its own, independent of what any individual issue says.
    """
    if not issue_logs:
        return None
    by_iteration: Dict[int, Counter] = {}
    for log in issue_logs:
        iteration = log.get("iteration")
        if not isinstance(iteration, int):
            continue
        counts = by_iteration.setdefault(iteration, Counter())
        for issue in log.get("issues") or []:
            severity = issue.get("severity")
            if severity:
                counts[severity] += 1
    if not by_iteration:
        return None

    iterations = sorted(by_iteration.keys())
    severities = [("FATAL", "critical"), ("ERROR", "serious"), ("WARN", "warning")]
    fig, ax = _new_figure(figsize=(8, 4))
    bottom = [0] * len(iterations)
    for severity, status_key in severities:
        values = [by_iteration[i].get(severity, 0) for i in iterations]
        if any(values):
            ax.bar(iterations, values, bottom=bottom, color=STATUS_COLORS[status_key], label=severity, zorder=3)
            bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xlabel("iteration")
    ax.set_ylabel("issue count")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    return _finalize(fig, path, title="Pipeline validation issues (this run)")


def chart_token_fingerprint_scalars_evolution(all_metrics: List[Dict[str, Any]], path: Path) -> Optional[Path]:
    """Tier 2 diagnostic: attn_distance_slope and induction_score (the two
    scalar, not per-layer, fingerprint fields) over report history -- same
    "evolution over the campaign" style as chart_hyperparameter_importance_evolution,
    which every other Tier 2 quantity lacked before this (chart_token_fingerprint
    only ever showed one run's snapshot).
    """
    slope_xs, slope_ys = [], []
    induction_xs, induction_ys = [], []
    for i, item in enumerate(all_metrics):
        fp = item.get("token_fingerprint") or {}
        slope = fp.get("attn_distance_slope")
        if isinstance(slope, (int, float)) and math.isfinite(slope):
            slope_xs.append(i)
            slope_ys.append(float(slope))
        induction = fp.get("induction_score")
        if isinstance(induction, (int, float)) and math.isfinite(induction):
            induction_xs.append(i)
            induction_ys.append(float(induction))
    if not slope_ys and not induction_ys:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    panels = [("attn_distance_slope", slope_xs, slope_ys), ("induction_score", induction_xs, induction_ys)]
    for ax, (label, xs, ys) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        if ys:
            ax.plot(xs, ys, color=SEQUENTIAL_BLUE, marker="o", markersize=4, linewidth=1.5, zorder=3)
            if label == "attn_distance_slope":
                ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        ax.set_title(label, color=INK_PRIMARY, fontsize=9, loc="left")
        ax.set_xlabel("report index (chronological)", fontsize=8, color=INK_SECONDARY)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return _finalize(fig, path, title="Tier 2 scalar fingerprint fields over the campaign")


def chart_noise_floor_trend(history: List[Dict[str, Any]], path: Path) -> Optional[Path]:
    """scripts/noise_floor.py diagnostic: mean +/- std per measurement over
    time, now that noise_floor.json tracks a real history instead of a
    single overwritten snapshot. A single point is a valid (if sparse)
    chart -- still worth seeing on its own rather than only as a passive
    band on the val_bpb trend.
    """
    if not history:
        return None
    xs = list(range(len(history)))
    means = [h.get("mean") for h in history]
    stds = [h.get("std") for h in history]
    if not all(isinstance(m, (int, float)) for m in means) or not all(isinstance(s, (int, float)) for s in stds):
        return None

    fig, ax = _new_figure(figsize=(7, 4))
    ax.errorbar(xs, means, yerr=stds, color=SEQUENTIAL_BLUE, marker="o", markersize=5, linewidth=1.5,
                capsize=4, ecolor=INK_SECONDARY, zorder=3)
    ax.set_xlabel("measurement # (chronological)")
    ax.set_ylabel("val_bpb (mean ± std across repeats)")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return _finalize(fig, path, title="Noise floor over time")


# ---------------------------------------------------------------------------
# Part D -- optimization landscape (state/landscape.py)
# ---------------------------------------------------------------------------

# Predicted terrain gets its own fixed hue, deliberately chosen from the
# categorical theme rather than the sequential/diverging ramps: height already
# encodes predicted val_bpb, so color here is doing pure "which point set is
# this" work. Purple is distinct from the blue real runs, from every
# STATUS_COLORS entry, and from DIVERGING_POS/NEG -- so it can never be
# misread as a good/bad judgement.
PREDICTED_TERRAIN = CATEGORICAL[6]  # "#4a3aa7"

# Confidence maps to opacity, floored/capped so the least-certain terrain is
# still faintly visible and the most-certain never reads as solid (which is
# reserved for measured runs).
PREDICTED_ALPHA_RANGE = (0.15, 0.85)

# Region flags written by the registry (state/regions.py). Fixed
# marker + color per flag, same "one lookup table, never improvised" pattern
# as STATUS_COLORS.
# Keys are state/regions.py's lifecycle flags. "investigating" used to be
# here too and is gone with the exploration-window model: a region being
# searched is simply "currently_exploiting" now, since several are searched at
# once and there is no transient "we are checking this one out" state. "merged"
# never reaches a chart -- RegionRegistry.flags_snapshot() omits absorbed
# regions, whose anchor is by definition on top of the survivor's.
REGION_FLAG_STYLES = {
    "currently_exploiting": {"marker": "*", "color": STATUS_COLORS["good"], "label": "exploiting now"},
    "no_optimum":           {"marker": "x", "color": INK_MUTED, "label": "no optimum found"},
    "local_optimum":        {"marker": "^", "color": DIVERGING_POS, "label": "local optimum"},
    "exploitation_paused":  {"marker": "s", "color": INK_MUTED, "label": "exploitation paused"},
    "capacity_paused":      {"marker": "d", "color": INK_SECONDARY, "label": "paused: no GPU free"},
}
_REGION_FLAG_FALLBACK = {"marker": "P", "color": INK_MUTED, "label": "flagged"}


def _hex_to_rgb(value: str) -> Tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def chart_optimization_landscape(
    landscape: Optional[Dict[str, Any]],
    path: Path,
    region_flags: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """The optimization landscape as terrain: a 2D PCA-compressed floor of
    the tuned hyperparameters, with val_bpb as height.

    Two overlaid sources, per state/landscape.py::build_landscape:
      - measured runs, drawn solid in the sequential blue;
      - the surrogate's predicted surface over never-tried configurations,
        drawn in one fixed hue whose *opacity* is its prediction confidence
        (across-tree spread) -- faint where the model is guessing.

    Region flags from Agent 4 (local optimum / exploitation paused / no
    optimum / currently exploiting) are placed by re-projecting their raw
    hyperparameters onto this landscape's current PCA basis; a flag that
    can't be projected is skipped rather than drawn somewhere invented.

    Returns None (never a fabricated chart) when there's no landscape.

    Read this as an approximation, not ground truth: compressing 11
    dimensions to 2 and inverse-transforming back is lossy, so the surface
    is a projection of the surrogate's belief. The title carries that
    caveat and the explained-variance fraction that quantifies it.
    """
    if not landscape:
        return None
    real_points = landscape.get("real_points") or []
    grid_x = landscape.get("grid_x") or []
    grid_y = landscape.get("grid_y") or []
    grid_z = landscape.get("grid_z_mean") or []
    if not real_points or not grid_x or not grid_y or not grid_z:
        return None

    import numpy as np  # local: only this chart needs it, matching module convention

    fig = plt.figure(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(SURFACE)
    # _new_figure is 2D-only (it styles left/bottom spines); replicate the
    # equivalent recessive styling for the three 3D panes by hand.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((*_hex_to_rgb(SURFACE), 1.0))
        axis._axinfo["grid"].update(color=GRIDLINE, linewidth=0.6)
        axis.line.set_color(BASELINE)
        axis.label.set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)

    # -- predicted terrain: one surface, per-face alpha = confidence --------
    mesh_x, mesh_y = np.meshgrid(np.array(grid_x), np.array(grid_y))
    mesh_z = np.array(grid_z)
    confidence = np.array(landscape.get("grid_confidence") or np.full(mesh_z.shape, 0.5))
    lo, hi = PREDICTED_ALPHA_RANGE
    alphas = lo + np.clip(confidence, 0.0, 1.0) * (hi - lo)
    rgb = _hex_to_rgb(PREDICTED_TERRAIN)
    facecolors = np.empty(mesh_z.shape + (4,))
    facecolors[..., 0], facecolors[..., 1], facecolors[..., 2] = rgb
    facecolors[..., 3] = alphas
    ax.plot_surface(
        mesh_x, mesh_y, mesh_z, facecolors=facecolors, shade=False,
        linewidth=0, antialiased=True, rstride=1, cstride=1, zorder=1,
    )

    # -- measured runs: solid, opaque, on top -------------------------------
    ax.scatter(
        [p["x"] for p in real_points], [p["y"] for p in real_points], [p["z"] for p in real_points],
        color=SEQUENTIAL_BLUE, s=26, alpha=1.0, depthshade=False, zorder=5,
    )

    # -- Agent 4 region flags ------------------------------------------------
    drawn_flags = {}
    if region_flags:
        from state.landscape import project_point
        # Flags float above everything (terrain and measured runs alike) with
        # a stem dropped to the terrain: at a marker's own height inside the
        # cloud they'd read as data points, and 3D perspective makes a bare
        # floating marker's (x, y) genuinely ambiguous.
        z_floor = min(float(np.min(mesh_z)), min(p["z"] for p in real_points))
        z_ceiling = max(float(np.max(mesh_z)), max(p["z"] for p in real_points))
        z_flag = z_ceiling + (z_ceiling - z_floor) * 0.12
        for entry in region_flags:
            flag = entry.get("flag")
            point = project_point(entry.get("hyperparams") or {}, landscape)
            if point is None:
                continue  # unplaceable -- skip rather than guess a position
            style = REGION_FLAG_STYLES.get(flag, _REGION_FLAG_FALLBACK)
            ax.plot([point[0], point[0]], [point[1], point[1]], [z_floor, z_flag],
                    color=style["color"], linewidth=1.0, alpha=0.55, zorder=6)
            ax.scatter([point[0]], [point[1]], [z_flag], marker=style["marker"],
                       color=style["color"], s=140, depthshade=False, zorder=7)
            drawn_flags[flag] = style

    # -- legend: identity is never color-alone ------------------------------
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=SEQUENTIAL_BLUE,
                   markersize=7, label=f"measured runs (n={len(real_points)})"),
        plt.Line2D([], [], marker="s", linestyle="none", color=PREDICTED_TERRAIN,
                   markersize=8, alpha=0.7, label="surrogate prediction (opacity = confidence)"),
    ]
    for flag, style in drawn_flags.items():
        handles.append(plt.Line2D([], [], marker=style["marker"], linestyle="none",
                                  color=style["color"], markersize=8, label=style["label"]))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7,
              labelcolor=INK_SECONDARY, bbox_to_anchor=(0.0, 1.0))

    ax.set_xlabel("PCA component 1 (compressed hyperparameters)", fontsize=8)
    ax.set_ylabel("PCA component 2", fontsize=8)
    ax.set_zlabel("val_bpb (lower is better)", fontsize=8)
    # Fixed viewing angle so charts from different points in the campaign are
    # visually comparable to each other rather than arbitrarily rotated.
    ax.view_init(elev=25, azim=-60)

    variance = sum(landscape.get("explained_variance_ratio") or [])
    title = (f"Optimization landscape — approximate (PCA-compressed, "
             f"{landscape.get('n_real', len(real_points))} real runs, "
             f"{variance:.0%} of hyperparameter variance explained)")
    return _finalize(fig, path, title=title)
