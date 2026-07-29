"""Standalone aggregate campaign dashboard: reads results.tsv, every Agent 2
report, and (if present) the latest Tier 1 search-plan JSON, and renders one
self-contained reports/visuals/dashboard.html -- no external CDN/JS
dependency, inline SVG charts, hover tooltips (native SVG <title>, robust
and zero-JS), dual light/dark mode via CSS custom properties, a legend on
every multi-series chart, and a table fallback for the status distribution.

Not wired into the orchestrator loop (unlike the per-report/summary charts
in state/visualize.py, which run every report/summary) -- this aggregates
everything and is cheap to regenerate on demand instead.

Usage:
    uv run python scripts/visualize_dashboard.py
"""

import html
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.agent3_report_analyst import Agent3ReportAnalyst
from agents.search_planner import STATE_PATH_DEFAULT, SearchPlannerState
from state.clustering import cluster_attention_trajectories, cluster_fingerprints
from state.results_analysis import HYPERPARAM_COLUMNS, load_results

OUTPUT_PATH = Path("reports/visuals/dashboard.html")
NOISE_FLOOR_PATH = Path("state/noise_floor.json")
SEARCH_PLAN_DIR = Path("reports/agent1_search_plan")
SEARCH_PLANNER_STATE_PATH = Path(STATE_PATH_DEFAULT)

# --- Reference palette (dataviz skill, references/palette.md) --------------
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#2a78d6", "#f0efec", "#e34948"
STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_MAP = {
    "remote_ok": "good", "ok": "good",
    "simulated": "warning", "dry_run": "warning",
    "remote_error": "critical", "timeout": "critical",
}


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c))) for c in rgb)


def _lerp_color(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t))


def _diverging_color(value: float, bound: float) -> str:
    if bound <= 0 or not math.isfinite(value):
        return DIVERGING_MID
    t = max(-1.0, min(1.0, value / bound))
    return _lerp_color(DIVERGING_MID, DIVERGING_POS, t) if t >= 0 else _lerp_color(DIVERGING_MID, DIVERGING_NEG, -t)


def _sequential_color(value: float, vmax: float) -> str:
    """Sequential (not diverging): for magnitude-only, always-non-negative
    data (e.g. RF feature importances) -- diverging color implies a
    meaningful zero-crossing/polarity these values don't have."""
    if vmax <= 0 or not math.isfinite(value):
        return "#f0efec"
    t = max(0.0, min(1.0, value / vmax))
    return _lerp_color("#f0efec", CATEGORICAL[0], t)


# ---------------------------------------------------------------------------
# SVG chart builders. Tooltips are native SVG <title> elements -- a real
# hover tooltip mechanism with zero JS, works in every browser, fully
# offline. The only JS on the page is the light/dark theme toggle.
# ---------------------------------------------------------------------------

def svg_line_chart(
    series: List[Dict[str, Any]],
    scatter: Optional[List[Tuple[float, float]]] = None,
    band: Optional[Tuple[float, float]] = None,
    highlight: Optional[List[Tuple[float, float]]] = None,
    highlight_label: str = "highlight",
    width: int = 760, height: int = 320, pad: int = 50,
) -> str:
    """highlight: a second, distinctly-styled scatter overlay (diamond
    marker, accent-red) for sparse points that need to stand out from the
    regular per-run dots -- e.g. holdout_val_bpb among ordinary val_bpb runs."""
    all_xs = [x for s in series for x, _ in s["points"]]
    all_ys = [y for s in series for _, y in s["points"]]
    if scatter:
        all_xs += [x for x, _ in scatter]
        all_ys += [y for _, y in scatter]
    if highlight:
        all_xs += [x for x, _ in highlight]
        all_ys += [y for _, y in highlight]
    if not all_xs:
        return '<p class="empty">No data yet.</p>'
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    if band:
        y_min, y_max = min(y_min, band[0]), max(y_max, band[1])
    x_range = (x_max - x_min) or 1
    y_pad = ((y_max - y_min) or abs(y_max) or 1) * 0.1
    y_min, y_max = y_min - y_pad, y_max + y_pad
    y_range = (y_max - y_min) or 1

    def X(x): return pad + (x - x_min) / x_range * (width - 2 * pad)
    def Y(y): return height - pad - (y - y_min) / y_range * (height - 2 * pad)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i in range(5):
        gy = pad + i * (height - 2 * pad) / 4
        parts.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" class="gridline" />')
    if band:
        y1, y2 = Y(band[1]), Y(band[0])
        parts.append(f'<rect x="{pad}" y="{y1:.1f}" width="{width - 2 * pad}" height="{(y2 - y1):.1f}" class="band" />')
    if scatter:
        for x, y in scatter:
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" class="dot-muted">'
                          f'<title>run {x:.0f}: {y:.6f}</title></circle>')
    for s in series:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in s["points"])
        dash = ' stroke-dasharray="6,4"' if s.get("style") == "dashed" else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" stroke-width="2.5"{dash} />')
        for x, y in s["points"]:
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{s["color"]}" class="hover-dot">'
                          f'<title>{_esc(s["label"])} — {x:.0f}: {y:.6f}</title></circle>')
    if highlight:
        for x, y in highlight:
            cx, cy = X(x), Y(y)
            r = 5.5
            points = f"{cx},{cy - r} {cx + r},{cy} {cx},{cy + r} {cx - r},{cy}"
            parts.append(f'<polygon points="{points}" fill="{DIVERGING_POS}" stroke="var(--surface-1)" stroke-width="1">'
                          f'<title>{_esc(highlight_label)} — {x:.0f}: {y:.6f}</title></polygon>')
    parts.append(f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="axis" />')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" class="axis" />')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_stacked_bar_chart_v(
    x_labels: List[str],
    series: List[Tuple[str, List[float], str]],
    width: int = 760, height: int = 280, pad: int = 46,
) -> str:
    """Vertical stacked bars, one bar per x_labels entry, one segment per
    (name, values, color) in series -- used for FATAL/ERROR/WARN issue
    counts per iteration."""
    if not x_labels or not series:
        return '<p class="empty">No data yet.</p>'
    totals = [sum(vals[i] for _, vals, _ in series) for i in range(len(x_labels))]
    max_total = max(totals) or 1
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    bar_w = plot_w / len(x_labels) * 0.7
    gap = plot_w / len(x_labels)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i in range(5):
        gy = pad + i * plot_h / 4
        parts.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{width - pad}" y2="{gy:.1f}" class="gridline" />')
    for i, label in enumerate(x_labels):
        x0 = pad + i * gap + (gap - bar_w) / 2
        y_cursor = height - pad
        for name, values, color in series:
            v = values[i] if i < len(values) else 0
            if v <= 0:
                continue
            seg_h = (v / max_total) * plot_h
            y0 = y_cursor - seg_h
            parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{seg_h:.1f}" fill="{color}">'
                          f'<title>iteration {label} — {_esc(name)}: {v:.0f}</title></rect>')
            y_cursor = y0
        parts.append(f'<text x="{x0 + bar_w / 2:.1f}" y="{height - pad + 14}" text-anchor="middle" class="bar-label">{_esc(label)}</text>')
    parts.append(f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="axis" />')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" class="axis" />')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_scatter_chart(
    points: List[Tuple[float, float]],
    ref_line: Optional[List[Tuple[float, float]]] = None,
    highlight_idx: Optional[int] = None,
    width: int = 320, height: int = 260, pad: int = 40,
) -> str:
    """A single scatter plot, optionally with a reference line (e.g. y=x)
    and one highlighted point (e.g. the chosen EI candidate) -- the small-
    multiples building block for predicted-vs-actual, EI candidates, and
    Sobol coverage, each of which just calls this once per panel."""
    if not points:
        return '<p class="empty">No data yet.</p>'
    all_xs = [x for x, _ in points] + ([x for x, _ in ref_line] if ref_line else [])
    all_ys = [y for _, y in points] + ([y for _, y in ref_line] if ref_line else [])
    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)
    x_range = (x_max - x_min) or 1
    y_range = (y_max - y_min) or 1

    def X(x): return pad + (x - x_min) / x_range * (width - 2 * pad)
    def Y(y): return height - pad - (y - y_min) / y_range * (height - 2 * pad)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    if ref_line:
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in ref_line)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--baseline)" stroke-width="1.5" stroke-dasharray="5,4" />')
    for i, (x, y) in enumerate(points):
        if highlight_idx is not None and i == highlight_idx:
            continue
        parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.5" class="dot-muted">'
                      f'<title>{x:.4g}, {y:.4g}</title></circle>')
    if highlight_idx is not None and 0 <= highlight_idx < len(points):
        hx, hy = points[highlight_idx]
        parts.append(f'<circle cx="{X(hx):.1f}" cy="{Y(hy):.1f}" r="5" fill="{DIVERGING_POS}" class="hover-dot">'
                      f'<title>chosen: {hx:.4g}, {hy:.4g}</title></circle>')
    parts.append(f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="axis" />')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" class="axis" />')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_small_multiples(panels: List[Tuple[str, str]]) -> str:
    """Wraps N (title, svg_or_html) panels in a responsive grid -- shared
    layout for the EI-candidates and Sobol-coverage sections, each of which
    is naturally one mini-chart per parameter rather than one combined SVG."""
    if not panels:
        return '<p class="empty">No data yet.</p>'
    cells = "".join(
        f'<div class="mini-panel"><p class="caption">{_esc(title)}</p>{content}</div>'
        for title, content in panels
    )
    return f'<div class="mini-grid">{cells}</div>'


def svg_bar_chart_h(
    items: List[Tuple[str, float, str]],
    width: int = 720, bar_height: int = 26, gap: int = 10, label_width: int = 160,
    value_fmt: str = "{:.3f}",
) -> str:
    if not items:
        return '<p class="empty">No data yet.</p>'
    max_val = max(v for _, v, _ in items) or 1.0
    pad_top, pad_bottom = 10, 30
    height = pad_top + len(items) * (bar_height + gap) + pad_bottom
    plot_width = width - label_width - 70

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for i, (label, value, color) in enumerate(items):
        y = pad_top + i * (bar_height + gap)
        bar_w = max(1.0, (value / max_val) * plot_width)
        parts.append(f'<text x="{label_width - 10}" y="{y + bar_height / 2 + 4}" text-anchor="end" class="bar-label">{_esc(label)}</text>')
        parts.append(f'<rect x="{label_width}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" fill="{color}" rx="3">'
                      f'<title>{_esc(label)}: {value_fmt.format(value)}</title></rect>')
        parts.append(f'<text x="{label_width + bar_w + 8:.1f}" y="{y + bar_height / 2 + 4}" class="bar-value">{value_fmt.format(value)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_heatmap(
    grid: List[List[Optional[float]]], row_labels: List[str], col_labels: List[str],
    cell: int = 42, pad_left: int = 46, pad_top: int = 24,
    diverging: bool = True,
) -> str:
    """diverging=False switches to _sequential_color for magnitude-only,
    always-non-negative data (e.g. the interaction-matrix section's RF
    product-term importances) instead of implying a polarity these values
    don't have."""
    if not grid or not grid[0]:
        return '<p class="empty">No data yet.</p>'
    finite = [v for row in grid for v in row if v is not None and math.isfinite(v)]
    if not finite:
        return '<p class="empty">No data yet.</p>'
    bound = max(abs(min(finite)), abs(max(finite)), 1e-9) if diverging else max(max(finite), 1e-9)
    width = pad_left + len(col_labels) * cell + 10
    height = pad_top + len(row_labels) * cell + 10

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']
    for c, label in enumerate(col_labels):
        parts.append(f'<text x="{pad_left + c * cell + cell / 2}" y="{pad_top - 8}" text-anchor="middle" class="bar-label">{_esc(label)}</text>')
    for r, label in enumerate(row_labels):
        parts.append(f'<text x="{pad_left - 8}" y="{pad_top + r * cell + cell / 2 + 4}" text-anchor="end" class="bar-label">{_esc(label)}</text>')
        for c in range(len(col_labels)):
            value = grid[r][c] if c < len(grid[r]) else None
            if value is not None:
                color = _diverging_color(value, bound) if diverging else _sequential_color(value, bound)
            else:
                color = "var(--gridline)"
            x, y = pad_left + c * cell, pad_top + r * cell
            title = f"{row_labels[r]} × {col_labels[c]}: {value:.6f}" if value is not None else "no data"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{color}" rx="2">'
                          f'<title>{_esc(title)}</title></rect>')
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data loading (reuses Agent3's existing report loader/parser -- no new
# report-parsing logic duplicated here).
# ---------------------------------------------------------------------------

def _load_data() -> Dict[str, Any]:
    results = load_results("results.tsv")
    agent3 = Agent3ReportAnalyst()
    all_reports = agent3._load_all_reports()
    all_metrics = [agent3._extract_structured_metrics(content) for _, content in all_reports]

    latest_ablation = None
    for item in reversed(all_metrics):
        if item.get("ablation_ran") and item.get("head_importance"):
            latest_ablation = item
            break

    latest_plan = None
    if SEARCH_PLAN_DIR.exists():
        plan_files = sorted(SEARCH_PLAN_DIR.glob("plan_*.json"))
        if plan_files:
            try:
                latest_plan = json.loads(plan_files[-1].read_text())
            except (json.JSONDecodeError, OSError):
                latest_plan = None

    sigma = None
    noise_floor_history: List[Dict[str, Any]] = []
    if NOISE_FLOOR_PATH.exists():
        try:
            noise_floor_payload = json.loads(NOISE_FLOOR_PATH.read_text())
            sigma = float(noise_floor_payload["std"])
            noise_floor_history = noise_floor_payload.get("history", [])
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            sigma = None

    # dev/checks.txt visualization-gaps pass: Tier 4 decision logs and
    # pipeline_validator's per-run issues, via Agent3's own loaders so the
    # parsing/globbing convention isn't duplicated here.
    decision_logs = agent3._load_all_decision_logs()
    issue_logs = agent3._load_latest_run_issues()

    # Tier 3 clusters: recomputed from all_metrics the same way Agent3 does
    # internally (_format_statistical_summary) -- there's no persisted
    # cluster artifact to read, clustering is cheap enough to redo on demand.
    fingerprint_rows = [
        {**(item.get("token_fingerprint") or {}), "val_bpb": item.get("val_bpb")}
        for item in all_metrics
        if item.get("token_fingerprint")
    ]
    overall_clusters = cluster_fingerprints(fingerprint_rows, min_n=agent3.min_cluster_n)
    trajectory_clusters = cluster_attention_trajectories(fingerprint_rows, min_n=agent3.min_cluster_n)

    cold_start_points: List[Dict[str, Any]] = []
    if SEARCH_PLANNER_STATE_PATH.exists():
        try:
            cold_start_points = SearchPlannerState.load(str(SEARCH_PLANNER_STATE_PATH)).cold_start_points
        except Exception:
            cold_start_points = []

    return {
        "results": results,
        "all_metrics": all_metrics,
        "latest_ablation": latest_ablation,
        "latest_plan": latest_plan,
        "sigma": sigma,
        "noise_floor_history": noise_floor_history,
        "decision_logs": decision_logs,
        "issue_logs": issue_logs,
        "overall_clusters": overall_clusters,
        "trajectory_clusters": trajectory_clusters,
        "cold_start_points": cold_start_points,
        "search_params": list(HYPERPARAM_COLUMNS),
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section_trend(
    results: List[Dict[str, Any]], sigma: Optional[float],
    all_metrics: Optional[List[Dict[str, Any]]] = None,
) -> str:
    points = [(i, r["val_bpb"]) for i, r in enumerate(results) if "val_bpb" in r and math.isfinite(r["val_bpb"])]
    if not points:
        return svg_line_chart([])
    frontier, running_min = [], math.inf
    for x, y in points:
        running_min = min(running_min, y)
        frontier.append((x, running_min))
    band = (frontier[-1][1] - 2 * sigma, frontier[-1][1] + 2 * sigma) if sigma else None
    # holdout_val_bpb is sparse (top-K candidates only, see
    # scripts/holdout_eval.py) and lives in Agent 2's report metadata, not
    # results.tsv -- report index and results.tsv row index track each
    # other 1:1 in normal operation (one accepted run produces one of each).
    holdout_points = []
    for i, item in enumerate(all_metrics or []):
        holdout = (item.get("metadata") or {}).get("holdout_val_bpb")
        if isinstance(holdout, (int, float)) and math.isfinite(holdout):
            holdout_points.append((i, float(holdout)))
    chart = svg_line_chart(
        series=[{"label": "best so far", "color": CATEGORICAL[0], "points": frontier}],
        scatter=points, band=band,
        highlight=holdout_points or None, highlight_label="holdout_val_bpb",
    )
    legend = (f'<p class="legend"><span class="swatch" style="background:{CATEGORICAL[0]}"></span> best so far '
              f'&nbsp;&nbsp;<span class="swatch-dot"></span> individual run'
              + (f" &nbsp;&nbsp;<span class=\"swatch-band\"></span> ±2σ noise floor ({sigma:.4f})" if sigma else "")
              + (" &nbsp;&nbsp;◆ holdout_val_bpb" if holdout_points else "")
              + "</p>")
    return chart + legend


def section_importance(all_metrics: List[Dict[str, Any]]) -> str:
    latest = next((m for m in reversed(all_metrics) if m.get("hyperparameter_importance")), None)
    if not latest:
        return '<p class="empty">Not enough historical runs yet.</p>'
    importance = latest["hyperparameter_importance"]
    items = sorted(importance.items(), key=lambda kv: -kv[1])
    return svg_bar_chart_h([(k, v, CATEGORICAL[0]) for k, v in items])


def section_importance_evolution(all_metrics: List[Dict[str, Any]]) -> str:
    per_param: Dict[str, List[Tuple[int, float]]] = {}
    for i, item in enumerate(all_metrics):
        for param, score in (item.get("hyperparameter_importance") or {}).items():
            try:
                per_param.setdefault(param, []).append((i, float(score)))
            except (TypeError, ValueError):
                continue
    if not per_param:
        return '<p class="empty">Not enough historical runs yet.</p>'
    ranked = sorted(per_param.items(), key=lambda kv: kv[1][-1][1], reverse=True)
    top, rest = ranked[:6], ranked[6:]
    series = [{"label": param, "color": color, "points": points}
              for (param, points), color in zip(top, CATEGORICAL)]
    if rest:
        by_index: Dict[int, List[float]] = {}
        for _param, points in rest:
            for i, score in points:
                by_index.setdefault(i, []).append(score)
        other_points = [(i, sum(v) / len(v)) for i, v in sorted(by_index.items())]
        series.append({"label": f"Other ({len(rest)})", "color": "#898781", "points": other_points, "style": "dashed"})
    chart = svg_line_chart(series)
    legend = '<p class="legend">' + " &nbsp;&nbsp;".join(
        f'<span class="swatch" style="background:{s["color"]}"></span> {_esc(s["label"])}' for s in series
    ) + "</p>"
    return chart + legend


def section_status(results: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for r in results:
        status = str(r.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    if not counts:
        return '<p class="empty">No runs yet.</p>'
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    bars = svg_bar_chart_h(
        [(label, count, STATUS_COLORS.get(STATUS_MAP.get(label, ""), "#898781")) for label, count in items],
        value_fmt="{:.0f}",
    )
    table_rows = "".join(f"<tr><td>{_esc(label)}</td><td>{count}</td></tr>" for label, count in items)
    table = (f'<table class="data-table"><caption>Status distribution (table view)</caption>'
             f"<thead><tr><th>status</th><th>runs</th></tr></thead><tbody>{table_rows}</tbody></table>")
    return bars + table


def section_head_heatmap(latest_ablation: Optional[Dict[str, Any]]) -> str:
    if not latest_ablation:
        return '<p class="empty">No run with ablation data yet.</p>'
    head_importance = latest_ablation["head_importance"]
    hyperparams = latest_ablation.get("hyperparams", {})
    n_layer = int(hyperparams.get("n_layer", 0) or 0)
    n_head = int(hyperparams.get("n_head", 0) or 0)
    if not n_layer or not n_head:
        return '<p class="empty">Missing architecture info for the latest ablation run.</p>'
    grid: List[List[Optional[float]]] = [[None] * n_head for _ in range(n_layer)]
    for key, impact in head_importance.items():
        try:
            layer_str, head_str = key.split("_")
            li, hi = int(layer_str[1:]), int(head_str[1:])
        except (ValueError, IndexError):
            continue
        if 0 <= li < n_layer and 0 <= hi < n_head:
            grid[li][hi] = impact
    caption = f'<p class="caption">Model: {_esc(latest_ablation.get("model_id", "?"))}</p>'
    return caption + svg_heatmap(grid, [f"L{i}" for i in range(n_layer)], [f"H{i}" for i in range(n_head)])


def section_search_plan(latest_plan: Optional[Dict[str, Any]]) -> str:
    if not latest_plan:
        return '<p class="empty">Tier 1 surrogate has not produced a search plan yet.</p>'
    main_effect = latest_plan.get("main_effect", {})
    if not main_effect:
        return '<p class="empty">No sensitivity data in the latest search plan.</p>'
    items = sorted(main_effect.items(), key=lambda kv: -kv[1])
    frozen = set(latest_plan.get("frozen", []))
    bars = svg_bar_chart_h(
        [(k, v, "#898781" if k in frozen else CATEGORICAL[0]) for k, v in items],
        value_fmt="{:.4f}",
    )
    active = latest_plan.get("active_block", [])
    note = f'<p class="caption">Active block this cycle: {_esc(active)} — iteration {latest_plan.get("iteration")}</p>'
    return bars + note


def section_predicted_vs_actual(latest_plan: Optional[Dict[str, Any]]) -> str:
    if not latest_plan:
        return '<p class="empty">Tier 1 surrogate has not produced a search plan yet.</p>'
    actual = latest_plan.get("oob_actual") or []
    predicted = latest_plan.get("oob_predicted") or []
    if not actual or not predicted or len(actual) != len(predicted):
        return '<p class="empty">No out-of-bag surrogate diagnostics in the latest search plan.</p>'
    points = list(zip(actual, predicted))
    lo = min(min(actual), min(predicted))
    hi = max(max(actual), max(predicted))
    chart = svg_scatter_chart(points, ref_line=[(lo, lo), (hi, hi)], width=480, height=420)
    legend = '<p class="legend">dashed line: y = x (perfect prediction)</p>'
    return chart + legend


def section_interaction_matrix(latest_plan: Optional[Dict[str, Any]]) -> str:
    if not latest_plan:
        return '<p class="empty">Tier 1 surrogate has not produced a search plan yet.</p>'
    raw = latest_plan.get("interaction_matrix") or {}
    main_effect = latest_plan.get("main_effect") or {}
    params = sorted(main_effect.keys())
    if not raw or len(params) < 2:
        return '<p class="empty">Not enough kept parameters for an interaction matrix yet.</p>'
    idx = {p: i for i, p in enumerate(params)}
    grid: List[List[Optional[float]]] = [[None] * len(params) for _ in range(len(params))]
    for key, score in raw.items():
        pair = key.split("|")
        if len(pair) != 2:
            continue
        a, b = pair
        if a in idx and b in idx:
            grid[idx[a]][idx[b]] = score
            grid[idx[b]][idx[a]] = score
    return svg_heatmap(grid, params, params, diverging=False)


def section_ei_candidates(latest_plan: Optional[Dict[str, Any]]) -> str:
    if not latest_plan:
        return '<p class="empty">Tier 1 surrogate has not produced a search plan yet.</p>'
    diagnostics = latest_plan.get("ei_diagnostics") or {}
    free_params = diagnostics.get("free_params") or []
    candidate_values = diagnostics.get("candidate_values") or {}
    eis = diagnostics.get("eis") or []
    best_idx = diagnostics.get("best_idx")
    if not free_params or not eis:
        return '<p class="empty">No EI diagnostics in the latest search plan.</p>'
    panels = []
    for param in free_params:
        values = candidate_values.get(param) or []
        if len(values) != len(eis):
            continue
        panels.append((param, svg_scatter_chart(list(zip(values, eis)), highlight_idx=best_idx)))
    if not panels:
        return '<p class="empty">No EI diagnostics in the latest search plan.</p>'
    return svg_small_multiples(panels)


def section_sobol_coverage(cold_start_points: List[Dict[str, Any]], params: List[str]) -> str:
    if not cold_start_points:
        return '<p class="empty">No cold-start (Sobol) design recorded yet.</p>'
    panels = []
    for param in params:
        values = [p.get(param) for p in cold_start_points if isinstance(p.get(param), (int, float))]
        if not values:
            continue
        panels.append((param, svg_scatter_chart([(v, 0.0) for v in values])))
    if not panels:
        return '<p class="empty">No cold-start (Sobol) design recorded yet.</p>'
    return svg_small_multiples(panels)


def section_noise_floor_trend(history: List[Dict[str, Any]]) -> str:
    if not history:
        return '<p class="empty">No noise-floor measurements recorded yet.</p>'
    points = [(i, h.get("mean")) for i, h in enumerate(history) if isinstance(h.get("mean"), (int, float))]
    if not points:
        return '<p class="empty">No noise-floor measurements recorded yet.</p>'
    chart = svg_line_chart([{"label": "mean val_bpb (repeats)", "color": CATEGORICAL[0], "points": points}])
    latest = history[-1]
    caption = f'<p class="caption">Latest: mean={latest.get("mean")}, std={latest.get("std")}, n={latest.get("n")}</p>'
    return chart + caption


def section_token_fingerprint_evolution(all_metrics: List[Dict[str, Any]]) -> str:
    slope_points, induction_points = [], []
    for i, item in enumerate(all_metrics):
        fp = item.get("token_fingerprint") or {}
        slope = fp.get("attn_distance_slope")
        if isinstance(slope, (int, float)) and math.isfinite(slope):
            slope_points.append((i, float(slope)))
        induction = fp.get("induction_score")
        if isinstance(induction, (int, float)) and math.isfinite(induction):
            induction_points.append((i, float(induction)))
    if not slope_points and not induction_points:
        return '<p class="empty">No token_fingerprint data yet (token_xai_enabled has not run on any historical run).</p>'
    panels = []
    if slope_points:
        panels.append(("attn_distance_slope", svg_line_chart(
            [{"label": "attn_distance_slope", "color": CATEGORICAL[0], "points": slope_points}],
            width=480, height=260,
        )))
    if induction_points:
        panels.append(("induction_score", svg_line_chart(
            [{"label": "induction_score", "color": CATEGORICAL[1], "points": induction_points}],
            width=480, height=260,
        )))
    return svg_small_multiples(panels)


def section_fingerprint_clusters_overall(overall_clusters: Optional[Dict[str, Any]]) -> str:
    if not overall_clusters or not overall_clusters.get("clusters"):
        return '<p class="empty">Not enough historical fingerprints yet to cluster overall.</p>'
    clusters = [c for c in overall_clusters["clusters"] if c.get("mean_val_bpb") is not None]
    if not clusters:
        return '<p class="empty">Not enough historical fingerprints yet to cluster overall.</p>'
    items = [
        (f"Cluster {c['cluster_id']} (n={c['n']})", c["mean_val_bpb"], CATEGORICAL[i % len(CATEGORICAL)])
        for i, c in enumerate(clusters)
    ]
    bars = svg_bar_chart_h(items, value_fmt="{:.6f}")
    caption = f'<p class="caption">k={overall_clusters.get("k")}, silhouette={overall_clusters.get("silhouette", 0):.3f}</p>'
    return bars + caption


def section_fingerprint_clusters_trajectory(trajectory_clusters: Optional[Dict[str, Any]]) -> str:
    if not trajectory_clusters or not trajectory_clusters.get("clusters"):
        return '<p class="empty">Not enough historical fingerprints yet to cluster trajectory shapes.</p>'
    clusters = [c for c in trajectory_clusters["clusters"] if c.get("mean_shape")]
    if not clusters:
        return '<p class="empty">Not enough historical fingerprints yet to cluster trajectory shapes.</p>'
    n_resample = trajectory_clusters.get("n_resample") or len(clusters[0]["mean_shape"])
    series = []
    for i, c in enumerate(clusters):
        shape = c["mean_shape"]
        xs = [j / (n_resample - 1) for j in range(len(shape))] if n_resample > 1 else [0.0]
        series.append({
            "label": f"Cluster {c['cluster_id']} (n={c['n']})",
            "color": CATEGORICAL[i % len(CATEGORICAL)],
            "points": list(zip(xs, shape)),
        })
    chart = svg_line_chart(series)
    legend = '<p class="legend">' + " &nbsp;&nbsp;".join(
        f'<span class="swatch" style="background:{s["color"]}"></span> {_esc(s["label"])}' for s in series
    ) + "</p>"
    caption = f'<p class="caption">k={trajectory_clusters.get("k")}, silhouette={trajectory_clusters.get("silhouette", 0):.3f}</p>'
    return chart + legend + caption


def section_fingerprint_adjustments(decision_logs: List[Dict[str, Any]]) -> str:
    if not decision_logs:
        return '<p class="empty">No fingerprint-driven adjustments fired yet.</p>'
    series_map: Dict[str, List[Tuple[int, float]]] = {}
    for log in decision_logs:
        iteration = log.get("iteration")
        if not isinstance(iteration, int):
            continue
        for entry in log.get("fingerprint_adjustments") or []:
            param, delta = entry.get("param"), entry.get("delta")
            if param and isinstance(delta, (int, float)):
                series_map.setdefault(param, []).append((iteration, float(delta)))
    if not series_map:
        return '<p class="empty">No fingerprint-driven adjustments fired yet.</p>'
    series = []
    for i, (param, points) in enumerate(sorted(series_map.items())):
        points.sort(key=lambda p: p[0])
        series.append({"label": param, "color": CATEGORICAL[i % len(CATEGORICAL)], "points": points})
    chart = svg_line_chart(series)
    legend = '<p class="legend">' + " &nbsp;&nbsp;".join(
        f'<span class="swatch" style="background:{s["color"]}"></span> {_esc(s["label"])}' for s in series
    ) + "</p>"
    return chart + legend


def section_pipeline_issues(issue_logs: List[Dict[str, Any]]) -> str:
    if not issue_logs:
        return '<p class="empty">No pipeline_validator issues recorded for this run.</p>'
    by_iteration: Dict[int, Dict[str, int]] = {}
    for log in issue_logs:
        iteration = log.get("iteration")
        if not isinstance(iteration, int):
            continue
        counts = by_iteration.setdefault(iteration, {"FATAL": 0, "ERROR": 0, "WARN": 0})
        for issue in log.get("issues") or []:
            severity = issue.get("severity")
            if severity in counts:
                counts[severity] += 1
    if not by_iteration:
        return '<p class="empty">No pipeline_validator issues recorded for this run.</p>'
    iterations = sorted(by_iteration.keys())
    x_labels = [str(i) for i in iterations]
    series = [
        ("FATAL", [by_iteration[i]["FATAL"] for i in iterations], STATUS_COLORS["critical"]),
        ("ERROR", [by_iteration[i]["ERROR"] for i in iterations], STATUS_COLORS["serious"]),
        ("WARN", [by_iteration[i]["WARN"] for i in iterations], STATUS_COLORS["warning"]),
    ]
    chart = svg_stacked_bar_chart_v(x_labels, series)
    legend = '<p class="legend">' + " &nbsp;&nbsp;".join(
        f'<span class="swatch" style="background:{color}"></span> {name}' for name, _, color in series
    ) + "</p>"
    return chart + legend


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Autoresearch campaign dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
    --gridline: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --dot-muted: #898781; --band: rgba(42,120,214,0.10);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
      --gridline: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --dot-muted: #898781; --band: rgba(57,135,229,0.15);
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --gridline: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --dot-muted: #898781; --band: rgba(57,135,229,0.15);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px; background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; max-width: 1100px; margin-inline: auto; }}
  h1 {{ font-size: 20px; margin: 0; }}
  .generated {{ color: var(--text-muted); font-size: 12px; }}
  button#theme-toggle {{
    background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 12px; font-size: 12px; cursor: pointer;
  }}
  main {{ max-width: 1100px; margin-inline: auto; display: grid; gap: 20px; }}
  section {{
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px;
  }}
  section.hero {{ grid-column: 1 / -1; }}
  h2 {{ font-size: 14px; margin: 0 0 10px; color: var(--text-primary); }}
  .chart {{ width: 100%; height: auto; }}
  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .axis {{ stroke: var(--baseline); stroke-width: 1; }}
  .band {{ fill: var(--band); }}
  .dot-muted {{ fill: var(--dot-muted); opacity: 0.55; }}
  .hover-dot {{ cursor: pointer; }}
  .bar-label {{ font-size: 11px; fill: var(--text-secondary); }}
  .bar-value {{ font-size: 11px; fill: var(--text-secondary); }}
  .legend {{ font-size: 12px; color: var(--text-secondary); margin-top: 8px; }}
  .caption {{ font-size: 12px; color: var(--text-muted); }}
  .empty {{ color: var(--text-muted); font-size: 13px; font-style: italic; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
  .swatch-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--dot-muted); margin-right: 4px; }}
  .swatch-band {{ display: inline-block; width: 10px; height: 10px; background: var(--band); margin-right: 4px; }}
  table.data-table {{ border-collapse: collapse; margin-top: 12px; font-size: 12px; }}
  table.data-table caption {{ text-align: left; color: var(--text-muted); margin-bottom: 4px; }}
  table.data-table th, table.data-table td {{ border: 1px solid var(--border); padding: 4px 10px; text-align: left; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 720px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .mini-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
  .mini-panel {{ min-width: 0; }}
</style>
</head>
<body>
<header>
  <h1>Autoresearch campaign dashboard</h1>
  <div>
    <span class="generated">generated {generated}</span>
    <button id="theme-toggle" type="button">Toggle theme</button>
  </div>
</header>
<main>
  <section class="hero">
    <h2>val_bpb over the campaign</h2>
    {trend}
  </section>
  <div class="grid-2">
    <section>
      <h2>Hyperparameter importance (latest)</h2>
      {importance}
    </section>
    <section>
      <h2>Run status distribution</h2>
      {status}
    </section>
  </div>
  <section>
    <h2>Hyperparameter importance evolution</h2>
    {importance_evolution}
  </section>
  <div class="grid-2">
    <section>
      <h2>Head importance (latest ablation run)</h2>
      {heads}
    </section>
    <section>
      <h2>Tier 1 surrogate sensitivity (latest search plan)</h2>
      {search_plan}
    </section>
  </div>
  <div class="grid-2">
    <section>
      <h2>Surrogate: predicted vs. actual (out-of-bag)</h2>
      {predicted_vs_actual}
    </section>
    <section>
      <h2>Parameter interaction matrix</h2>
      {interaction_matrix}
    </section>
  </div>
  <section>
    <h2>EI acquisition: candidates considered this cycle</h2>
    {ei_candidates}
  </section>
  <section>
    <h2>Sobol cold-start design coverage</h2>
    {sobol_coverage}
  </section>
  <section>
    <h2>Noise floor over time</h2>
    {noise_floor_trend}
  </section>
  <section>
    <h2>Tier 2 scalar fingerprint fields over the campaign</h2>
    {token_fingerprint_evolution}
  </section>
  <div class="grid-2">
    <section>
      <h2>Fingerprint clusters vs. val_bpb (Tier 3)</h2>
      {fingerprint_clusters_overall}
    </section>
    <section>
      <h2>Attention-reach trajectory clusters (Tier 3)</h2>
      {fingerprint_clusters_trajectory}
    </section>
  </div>
  <section>
    <h2>Tier 4: fingerprint-driven architecture adjustments</h2>
    {fingerprint_adjustments}
  </section>
  <section>
    <h2>Pipeline validation issues (this run)</h2>
    {pipeline_issues}
  </section>
</main>
<script>
  var btn = document.getElementById("theme-toggle");
  btn.addEventListener("click", function () {{
    var root = document.documentElement;
    var current = root.getAttribute("data-theme");
    root.setAttribute("data-theme", current === "dark" ? "light" : "dark");
  }});
</script>
</body>
</html>
"""


def main():
    data = _load_data()

    doc = PAGE_TEMPLATE.format(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        trend=section_trend(data["results"], data["sigma"], data["all_metrics"]),
        importance=section_importance(data["all_metrics"]),
        status=section_status(data["results"]),
        importance_evolution=section_importance_evolution(data["all_metrics"]),
        heads=section_head_heatmap(data["latest_ablation"]),
        search_plan=section_search_plan(data["latest_plan"]),
        predicted_vs_actual=section_predicted_vs_actual(data["latest_plan"]),
        interaction_matrix=section_interaction_matrix(data["latest_plan"]),
        ei_candidates=section_ei_candidates(data["latest_plan"]),
        sobol_coverage=section_sobol_coverage(data["cold_start_points"], data["search_params"]),
        noise_floor_trend=section_noise_floor_trend(data["noise_floor_history"]),
        token_fingerprint_evolution=section_token_fingerprint_evolution(data["all_metrics"]),
        fingerprint_clusters_overall=section_fingerprint_clusters_overall(data["overall_clusters"]),
        fingerprint_clusters_trajectory=section_fingerprint_clusters_trajectory(data["trajectory_clusters"]),
        fingerprint_adjustments=section_fingerprint_adjustments(data["decision_logs"]),
        pipeline_issues=section_pipeline_issues(data["issue_logs"]),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(doc, encoding="utf-8")
    print(f"[visualize_dashboard] Wrote {OUTPUT_PATH} "
          f"({len(data['results'])} results.tsv rows, {len(data['all_metrics'])} reports)")


if __name__ == "__main__":
    main()
