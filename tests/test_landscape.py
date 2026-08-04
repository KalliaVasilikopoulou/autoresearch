"""Synthetic-data tests for state/landscape.py -- the PCA-compressed
optimization landscape (real runs + surrogate-predicted grid).

Same discipline as tests/test_surrogate.py and tests/test_clustering.py:
never assert on exact numeric output of a fitted model, assert on the
contracts that matter (None-vs-real-dict, shapes, and the "every grid cell
must describe a model train.py could actually build" invariant).
"""

import json
import math

import pytest

from state.landscape import (
    DEFAULT_GRID_RESOLUTION,
    LANDSCAPE_DEPS_AVAILABLE,
    build_landscape,
    load_region_flags,
    project_point,
    region_members,
    save_region_flags,
)
from state.results_analysis import HYPERPARAM_COLUMNS
from state.surrogate import fit_surrogate

requires_deps = pytest.mark.skipif(
    not LANDSCAPE_DEPS_AVAILABLE, reason="scikit-learn not installed"
)

# A deliberately narrow SEARCH_SPACE-shaped dict, so the hard-bounds test can
# tell "clamped" apart from "happened to land inside anyway".
HARD_BOUNDS = {
    "n_layer": (1, 24),
    "n_embd": (128, 1024),
    "n_head": (2, 16),
    "window_s_fraction": (0.1, 1.0),
    "embedding_lr": (0.01, 1.0),
    "unembedding_lr": (0.0001, 0.05),
    "matrix_lr": (0.001, 0.1),
    "scalar_lr": (0.001, 1.0),
    "weight_decay": (0.0, 0.5),
    "warmup_ratio": (0.0, 0.5),
    "batch_size": (1024, 32768),
}


def _rows(n=30):
    """n distinct, plausible runs -- varied enough that PCA has real
    structure to find and the surrogate has something to fit."""
    rows = []
    for i in range(n):
        rows.append({
            "n_layer": 4 + (i % 9),
            "n_embd": 256 + (i % 6) * 64,
            "n_head": 4 + (i % 3) * 2,
            "window_s_fraction": 0.2 + (i % 5) * 0.15,
            "embedding_lr": 0.05 * (1 + i % 7),
            "unembedding_lr": 0.001 * (1 + i % 4),
            "matrix_lr": 0.005 * (1 + i % 6),
            "scalar_lr": 0.02 * (1 + i % 5),
            "weight_decay": 0.01 * (i % 8),
            "warmup_ratio": 0.02 * (i % 6),
            "batch_size": 2048 * (1 + i % 4),
            "val_bpb": 1.2 + 0.03 * (i % 11),
            "status": "remote_ok",
        })
    return rows


def _landscape(n=30, **kwargs):
    rows = _rows(n)
    sm = fit_surrogate(rows)
    assert sm is not None, "fixture rows must be enough to fit a surrogate"
    return rows, build_landscape(rows, sm, hard_bounds=HARD_BOUNDS, **kwargs)


# --- the "never fabricate" contract ----------------------------------------

def test_build_landscape_none_when_surrogate_is_none():
    assert build_landscape(_rows(30), None) is None


@requires_deps
def test_build_landscape_none_when_too_few_usable_rows():
    rows = _rows(30)
    sm = fit_surrogate(rows)
    assert build_landscape(rows[:5], sm, min_n=15) is None


@requires_deps
def test_build_landscape_none_when_rows_missing_feature_columns():
    """Rows that don't carry every hyperparameter column aren't usable --
    the same filter the surrogate itself applies, so the landscape is never
    built over a different run set than the model it's visualizing."""
    rows = _rows(30)
    sm = fit_surrogate(rows)
    partial = [{"val_bpb": r["val_bpb"], "n_layer": r["n_layer"]} for r in rows]
    assert build_landscape(partial, sm) is None


def test_build_landscape_none_when_deps_unavailable(monkeypatch):
    import state.landscape as landscape_module
    monkeypatch.setattr(landscape_module, "LANDSCAPE_DEPS_AVAILABLE", False)
    rows = _rows(30)
    assert landscape_module.build_landscape(rows, object()) is None


# --- shape / content contracts ---------------------------------------------

@requires_deps
def test_build_landscape_real_points_carry_measured_val_bpb():
    rows, ls = _landscape()
    assert ls is not None
    assert len(ls["real_points"]) == len(rows)
    assert sorted(p["z"] for p in ls["real_points"]) == sorted(r["val_bpb"] for r in rows)


@requires_deps
def test_build_landscape_grid_shapes_match_resolution():
    _, ls = _landscape(grid_resolution=8)
    assert len(ls["grid_x"]) == 8 and len(ls["grid_y"]) == 8
    for key in ("grid_z_mean", "grid_z_std", "grid_confidence", "grid_hyperparams"):
        assert len(ls[key]) == 8, key
        assert all(len(row) == 8 for row in ls[key]), key


@requires_deps
def test_build_landscape_default_grid_resolution():
    _, ls = _landscape()
    assert len(ls["grid_x"]) == DEFAULT_GRID_RESOLUTION


@requires_deps
def test_build_landscape_grid_hyperparams_respect_hard_bounds():
    """PCA inverse-transform routinely lands outside the legal range; every
    cell must be clamped back before it's shown or predicted on."""
    _, ls = _landscape(grid_resolution=8)
    for row in ls["grid_hyperparams"]:
        for cell in row:
            for param, (lo, hi) in HARD_BOUNDS.items():
                assert lo <= cell[param] <= hi, f"{param}={cell[param]} outside {(lo, hi)}"


@requires_deps
def test_build_landscape_grid_cells_are_trainable_configs():
    """Every grid cell must describe a model train.py could actually build:
    integer int-params, and an even head_dim (n_embd / n_head) because RoPE
    splits each head into two equal halves."""
    _, ls = _landscape(grid_resolution=8)
    for row in ls["grid_hyperparams"]:
        for cell in row:
            for param in ("n_layer", "n_head", "n_embd", "batch_size"):
                assert isinstance(cell[param], int), f"{param} is {type(cell[param])}"
            head_dim = cell["n_embd"] / cell["n_head"]
            assert head_dim == int(head_dim), "n_head must divide n_embd"
            assert int(head_dim) % 2 == 0, f"head_dim {head_dim} is odd -- RoPE would crash"


@requires_deps
def test_build_landscape_confidence_in_unit_range():
    _, ls = _landscape(grid_resolution=8)
    flat = [c for row in ls["grid_confidence"] for c in row]
    assert flat and all(0.0 <= c <= 1.0 for c in flat)


@requires_deps
def test_build_landscape_confidence_is_inverse_of_std():
    """The whole point of the gradient: the least-certain cell must be the
    least opaque one."""
    _, ls = _landscape(grid_resolution=8)
    pairs = [
        (ls["grid_z_std"][r][c], ls["grid_confidence"][r][c])
        for r in range(8) for c in range(8)
    ]
    highest_std = max(pairs, key=lambda p: p[0])
    lowest_std = min(pairs, key=lambda p: p[0])
    assert highest_std[1] <= lowest_std[1]


@requires_deps
def test_build_landscape_reports_explained_variance():
    _, ls = _landscape()
    ratios = ls["explained_variance_ratio"]
    assert len(ratios) == 2
    assert 0.0 <= sum(ratios) <= 1.0 + 1e-9


@requires_deps
def test_build_landscape_grid_extends_beyond_real_points():
    """The predicted surface has to cover ground the real runs don't, or it
    shows nothing the trend chart doesn't already."""
    _, ls = _landscape(grid_resolution=8)
    xs = [p["x"] for p in ls["real_points"]]
    assert ls["grid_x"][0] < min(xs) and ls["grid_x"][-1] > max(xs)


# --- project_point ---------------------------------------------------------

@requires_deps
def test_project_point_roundtrips_a_real_row():
    rows, ls = _landscape()
    projected = project_point(rows[0], ls)
    stored = ls["real_points"][0]
    assert projected == pytest.approx((stored["x"], stored["y"]), abs=1e-6)


@requires_deps
def test_project_point_returns_none_for_incomplete_hyperparams():
    _, ls = _landscape()
    assert project_point({"n_layer": 6}, ls) is None


def test_project_point_returns_none_for_empty_landscape():
    assert project_point({"n_layer": 6}, {}) is None


# --- region_members --------------------------------------------------------

@requires_deps
def test_region_members_includes_the_center_itself():
    rows, ls = _landscape()
    members = region_members(rows, rows[0], ls, radius=0.1)
    assert any(m is rows[0] for m in members)


@requires_deps
def test_region_members_radius_is_monotonic():
    rows, ls = _landscape()
    small = region_members(rows, rows[0], ls, radius=0.05)
    large = region_members(rows, rows[0], ls, radius=0.9)
    assert len(small) <= len(large)
    assert len(large) == len(rows)  # a radius spanning the whole grid catches everything


@requires_deps
def test_region_members_empty_when_center_unprojectable():
    rows, ls = _landscape()
    assert region_members(rows, {"n_layer": 6}, ls, radius=0.5) == []


# --- region flags ----------------------------------------------------------

def test_load_region_flags_missing_file_returns_empty(tmp_path):
    assert load_region_flags(tmp_path / "nope.json") == []


def test_load_region_flags_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "flags.json"
    path.write_text("not valid json{{{", encoding="utf-8")
    assert load_region_flags(path) == []


def test_load_region_flags_malformed_regions_key_returns_empty(tmp_path):
    path = tmp_path / "flags.json"
    path.write_text(json.dumps({"regions": "not a list"}), encoding="utf-8")
    assert load_region_flags(path) == []


def test_load_region_flags_skips_entries_without_hyperparams(tmp_path):
    """An entry with no hyperparams can't be placed on the landscape at all
    -- dropped rather than drawn at a guessed position."""
    path = tmp_path / "flags.json"
    path.write_text(json.dumps({"regions": [
        {"flag": "local_optimum"},
        {"hyperparams": {"n_layer": 6}, "flag": "currently_exploiting"},
    ]}), encoding="utf-8")
    loaded = load_region_flags(path)
    assert len(loaded) == 1 and loaded[0]["flag"] == "currently_exploiting"


def test_save_load_region_flags_roundtrip(tmp_path):
    path = tmp_path / "state" / "flags.json"
    regions = [{"hyperparams": {"n_layer": 6, "n_embd": 512}, "flag": "local_optimum",
                "since_iteration": 120, "n_runs": 14}]
    save_region_flags(path, regions)
    assert load_region_flags(path) == regions
