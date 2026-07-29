"""Tier 1 (agents/search_planner.py): the persisted search loop that calls
state/surrogate.py. Never had dedicated tests before this -- synthetic-data
tests only, same discipline used for Tiers 2-4.
"""
import json
import random

import pytest

import agents.search_planner as search_planner_module
from agents.agent1_training_specialist import SEARCH_SPACE
from agents.search_planner import (
    DEFAULT_SIGMA,
    SearchPlannerState,
    _load_sigma,
    propose_next,
    render_report,
)
from state.results_analysis import HYPERPARAM_COLUMNS
from state.surrogate import SURROGATE_DEPS_AVAILABLE

pytestmark = pytest.mark.skipif(not SURROGATE_DEPS_AVAILABLE, reason="numpy/scikit-learn not installed")


def _default_best_hyperparams():
    hp = {name: (lo + hi) / 2 for name, (lo, hi) in SEARCH_SPACE.items()}
    hp["ablation_k"] = 3  # a pass-through key, not in SEARCH_SPACE -- exercises that path too
    return hp


def _random_hp(rng):
    return {name: rng.uniform(lo, hi) for name, (lo, hi) in SEARCH_SPACE.items()}


def _synthetic_rows(n, seed=0, effect_param="n_layer", effect_coef=0.05, noise=0.001):
    """val_bpb depends only on `effect_param` (linearly, over its real
    SEARCH_SPACE range) -- every other HYPERPARAM_COLUMNS entry is present
    (required for a row to count as "usable") but has zero true effect.
    """
    rng = random.Random(seed)
    lo, _hi = SEARCH_SPACE[effect_param]
    rows = []
    for _ in range(n):
        hp = _random_hp(rng)
        y = 1.0 + effect_coef * (hp[effect_param] - lo) + rng.uniform(-noise, noise)
        row = dict(hp)
        row["val_bpb"] = y
        rows.append(row)
    return rows


def _write_noise_floor(tmp_path, sigma):
    path = tmp_path / "noise_floor.json"
    path.write_text(json.dumps({"std": sigma}))
    return str(path)


# ---------------------------------------------------------------------------
# SearchPlannerState
# ---------------------------------------------------------------------------

def test_state_load_save_roundtrip(tmp_path):
    path = str(tmp_path / "state.json")
    state = SearchPlannerState(cold_start_used=3, frozen={"x": 5}, active_block=["a", "b"], budget_used_in_block=2)
    state.save(path)
    loaded = SearchPlannerState.load(path)
    assert loaded == state


def test_state_load_missing_file_returns_fresh(tmp_path):
    loaded = SearchPlannerState.load(str(tmp_path / "does_not_exist.json"))
    assert loaded == SearchPlannerState()


def test_state_load_corrupt_file_returns_fresh(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    loaded = SearchPlannerState.load(str(path))
    assert loaded == SearchPlannerState()


# ---------------------------------------------------------------------------
# _load_sigma
# ---------------------------------------------------------------------------

def test_load_sigma_fallback_when_missing(tmp_path):
    assert _load_sigma(str(tmp_path / "missing.json")) == DEFAULT_SIGMA


def test_load_sigma_reads_real_value(tmp_path):
    path = _write_noise_floor(tmp_path, 0.0321)
    assert _load_sigma(path) == pytest.approx(0.0321)


# ---------------------------------------------------------------------------
# propose_next -- cold start
# ---------------------------------------------------------------------------

def test_propose_next_cold_start_returns_distinct_sequential_points(tmp_path):
    state_path = str(tmp_path / "state.json")
    common_kwargs = dict(
        rows=[], current_best_hyperparams=_default_best_hyperparams(), current_best_val_bpb=1.0,
        cold_start_n=5, state_path=state_path, noise_floor_path=str(tmp_path / "nf.json"),
        report_dir=str(tmp_path / "reports"),
    )
    p1 = propose_next(iteration=0, **common_kwargs)
    p2 = propose_next(iteration=1, **common_kwargs)
    assert p1 is not None and p2 is not None
    assert any(p1[k] != p2[k] for k in SEARCH_SPACE)  # distinct Sobol points

    state = SearchPlannerState.load(state_path)
    assert state.cold_start_used == 2


def test_propose_next_cold_start_exhausted_generates_extra_point_not_stalling(tmp_path):
    state_path = str(tmp_path / "state.json")
    common_kwargs = dict(
        rows=[], current_best_hyperparams=_default_best_hyperparams(), current_best_val_bpb=1.0,
        cold_start_n=2, state_path=state_path, noise_floor_path=str(tmp_path / "nf.json"),
        report_dir=str(tmp_path / "reports"),
    )
    for i in range(2):
        assert propose_next(iteration=i, **common_kwargs) is not None
    # batch of 2 Sobol points now exhausted, but rows=[] still means 0 usable
    # rows (< cold_start_n) -- must generate an ad hoc point, not return None.
    p3 = propose_next(iteration=2, **common_kwargs)
    assert p3 is not None
    assert all(k in p3 for k in SEARCH_SPACE)


# ---------------------------------------------------------------------------
# propose_next -- surrogate-driven path
# ---------------------------------------------------------------------------

def test_propose_next_surrogate_path_freezes_the_no_effect_param(tmp_path):
    rows = _synthetic_rows(200, seed=1, effect_param="n_layer", effect_coef=0.1)
    noise_floor_path = _write_noise_floor(tmp_path, sigma=0.1)  # 2*sigma=0.2; n_layer's total effect ~2.0, everything else ~0
    result = propose_next(
        rows=rows, current_best_hyperparams=_default_best_hyperparams(), current_best_val_bpb=min(r["val_bpb"] for r in rows),
        iteration=100, cold_start_n=15, state_path=str(tmp_path / "state.json"),
        noise_floor_path=noise_floor_path, report_dir=str(tmp_path / "reports"),
    )
    assert result is not None
    assert all(k in result for k in SEARCH_SPACE)
    assert result["ablation_k"] == 3  # pass-through key preserved

    report = json.loads((tmp_path / "reports" / "plan_0100.json").read_text())
    assert "weight_decay" in report["frozen"]
    assert "n_layer" not in report["frozen"]


def test_propose_next_returns_none_without_deps(monkeypatch):
    monkeypatch.setattr(search_planner_module.surrogate, "SURROGATE_DEPS_AVAILABLE", False)
    result = propose_next(
        rows=[], current_best_hyperparams=_default_best_hyperparams(), current_best_val_bpb=1.0, iteration=0,
    )
    assert result is None


def test_propose_next_returns_none_when_everything_is_frozen(tmp_path):
    rng = random.Random(7)
    rows = []
    for _ in range(200):
        hp = _random_hp(rng)
        row = dict(hp)
        row["val_bpb"] = 1.0 + rng.uniform(-0.001, 0.001)  # pure noise, no real relationship to any param
        rows.append(row)
    noise_floor_path = _write_noise_floor(tmp_path, sigma=0.05)  # 2*sigma=0.1, well above any spurious RF sensitivity to pure noise
    result = propose_next(
        rows=rows, current_best_hyperparams=_default_best_hyperparams(), current_best_val_bpb=min(r["val_bpb"] for r in rows),
        iteration=200, cold_start_n=15, state_path=str(tmp_path / "state.json"),
        noise_floor_path=noise_floor_path, report_dir=str(tmp_path / "reports"),
    )
    assert result is None


def test_propose_next_age_based_reprobe_gives_frozen_param_another_chance(tmp_path):
    state_path = tmp_path / "state.json"
    # Hand-seed a state where "weight_decay" has been frozen since iteration 0.
    SearchPlannerState(frozen={"weight_decay": 0}).save(str(state_path))

    rows = _synthetic_rows(200, seed=3, effect_param="n_layer", effect_coef=0.1)
    noise_floor_path = _write_noise_floor(tmp_path, sigma=0.1)
    result = propose_next(
        rows=rows, current_best_hyperparams=_default_best_hyperparams(), current_best_val_bpb=min(r["val_bpb"] for r in rows),
        iteration=25, cold_start_n=15, reprobe_every=20,  # 25 - 0 >= 20 -> reprobe fires
        state_path=str(state_path), noise_floor_path=noise_floor_path, report_dir=str(tmp_path / "reports"),
    )
    assert result is not None
    report = json.loads((tmp_path / "reports" / "plan_0025.json").read_text())
    assert "weight_decay" not in report["frozen"]  # got another chance this round, not re-frozen outright


def test_propose_next_block_rotation_advances_after_budget_exhausted(tmp_path):
    # Two independent, both-meaningful params -> (most likely) two separate
    # blocks; small cycle_runs so the active block's budget is tiny and
    # rotation is observable within a handful of calls.
    rng = random.Random(9)
    rows = []
    for _ in range(300):
        hp = _random_hp(rng)
        n_layer_lo = SEARCH_SPACE["n_layer"][0]
        matrix_lr_lo = SEARCH_SPACE["matrix_lr"][0]
        y = (
            1.0
            + 0.1 * (hp["n_layer"] - n_layer_lo)
            + 0.1 * (hp["matrix_lr"] - matrix_lr_lo)
            + rng.uniform(-0.001, 0.001)
        )
        row = dict(hp)
        row["val_bpb"] = y
        rows.append(row)

    state_path = str(tmp_path / "state.json")
    noise_floor_path = _write_noise_floor(tmp_path, sigma=0.05)
    seen_active_blocks = []
    for i in range(8):
        propose_next(
            rows=rows, current_best_hyperparams=_default_best_hyperparams(), current_best_val_bpb=min(r["val_bpb"] for r in rows),
            iteration=100 + i, cold_start_n=15, cycle_runs=2, state_path=state_path,
            noise_floor_path=noise_floor_path, report_dir=str(tmp_path / "reports"),
        )
        seen_active_blocks.append(tuple(SearchPlannerState.load(state_path).active_block))

    # With cycle_runs=2, budget per block is small -- across 8 calls the
    # active block must have rotated (not stayed on one block forever),
    # unless everything collapsed into a single block (still worth knowing).
    assert len(set(seen_active_blocks)) >= 1
    if len(set(f for block in seen_active_blocks for f in block)) > 1:
        assert len(set(seen_active_blocks)) >= 2, (
            f"expected block rotation across {seen_active_blocks} given >1 distinct kept param"
        )


# ---------------------------------------------------------------------------
# report generation
# ---------------------------------------------------------------------------

def test_propose_next_writes_report_files_with_expected_structure(tmp_path):
    rows = _synthetic_rows(200, seed=5, effect_param="n_layer", effect_coef=0.1)
    noise_floor_path = _write_noise_floor(tmp_path, sigma=0.1)
    report_dir = tmp_path / "reports"
    result = propose_next(
        rows=rows, current_best_hyperparams=_default_best_hyperparams(), current_best_val_bpb=min(r["val_bpb"] for r in rows),
        iteration=42, cold_start_n=15, state_path=str(tmp_path / "state.json"),
        noise_floor_path=noise_floor_path, report_dir=str(report_dir),
    )
    assert result is not None
    md = (report_dir / "plan_0042.md").read_text()
    assert "iteration 42" in md
    assert "Sensitivity" in md
    assert "This iteration's proposal" in md

    payload = json.loads((report_dir / "plan_0042.json").read_text())
    for key in ("iteration", "main_effect", "blocks", "variance_share", "frozen", "active_block", "proposal"):
        assert key in payload


def test_render_report_contains_expected_sections():
    class _FakeSurrogate:
        n_train = 42

    report = render_report(
        iteration=5, surrogate_model=_FakeSurrogate(),
        main_effect={"a": 0.5, "b": 0.1},
        blocks=[["a"], ["b"]],
        variance_share={("a",): 0.8, ("b",): 0.2},
        frozen=["b"],
        active_block=["a"],
        proposal={"a": 1.0},
    )
    assert "iteration 5" in report
    assert "frozen (< 2σ)" in report
    assert "(active this cycle)" in report
    assert '"a": 1.0' in report
