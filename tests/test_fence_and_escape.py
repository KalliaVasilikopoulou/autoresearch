"""Step 3b: proposals stay inside their region, and wanting to leave is recorded.

Before this, `propose_via_ei` sampled the active block across each parameter's
FULL observed range. A single proposal could therefore move a region's centre
6-10x its own radius, on essentially every iteration -- so "region" named a
starting point rather than a place, and `sigma_region` described a
neighbourhood nothing respected.

The fence restricts what may be RUN. The unfenced candidates are still
generated and scored, because the gap between "best inside" and "best ignoring
the fence" is the escape signal, and it costs nothing extra.
"""

import math

import pytest

from state import surrogate
from state.regions import distance
from state.results_analysis import TUNABLE_COLUMNS

pytestmark = pytest.mark.skipif(
    not surrogate.SURROGATE_DEPS_AVAILABLE, reason="scipy/scikit-learn not installed"
)

FEATURES = ("n_layer", "n_embd", "matrix_lr", "batch_size")
CENTER = {"n_layer": 8, "n_embd": 512, "matrix_lr": 0.04, "batch_size": 8192}


def _fitted():
    """A surrogate over a spread of runs, so bounds are wide enough for a fence
    to actually constrain anything."""
    rows = []
    for i in range(40):
        rows.append({
            "n_layer": 4 + (i % 12), "n_embd": 256 + 32 * (i % 20),
            "matrix_lr": 0.006 + 0.008 * (i % 20),
            "batch_size": 2048 + 1024 * (i % 24),
            "val_bpb": 1.20 + 0.004 * (i % 11),
        })
    return surrogate.fit_surrogate(rows, feature_columns=FEATURES, min_n=5,
                                   exclude_compute_starved=False), rows


def _tunable_distance(a, b, bounds):
    """Distance over just the free params, on the region's scale (divided by
    sqrt of the full tunable dimension count)."""
    return math.sqrt(sum(
        (surrogate.normalized_value(p, float(a[p]), bounds)
         - surrogate.normalized_value(p, float(b[p]), bounds)) ** 2
        for p in ("matrix_lr", "batch_size"))) / math.sqrt(len(TUNABLE_COLUMNS))


# --- the fence --------------------------------------------------------------


def test_without_a_fence_proposals_roam_the_whole_range():
    """The behaviour being fixed, pinned so the fix is visibly a change."""
    sm, _ = _fitted()
    far = 0
    for seed in range(12):
        p = surrogate.propose_via_ei(
            sm, f_best=1.20, bounds=sm.bounds,
            free_params=["matrix_lr", "batch_size"], fixed_values=CENTER,
            n_candidates=200, seed=seed,
        )
        if _tunable_distance(p, CENTER, sm.bounds) > 0.05:
            far += 1
    assert far > 0, "unfenced search should sometimes land far from the centre"


def test_a_fenced_proposal_stays_inside_the_radius():
    sm, _ = _fitted()
    radius = 0.03
    for seed in range(12):
        p = surrogate.propose_via_ei(
            sm, f_best=1.20, bounds=sm.bounds,
            free_params=["matrix_lr", "batch_size"], fixed_values=CENTER,
            n_candidates=200, seed=seed,
            fence_center=CENTER, fence_radius=radius, fence_dims=len(TUNABLE_COLUMNS),
        )
        d = _tunable_distance(p, CENTER, sm.bounds)
        # Snapping to legal values (batch_size rounds to a multiple of 2048)
        # can push a candidate a hair past the boundary; the fence constrains
        # where we SAMPLE, and the snap is applied afterwards.
        assert d <= radius * 1.35, f"seed {seed}: {d:.4f} vs radius {radius}"


def test_a_tighter_fence_gives_tighter_proposals():
    sm, _ = _fitted()

    def spread(radius):
        return max(
            _tunable_distance(
                surrogate.propose_via_ei(
                    sm, f_best=1.20, bounds=sm.bounds,
                    free_params=["matrix_lr", "batch_size"], fixed_values=CENTER,
                    n_candidates=200, seed=s,
                    fence_center=CENTER, fence_radius=radius,
                    fence_dims=len(TUNABLE_COLUMNS)),
                CENTER, sm.bounds)
            for s in range(10))

    assert spread(0.01) < spread(0.06)


def test_the_fence_is_off_by_default_so_existing_callers_are_unchanged():
    """Agent 4 proposes new regions with every parameter free and must not be
    fenced; only Agent 1, scoped to a region, passes one."""
    sm, _ = _fitted()
    a = surrogate.propose_via_ei(
        sm, f_best=1.20, bounds=sm.bounds, free_params=["matrix_lr"],
        fixed_values=CENTER, n_candidates=100, seed=3)
    b = surrogate.propose_via_ei(
        sm, f_best=1.20, bounds=sm.bounds, free_params=["matrix_lr"],
        fixed_values=CENTER, n_candidates=100, seed=3,
        fence_center=None, fence_radius=None)
    assert a == b


# --- escape pressure --------------------------------------------------------


def test_escape_is_recorded_but_not_run():
    """The whole point: we learn where the search wanted to go WITHOUT going
    there, and it costs no extra training."""
    sm, _ = _fitted()
    _p, diag = surrogate.propose_via_ei(
        sm, f_best=1.20, bounds=sm.bounds,
        free_params=["matrix_lr", "batch_size"], fixed_values=CENTER,
        n_candidates=400, seed=1, return_diagnostics=True,
        fence_center=CENTER, fence_radius=0.01, fence_dims=len(TUNABLE_COLUMNS),
    )
    esc = diag["escape"]
    assert diag["fenced"] is True
    assert set(esc["direction"]) == {"matrix_lr", "batch_size"}
    assert esc["distance"] >= 0.0
    assert esc["radius"] == pytest.approx(0.01)
    # The run proposal is the fenced one, never the escapee.
    assert diag["best_idx"] != esc["unfenced_best_idx"]


def test_escape_direction_is_signed_so_it_can_be_averaged():
    """One escape is noise; a consistent direction over several iterations is
    the signal Agent 4 acts on. That only works if the direction keeps its
    sign rather than being reported as a magnitude."""
    sm, _ = _fitted()
    _p, diag = surrogate.propose_via_ei(
        sm, f_best=1.20, bounds=sm.bounds,
        free_params=["matrix_lr", "batch_size"], fixed_values=CENTER,
        n_candidates=400, seed=2, return_diagnostics=True,
        fence_center=CENTER, fence_radius=0.01, fence_dims=len(TUNABLE_COLUMNS),
    )
    values = list(diag["escape"]["direction"].values())
    assert any(v < 0 for v in values) or any(v > 0 for v in values)
    assert all(-1.0 <= v <= 1.0 for v in values)


def test_no_escape_block_when_unfenced():
    sm, _ = _fitted()
    _p, diag = surrogate.propose_via_ei(
        sm, f_best=1.20, bounds=sm.bounds, free_params=["matrix_lr"],
        fixed_values=CENTER, n_candidates=100, seed=1, return_diagnostics=True)
    assert diag.get("fenced") is False
    assert "escape" not in diag


# --- the ball sampler -------------------------------------------------------


def test_ball_samples_respect_the_radius_and_fill_it():
    """Uniform by VOLUME, not by length -- otherwise candidates pile near the
    centre and the edge of the region, which is where a search under pressure
    wants to go, is barely probed."""
    import numpy as np

    rng = np.random.default_rng(0)
    pts = surrogate._sample_in_ball([0.5, 0.5, 0.5], max_euclid=0.4, n=4000, rng=rng)
    d = np.linalg.norm(pts - 0.5, axis=1)

    assert d.max() <= 0.4 + 1e-9
    assert (d > 0.2).mean() > 0.5, "the outer shell holds most of the volume"


def test_ball_samples_stay_legal_near_an_edge():
    """A region anchored at a parameter's limit genuinely has less room on that
    side; candidates must stay in [0,1] rather than proposing values the
    clamps would reject."""
    import numpy as np

    rng = np.random.default_rng(0)
    pts = surrogate._sample_in_ball([0.0, 1.0], max_euclid=0.5, n=500, rng=rng)

    assert pts.min() >= 0.0 and pts.max() <= 1.0
