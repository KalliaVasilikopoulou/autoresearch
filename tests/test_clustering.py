import random

import pytest

from state.clustering import (
    MIN_CLUSTER_N,
    POS_SALIENCY_LEN,
    _resample_curve,
    cluster_attention_trajectories,
    cluster_fingerprints,
)


def _fingerprint(n_layer, attn_entropy, attn_distance, dla, x0_lambda, attn_distance_slope,
                  induction_score, pos_saliency=None, val_bpb=1.0):
    return {
        "attn_entropy": attn_entropy,
        "attn_distance": attn_distance,
        "attn_distance_slope": attn_distance_slope,
        "dla": dla,
        "x0_lambda": x0_lambda,
        "induction_score": induction_score,
        "pos_saliency": pos_saliency or [0.0] * POS_SALIENCY_LEN,
        "val_bpb": val_bpb,
    }


def test_resample_curve_hand_computed():
    # min-max normalize [0, 10] -> [0.0, 1.0], then resample 2 points to 3
    # via linear interpolation over normalized depth: [0.0, 0.5, 1.0]
    assert _resample_curve([0.0, 10.0], 3) == pytest.approx([0.0, 0.5, 1.0])


def test_resample_curve_constant_curve_is_flat_half():
    # zero span -> every point maps to 0.5 (the span>1e-12 guard branch)
    assert _resample_curve([5.0, 5.0, 5.0], 4) == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_resample_curve_preserves_shape_regardless_of_scale():
    # two curves with the same relative shape but different absolute scale
    # should resample to (near) identical normalized curves
    a = _resample_curve([0.0, 5.0, 10.0], 5)
    b = _resample_curve([100.0, 150.0, 200.0], 5)  # same linear ramp, different offset/scale
    assert a == pytest.approx(b, abs=1e-9)


def test_cluster_fingerprints_returns_none_below_min_n():
    rows = [_fingerprint(4, [1, 2, 3, 4], [1, 2, 3, 4], [0.1] * 4, [1, 2, 3, 4], 0.5, 0.1) for _ in range(MIN_CLUSTER_N - 1)]
    assert cluster_fingerprints(rows) is None


def test_cluster_attention_trajectories_returns_none_below_min_n():
    rows = [_fingerprint(4, [1, 2, 3, 4], [1, 2, 3, 4], [0.1] * 4, [1, 2, 3, 4], 0.5, 0.1) for _ in range(MIN_CLUSTER_N - 1)]
    assert cluster_attention_trajectories(rows) is None


def test_cluster_fingerprints_skips_rows_with_wrong_pos_saliency_length():
    # Two distinguishable sub-groups within "good" so the clustering itself
    # succeeds (all-identical rows have zero variance and can't be split
    # into >=2 clusters at all) -- the thing under test is that `bad` rows
    # (wrong pos_saliency length) get silently skipped, not guessed. No
    # jitter within a row (attn_entropy is a flat [v,v,v,v]) so mean is the
    # ONLY real signal dimension (std/slope collapse to 0 for every row) --
    # avoids Ward being thrown off by noise in the other summary-stat
    # dimensions when there's only one clean separating feature intended.
    good = [
        _fingerprint(4, [1.0] * 4, [1, 2, 3, 4], [0.1] * 4, [1, 2, 3, 4], 0.5, 0.1)
        for _ in range(MIN_CLUSTER_N // 2)
    ] + [
        _fingerprint(4, [5.0] * 4, [1, 2, 3, 4], [0.1] * 4, [1, 2, 3, 4], 0.5, 0.1)
        for _ in range(MIN_CLUSTER_N // 2)
    ]
    bad = [_fingerprint(4, [1, 2, 3, 4], [1, 2, 3, 4], [0.1] * 4, [1, 2, 3, 4], 0.5, 0.1,
                         pos_saliency=[0.0] * 3) for _ in range(5)]
    result = cluster_fingerprints(good + bad)
    assert result is not None
    assert result["n_total"] == len(good)


def test_cluster_fingerprints_recovers_two_well_separated_groups():
    random.seed(0)
    rows = []
    # Group A: low entropy/distance/dla, low val_bpb (the "good" cluster)
    for _ in range(15):
        rows.append(_fingerprint(
            n_layer=6,
            attn_entropy=[1.0 + random.uniform(-0.05, 0.05) for _ in range(6)],
            attn_distance=[5.0 + random.uniform(-0.5, 0.5) for _ in range(6)],
            dla=[0.01 + random.uniform(-0.005, 0.005) for _ in range(6)],
            x0_lambda=[1.0 + random.uniform(-0.1, 0.1) for _ in range(6)],
            attn_distance_slope=0.1 + random.uniform(-0.02, 0.02),
            induction_score=0.05 + random.uniform(-0.01, 0.01),
            val_bpb=0.90 + random.uniform(-0.01, 0.01),
        ))
    # Group B: high entropy/distance/dla, high val_bpb (the "bad" cluster)
    for _ in range(15):
        rows.append(_fingerprint(
            n_layer=6,
            attn_entropy=[4.0 + random.uniform(-0.05, 0.05) for _ in range(6)],
            attn_distance=[50.0 + random.uniform(-0.5, 0.5) for _ in range(6)],
            dla=[0.5 + random.uniform(-0.005, 0.005) for _ in range(6)],
            x0_lambda=[10.0 + random.uniform(-0.1, 0.1) for _ in range(6)],
            attn_distance_slope=-0.1 + random.uniform(-0.02, 0.02),
            induction_score=0.5 + random.uniform(-0.01, 0.01),
            val_bpb=1.30 + random.uniform(-0.01, 0.01),
        ))

    result = cluster_fingerprints(rows)
    assert result is not None
    assert result["k"] == 2
    # Real, non-trivial separation (>0.25 is "weak but real structure" by
    # the usual silhouette rule of thumb) -- not >0.5, since several of the
    # ~26 features are per-row internal std/slope of a tiny jittered array,
    # similar magnitude in both groups, which dilutes the otherwise very
    # strong mean-based separation once every feature gets equal weight
    # after z-score standardization. The real correctness signals are k==2
    # and the exact group sizes/val_bpb ordering asserted below.
    assert result["silhouette"] > 0.25
    assert result["n_total"] == 30

    clusters = sorted(result["clusters"], key=lambda c: c["mean_val_bpb"])
    low_cluster, high_cluster = clusters
    assert low_cluster["n"] == 15
    assert high_cluster["n"] == 15
    assert low_cluster["mean_val_bpb"] == pytest.approx(0.90, abs=0.02)
    assert high_cluster["mean_val_bpb"] == pytest.approx(1.30, abs=0.02)


def test_cluster_attention_trajectories_recovers_shapes_across_varying_n_layer():
    random.seed(1)
    rows = []
    # Shape A: steady ramp (attn_distance grows roughly linearly with depth),
    # at DIFFERENT n_layer per row (10..14) -- must still cluster together by shape.
    for _ in range(15):
        n_layer = random.choice([10, 12, 14])
        curve = [float(i) + random.uniform(-0.3, 0.3) for i in range(n_layer)]
        rows.append(_fingerprint(
            n_layer=n_layer, attn_entropy=[1.0] * n_layer, attn_distance=curve,
            dla=[0.1] * n_layer, x0_lambda=[1.0] * n_layer,
            attn_distance_slope=1.0, induction_score=0.1, val_bpb=0.95,
        ))
    # Shape B: early saturation (attn_distance plateaus after the first few layers),
    # also at varying n_layer.
    for _ in range(15):
        n_layer = random.choice([10, 12, 14])
        curve = [min(float(i), 3.0) + random.uniform(-0.3, 0.3) for i in range(n_layer)]
        rows.append(_fingerprint(
            n_layer=n_layer, attn_entropy=[1.0] * n_layer, attn_distance=curve,
            dla=[0.1] * n_layer, x0_lambda=[1.0] * n_layer,
            attn_distance_slope=0.1, induction_score=0.1, val_bpb=1.15,
        ))

    result = cluster_attention_trajectories(rows)
    assert result is not None
    assert result["k"] == 2
    assert result["n_total"] == 30

    clusters = sorted(result["clusters"], key=lambda c: c["mean_val_bpb"])
    ramp_cluster, saturating_cluster = clusters
    assert ramp_cluster["n"] == 15
    assert saturating_cluster["n"] == 15
    # ramp shape's mean_shape should be monotonically increasing end-to-end;
    # saturating shape should flatten out in the back half.
    ramp_shape = ramp_cluster["mean_shape"]
    saturating_shape = saturating_cluster["mean_shape"]
    assert ramp_shape[-1] - ramp_shape[0] > saturating_shape[-1] - saturating_shape[0]
