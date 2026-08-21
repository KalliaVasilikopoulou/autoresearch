"""Budget-aware exploration: EI's uncertainty term is only worth paying for
while runs remain to exploit what it finds.

THE MEASURED FAILURE these pin. On a 30-run campaign EI chose candidates it
predicted at 1.5688 / 1.5638 / 1.5701 at iterations 22 / 25 / 28, while
candidates predicted at 1.4532 / 1.4809 / 1.4898 sat in the same batch. Sigma at
the chosen points was 0.186 / 0.087 / 0.067 -- up to 64x the 0.0029 noise floor
-- so the uncertainty term won every time. The 1.4463 found at iteration 19 was
never revisited, and 8 of 30 runs went into ground there was no budget left to
use.
"""

import pytest

from state import surrogate
from state.landscape import LANDSCAPE_DEPS_AVAILABLE

requires_deps = pytest.mark.skipif(
    not LANDSCAPE_DEPS_AVAILABLE, reason="scikit-learn not installed")

BASE = {
    "n_layer": 8, "n_embd": 512, "n_head": 4, "window_s_fraction": 0.75,
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192,
}
FEATURES = ["matrix_lr", "embedding_lr", "batch_size", "weight_decay"]


def _rows(n=40):
    out = []
    for i in range(n):
        r = dict(BASE)
        r["matrix_lr"] = 0.005 * (1 + i % 7)
        r["embedding_lr"] = 0.05 * (1 + i % 5)
        r["batch_size"] = 2048 * (1 + i % 4)
        r["val_bpb"] = 1.4 + 0.02 * (i % 9)
        r["status"] = "remote_ok"
        out.append(r)
    return out


@requires_deps
def _fit():
    return surrogate.fit_surrogate(_rows(), feature_columns=FEATURES, min_n=15)


@requires_deps
def test_zero_weight_is_pure_exploitation():
    """At weight 0 the acquisition collapses to max(f_best - mu, 0), which is
    monotone in -mu -- so it must pick the lowest predicted mean available."""
    sm = _fit()
    _, d = surrogate.propose_via_ei(
        sm, f_best=1.45, bounds=sm.bounds, free_params=["matrix_lr", "embedding_lr"],
        fixed_values=dict(BASE), n_candidates=400, seed=3,
        return_diagnostics=True, exploration_weight=0.0)
    assert d["mus"][d["best_idx"]] == pytest.approx(min(d["mus"]))


@requires_deps
def test_full_weight_may_take_a_worse_mean_for_the_information():
    """Textbook EI. Not a defect -- it is the behaviour that must be ANNEALED,
    not removed, or the search stops exploring at all."""
    sm = _fit()
    _, d = surrogate.propose_via_ei(
        sm, f_best=1.45, bounds=sm.bounds, free_params=["matrix_lr", "embedding_lr"],
        fixed_values=dict(BASE), n_candidates=400, seed=3,
        return_diagnostics=True, exploration_weight=1.0)
    assert d["mus"][d["best_idx"]] >= min(d["mus"])


@requires_deps
def test_the_incumbent_is_recorded():
    """`f_best used = None` appeared in every plan on disk, so the log could not
    show what improvement was being measured against -- the same blind spot as
    the migration gain."""
    sm = _fit()
    _, d = surrogate.propose_via_ei(
        sm, f_best=1.4321, bounds=sm.bounds, free_params=["matrix_lr"],
        fixed_values=dict(BASE), n_candidates=100, seed=1,
        return_diagnostics=True, exploration_weight=0.5)
    assert d["f_best"] == pytest.approx(1.4321)
    assert d["exploration_weight"] == pytest.approx(0.5)


@requires_deps
def test_a_negative_weight_is_clamped_not_trusted():
    sm = _fit()
    _, d = surrogate.propose_via_ei(
        sm, f_best=1.45, bounds=sm.bounds, free_params=["matrix_lr"],
        fixed_values=dict(BASE), n_candidates=50, seed=1,
        return_diagnostics=True, exploration_weight=-5.0)
    assert d["exploration_weight"] == 0.0
