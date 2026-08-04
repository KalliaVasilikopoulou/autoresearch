"""Excluding runs that were robbed of training steps by GPU contention.

Measured on this project's shared DGX: num_steps is 91% predictable from the
config alone (OOB R2=0.909), yet 29% of historical runs came in >10% short and
14% >20% short. An hour of identical-config repeats saw step counts fall
1688 -> 1304 as other tenants arrived, moving val_bpb by 0.028 -- about the
whole elite-to-best gap of the campaign.

Such a run measures "this config robbed of a fifth of its training", which is
not the question the search asks. Same category as the dry_run placeholders
SYNTHETIC_STATUSES already excludes.
"""

import pytest

from state.surrogate import (
    SURROGATE_DEPS_AVAILABLE,
    STEP_DEFICIT_THRESHOLD,
    fit_surrogate,
    step_deficits,
    without_compute_starved,
)

requires_deps = pytest.mark.skipif(not SURROGATE_DEPS_AVAILABLE, reason="sklearn not installed")


def _row(i, steps=1000, val_bpb=1.3, **over):
    r = {
        "n_layer": 4 + (i % 9), "n_embd": 256 + (i % 6) * 64, "n_head": 4 + (i % 3) * 2,
        "window_s_fraction": 0.2 + (i % 5) * 0.15,
        "embedding_lr": 0.05 * (1 + i % 7), "unembedding_lr": 0.001 * (1 + i % 4),
        "matrix_lr": 0.005 * (1 + i % 6), "scalar_lr": 0.02 * (1 + i % 5),
        "weight_decay": 0.01 * (i % 8), "warmup_ratio": 0.02 * (i % 6),
        "batch_size": 2048 * (1 + i % 4),
        "num_steps": steps, "val_bpb": val_bpb,
    }
    r.update(over)
    return r


def _healthy(n=40):
    """Step count a clean function of n_layer, so the model can learn it."""
    return [_row(i, steps=2000 - 100 * (4 + (i % 9)), val_bpb=1.2 + 0.01 * (i % 7)) for i in range(n)]


# --- the never-fabricate contract ----------------------------------------

def test_returns_rows_unchanged_when_it_cannot_judge():
    """No num_steps logged anywhere -- nothing can be concluded, so nothing
    is dropped on suspicion."""
    rows = [{k: v for k, v in _row(i).items() if k != "num_steps"} for i in range(40)]
    assert without_compute_starved(rows) == rows


def test_returns_rows_unchanged_below_min_n():
    assert len(without_compute_starved(_healthy(5))) == 5


def test_step_deficits_none_when_deps_unavailable(monkeypatch):
    import state.surrogate as s
    monkeypatch.setattr(s, "SURROGATE_DEPS_AVAILABLE", False)
    assert s.step_deficits(_healthy(40)) is None


@requires_deps
def test_unjudgeable_rows_are_kept_not_dropped():
    """A row missing num_steps sits alongside judgeable ones -- it must
    survive, since absence of evidence isn't evidence of starvation."""
    rows = _healthy(40)
    orphan = {k: v for k, v in _row(0).items() if k != "num_steps"}
    rows.append(orphan)
    kept = without_compute_starved(rows)
    assert orphan in kept


# --- it actually catches starvation --------------------------------------

@requires_deps
def test_flags_a_run_robbed_of_steps():
    rows = _healthy(40)
    starved = _row(0, steps=int((2000 - 100 * 4) * 0.5), val_bpb=1.9)  # half its due
    rows.append(starved)
    kept = without_compute_starved(rows)
    assert starved not in kept
    assert len(kept) == len(rows) - 1


@requires_deps
def test_does_not_flag_a_config_that_is_merely_slow():
    """The whole point: an expensive architecture legitimately earns fewer
    steps. Judging against the config's OWN prediction is what separates
    that from a contended run."""
    rows = _healthy(40)
    # n_layer=12 -> the model learns this config only ever gets 800 steps
    slow = [_row(i, steps=800, val_bpb=1.35, n_layer=12) for i in range(40, 55)]
    kept = without_compute_starved(rows + slow)
    assert all(r in kept for r in slow), "a legitimately slow config was wrongly excluded"


@requires_deps
def test_threshold_is_respected():
    rows = _healthy(40)
    due = 2000 - 100 * 4
    mild = _row(0, steps=int(due * 0.95), val_bpb=1.3)      # 5% short -> keep
    severe = _row(0, steps=int(due * 0.4), val_bpb=1.9)     # 60% short -> drop
    kept = without_compute_starved(rows + [mild, severe], threshold=STEP_DEFICIT_THRESHOLD)
    assert mild in kept and severe not in kept


@requires_deps
def test_deficits_are_aligned_with_rows_and_signed():
    rows = _healthy(40)
    deficits = step_deficits(rows)
    assert len(deficits) == len(rows)
    assert any(d is not None for d in deficits)
    # A healthy set should hover near zero, not be systematically starved.
    real = [d for d in deficits if d is not None]
    assert abs(sum(real) / len(real)) < 0.15


# --- integration with fit_surrogate --------------------------------------

@requires_deps
def test_fit_surrogate_excludes_starved_runs_by_default():
    """A contended run in a good region reports a bad val_bpb. Left in, it
    teaches the surrogate that the region is bad."""
    rows = _healthy(40)
    liars = [_row(0, steps=100, val_bpb=9.9) for _ in range(6)]
    clean = fit_surrogate(rows + liars)
    dirty = fit_surrogate(rows + liars, exclude_compute_starved=False)
    assert clean is not None and dirty is not None
    assert clean.n_train < dirty.n_train
    probe = {k: v for k, v in _row(0).items() if k not in ("num_steps", "val_bpb")}
    assert clean.predict(probe)[0] < dirty.predict(probe)[0]


@requires_deps
def test_fit_surrogate_opt_out_preserves_old_behaviour():
    rows = _healthy(40)
    assert fit_surrogate(rows, exclude_compute_starved=False).n_train == len(rows)


@requires_deps
def test_fit_surrogate_still_fits_when_nothing_is_starved():
    rows = _healthy(40)
    assert fit_surrogate(rows).n_train == len(rows)


# --- token-budget regime: truncated runs -----------------------------------
# Under prepare.py's TOKEN_BUDGET, num_steps becomes a deterministic function
# of batch_size, so the inferred step-deficit above has nothing left to
# detect. The signal that matters becomes train.py's directly-reported
# budget_shortfall_pct: >0 means the run hit the wall-clock safety cap and
# trained on LESS data than everything it would be compared against.

def test_truncated_runs_are_excluded_without_needing_a_model():
    """Direct evidence, so it must work even with too little history to fit
    the step model at all."""
    rows = [_row(i, val_bpb=1.3) for i in range(3)]
    rows.append(_row(9, val_bpb=1.8, budget_shortfall_pct=37.5))
    kept = without_compute_starved(rows)
    assert len(kept) == 3
    assert all(r.get("budget_shortfall_pct") is None for r in kept)


def test_a_complete_run_reports_zero_shortfall_and_is_kept():
    rows = [_row(i, val_bpb=1.3, budget_shortfall_pct=0.0) for i in range(3)]
    assert without_compute_starved(rows) == rows


@requires_deps
def test_truncated_runs_are_excluded_alongside_step_deficits():
    rows = _healthy(40)
    truncated = _row(0, val_bpb=1.9, budget_shortfall_pct=12.0)
    starved = _row(0, steps=int((2000 - 100 * 4) * 0.4), val_bpb=1.9)
    kept = without_compute_starved(rows + [truncated, starved])
    assert truncated not in kept and starved not in kept


def test_non_numeric_shortfall_does_not_crash_or_drop():
    """results.tsv stores blanks for columns a run never reported."""
    rows = [_row(i, val_bpb=1.3, budget_shortfall_pct="") for i in range(3)]
    assert without_compute_starved(rows) == rows
