"""Two changes the size sweep forced, tested together because they are the
same mistake seen from opposite sides: an architecture number written in more
than one place.

RAISING THE BOX. `reports/size_sweep.md` measured val_bpb falling at every one
of five steps across 189x of model size, and still falling at the old ceiling
-- so what limited size was the search space, not the search. The ceilings
moved (n_layer 24 -> 28, n_embd 1024 -> 1280), but they used to appear as
hardcoded literals in eight places across Agent 1's non-surrogate paths, so a
raise in one spot would have left every one of those paths dragging the model
back to 1024 whenever it ran.

KEEPING A REGION'S IDENTITY. The same paths write n_layer/n_embd/n_head
directly. A region IS an exact architecture plus a neighbourhood of the eight
tunables, so a mid-region architecture change produces a run whose initial
weights differ from the rest of its own region -- destroying the pairing that
steps 1-3 exist to establish. Step 5b closed this on the fingerprint path;
these tests cover the rest of them.
"""

import pytest

from agents import agent1_training_specialist as a1mod
from agents.agent1_training_specialist import (
    ARCH_SAFE_RANGES,
    MAX_N_EMBD,
    MAX_N_HEAD,
    MAX_N_LAYER,
    SEARCH_SPACE,
    Agent1TrainingSpecialist,
)
from state.regions import RegionRegistry
from state.results_analysis import ARCHITECTURE_COLUMNS

BASE = {
    "n_layer": 8, "n_embd": 512, "n_head": 4, "window_s_fraction": 0.75,
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192,
}


def hp(**overrides):
    out = dict(BASE)
    out.update(overrides)
    return out


@pytest.fixture
def agent(tmp_path):
    for sub in ("state", "reports"):
        (tmp_path / sub).mkdir()
    return Agent1TrainingSpecialist(
        config_path="agents_config.yaml", root_dir=str(tmp_path),
        state_dir=str(tmp_path / "state"), reports_dir=str(tmp_path / "reports"),
    )


@pytest.fixture
def registry(tmp_path):
    return RegionRegistry(str(tmp_path / "state" / "regions.json"))


# --- one definition of the ceiling -------------------------------------------


def test_the_ceilings_track_the_search_space():
    """Derived, not repeated. This is the property that makes the next raise a
    one-line change instead of an eight-line one with seven chances to miss."""
    assert MAX_N_LAYER == SEARCH_SPACE["n_layer"][1]
    assert MAX_N_EMBD == SEARCH_SPACE["n_embd"][1]
    assert MAX_N_HEAD == SEARCH_SPACE["n_head"][1]


def test_the_box_is_wider_than_the_ladder_the_sweep_measured():
    """The sweep's top rung was n_layer=21 / n_embd=960 and was still
    improving. A box that cannot exceed it leaves the measured gain
    unreachable."""
    assert ARCH_SAFE_RANGES["n_layer"][1] > 21
    assert ARCH_SAFE_RANGES["n_embd"][1] > 960


def test_the_box_stays_inside_what_train_py_will_run():
    """train.py clamps n_layer to 48, n_head to 64 and n_embd to 8192. A box
    outside that would propose configurations train.py silently rewrites, and
    then "requested" and "actually used" diverge in results.tsv."""
    assert ARCH_SAFE_RANGES["n_layer"][1] <= 48
    assert ARCH_SAFE_RANGES["n_head"][1] <= 64
    assert ARCH_SAFE_RANGES["n_embd"][1] <= 8192


def test_the_worst_case_model_still_fits_the_wall_clock():
    """Sized against measurement, not taste. From the size sweep's own
    time-vs-size fit (~259s + 1.72s per M non-embedding params), the largest
    model the box now allows must leave real margin under train.py's 1800s
    MAX_TRAIN_SECONDS -- a run that trips the cap is excluded as incomplete,
    so it costs a GPU slot and returns nothing.
    """
    params_m = 12 * MAX_N_LAYER * MAX_N_EMBD ** 2 / 1e6
    projected_seconds = 259 + 1.72 * params_m
    assert projected_seconds < 0.8 * 1800


#: The old hardcoded ceilings, kept here as the thing being regressed against.
OLD_MAX_N_LAYER, OLD_MAX_N_EMBD = 24, 1024


def test_the_evidence_path_can_now_grow_past_the_old_ceiling(agent):
    """THE REGRESSION THE EIGHT LITERALS WOULD HAVE CAUSED. Start at the old
    ceiling and ask, through evidence, for a bigger model. Under the hardcoded
    24/1024 this could not move at all, however strong the evidence -- so the
    box raise would have applied to the surrogate path only."""
    agent.current_hyperparams = hp(n_layer=OLD_MAX_N_LAYER, n_embd=OLD_MAX_N_EMBD,
                                   n_head=8)
    out = agent._evidence_adjustment(
        latest_summary=None,
        evidence=[{"report_id": "r", "hyperparameter_importance":
                   {"n_layer": 1.0, "n_embd": 1.0}}],
        stuck_signal=False, iteration=3)

    assert out["n_layer"] > OLD_MAX_N_LAYER
    assert out["n_embd"] > OLD_MAX_N_EMBD


def test_no_path_ever_leaves_the_box(agent):
    """The ceiling still has to hold. Push every non-surrogate path hard from
    a model already at the top and check none of them steps outside."""
    agent.current_hyperparams = hp(n_layer=MAX_N_LAYER, n_embd=MAX_N_EMBD,
                                   n_head=MAX_N_HEAD)
    pushy = [{"report_id": "r", "hyperparameter_importance":
              {"n_layer": 1.0, "n_embd": 1.0, "n_head": 1.0}}]

    for out in (agent._evidence_adjustment(latest_summary=None, evidence=pushy,
                                           stuck_signal=False, iteration=3),
                agent._heuristic_adjustment(None, False, 3),
                agent._heuristic_adjustment(None, False, 30)):
        for col in ARCHITECTURE_COLUMNS:
            lo, hi = SEARCH_SPACE[col]
            assert lo <= out[col] <= hi, f"{col}={out[col]} left the box"


def test_a_random_restart_can_still_reach_the_top_of_the_box(agent, monkeypatch):
    """_radical_change picks n_embd from a fixed list. If that list stops at
    the old ceiling, a restart can never land in the part of the space the
    sweep says is best."""
    monkeypatch.setattr(a1mod.random, "choice", lambda seq: seq[-1])
    out = agent._radical_change(hp())
    assert out["n_embd"] == MAX_N_EMBD > OLD_MAX_N_EMBD


# --- a region's architecture survives every path -----------------------------


def _decide_inside(agent, region, **kwargs):
    with agent.search_region(region):
        return agent.decide_next_hyperparams(**kwargs)


def test_the_evidence_path_cannot_change_a_regions_architecture(agent, registry):
    """The fallback paths are not exotic -- they are what runs during a
    campaign's opening iterations, before the surrogate has enough data to
    fit, which is exactly when a region is youngest."""
    region = registry.open_region(hp(), at_run=0)
    agent.current_hyperparams = hp()
    agent.use_surrogate = False

    out = _decide_inside(agent, region, latest_summary=None,
                         evidence=[{"report_id": "r1", "hyperparameter_importance":
                                    {"n_layer": 1.0, "n_embd": 1.0, "n_head": 1.0}}],
                         iteration=2)
    for col in ARCHITECTURE_COLUMNS:
        assert out[col] == region.anchor[col], f"{col} moved inside its own region"


def test_the_heuristic_path_cannot_change_a_regions_architecture(agent, registry):
    region = registry.open_region(hp(), at_run=0)
    agent.current_hyperparams = hp()
    agent.use_surrogate = False

    out = _decide_inside(agent, region, latest_summary=None, evidence=None, iteration=2)
    for col in ARCHITECTURE_COLUMNS:
        assert out[col] == region.anchor[col]


def test_the_architecture_is_restored_from_the_anchor_not_the_center(agent, registry):
    """The anchor is what makes a region a fixed place. Restoring from the
    center would let a region's identity walk, one proposal at a time."""
    region = registry.open_region(hp(n_layer=8, n_embd=512, n_head=4), at_run=0)
    region.center = dict(region.center, n_layer=19, n_embd=1024, n_head=13)
    agent.current_hyperparams = hp()
    agent.use_surrogate = False

    out = _decide_inside(agent, region, latest_summary=None, evidence=None, iteration=2)
    assert (out["n_layer"], out["n_embd"], out["n_head"]) == (8, 512, 4)


def test_tunables_are_still_free_to_move_inside_a_region(agent, registry):
    """The guard must pin the architecture and nothing else -- the eight
    tunables are the whole point of searching a region."""
    region = registry.open_region(hp(), at_run=0)
    agent.current_hyperparams = hp()
    agent.use_surrogate = False

    out = _decide_inside(agent, region, latest_summary=None, evidence=None, iteration=2)
    assert any(out[c] != BASE[c] for c in
               ("embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr",
                "weight_decay", "warmup_ratio", "batch_size", "window_s_fraction"))


def test_outside_a_region_the_architecture_is_free(agent):
    """Nothing to preserve, so the paths keep their freedom -- that is how a
    campaign moves between architectures at all."""
    agent.current_hyperparams = hp()
    agent.use_surrogate = False
    assert agent._active_region is None

    out = agent.decide_next_hyperparams(
        latest_summary=None,
        evidence=[{"report_id": "r1", "hyperparameter_importance":
                   {"n_layer": 1.0, "n_embd": 1.0, "n_head": 1.0}}],
        iteration=2)
    assert any(out[c] != BASE[c] for c in ARCHITECTURE_COLUMNS)
