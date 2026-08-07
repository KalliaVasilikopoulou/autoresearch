"""Tests for the per-region stagnation / stuck / stop path.

These cover the single most dangerous leftover of the old single-search
design. The chain was:

    orchestrator passes the campaign's last 3 results
      -> _detect_stagnation compares them to each other (hardcoded 0.01)
      -> effective_stuck_signal
      -> _evidence_adjustment(stuck_signal=True)
      -> _radical_change -> random.randint(8, 20) on n_layer

Under a multi-GPU wave those 3 results come from 3 different regions, so the
comparison measures the distance between two places in the space, calls it
"no progress", and jumps the search center out of the region it was meant to
be exploiting. Region identity would not survive a single wave.
"""

import math

import pytest

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from state.regions import RegionRegistry


BASE = {
    "n_layer": 8, "n_embd": 512, "n_head": 4, "window_s_fraction": 0.75,
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192,
}
SIGMA = 0.00919


def hp(**overrides):
    out = dict(BASE)
    out.update(overrides)
    return out


@pytest.fixture
def agent(tmp_path):
    for sub in ("state", "reports"):
        (tmp_path / sub).mkdir()
    (tmp_path / "state" / "noise_floor.json").write_text(
        '{"std": %s, "mean": 1.2992, "n": 10}' % SIGMA, encoding="utf-8")
    return Agent1TrainingSpecialist(
        config_path="agents_config.yaml",
        root_dir=str(tmp_path),
        state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"),
    )


@pytest.fixture
def registry(tmp_path):
    return RegionRegistry(str(tmp_path / "state" / "regions.json"))


def results(*vals):
    return [{"val_bpb": v} for v in vals]


# -- the trend is read from the region, not from the campaign ---------------


def test_stagnation_ignores_other_regions_runs(agent, registry):
    """The campaign's recent results say "flat"; this region's own history
    says "improving fast". The region wins."""
    r = registry.open_region(hp(), at_run=0)
    for v in (1.50, 1.40, 1.30):
        registry.assign_run(r.region_id, "x", v)

    interleaved = results(1.30, 1.31, 1.30)  # three different regions
    assert agent._detect_stagnation(interleaved, 1.30) is True, "campaign-wide: flat"
    with agent.search_region(r):
        assert agent._detect_stagnation(interleaved, 1.30) is False


def test_a_flat_region_is_still_detected_as_flat(agent, registry):
    r = registry.open_region(hp(), at_run=0)
    for v in (1.30, 1.3005, 1.2999):
        registry.assign_run(r.region_id, "x", v)
    with agent.search_region(r):
        assert agent._detect_stagnation(None, 1.2999) is True


def test_a_region_with_too_little_history_is_not_called_stagnant(agent, registry):
    r = registry.open_region(hp(), at_run=0)
    registry.assign_run(r.region_id, "x", 1.30)
    with agent.search_region(r):
        assert agent._detect_stagnation(results(1.3, 1.3, 1.3), 1.30) is False


# -- "improved" is sized against the noise floor ----------------------------


def test_a_sub_sigma_gain_is_not_progress(agent, registry):
    """The old hardcoded 0.01 was 1.09 sigma, so a run that was merely
    luckier than the last one cleared the bar."""
    r = registry.open_region(hp(), at_run=0)
    for v in (1.3000, 1.2960, 1.2920):  # steps of 0.004 = 0.44 sigma
        registry.assign_run(r.region_id, "x", v)
    with agent.search_region(r):
        assert agent._detect_stagnation(None, 1.2920) is True


def test_a_real_multi_sigma_gain_is_progress(agent, registry):
    r = registry.open_region(hp(), at_run=0)
    for v in (1.3000, 1.2700, 1.2400):  # steps of 0.03 = 3.3 sigma
        registry.assign_run(r.region_id, "x", v)
    with agent.search_region(r):
        assert agent._detect_stagnation(None, 1.2400) is False


def test_the_margin_follows_the_measured_noise_floor(agent, registry):
    """The noise floor has already moved ~3x once (time budget -> token
    budget). A hardcoded margin would silently change meaning; a sigma
    multiple does not."""
    r = registry.open_region(hp(), at_run=0)
    for v in (1.3000, 1.2900, 1.2800):  # steps of 0.01 = 1.09 sigma
        registry.assign_run(r.region_id, "x", v)
    with agent.search_region(r):
        agent.stagnation_sigma_multiple = 0.5   # margin 0.0046 -> a real gain
        assert agent._detect_stagnation(None, 1.2800) is False
        agent.stagnation_sigma_multiple = 4.0   # margin 0.0368 -> noise
        assert agent._detect_stagnation(None, 1.2800) is True


# -- the teleport is gone ---------------------------------------------------


def test_a_stuck_region_does_not_jump_its_center_out_of_the_region(agent, registry):
    """The whole point. _radical_change would randomize n_layer and n_embd,
    landing the search outside the region it was scoped to."""
    r = registry.open_region(hp(n_layer=6, n_embd=512), at_run=0)
    for v in (1.30, 1.3001, 1.2999):
        registry.assign_run(r.region_id, "x", v)

    with agent.search_region(r):
        proposal = agent.decide_next_hyperparams(
            recent_results=results(1.30, 1.3001, 1.2999), latest_val_bpb=1.2999,
            iteration=5,
        )

    assert proposal is not None
    assert agent.last_region_stuck is True, "the signal is reported..."
    assert agent.last_decision_log["path_taken"] != "radical_change", "...not acted on by teleporting"


def test_stuck_outside_a_region_still_takes_the_radical_path(agent):
    """The single-search path is unchanged -- with no region to preserve,
    a radical jump is still the right response to being stuck."""
    agent.decide_next_hyperparams(
        recent_results=results(1.30, 1.3001, 1.2999), latest_val_bpb=1.2999,
        iteration=5,
    )
    assert agent.last_decision_log["path_taken"] == "radical_change"


def test_the_stuck_flag_is_cleared_when_a_region_scope_opens(agent, registry):
    a = registry.open_region(hp(), at_run=0)
    b = registry.open_region(hp(n_layer=16), at_run=0)
    for v in (1.30, 1.3001, 1.2999):
        registry.assign_run(a.region_id, "x", v)

    with agent.search_region(a):
        agent.decide_next_hyperparams(recent_results=None, latest_val_bpb=1.2999, iteration=1)
    assert agent.last_region_stuck is True

    with agent.search_region(b):
        assert agent.last_region_stuck is False, "b must not inherit a's verdict"


# -- one region running out of road is not a campaign stop ------------------


def test_a_stalled_region_reports_itself_instead_of_stopping_the_campaign(agent, registry):
    r = registry.open_region(hp(), at_run=0)
    for v in (1.30, 1.30, 1.30, 1.30, 1.30):
        registry.assign_run(r.region_id, "x", v)

    with agent.search_region(r):
        assert agent._should_stop_early(None, 1.30) is False
        assert agent.last_region_stalled is True


def test_a_healthy_region_is_not_reported_as_stalled(agent, registry):
    r = registry.open_region(hp(), at_run=0)
    for v in (1.50, 1.40, 1.30, 1.20, 1.10):
        registry.assign_run(r.region_id, "x", v)
    with agent.search_region(r):
        assert agent._should_stop_early(None, 1.10) is False
        assert agent.last_region_stalled is False


def test_the_campaign_wide_stop_still_works_outside_a_region(agent):
    flat = results(*([1.30] * 6))
    assert agent._should_stop_early(flat, 1.30) is True


def test_a_new_record_is_never_reported_as_a_stall(agent):
    """Latent bug the region path made reachable: the stall baseline used to
    include the latest run itself, so `improvement` was zero exactly when the
    newest run set a record -- a steadily improving search called itself
    stalled. Unreachable before only because this needs 4 values and the
    orchestrator passes 3."""
    improving = results(1.50, 1.40, 1.30, 1.20, 1.10)
    assert agent._should_stop_early(improving, 1.10) is False


def test_a_stalled_region_never_returns_none_from_the_decision(agent, registry):
    """None means STOP THE CAMPAIGN. A single exhausted region must not be
    able to say that while three other regions are still improving."""
    r = registry.open_region(hp(), at_run=0)
    for v in ([1.30] * 6):
        registry.assign_run(r.region_id, "x", v)
    with agent.search_region(r):
        proposal = agent.decide_next_hyperparams(
            recent_results=results(*([1.30] * 6)), latest_val_bpb=1.30, iteration=9,
        )
    assert proposal is not None
    assert agent.last_region_stalled is True
