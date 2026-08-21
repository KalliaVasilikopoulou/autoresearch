"""Step 5b, second half: interpretability steers ARCHITECTURE, and only there.

Everything Agent 2 measures is a statement about depth, width or heads -- which
are Agent 4's parameters. The surrogate cannot see any of it: it knows settings
and scores, so it cannot tell "bad because too deep" from "bad because the
learning rate was wrong".

It biases rather than overrules, scaled by the surrogate's own out-of-bag
accuracy: trust the model where it predicts well, lean on direct observation
where it does not.
"""

import math

import pytest

from agents import xai_direction
from state.results_analysis import ARCHITECTURE_COLUMNS


def _fingerprint(**over):
    """A fingerprint with no rule tripped, so a test can trip exactly one."""
    fp = {"dla": [1.0, 1.0, 1.0, 1.0], "x0_lambda": [1.0, 1.0, 1.0, 1.0],
          "attn_entropy": [5.0, 5.0, 5.0, 5.0], "attn_distance": [1.0, 5.0, 9.0, 20.0],
          "induction_score": 0.0}
    fp.update(over)
    return fp


def _evidence(n=1, **over):
    """`n` copies of the same fingerprint. architecture_votes now requires
    MIN_AGREEING_FINGERPRINTS readings to agree before a direction counts, so a
    test of what one SIGNAL means either supplies enough agreeing readings or
    passes min_agreeing=1 to isolate the rule from the gate."""
    return [{"token_fingerprint": _fingerprint(**over)} for _ in range(n)]


# --- which signals produce which direction ----------------------------------


def test_late_layers_contributing_nothing_votes_to_reduce_depth():
    """The most actionable signal we have: if the last layers do not write to
    the output, the model is too deep."""
    votes = xai_direction.architecture_votes(
        _evidence(dla=[1.0, 1.0, 1.0, 0.001]), min_agreeing=1)
    assert votes["n_layer"] < 0


def test_narrow_attention_votes_fewer_heads_and_more_width():
    votes = xai_direction.architecture_votes(
        _evidence(attn_entropy=[0.2, 0.2, 0.2, 0.2]), min_agreeing=1)
    assert votes["n_head"] < 0
    assert votes["n_embd"] > 0


def test_only_architecture_votes_come_through():
    """window_s_fraction stays with Agent 1 -- it changes no weights, so it can
    vary safely inside a region (step 1: 0.05 vs 0.95 gave 46/46 identical
    tensors)."""
    votes = xai_direction.architecture_votes(
        _evidence(attn_distance=[9.0, 9.5, 9.9, 10.0]),  # trips the window rule
        min_agreeing=1)
    assert set(votes) <= set(ARCHITECTURE_COLUMNS)
    assert "window_s_fraction" not in votes


def test_no_fingerprint_gives_no_direction():
    """The common case -- fingerprints are computed on a cadence, not every
    run, because they cost real GPU time."""
    assert xai_direction.architecture_votes(None) == {}
    assert xai_direction.architecture_votes([]) == {}
    assert xai_direction.architecture_votes([{"report_id": "r1"}]) == {}


# --- the agreement gate -----------------------------------------------------


def test_one_fingerprint_alone_steers_nothing():
    """A fingerprint is a measurement of a noisy training run. The same
    variation that makes val_bpb differences under 0.0138 unreadable at one
    seed also moves dead-head counts and per-layer contributions -- and an
    architecture vote is expensive, since it opens a whole new region rather
    than nudging a knob inside one."""
    once = _evidence(n=1, dla=[1.0, 1.0, 1.0, 0.001])
    assert xai_direction.architecture_votes(once) == {}
    assert xai_direction.architecture_votes(once, min_agreeing=1)["n_layer"] < 0


def test_two_agreeing_fingerprints_do_steer():
    twice = _evidence(n=2, dla=[1.0, 1.0, 1.0, 0.001])
    assert xai_direction.architecture_votes(twice)["n_layer"] < 0


def test_disagreeing_fingerprints_cancel_rather_than_average():
    """"The evidence does not know" is the honest answer, and it is different
    from "the evidence says zero" -- averaging two opposite readings would
    invent a confident middle."""
    deep = {"token_fingerprint": _fingerprint(dla=[1.0, 1.0, 1.0, 0.001])}
    shallow = {"token_fingerprint": _fingerprint(dla=[0.001, 1.0, 1.0, 1.0])}
    votes = xai_direction.architecture_votes([shallow, deep])
    assert "n_layer" not in votes


def test_head_ablation_is_exempt_from_the_agreement_gate():
    """It does not INFER redundancy -- it switches a head off and measures what
    happens. dead_head_vote already requires a majority of probed heads to be
    free, which is agreement within a single measurement."""
    # One head that matters, three that can be switched off for free. All-zero
    # impacts give peak == 0 and the rule abstains, which is correct: nothing
    # was measured, not "every head is dead".
    evidence = [{"head_ablation_impacts": {"0": 0.5, "1": 0.001, "2": 0.001, "3": 0.001}}]
    assert xai_direction.architecture_votes(evidence)["n_head"] < 0


# --- head ablation: the direct evidence -------------------------------------


def test_mostly_dead_heads_vote_to_reduce_them():
    """Ablation does not infer that heads are redundant -- it removes one and
    measures what happens."""
    impacts = {f"L0_H{i}": (0.05 if i < 2 else 0.0001) for i in range(8)}
    assert xai_direction.dead_head_vote([{"head_ablation_impacts": impacts}]) == -1


def test_heads_that_all_matter_produce_no_vote():
    impacts = {f"L0_H{i}": 0.04 + 0.001 * i for i in range(8)}
    assert xai_direction.dead_head_vote([{"head_ablation_impacts": impacts}]) is None


def test_ablation_deadness_is_relative_not_absolute():
    """Impacts scale with the model, so the threshold is a fraction of the
    largest observed impact rather than a fixed number."""
    big = {f"L0_H{i}": (5.0 if i < 2 else 0.001) for i in range(8)}
    small = {f"L0_H{i}": (0.005 if i < 2 else 1e-6) for i in range(8)}
    assert xai_direction.dead_head_vote([{"head_ablation_impacts": big}]) == -1
    assert xai_direction.dead_head_vote([{"head_ablation_impacts": small}]) == -1


# --- weighting against the surrogate ----------------------------------------


class _FakeSurrogate:
    def __init__(self, actual, predicted):
        self.oob_actual, self.oob_predicted = actual, predicted


def test_accuracy_is_high_when_predictions_track_reality():
    sm = _FakeSurrogate([1.20, 1.25, 1.30, 1.35], [1.21, 1.24, 1.31, 1.34])
    assert xai_direction.surrogate_accuracy(sm) > 0.9


def test_accuracy_floors_at_zero_for_a_useless_model():
    """A negative R^2 means worse than predicting the mean. For "how much
    should we trust it" that is no trust, not negative trust."""
    sm = _FakeSurrogate([1.20, 1.25, 1.30, 1.35], [9.0, 0.1, 9.0, 0.1])
    assert xai_direction.surrogate_accuracy(sm) == 0.0


def test_accuracy_is_none_without_enough_data():
    assert xai_direction.surrogate_accuracy(_FakeSurrogate([], [])) is None
    assert xai_direction.surrogate_accuracy(_FakeSurrogate([1.2], [1.2])) is None


def test_a_good_surrogate_silences_xai():
    """Bias, not override: where the model predicts well, leave it alone."""
    steps = xai_direction.weighted_step({"n_layer": -1}, accuracy=1.0, base_weight=0.25)
    assert steps == {}


def test_a_poor_surrogate_lets_xai_through_at_the_base_weight():
    steps = xai_direction.weighted_step({"n_layer": -1}, accuracy=0.0, base_weight=0.25)
    assert steps["n_layer"] == pytest.approx(-0.25)


def test_no_accuracy_yet_means_lean_on_observation():
    """The early phase: the surrogate has nothing to say and the direct
    measurement is all there is."""
    steps = xai_direction.weighted_step({"n_layer": -1}, accuracy=None, base_weight=0.25)
    assert steps["n_layer"] == pytest.approx(-0.25)


def test_votes_are_capped_so_one_reading_cannot_leap():
    steps = xai_direction.weighted_step({"n_layer": -9}, accuracy=0.0,
                                        base_weight=1.0, max_step=2)
    assert steps["n_layer"] == pytest.approx(-2.0)


# --- Agent 1 must no longer touch architecture ------------------------------


def test_agent1_no_longer_moves_architecture_from_a_fingerprint(tmp_path):
    """THE LEAK THIS CLOSED. _fingerprint_adjustment ran on every decision
    path, including the surrogate one, so it bypassed the fence 3b put on the
    EI search -- and a changed architecture inside a region means a run whose
    weights differ from the rest of its own region."""
    from agents.agent1_training_specialist import Agent1TrainingSpecialist

    a1 = Agent1TrainingSpecialist(config_path=str(tmp_path / "missing.yaml"),
                                  root_dir=str(tmp_path))
    before = {"n_layer": 12, "n_embd": 512, "n_head": 8, "window_s_fraction": 0.5}
    after = a1._fingerprint_adjustment(dict(before), _evidence(
        dla=[1.0, 1.0, 1.0, 0.001],          # would have voted n_layer down
        attn_entropy=[0.2, 0.2, 0.2, 0.2],   # would have voted n_head down, n_embd up
        attn_distance=[9.0, 9.5, 9.9, 10.0],  # window rule -- still Agent 1's
    ))

    for col in ARCHITECTURE_COLUMNS:
        assert after[col] == before[col], f"Agent 1 must not move {col}"
    assert after["window_s_fraction"] != before["window_s_fraction"]
