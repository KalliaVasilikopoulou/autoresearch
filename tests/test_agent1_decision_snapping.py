"""decide_next_hyperparams applies a final, path-independent n_embd snap
(matching train.py's own head_dim-parity snap, see state/surrogate.py's
snap_n_embd) regardless of which internal path produced the proposal --
mirrors the existing LR re-clamp pass right above it in
agents/agent1_training_specialist.py. Before this fix, only the Tier 1
surrogate's EI/cold-start path applied this snap; every other path
(heuristic, evidence-based, Tier 4 fingerprint votes, Claude's free-form
suggestion) could propose an n_embd/n_head combo train.py would silently
re-snap at train time, which pipeline_validator was correctly flagging as
an ERROR ("train.py clamped n_embd") on nearly every iteration in practice.
"""

import pytest

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from state.surrogate import snap_n_embd


def _base_hyperparams():
    return {
        "n_layer": 12, "n_head": 8, "n_embd": 512, "window_s_fraction": 0.75,
        "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04, "scalar_lr": 0.5,
        "batch_size": 8192, "warmup_ratio": 0.1, "weight_decay": 0.1,
    }


@pytest.fixture
def specialist(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("agent1:\n  use_llm: false\n")
    return Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path))


def test_decide_next_hyperparams_snaps_n_embd_from_heuristic_path(specialist, monkeypatch):
    specialist.use_surrogate = False  # force the evidence/heuristic fallback path
    bad_params = _base_hyperparams()
    bad_params["n_embd"] = 100
    bad_params["n_head"] = 3  # head_dim = round(100/3) = 33 (odd) -> snaps to 34 -> n_embd=102
    monkeypatch.setattr(specialist, "_heuristic_adjustment", lambda *a, **k: dict(bad_params))

    result = specialist.decide_next_hyperparams(latest_summary=None, iteration=0)

    assert result is not None
    assert result["n_head"] == 3
    assert result["n_embd"] == snap_n_embd(100, 3) == 102


def test_decide_next_hyperparams_leaves_already_valid_n_embd_unchanged(specialist, monkeypatch):
    specialist.use_surrogate = False
    good_params = _base_hyperparams()
    good_params["n_embd"] = 512
    good_params["n_head"] = 8  # head_dim = 64, already even -> unchanged
    monkeypatch.setattr(specialist, "_heuristic_adjustment", lambda *a, **k: dict(good_params))

    result = specialist.decide_next_hyperparams(latest_summary=None, iteration=0)

    assert result is not None
    assert result["n_embd"] == 512


# --- relocate_search_center (Agent 4 commit hook) ------------------------
# The one mechanism by which Agent 4's "this other region is better" verdict
# actually changes Agent 1's future proposals: agents/search_planner.py pins
# `center = dict(current_best_hyperparams)` every call, so moving that dict
# is the only way the EI search leaves a basin.

def test_relocate_search_center_moves_current_hyperparams(specialist):
    target = _base_hyperparams()
    target["n_layer"] = 3
    target["matrix_lr"] = 0.017

    specialist.relocate_search_center(target)

    assert specialist.current_hyperparams["n_layer"] == 3
    assert specialist.current_hyperparams["matrix_lr"] == 0.017


def test_relocate_search_center_persists_to_yaml(specialist):
    import yaml
    specialist.relocate_search_center({**_base_hyperparams(), "n_layer": 3})
    saved = yaml.safe_load(specialist.model_config_path.read_text())
    assert saved["n_layer"] == 3


def test_relocate_search_center_does_not_rewrite_best_val_bpb(specialist):
    """best_val_bpb is the record of what was actually measured -- still the
    right EI f_best and stop-condition reference. Only the search center
    moves; the achievement record does not."""
    specialist.best_val_bpb = 1.2065
    specialist.relocate_search_center({**_base_hyperparams(), "n_layer": 3})
    assert specialist.best_val_bpb == 1.2065


def test_relocate_search_center_copies_rather_than_aliases(specialist):
    """A later mutation of Agent 4's own dict must not silently rewrite
    Agent 1's live search center."""
    target = _base_hyperparams()
    specialist.relocate_search_center(target)
    target["n_layer"] = 999
    assert specialist.current_hyperparams["n_layer"] != 999


# --- train_model trains what it is GIVEN ---------------------------------
# train_model writes model_hyperparams.yaml and then points train.py at that
# file, so whatever lands there is what actually gets trained. It used to
# save self.current_hyperparams and ignore its argument -- invisible while
# Agent 1 was the only decider (decide_next_hyperparams assigns the same dict
# object it returns), catastrophic once Agent 4 proposes probes from its own
# dict: every probe would have trained Agent 1's stale center while
# results.tsv recorded Agent 4's proposal.

def test_train_model_persists_the_hyperparams_it_was_given(specialist):
    import yaml
    specialist.current_hyperparams = {**_base_hyperparams(), "n_layer": 99}
    given = {**_base_hyperparams(), "n_layer": 3, "matrix_lr": 0.017}

    specialist.train_model(given, dry_run=True, iteration=0)

    saved = yaml.safe_load(specialist.model_config_path.read_text())
    assert saved["n_layer"] == 3, "trained the agent's own state instead of the argument"
    assert saved["matrix_lr"] == 0.017


def test_train_model_still_persists_own_state_when_they_are_the_same_object(specialist):
    """Agent 1's own path is unchanged: it passes the very dict it assigned
    to current_hyperparams, including late mutations like token_xai_enabled."""
    import yaml
    hp = {**_base_hyperparams(), "n_layer": 7}
    specialist.current_hyperparams = hp
    hp["token_xai_enabled"] = True  # mutated after the decision, as the orchestrator does

    specialist.train_model(hp, dry_run=True, iteration=0)

    saved = yaml.safe_load(specialist.model_config_path.read_text())
    assert saved["n_layer"] == 7
    assert saved["token_xai_enabled"] is True
