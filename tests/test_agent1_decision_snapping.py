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
