"""Tests for Agent1TrainingSpecialist.search_region -- running one decision
as one region, so several regions can be searched concurrently by a single
Agent 1.

The properties that matter are all about *isolation*. Before this existed,
agents/search_planner.py kept one cold-start counter, one frozen set and one
active Gauss-Southwell block in a single state file, and Agent 1 kept a single
center. Four concurrent regions sharing those would interleave into one
incoherent search while looking, from the outside, like four.
"""

import json
from pathlib import Path

import pytest
import yaml

from agents.agent1_training_specialist import Agent1TrainingSpecialist
from state.regions import RegionRegistry


BASE = {
    "n_layer": 8, "n_embd": 512, "n_head": 4, "window_s_fraction": 0.75,
    "embedding_lr": 0.6, "unembedding_lr": 0.004, "matrix_lr": 0.04,
    "scalar_lr": 0.5, "weight_decay": 0.2, "warmup_ratio": 0.0,
    "batch_size": 8192,
}


@pytest.fixture
def agent(tmp_path):
    for sub in ("state", "reports"):
        (tmp_path / sub).mkdir()
    return Agent1TrainingSpecialist(
        config_path="agents_config.yaml",
        root_dir=str(tmp_path),
        state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"),
    )


@pytest.fixture
def registry(tmp_path):
    return RegionRegistry(str(tmp_path / "state" / "regions.json"))


def hp(**overrides):
    out = dict(BASE)
    out.update(overrides)
    return out


def test_each_region_searches_from_its_own_center(agent, registry):
    a = registry.open_region(hp(n_layer=4), at_run=0)
    b = registry.open_region(hp(n_layer=20), at_run=0)

    with agent.search_region(a):
        assert agent.current_hyperparams["n_layer"] == 4
    with agent.search_region(b):
        assert agent.current_hyperparams["n_layer"] == 20


def test_regions_do_not_share_planner_state(agent, registry):
    """The core of the refactor. Two regions must not write the same
    SearchPlannerState file, or one region's frozen parameters and block
    rotation silently apply to the other."""
    a = registry.open_region(hp(), at_run=0)
    b = registry.open_region(hp(n_layer=16), at_run=0)

    with agent.search_region(a):
        path_a = agent._search_planner_state_path
        dir_a = agent._search_plan_report_dir
    with agent.search_region(b):
        path_b = agent._search_planner_state_path
        dir_b = agent._search_plan_report_dir

    assert path_a != path_b
    assert dir_a != dir_b
    assert a.region_id in path_a and b.region_id in path_b


def test_ei_reference_is_the_region_best_not_the_campaign_best(agent, registry):
    """EI's f_best comes from best_val_bpb. With the campaign-wide value, a
    non-champion region sees ~zero improvement probability everywhere and its
    acquisition argmax degenerates into noise."""
    agent.best_val_bpb = 1.20
    r = registry.open_region(hp(), at_run=0)
    registry.assign_run(r.region_id, "run_0", 1.45)

    with agent.search_region(r):
        assert agent.best_val_bpb == pytest.approx(1.45)
    assert agent.best_val_bpb == pytest.approx(1.20), "campaign record restored"


def test_a_region_with_no_runs_yet_has_no_reference(agent, registry):
    agent.best_val_bpb = 1.20
    r = registry.open_region(hp(), at_run=0)
    with agent.search_region(r):
        assert agent.best_val_bpb == float("inf")


def test_a_genuine_campaign_record_found_inside_a_region_propagates_out(agent, registry):
    """The search is local; 'the best model anyone trained' is not."""
    agent.best_val_bpb = 1.30
    r = registry.open_region(hp(), at_run=0)
    registry.assign_run(r.region_id, "run_0", 1.25)
    with agent.search_region(r):
        agent.best_val_bpb = 1.19  # a probe in here beat the campaign record
    assert agent.best_val_bpb == pytest.approx(1.19)


def test_leaving_a_region_restores_the_global_center(agent, registry):
    before = dict(agent.current_hyperparams)
    r = registry.open_region(hp(n_layer=2), at_run=0)
    with agent.search_region(r):
        agent.current_hyperparams["n_layer"] = 7
    assert agent.current_hyperparams == before


def test_the_proposal_becomes_the_regions_new_center_but_not_its_anchor(agent, registry):
    r = registry.open_region(hp(n_layer=4), at_run=0)
    with agent.search_region(r):
        agent.current_hyperparams = hp(n_layer=9)
    assert r.center["n_layer"] == 9, "the local search drifts, by design"
    assert r.anchor["n_layer"] == 4, "identity does not"


def test_state_is_restored_even_when_the_decision_raises(agent, registry):
    """A failed decision in one region must not leave Agent 1 pointed at that
    region's planner state for every subsequent region."""
    before_center = dict(agent.current_hyperparams)
    before_path = agent._search_planner_state_path
    r = registry.open_region(hp(n_layer=3), at_run=0)

    with pytest.raises(RuntimeError):
        with agent.search_region(r):
            raise RuntimeError("surrogate blew up")

    assert agent.current_hyperparams == before_center
    assert agent._search_planner_state_path == before_path


def test_region_scoped_decisions_do_not_fight_over_model_hyperparams_yaml(agent, registry):
    """Several regions decided back-to-back in one wave would each overwrite
    the file train.py reads, leaving it describing whichever was last."""
    agent._save_hyperparams()
    original = yaml.safe_load(Path(agent.model_config_path).read_text(encoding="utf-8"))

    r = registry.open_region(hp(n_layer=23), at_run=0)
    with agent.search_region(r):
        agent.current_hyperparams = hp(n_layer=23)
        agent._save_hyperparams()

    after = yaml.safe_load(Path(agent.model_config_path).read_text(encoding="utf-8"))
    assert after == original


def test_an_explicit_save_is_still_honored_inside_a_region(agent, registry):
    """train_model passes the exact dict it is about to train; that must
    reach disk regardless of any region scope in effect, or results.tsv would
    record a configuration different from the one that ran."""
    r = registry.open_region(hp(), at_run=0)
    with agent.search_region(r):
        agent._save_hyperparams(hp(n_layer=21))
    after = yaml.safe_load(Path(agent.model_config_path).read_text(encoding="utf-8"))
    assert after["n_layer"] == 21


def test_sequential_path_can_still_persist_its_center(agent, registry):
    r = registry.open_region(hp(n_layer=13), at_run=0)
    with agent.search_region(r, save_hyperparams=True):
        agent.current_hyperparams = hp(n_layer=13)
        agent._save_hyperparams()
    after = yaml.safe_load(Path(agent.model_config_path).read_text(encoding="utf-8"))
    assert after["n_layer"] == 13


def test_nesting_is_unwound_in_order(agent, registry):
    """Not a pattern the orchestrator uses, but the save-suppression flag is
    stack-like and a bug here would surface as a mysteriously unwritten
    model_hyperparams.yaml much later."""
    a = registry.open_region(hp(n_layer=4), at_run=0)
    b = registry.open_region(hp(n_layer=18), at_run=0)
    with agent.search_region(a, save_hyperparams=True):
        assert agent._suppress_hyperparams_save is False
        with agent.search_region(b):
            assert agent._suppress_hyperparams_save is True
        assert agent._suppress_hyperparams_save is False
    assert agent._suppress_hyperparams_save is False


def test_orchestration_flags_do_not_become_part_of_a_regions_center(agent, registry):
    """region_id / token_xai_enabled / holdout_eval_if_below describe one
    dispatch, not a configuration. Left in the center they make regions.json
    read as a snapshot of orchestration state, and a stale
    holdout_eval_if_below could survive into a later run on the one path that
    does not overwrite it (no finite best yet)."""
    r = registry.open_region(hp(), at_run=0)
    with agent.search_region(r):
        agent.current_hyperparams = dict(
            hp(), region_id="r0009", token_xai_enabled=True,
            holdout_eval_if_below=1.23, ablation_k=10,
        )
    assert "region_id" not in r.center
    assert "token_xai_enabled" not in r.center
    assert "holdout_eval_if_below" not in r.center
    assert r.center["ablation_k"] == 10, "real pass-through keys are kept"
    assert r.center["n_layer"] == hp()["n_layer"]
