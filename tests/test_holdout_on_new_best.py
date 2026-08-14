"""Continuous tracking of validation-shard selection bias.

The search compares every run against ONE pinned val shard and has made
500+ accept/reject decisions against it -- the multiple-comparisons problem
prepare.py's HOLDOUT_SHARD exists to detect. The holdout machinery was
fully built (pinned shard, evaluate_bpb_holdout, a holdout_eval flag, a
results.tsv column, an Agent 2 parser, scripts/holdout_eval.py) but was only
reachable via a manual script, so across 564 runs it had never once been
used. These cover the orchestrator half of wiring it into the loop.

No GPU, no SSH: only the hyperparams dict handed to train.py is asserted on.
"""

import math

import pytest

from agents.orchestrator import Orchestrator
from state.regions import ACTIVE, Region


def _config(tmp_path, holdout_on_new_best=None):
    flag = "" if holdout_on_new_best is None else f"\n  holdout_on_new_best: {str(holdout_on_new_best).lower()}"
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text(f"""
agent1:
  use_llm: false
  accuracy_threshold: 0.01
  cost_limit_usd: 50.0
  training_budget_seconds: 60{flag}

agent2:
  xai_method: fast
  use_llm: false

agent3:
  batch_size: 100
  use_llm: false
  generate_charts: false

agent4:
  enabled: false

llm:
  backend: none

orchestrator:
  parallel: false
""".strip(), encoding="utf-8")
    return config_path


def _make_orchestrator(tmp_path, **kwargs):
    return Orchestrator(
        config_path=str(_config(tmp_path, **kwargs)),
        state_dir=str(tmp_path / "state"), reports_dir=str(tmp_path / "reports"),
        root_dir=str(tmp_path), dry_run=True,
    )


def test_threshold_is_the_current_best_val_bpb(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.agent1.best_val_bpb = 1.206506
    hp = {"n_layer": 6}
    orch._set_holdout_threshold(hp)
    assert hp["holdout_eval_if_below"] == 1.206506


def test_no_threshold_before_a_finite_best_exists(tmp_path):
    """best_val_bpb starts at infinity. Passing that through would make every
    opening run of a campaign beat it and trigger a holdout eval -- four of
    them at once in a parallel wave."""
    orch = _make_orchestrator(tmp_path)
    orch.agent1.best_val_bpb = float("inf")
    hp = {"n_layer": 6}
    orch._set_holdout_threshold(hp)
    assert "holdout_eval_if_below" not in hp


def test_disabled_by_config(tmp_path):
    orch = _make_orchestrator(tmp_path, holdout_on_new_best=False)
    orch.agent1.best_val_bpb = 1.2
    hp = {"n_layer": 6}
    orch._set_holdout_threshold(hp)
    assert "holdout_eval_if_below" not in hp


def test_enabled_by_default(tmp_path):
    """The whole point is that it stops being opt-in -- it was already
    opt-in via scripts/holdout_eval.py and therefore never used."""
    orch = _make_orchestrator(tmp_path)
    assert orch.holdout_on_new_best is True


def _seed_region(orch, region_id, values):
    r = Region(region_id=region_id, anchor={}, center={}, flag=ACTIVE)
    for v in values:
        r.record(f"{region_id}_x", v)
    orch.registry.regions.append(r)
    return r


def test_the_best_region_can_trigger_a_holdout_without_setting_a_record(tmp_path):
    """The measured failure: r0001's record 1.429945 came from 82 draws, so
    r0008 -- better by every robust measure -- never cleared the bar in 9 runs
    and the strongest result in the campaign was never holdout-verified.
    best-of-n falls as n grows; a quantile does not."""
    orch = _make_orchestrator(tmp_path)
    orch.agent1.best_val_bpb = 1.429945
    _seed_region(orch, "r0008", (1.4325, 1.4332, 1.4342, 1.4355, 1.4395,
                                 1.4427, 1.4480, 1.4723, 1.4797))

    hp = {"n_layer": 6}
    orch._set_holdout_threshold(hp)
    # elite_score of that region is 1.4332 -- above the record, so r0008's own
    # good runs now qualify where before none of them could.
    assert hp["holdout_eval_if_below"] > 1.429945
    assert hp["holdout_eval_if_below"] == orch.registry.champion().elite_score()


def test_the_record_still_sets_the_bar_when_no_region_can_be_judged(tmp_path):
    """The champion is an ADDITIONAL bar, not a replacement. Early in a
    campaign no region has enough runs for a quantile, and the record has to
    keep working or the holdout goes back to never firing."""
    orch = _make_orchestrator(tmp_path)
    orch.agent1.best_val_bpb = 1.20
    _seed_region(orch, "r0001", (1.40, 1.41))  # below MIN_RUNS_FOR_ELITE_SCORE

    assert orch.registry.champion() is None
    hp = {"n_layer": 6}
    orch._set_holdout_threshold(hp)
    assert hp["holdout_eval_if_below"] == 1.20


def test_the_bar_is_the_looser_of_the_two(tmp_path):
    """Either claim is worth verifying, so the run only has to clear one."""
    orch = _make_orchestrator(tmp_path)
    orch.agent1.best_val_bpb = 1.20
    _seed_region(orch, "r0001", (1.40, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47))

    hp = {"n_layer": 6}
    orch._set_holdout_threshold(hp)
    # top quartile of 8 is 2 runs -> median of {1.40, 1.41}
    assert orch.registry.champion().elite_score() == pytest.approx(1.405)
    assert hp["holdout_eval_if_below"] == pytest.approx(1.405)


def test_threshold_reaches_the_hyperparams_handed_to_training(tmp_path, monkeypatch):
    """End-to-end through the real sequential loop: the flag has to survive
    into the dict train.py actually receives, not just exist in isolation."""
    orch = _make_orchestrator(tmp_path)
    orch.agent1.best_val_bpb = 1.30

    seen = []
    real_train = orch.agent1.train_model

    def spy(hyperparams, **kwargs):
        seen.append(dict(hyperparams))
        return real_train(hyperparams, **kwargs)

    monkeypatch.setattr(orch.agent1, "train_model", spy)
    orch.run(max_iterations=1)

    assert seen, "training was never invoked"
    assert seen[0]["holdout_eval_if_below"] == 1.30
