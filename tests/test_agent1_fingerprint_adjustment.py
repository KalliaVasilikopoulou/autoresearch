"""Tier 4: agents/agent1_training_specialist.py's fingerprint-driven
architecture nudges (see dev/INNOVATION_PLAN.md, Tier 4). Synthetic-data
tests only -- there's no real fingerprint history yet; real validation is a
separate, later exercise once enough real runs exist.
"""
import sys
from pathlib import Path

import pytest

from agents.agent1_training_specialist import Agent1TrainingSpecialist, ARCH_SAFE_RANGES

REPO_ROOT = Path(__file__).resolve().parent.parent


def _neutral_fingerprint(n_layer=8):
    """A fingerprint that trips none of the 5 rules -- late-layer slices
    stay well above their "~=0" thresholds, entropy stays well above the
    low-entropy threshold, induction_score stays low, attn_distance keeps
    growing (no early saturation)."""
    return {
        "dla": [0.1 + 0.05 * i for i in range(n_layer)],           # growing, no late collapse
        "x0_lambda": [5.0 + i for i in range(n_layer)],             # growing, no late collapse
        "attn_entropy": [2.0] * n_layer,                            # well above ln(4)~=1.386
        "induction_score": 0.1,                                     # well below 0.5
        "attn_distance": [1.0 + i for i in range(n_layer)],         # steadily growing, no early plateau
        "pos_saliency": [0.0] * 16,
    }


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


def _evidence_with(fingerprint):
    return [{"token_fingerprint": fingerprint}]


# ---------------------------------------------------------------------------
# No-op cases
# ---------------------------------------------------------------------------

def test_no_op_when_evidence_is_none(specialist):
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), None)
    assert result == params


def test_no_op_when_evidence_is_empty(specialist):
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), [])
    assert result == params


def test_no_op_when_no_evidence_entry_has_a_fingerprint(specialist):
    params = _base_hyperparams()
    evidence = [{"hyperparameter_importance": {"n_layer": 0.9}}, {"token_fingerprint": {}}]
    result = specialist._fingerprint_adjustment(dict(params), evidence)
    assert result == params


def test_no_op_when_fingerprint_trips_no_rule(specialist):
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), _evidence_with(_neutral_fingerprint()))
    assert result == params


# ---------------------------------------------------------------------------
# The 5 rules, isolated
# ---------------------------------------------------------------------------

def test_rule_dead_late_layers_votes_to_reduce_n_layer(specialist):
    """The rule still lives here; APPLYING it moved to Agent 4, which owns
    architecture. Agent 1 applying it would change a region's architecture
    mid-flight and break the shared-weights pairing (see
    agents/xai_direction.py).  """
    fp = _neutral_fingerprint()
    fp["dla"] = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.001, 0.001]  # late 25% (last 2) ~= 0
    votes = specialist._fingerprint_votes(fp)
    assert sum(votes["n_layer"]) == -1
    # and Agent 1 does not act on it any more
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), _evidence_with(fp))
    assert result["n_layer"] == params["n_layer"]


def test_rule_late_x0_lambda_near_zero_votes_to_increase_n_layer(specialist):
    fp = _neutral_fingerprint()
    fp["x0_lambda"] = [50.0, 40.0, 30.0, 20.0, 10.0, 5.0, 0.001, 0.001]
    votes = specialist._fingerprint_votes(fp)
    assert sum(votes["n_layer"]) == 1
    # and Agent 1 does not act on it any more
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), _evidence_with(fp))
    assert result["n_layer"] == params["n_layer"]


def test_rule_low_entropy_votes_fewer_heads_more_embd(specialist):
    fp = _neutral_fingerprint()
    fp["attn_entropy"] = [0.3] * 8  # well below ln(4)
    votes = specialist._fingerprint_votes(fp)
    assert sum(votes["n_head"]) == -1 and sum(votes["n_embd"]) == 1
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), _evidence_with(fp))
    assert result["n_head"] == params["n_head"]
    assert result["n_embd"] == params["n_embd"]


def test_rule_high_induction_score_votes_to_increase_n_layer(specialist):
    fp = _neutral_fingerprint()
    fp["induction_score"] = 0.8
    votes = specialist._fingerprint_votes(fp)
    assert sum(votes["n_layer"]) == 1
    # and Agent 1 does not act on it any more
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), _evidence_with(fp))
    assert result["n_layer"] == params["n_layer"]


def test_rule_early_saturation_increases_window_s_fraction(specialist):
    fp = _neutral_fingerprint()
    fp["attn_distance"] = [3.0, 7.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # peak reached by index 2
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), _evidence_with(fp))
    assert result["window_s_fraction"] == pytest.approx(params["window_s_fraction"] + 0.1)
    for key in ("n_layer", "n_head", "n_embd"):
        assert result[key] == params[key]


# ---------------------------------------------------------------------------
# Combined / conflicting signals, clamping, most-recent-fingerprint selection
# ---------------------------------------------------------------------------

def test_conflicting_n_layer_votes_sum_to_net_zero(specialist):
    fp = _neutral_fingerprint()
    fp["dla"] = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.001, 0.001]         # votes n_layer -1
    fp["x0_lambda"] = [50.0, 40.0, 30.0, 20.0, 10.0, 5.0, 0.001, 0.001]  # votes n_layer +1
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), _evidence_with(fp))
    assert result["n_layer"] == params["n_layer"]  # -1 and +1 cancel out


def test_delta_clamps_to_search_space_bounds(specialist):
    fp = _neutral_fingerprint()
    fp["induction_score"] = 0.8  # votes n_layer +1
    params = _base_hyperparams()
    _, hi = ARCH_SAFE_RANGES["n_layer"]
    params["n_layer"] = hi  # already at the upper bound
    result = specialist._fingerprint_adjustment(dict(params), _evidence_with(fp))
    assert result["n_layer"] == hi  # clamped, not hi + 1


def test_uses_most_recent_fingerprint_bearing_evidence(specialist):
    old_fp = _neutral_fingerprint()
    old_fp["induction_score"] = 0.8  # would vote n_layer +1 if used
    new_fp = _neutral_fingerprint()  # neutral -- no rule trips
    evidence = [{"token_fingerprint": old_fp}, {"hyperparameter_importance": {}}, {"token_fingerprint": new_fp}]
    params = _base_hyperparams()
    result = specialist._fingerprint_adjustment(dict(params), evidence)
    assert result["n_layer"] == params["n_layer"]  # the most recent (neutral) one wins, not the older trigger


def test_decision_log_records_fingerprint_adjustments(specialist):
    fp = _neutral_fingerprint()
    # attn_distance saturating early is the rule Agent 1 still ACTS on
    # (window_s_fraction changes no weights, so it may vary inside a
    # region). induction_score votes on n_layer, which is Agent 4's now.
    fp["attn_distance"] = [9.0, 9.5, 9.9, 10.0]
    params = _base_hyperparams()
    specialist._fingerprint_adjustment(dict(params), _evidence_with(fp))
    assert len(specialist._last_fingerprint_adjustments) == 1
    entry = specialist._last_fingerprint_adjustments[0]
    assert entry["param"] == "window_s_fraction"
    assert entry["delta"] > 0


# ---------------------------------------------------------------------------
# _build_window_pattern (train.py) -- exec-prefix technique, same as other
# tests in this repo, since train.py executes training eagerly at import time.
# ---------------------------------------------------------------------------

def _load_build_window_pattern():
    train_src = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    boundary = train_src.index("t_start = time.time()")
    ns = {"__name__": "train_defs_only", "__file__": str(REPO_ROOT / "train.py")}
    exec(compile(train_src[:boundary], str(REPO_ROOT / "train.py"), "exec"), ns)
    return ns["_build_window_pattern"]


def test_build_window_pattern_hand_computed():
    build = _load_build_window_pattern()
    # n_s = round(4*0.5) = 2, evenly interleaved: L,S,L,S (Bresenham-style)
    assert build(4, 0.5) == "LSLS"


def test_build_window_pattern_all_short():
    build = _load_build_window_pattern()
    assert build(6, 1.0) == "SSSSSS"


def test_build_window_pattern_all_long():
    build = _load_build_window_pattern()
    assert build(6, 0.0) == "LLLLLL"


def test_build_window_pattern_preserves_s_count():
    build = _load_build_window_pattern()
    for n_layer in (5, 7, 11, 18):
        for frac in (0.1, 0.3, 0.5, 0.75, 0.9):
            pattern = build(n_layer, frac)
            assert len(pattern) == n_layer
            assert set(pattern) <= {"S", "L"}
            assert pattern.count("S") == round(n_layer * frac)


# ---------------------------------------------------------------------------
# Path-independence: the fingerprint nudge must apply regardless of which
# decision path (surrogate vs. evidence/heuristic) produced new_hyperparams.
# ---------------------------------------------------------------------------

def test_fingerprint_adjustment_applies_on_evidence_fallback_path(tmp_path):
    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("agent1:\n  use_llm: false\n  use_surrogate: false\n")
    specialist = Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path))
    specialist.current_hyperparams = _base_hyperparams()

    fp = _neutral_fingerprint()
    # attn_distance saturating early is the rule Agent 1 still ACTS on
    # (window_s_fraction changes no weights, so it may vary inside a
    # region). induction_score votes on n_layer, which is Agent 4's now.
    fp["attn_distance"] = [9.0, 9.5, 9.9, 10.0]
    evidence = [{"hyperparameter_importance": {"n_layer": 0.9}}, {"token_fingerprint": fp}]

    before_n_layer = specialist.current_hyperparams["n_layer"]
    result = specialist.decide_next_hyperparams(
        latest_summary=None, evidence=evidence, iteration=0, latest_val_bpb=None, recent_results=None,
    )
    assert result is not None
    # The evidence path itself may also move n_layer from hyperparameter_importance;
    # what matters here is that the fingerprint's own +1 vote was applied on top --
    # verified directly via the audit trail rather than guessing the evidence path's own delta.
    assert any(e["param"] == "window_s_fraction"
               for e in specialist._last_fingerprint_adjustments)


def test_fingerprint_adjustment_applies_on_surrogate_path(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")

    config_path = tmp_path / "agents_config.yaml"
    config_path.write_text("agent1:\n  use_llm: false\n  use_surrogate: true\n  surrogate_min_observations: 5\n")
    specialist = Agent1TrainingSpecialist(config_path=str(config_path), root_dir=str(tmp_path))
    specialist.current_hyperparams = _base_hyperparams()

    # Seed enough synthetic results.tsv rows for the surrogate to fit and
    # take the "surrogate" path instead of falling back to evidence/heuristic.
    from state.results_logger import log_result
    import random
    random.seed(0)
    for i in range(10):
        hp = dict(_base_hyperparams())
        hp["n_layer"] = random.randint(8, 16)
        hp["matrix_lr"] = random.uniform(0.01, 0.1)
        log_result(f"run_{i:04d}", hp, {"val_bpb": random.uniform(0.9, 1.3), "status": "ok"},
                   results_path=str(tmp_path / "results.tsv"))

    fp = _neutral_fingerprint()
    # attn_distance saturating early is the rule Agent 1 still ACTS on
    # (window_s_fraction changes no weights, so it may vary inside a
    # region). induction_score votes on n_layer, which is Agent 4's now.
    fp["attn_distance"] = [9.0, 9.5, 9.9, 10.0]
    evidence = [{"token_fingerprint": fp}]

    result = specialist.decide_next_hyperparams(
        latest_summary=None, evidence=evidence, iteration=10, latest_val_bpb=1.0, recent_results=None,
    )
    assert result is not None
    assert specialist.last_decision_log["path_taken"] == "surrogate", (
        f"expected the surrogate path to be active for this test to be meaningful, got {specialist.last_decision_log['path_taken']}"
    )
    assert any(e["param"] == "window_s_fraction"
               for e in specialist._last_fingerprint_adjustments)
