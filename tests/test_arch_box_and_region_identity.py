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


def test_the_box_reaches_past_where_size_stops_paying():
    """The ladder's top rung is n_layer=21 / n_embd=960 = 232M non-embedding
    params, and at TOKEN_BUDGET=4.19M its final step (-0.0036) has fallen into
    the noise -- so size stops paying somewhere around 138-232M. The box has to
    reach past that, or the search cannot see the flattening for itself; it
    does not need to reach much further, because room the search cannot profit
    from still costs wall clock on every oversized model it tries."""
    box_params_m = 12 * MAX_N_LAYER * MAX_N_EMBD ** 2 / 1e6
    assert box_params_m > 232, "the box cannot even reach the ladder's top rung"
    assert box_params_m < 4 * 232, "far more room than the measurement justifies"


def test_the_box_stays_inside_what_train_py_will_run():
    """train.py clamps n_layer to 48, n_head to 64 and n_embd to 8192. A box
    outside that would propose configurations train.py silently rewrites, and
    then "requested" and "actually used" diverge in results.tsv."""
    assert ARCH_SAFE_RANGES["n_layer"][1] <= 48
    assert ARCH_SAFE_RANGES["n_head"][1] <= 64
    assert ARCH_SAFE_RANGES["n_embd"][1] <= 8192


def test_the_worst_case_model_still_fits_the_wall_clock():
    """Sized against measurement, not taste. A run that trips the cap is
    excluded as incomplete, so it costs a GPU slot and returns nothing.

    The fit is `81.0s + 0.565s per M non-embedding param`, least squares on the
    top three rungs of the 4.19M ladder (68.8M/118.8s, 138.2M/161.1s,
    232.2M/211.5s) -- the top, because that is the end the worst case lives at
    and the curve is visibly sub-linear below it. BOTH NUMBERS ARE
    BUDGET-SPECIFIC: training time scales with TOKEN_BUDGET, so re-derive them
    from a fresh size_sweep whenever the budget moves. MAX_TRAIN_SECONDS is
    read live for the same reason.
    """
    from prepare import MAX_TRAIN_SECONDS

    params_m = 12 * MAX_N_LAYER * MAX_N_EMBD ** 2 / 1e6
    projected_seconds = 81.0 + 0.565 * params_m
    assert projected_seconds < 0.6 * MAX_TRAIN_SECONDS


def test_every_path_follows_the_ceiling_wherever_it_is_set(agent, monkeypatch):
    """THE REGRESSION THE EIGHT LITERALS WOULD HAVE CAUSED, tested without
    naming a value -- because the value moves. It was raised to 28/1280 and put
    back to 24/1024 within a day, once the size ladder was re-run at a smaller
    token budget: how big a model is worth building depends on how much you
    train it, so this ceiling tracks the budget rather than the search.

    What must hold at every setting is that the ceiling has ONE definition.
    Raise it here and the non-surrogate paths have to follow; under the old
    hardcoded literals they could not, however strong the evidence.
    """
    ceiling_now = a1mod.MAX_N_EMBD
    monkeypatch.setattr(a1mod, "MAX_N_EMBD", ceiling_now + 256)
    monkeypatch.setattr(a1mod, "MAX_N_LAYER", a1mod.MAX_N_LAYER + 4)

    agent.current_hyperparams = hp(n_layer=MAX_N_LAYER, n_embd=ceiling_now, n_head=8)
    out = agent._evidence_adjustment(
        latest_summary=None,
        evidence=[{"report_id": "r", "hyperparameter_importance":
                   {"n_layer": 1.0, "n_embd": 1.0}}],
        stuck_signal=False, iteration=3)

    assert out["n_layer"] > MAX_N_LAYER
    assert out["n_embd"] > ceiling_now


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
    """_radical_change picks n_embd from a fixed list. If that list stops below
    the ceiling, a restart can never land in the top of the space -- so the
    list has to track the ceiling too, not just the clamps."""
    monkeypatch.setattr(a1mod, "MAX_N_EMBD", MAX_N_EMBD + 256)
    monkeypatch.setattr(a1mod.random, "choice", lambda seq: seq[-1])

    out = agent._radical_change(hp())
    assert out["n_embd"] == MAX_N_EMBD + 256


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


# --- run ids must never be reissued ------------------------------------------


def _write_results(path, run_ids):
    from state.results_logger import log_result

    for rid in run_ids:
        log_result(rid, hp(), {"val_bpb": 1.3, "status": "remote_ok"},
                   results_path=str(path))


def test_a_resumed_campaign_continues_the_run_numbering(tmp_path):
    """run_id is f"run_{iteration:04d}" and `iteration` used to start at 0
    every launch. Against an existing results.tsv that reissues ids already
    taken -- and load_results de-duplicates by run_id, so the collision does
    not error, it quietly drops one of the two runs."""
    from agents.orchestrator import Orchestrator

    results = tmp_path / "results.tsv"
    _write_results(results, [f"run_{i:04d}" for i in range(32)])
    orch = Orchestrator.__new__(Orchestrator)
    orch.results_path = results
    assert orch._next_run_index() == 32


def test_experiment_run_ids_do_not_move_the_campaign_numbering(tmp_path):
    """`geom_...` and `size_...` rows live in their own results files, but
    anything unexpected here must not be able to push the numbering somewhere
    strange."""
    from agents.orchestrator import Orchestrator

    results = tmp_path / "results.tsv"
    _write_results(results, ["run_0005", "size_h6", "geom_anchor_s42"])
    orch = Orchestrator.__new__(Orchestrator)
    orch.results_path = results
    assert orch._next_run_index() == 6


def test_a_fresh_campaign_still_starts_at_zero(tmp_path):
    from agents.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch.results_path = tmp_path / "results.tsv"   # does not exist yet
    assert orch._next_run_index() == 0


def test_the_numbering_reads_results_tsv_not_the_session_metadata(tmp_path):
    """The near-miss worth pinning. state_manager's metadata.json is
    per-session bookkeeping and starts empty in a fresh process, so reading it
    reports "0 recorded runs" against a results.tsv holding 32 -- and reissues
    every id. results.tsv is where log_result appends, so it is the only place
    two runs can actually collide."""
    from agents.orchestrator import Orchestrator

    results = tmp_path / "results.tsv"
    _write_results(results, [f"run_{i:04d}" for i in range(32)])

    orch = Orchestrator.__new__(Orchestrator)
    orch.results_path = results
    orch.state_mgr = type("S", (), {"get_all_results": staticmethod(lambda: [])})()
    assert orch._next_run_index() == 32


# --- the shared cluster's one-GPU policy -------------------------------------


def test_gpu_discovery_never_offers_more_than_the_policy_allows(monkeypatch):
    """The DGX allows one GPU per user. Enforced in discover_available_gpus
    because that is the single place every dispatcher asks what it may use --
    the orchestrator's wave planner AND each of the scripts/ experiments,
    which do not read agents_config.yaml at all."""
    from agents import remote_runner

    class _FakeStdout:
        @staticmethod
        def read():
            # seven idle GPUs, all comfortably free
            return b"\n".join(f"{i}, 0, 40960, 0".encode() for i in range(7))

    class _FakeClient:
        @staticmethod
        def exec_command(*a, **k):
            return None, _FakeStdout(), None

        @staticmethod
        def close():
            pass

    monkeypatch.setattr(remote_runner, "_PARAMIKO_AVAILABLE", True)
    monkeypatch.setattr(remote_runner, "_load_cfg",
                        lambda: {"host": "h", "user": "u", "repo": "/r", "password": "p"})

    available = remote_runner.discover_available_gpus(client=_FakeClient())
    assert len(available) <= remote_runner.MAX_CONCURRENT_GPUS
    assert remote_runner.MAX_CONCURRENT_GPUS == 1


def test_the_configured_wave_size_does_not_exceed_the_policy():
    """agents_config.yaml is the polite half of the same limit; it must not
    disagree with the binding one."""
    import yaml

    from agents import remote_runner

    cfg = yaml.safe_load(open("agents_config.yaml", encoding="utf-8"))
    configured = int(cfg["orchestrator"]["max_parallel_runs"])
    assert configured <= remote_runner.MAX_CONCURRENT_GPUS
