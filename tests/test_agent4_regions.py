"""Tests for Agent 4's two remaining jobs: proposing regions and judging them.

Replaces tests/test_agent4_landscape_explorer.py, which tested the
exploration *window* -- one search that Agent 4 temporarily seized every GPU
for. There is no window any more; several regions run at once and Agent 4
only decides where a new one should go and when a running one has earned a
lifecycle change.

The properties worth pinning here are the ones the old design got wrong:
regions proposed by three genuinely different criteria rather than three draws
from one, a region retired by comparison against its live rivals rather than
against an absolute bar, and "paused" staying recoverable.
"""

import json

import pytest

from agents.agent4_landscape_explorer import (
    KEEP,
    ORIGIN_HIGH_EI,
    ORIGIN_RIVAL_OPTIMUM,
    ORIGIN_UNEXPLORED,
    PROPOSAL_CRITERIA,
    Agent4LandscapeExplorer,
)
from state.landscape import LANDSCAPE_DEPS_AVAILABLE
from state.regions import ACTIVE, LOCAL_OPTIMUM, NO_OPTIMUM, PAUSED, RegionRegistry

requires_deps = pytest.mark.skipif(
    not LANDSCAPE_DEPS_AVAILABLE, reason="scikit-learn not installed"
)


def _config(tmp_path, **overrides):
    settings = {
        "min_runs_before_judgement": 5,
        "stuck_runs_pause": 5,
        "stuck_runs_retire": 15,
        "sigma_region": 0.0028,
        "retire_margin_sigma": 3.0,
        "improvement_sigma": 1.0,
        "region_radius": 0.05,
        "merge_radius": 0.025,
        "max_regions": 4,
    }
    settings.update(overrides)
    body = "\n".join(f"  {k}: {v}" for k, v in settings.items())
    path = tmp_path / "agents_config.yaml"
    path.write_text(
        f"agent4:\n  enabled: true\n  llm_mode: statistics\n{body}\n\nllm:\n  backend: none\n",
        encoding="utf-8")
    return path


@pytest.fixture
def agent4(tmp_path):
    return Agent4LandscapeExplorer(
        config_path=str(_config(tmp_path)),
        root_dir=str(tmp_path),
        state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"),
    )


@pytest.fixture
def registry(tmp_path):
    return RegionRegistry(str(tmp_path / "state" / "regions.json"))


def _hyperparams(i=0):
    return {
        "n_layer": 4 + (i % 9), "n_embd": 256 + (i % 6) * 64, "n_head": 4 + (i % 3) * 2,
        "window_s_fraction": 0.2 + (i % 5) * 0.15,
        "embedding_lr": 0.05 * (1 + i % 7), "unembedding_lr": 0.001 * (1 + i % 4),
        "matrix_lr": 0.005 * (1 + i % 6), "scalar_lr": 0.02 * (1 + i % 5),
        "weight_decay": 0.01 * (i % 8), "warmup_ratio": 0.02 * (i % 6),
        "batch_size": 2048 * (1 + i % 4),
    }


def _rows(n=40):
    rows = []
    for i in range(n):
        row = dict(_hyperparams(i))
        row["val_bpb"] = 1.2 + 0.03 * (i % 11)
        row["status"] = "remote_ok"
        rows.append(row)
    return rows


def _fill(registry, region, values, start=0, step=1):
    """Record `values` against a region using GLOBAL run ids.

    The ids matter: the orchestrator issues run_{iteration:04d}, and
    _absorb_history sorts on them to rebuild a true chronology when two
    regions merge. Region-prefixed ids would make every merge look like
    concatenation and a merged region read as stuck -- an artifact of the
    fixture rather than of the code.
    """
    for j, v in enumerate(values):
        registry.assign_run(region.region_id, f"run_{start + j * step:04d}", v)


# === job 1: proposing regions ==============================================


@requires_deps
def test_proposes_the_requested_number_of_regions(agent4, registry):
    opened = agent4.propose_regions(_rows(), registry, n=3, at_run=10)
    assert len(opened) == 3
    assert all(r.flag == ACTIVE for r in opened)
    assert len({r.region_id for r in opened}) == 3


@requires_deps
def test_the_three_criteria_are_genuinely_different(agent4, registry):
    """Ranking purely by surrogate uncertainty concentrates candidates at the
    edges of the sampled space -- three regions chosen that way are three
    versions of one idea. The criteria must rotate."""
    opened = agent4.propose_regions(_rows(), registry, n=3, at_run=10)
    assert {r.origin for r in opened} == set(PROPOSAL_CRITERIA)


@requires_deps
def test_proposed_regions_are_not_on_top_of_each_other(agent4, registry):
    from state.regions import distance

    opened = agent4.propose_regions(_rows(), registry, n=3, at_run=10)
    for i, a in enumerate(opened):
        for b in opened[i + 1:]:
            assert distance(a.anchor, b.anchor, registry.bounds) > agent4.region_radius


@requires_deps
def test_nothing_is_proposed_before_a_surrogate_can_fit(agent4, registry):
    """Returning nothing is the honest answer -- inventing a region to fill a
    GPU slot would put a run somewhere no criterion actually chose."""
    assert agent4.propose_regions(_rows(n=5), registry, n=2, at_run=1) == []
    assert registry.regions == []


@requires_deps
def test_a_ruled_out_area_is_not_re_opened(agent4, registry):
    """A no_optimum flag is a measurement that cost GPU time. Re-proposing
    into it would spend more re-deriving a conclusion already paid for."""
    first = agent4.propose_regions(_rows(), registry, n=1, at_run=10)
    assert len(first) == 1
    first[0].set_flag(NO_OPTIMUM, at_run=20)
    registry.save()

    again = agent4.propose_regions(_rows(), registry, n=1, at_run=30)
    from state.regions import distance
    for region in again:
        assert distance(region.anchor, first[0].anchor, registry.bounds) > agent4.region_radius


@requires_deps
def test_proposing_zero_regions_does_nothing(agent4, registry):
    assert agent4.propose_regions(_rows(), registry, n=0, at_run=10) == []


@requires_deps
def test_a_disabled_agent4_proposes_nothing(tmp_path, registry):
    path = tmp_path / "off.yaml"
    path.write_text("agent4:\n  enabled: false\n\nllm:\n  backend: none\n", encoding="utf-8")
    off = Agent4LandscapeExplorer(config_path=str(path), root_dir=str(tmp_path),
                                  state_dir=str(tmp_path / "state"),
                                  reports_dir=str(tmp_path / "reports"))
    assert off.propose_regions(_rows(), registry, n=2, at_run=10) == []


@requires_deps
def test_opening_regions_writes_a_decision_log(agent4, registry):
    agent4.propose_regions(_rows(), registry, n=2, at_run=10)
    log = json.loads((agent4.decisions_dir / "verdict_0010.json").read_text(encoding="utf-8"))
    assert log["action"] == "opened_regions"
    assert len(log["detail"]["opened"]) == 2


# === job 2: judging regions ================================================


def test_a_young_region_is_not_judged(agent4, registry):
    """Below min_runs_before_judgement there is not enough evidence for any
    verdict, including a negative one."""
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.9, 1.9, 1.9, 1.9])
    assert agent4.judge(r, registry, at_run=10) == KEEP
    assert r.flag == ACTIVE


def test_a_healthy_region_is_kept(agent4, registry):
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.40, 1.36, 1.32, 1.28, 1.24, 1.20])
    assert agent4.judge(r, registry, at_run=10) == KEEP


def test_a_region_stuck_for_a_while_is_paused_not_retired(agent4, registry):
    """"We have not improved here lately" and "there is nothing here" are
    different claims and only the first is supported at 5 runs."""
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.30] + [1.3005] * 5)
    assert agent4.judge(r, registry, at_run=10) == PAUSED
    assert r.flag == PAUSED


def test_a_region_stuck_for_a_long_time_is_retired_as_a_local_optimum(agent4, registry):
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.30] + [1.3005] * 16)
    # Somewhere else to go -- the last live region is never terminally
    # retired, so a verdict needs an alternative to be a verdict at all.
    alt = registry.open_region(_hyperparams(6), at_run=0)
    _fill(registry, alt, [1.29] * 6, start=100)
    assert agent4.judge(r, registry, at_run=10) == LOCAL_OPTIMUM


def test_the_stronger_claim_wins_when_a_region_qualifies_for_both(agent4, registry):
    """A long-stuck region that is also behind the field gets local_optimum,
    the better-supported label, not no_optimum."""
    good = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, good, [1.20] * 6)
    bad = registry.open_region(_hyperparams(5), at_run=0)
    _fill(registry, bad, [1.60] + [1.6005] * 16)
    assert agent4.judge(bad, registry, at_run=10) == LOCAL_OPTIMUM


def test_a_region_far_behind_its_live_rivals_is_retired(agent4, registry):
    good = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, good, [1.20, 1.19, 1.18, 1.17, 1.16])
    bad = registry.open_region(_hyperparams(5), at_run=0)
    _fill(registry, bad, [1.50, 1.49, 1.48, 1.47, 1.46])
    assert agent4.judge(bad, registry, at_run=10) == NO_OPTIMUM
    assert agent4.judge(good, registry, at_run=10) == KEEP


def test_a_region_only_slightly_behind_is_kept(agent4, registry):
    """The retire margin is 3 sigma_region = 0.0084. A region 0.002 behind is
    within noise of its rival and must not be thrown away."""
    a = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, a, [1.200, 1.201, 1.202, 1.203, 1.204])
    b = registry.open_region(_hyperparams(5), at_run=0)
    _fill(registry, b, [1.202, 1.203, 1.204, 1.205, 1.206])
    assert agent4.judge(b, registry, at_run=10) == KEEP


def test_the_only_region_is_never_retired_for_being_behind(agent4, registry):
    """Relative comparison needs something to compare against. With one
    region there is no rival, so 'worse than the field' is not a claim that
    can be made -- and retiring the only region would leave nothing running."""
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.90, 1.89, 1.88, 1.87, 1.86])
    assert agent4.judge(r, registry, at_run=10) == KEEP


def test_a_rival_with_too_little_evidence_does_not_condemn_anyone(agent4, registry):
    """A region with 1 lucky run must not be the bar another is retired
    against."""
    lucky = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, lucky, [1.10])
    established = registry.open_region(_hyperparams(5), at_run=0)
    _fill(registry, established, [1.50, 1.49, 1.48, 1.47, 1.46])
    assert agent4.judge(established, registry, at_run=10) == KEEP


def test_noise_sized_gains_do_not_count_as_improvement(agent4, registry):
    """improvement_sigma exists because Region.runs_since_improvement's
    default of 0.0 counts any decrease -- including pure noise -- as
    progress, so a region would never register as stuck."""
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.3000, 1.2999, 1.2998, 1.2997, 1.2996, 1.2995])
    assert agent4.judge(r, registry, at_run=10) == PAUSED


def test_a_retirement_is_logged_per_region(agent4, registry):
    """Several regions can be judged at the same run number; a filename
    keyed only on the iteration would leave the record showing whichever was
    judged last."""
    a = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, a, [1.20] * 6)
    b = registry.open_region(_hyperparams(5), at_run=0)
    _fill(registry, b, [1.60] * 6)
    agent4.judge(b, registry, at_run=42)
    assert (agent4.decisions_dir / f"verdict_0042_{b.region_id}.json").exists()


# === maintenance ===========================================================


def test_maintain_merges_before_judging(agent4, registry):
    """Judging first would compare a region against what is effectively
    itself, and could retire one arm of a duplicate pair on noise.

    The two regions are given scores a hair apart -- which is what a genuine
    duplicate looks like. The previous version of this test used 1.20 vs 1.60,
    a 143-sigma_region gap, so the loser was retired as legitimately worse and
    the ordering it claimed to check was never actually exercised.

    Two maintain() calls because a merge now needs the overlap to persist (see
    RegionRegistry.merge_overlapping): merging is irreversible, and one
    accidental crossing of two centers should not destroy a region.
    """
    a = registry.open_region(_hyperparams(0), at_run=0)
    # Same ARCHITECTURE, a hair apart in a tunable. Two different
    # architectures can never merge now, however close their tunables sit --
    # they cannot share initial weights, so pooling their histories would pool
    # two different models.
    near = dict(_hyperparams(0))
    near["matrix_lr"] = float(near["matrix_lr"]) * 1.02
    b = registry.open_region(near, at_run=0)
    # Interleaved in real time, as two concurrently-searched regions are.
    # Both improving steadily (so neither reads as stalled) and only ~0.001
    # apart (so neither can be retired as worse than the field, which needs
    # 3 * sigma_region = 0.0084).
    _fill(registry, a, [1.2100, 1.2060, 1.2020, 1.1980, 1.1940, 1.1900], start=0, step=2)
    _fill(registry, b, [1.2110, 1.2070, 1.2030, 1.1990, 1.1950, 1.1910], start=1, step=2)

    first = agent4.maintain(registry, at_run=10)
    assert first["merges"] == [], "one overlap is not yet convergence"
    assert all(r.merged_into is None and r.flag not in (NO_OPTIMUM, LOCAL_OPTIMUM)
               for r in registry.regions), "and neither arm may be thrown away meanwhile"

    out = agent4.maintain(registry, at_run=11)
    assert out["merges"] == [(b.region_id, a.region_id)]
    assert b.region_id not in out["verdicts"]
    assert len(registry.active()) == 1


def test_maintain_judges_every_live_region(agent4, registry):
    a = registry.open_region(_hyperparams(0), at_run=0)
    b = registry.open_region(_hyperparams(6), at_run=0)
    _fill(registry, a, [1.20] * 6)
    _fill(registry, b, [1.21] * 6)
    out = agent4.maintain(registry, at_run=10)
    assert set(out["verdicts"]) == {a.region_id, b.region_id}


# === resuming ==============================================================


def test_the_best_paused_region_can_be_resumed(agent4, registry):
    """Pausing is only worth distinguishing from retiring if something can
    undo it -- this is the "if nothing better exists, continue there" case."""
    worse = registry.open_region(_hyperparams(0), at_run=0)
    better = registry.open_region(_hyperparams(6), at_run=0)
    _fill(registry, worse, [1.50] * 3)
    _fill(registry, better, [1.20] * 3)
    worse.set_flag(PAUSED, 5)
    better.set_flag(PAUSED, 5)

    resumed = agent4.resume_best_paused(registry, at_run=10)
    assert resumed.region_id == better.region_id
    assert resumed.flag == ACTIVE


def test_a_retired_region_is_never_resumed(agent4, registry):
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.20] * 3)
    r.set_flag(LOCAL_OPTIMUM, 5)
    assert agent4.resume_best_paused(registry, at_run=10) is None


def test_resuming_with_nothing_paused_returns_none(agent4, registry):
    registry.open_region(_hyperparams(0), at_run=0)
    assert agent4.resume_best_paused(registry, at_run=10) is None


# === config ================================================================


def test_code_defaults_match_the_shipped_config():
    """A caller without an agent4: block must not quietly get an
    uncalibrated set of thresholds back."""
    import yaml

    shipped = yaml.safe_load(open("agents_config.yaml", encoding="utf-8"))["agent4"]
    defaults = Agent4LandscapeExplorer(config_path="does_not_exist.yaml")
    for key, attr in [
        ("region_radius", "region_radius"), ("merge_radius", "merge_radius"),
        ("sigma_region", "sigma_region"), ("retire_margin_sigma", "retire_margin_sigma"),
        ("improvement_sigma", "improvement_sigma"),
        ("min_runs_before_judgement", "min_runs_before_judgement"),
        ("stuck_runs_pause", "stuck_runs_pause"), ("stuck_runs_retire", "stuck_runs_retire"),
        ("max_regions", "max_regions"),
    ]:
        assert getattr(defaults, attr) == shipped[key], key


def test_a_region_still_cold_starting_is_exempt_from_the_stuck_rules(agent4, registry):
    """Sobol cold-start points are a space-filling sample, not a descent --
    consecutive draws have no reason to improve on each other, so
    runs_since_improvement measures nothing there. Found in a dry run: the
    bootstrap region churned paused -> resumed -> paused every wave while
    doing exactly what it was supposed to."""
    r = registry.open_region(_hyperparams(0), at_run=0, origin="bootstrap")
    _fill(registry, r, [1.30] + [1.3005] * 8)
    assert agent4.judge(r, registry, at_run=10) == KEEP
    assert r.flag == ACTIVE


def test_the_exemption_ends_once_the_cold_start_does(agent4, registry):
    """Past cold_start_n the bootstrap region is judged like any other -- but
    only on the runs AFTER its Sobol prefix."""
    boot = registry.open_region(_hyperparams(0), at_run=0, origin="bootstrap")
    _fill(registry, boot, [1.30] * agent4.cold_start_n + [1.3005] * 17)
    # A rival of comparable quality: close enough that the worse-than-field
    # rule stays out of the way, so this isolates the stuck rule.
    alt = registry.open_region(_hyperparams(6), at_run=0)
    _fill(registry, alt, [1.2995] * 6, start=100)
    assert agent4.judge(boot, registry, at_run=10) == LOCAL_OPTIMUM


def test_a_real_region_is_never_exempt_just_for_being_young(agent4, registry):
    """The exemption is tied to the bootstrap origin, not to a run count --
    every other region is opened after a surrogate already fits."""
    r = registry.open_region(_hyperparams(0), at_run=0, origin=ORIGIN_HIGH_EI)
    _fill(registry, r, [1.30] + [1.3005] * 5)
    assert agent4.judge(r, registry, at_run=10) == PAUSED,         "pausing the last region is fine -- it is recoverable"


# === the last region standing ==============================================


def test_the_only_live_region_is_never_retired(agent4, registry):
    """Terminally retiring it leaves nowhere to search, and the orchestrator's
    answer to "no live regions" is to open a fresh bootstrap region and re-run
    the whole Sobol cold start -- which reaches the same state and is retired
    again, forever. Seen on the first real 20-run campaign: r0001 retired as
    local_optimum at run 16, r0002 opened to cold-start from scratch.

    Pausing it is still allowed: that is recoverable and the orchestrator
    resumes the best paused region when it has nowhere better to look.
    """
    r = registry.open_region(_hyperparams(0), at_run=0, origin=ORIGIN_HIGH_EI)
    _fill(registry, r, [1.30] + [1.3005] * 20)
    verdict = agent4.judge(r, registry, at_run=30)
    assert verdict not in (LOCAL_OPTIMUM, NO_OPTIMUM)
    assert r.flag != LOCAL_OPTIMUM


def test_a_region_is_retired_once_there_is_somewhere_else_to_go(agent4, registry):
    """The same region, same evidence -- retired only because an alternative
    now exists. A verdict is a comparison."""
    doomed = registry.open_region(_hyperparams(0), at_run=0, origin=ORIGIN_HIGH_EI)
    _fill(registry, doomed, [1.30] + [1.3005] * 20)
    alt = registry.open_region(_hyperparams(6), at_run=0, origin=ORIGIN_UNEXPLORED)
    _fill(registry, alt, [1.20] * 6, start=100)
    assert agent4.judge(doomed, registry, at_run=30) == LOCAL_OPTIMUM


# === cold-start runs are not evidence of being stuck =======================


def test_sobol_draws_do_not_count_toward_being_stuck(agent4, registry):
    """Deferring the check until the cold start ended was not enough on its
    own: the accumulated no-improvement history was still there, so the first
    judgement after the exemption lifted saw runs_since_improvement = 15 and
    retired the region. Exactly the shape the real campaign produced."""
    boot = registry.open_region(_hyperparams(0), at_run=0, origin="bootstrap")
    _fill(registry, boot, [1.248] + [1.30 + 0.01 * (i % 5) for i in range(15)])
    # Comparable rival, so only the stuck rule is under test here.
    alt = registry.open_region(_hyperparams(6), at_run=0)
    _fill(registry, alt, [1.30] * 6, start=100)

    assert boot.n_measured > agent4.cold_start_n, "exemption has lifted"
    assert agent4.judge(boot, registry, at_run=20) == KEEP


def test_a_bootstrap_region_can_still_go_stuck_on_its_own_choices(agent4, registry):
    """Only the cold-start prefix is exempt -- runs the region actually chose
    still count."""
    boot = registry.open_region(_hyperparams(0), at_run=0, origin="bootstrap")
    cold = [1.248] + [1.30 + 0.01 * (i % 5) for i in range(agent4.cold_start_n - 1)]
    _fill(registry, boot, cold + [1.3005] * 8)
    alt = registry.open_region(_hyperparams(6), at_run=0)
    _fill(registry, alt, [1.30] * 6, start=100)
    assert agent4.judge(boot, registry, at_run=30) == PAUSED




def test_the_shipped_config_has_no_settings_the_code_ignores():
    """A setting that reads as live and does nothing is worse than no
    setting, because it gets tuned. agent4.min_regions and
    agent4.bad_tolerance were both in this state and are now deleted."""
    import yaml

    shipped = yaml.safe_load(open("agents_config.yaml", encoding="utf-8"))
    src = open("agents/agent4_landscape_explorer.py", encoding="utf-8").read()
    unread = [k for k in shipped["agent4"] if f'"{k}"' not in src]
    assert unread == [], f"agent4 config keys nothing reads: {unread}"

    orch_src = open("agents/orchestrator.py", encoding="utf-8").read()
    unread = [k for k in shipped["orchestrator"] if f'"{k}"' not in orch_src]
    assert unread == [], f"orchestrator config keys nothing reads: {unread}"
