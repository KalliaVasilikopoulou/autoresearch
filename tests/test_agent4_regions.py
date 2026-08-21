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
from state.regions import (
    ACTIVE, LOCAL_OPTIMUM, NO_OPTIMUM, PAUSED, TIED_FOR_BEST, RegionRegistry,
)

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


def _readable_signal(agent4, monkeypatch, a_within=0.010):
    """Pin the region-local noise so the saturation test has a real answer.
    The fixture has no measurement report on disk, so _a_within would return
    None and every saturation branch would be skipped."""
    monkeypatch.setattr(agent4, "_a_within", lambda region: a_within)


def test_a_stuck_region_is_kept_while_its_differences_are_still_measurable(
        agent4, registry, monkeypatch):
    """The measurement outranks the counter. `saturated is False` says the
    spread between configurations here is LARGER than the noise, so more runs
    can still rank them -- which directly contradicts what the stagnation
    pause asserts.

    Measured instance: r0008 was paused at exactly stuck_runs_pause=5 with a
    real signal 1.43x its own noise, while r0001 -- saturated at 0.98x -- ran
    82 runs under the bootstrap cold-start exemption."""
    _readable_signal(agent4, monkeypatch)
    r = registry.open_region(_hyperparams(0), at_run=0)
    # best is the first; the next five do not improve on it, but they are
    # spread far wider than the 0.010 noise floor
    _fill(registry, r, [1.40, 1.46, 1.44, 1.47, 1.43, 1.45])

    assert r.runs_since_improvement(agent4.improvement_margin) >= agent4.stuck_runs_pause
    assert r.is_saturated(0.010) is False
    assert agent4.judge(r, registry, at_run=10) == KEEP
    assert r.flag == ACTIVE


def test_a_stuck_region_is_still_paused_once_its_signal_is_gone(
        agent4, registry, monkeypatch):
    """The protection is conditional on the finding, not on being stuck."""
    _readable_signal(agent4, monkeypatch)
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.30] + [1.3005] * 5)  # spread far below the noise floor

    assert r.is_saturated(0.010) is True
    assert agent4.judge(r, registry, at_run=10) == PAUSED


def test_readable_signal_buys_runs_but_does_not_buy_forever(
        agent4, registry, monkeypatch):
    """Bounded at stuck_runs_retire: 15 runs of no improvement instead of 5,
    not an open-ended stay. Without a bound a lone unsaturated region would
    hold the only GPU indefinitely, since pausing is what frees the slot for
    Agent 4 to open somewhere new."""
    _readable_signal(agent4, monkeypatch)
    r = registry.open_region(_hyperparams(0), at_run=0)
    values = [1.40] + [1.43 + 0.01 * (i % 4) for i in range(agent4.stuck_runs_retire)]
    _fill(registry, r, values)

    assert r.runs_since_improvement(agent4.improvement_margin) >= agent4.stuck_runs_retire
    assert r.is_saturated(0.010) is False  # still readable, and still stopped
    assert agent4.judge(r, registry, at_run=10) == PAUSED


def _pin_local_noise(agent4, monkeypatch, value):
    """The fixture has no results.tsv, so the real _local_noise cannot compute
    distances between a region's runs and returns None. Pin it to a known
    wobble so the verdict logic is what is under test."""
    monkeypatch.setattr(agent4, "_local_noise", lambda region: value)


def test_improvement_is_measured_against_the_region_own_wobble(
        agent4, registry, monkeypatch):
    """A good region is a TIGHT one, so a campaign-wide margin ends up wider
    than the whole region and nothing in it can ever count as progress.

    Measured instance: r0010's six runs spanned 0.0058 with a best step of
    0.0030, against a campaign margin of 0.010579. No run counted, the counter
    never reset, and it paused at 6 runs holding the best score in the
    campaign -- two short of MIN_RUNS_FOR_ELITE_SCORE, so it was stopped before
    it could qualify to be ranked."""
    _pin_local_noise(agent4, monkeypatch, 0.0008)
    r = registry.open_region(_hyperparams(0), at_run=0)
    # Steps of ~0.0012, each bigger than this region's wobble (0.0008). The
    # running best only advances on an improvement, so against the campaign
    # margin (0.0028) the best stays at 1.4324 and nothing ever clears
    # 1.4296 -- the descent is real but invisible at that scale.
    _fill(registry, r, [1.4324, 1.4310, 1.4298, 1.4300, 1.4305, 1.4310])

    campaign = agent4.improvement_margin
    own = agent4.improvement_margin_for(r)
    assert own < campaign

    assert r.runs_since_improvement(campaign) >= agent4.stuck_runs_pause
    assert r.runs_since_improvement(own) < agent4.stuck_runs_pause
    assert agent4.judge(r, registry, at_run=10) == KEEP
    assert r.flag == ACTIVE


def test_the_margin_falls_back_to_the_campaign_wide_one(agent4, registry):
    """Where a region cannot measure its own noise, behaviour is unchanged.
    Same discipline as the saturation test: no measurement, no new rule."""
    empty = registry.open_region(_hyperparams(0), at_run=0)
    assert agent4._local_noise(empty) is None
    assert agent4.improvement_margin_for(empty) == agent4.improvement_margin
    assert agent4.improvement_margin_for(None) == agent4.improvement_margin


def test_a_local_margin_does_not_keep_an_exhausted_region_alive(
        agent4, registry, monkeypatch):
    """The point is a FAIR bar, not a lower one. A region whose runs are flat
    still registers as stuck against its own wobble -- otherwise this would
    just be a slower way of never stopping."""
    _pin_local_noise(agent4, monkeypatch, 0.0008)
    r = registry.open_region(_hyperparams(0), at_run=0)
    _fill(registry, r, [1.30] + [1.3005] * 6)

    assert r.runs_since_improvement(agent4.improvement_margin_for(r)) >= agent4.stuck_runs_pause
    assert agent4.judge(r, registry, at_run=10) == PAUSED


def _paused(registry, region_id, values):
    from state.regions import Region, PAUSED as _P
    r = Region(region_id=region_id, anchor=dict(_hyperparams(0)),
               center=dict(_hyperparams(0)), flag=_P)
    for v in values:
        r.record(f"{region_id}_x", v)
    registry.regions.append(r)
    return r


def test_a_paused_region_better_than_a_typical_new_one_is_reclaimed(agent4, registry):
    """Campaign 11, measured: r0010 sat at elite 1.428681 -- the best ground in
    the campaign -- paused and unrankable on 6 runs, while 30 runs opened four
    NEW regions scoring 1.4479, 1.4815, 1.4811 and 1.5548. The search had the
    answer and spent the budget re-finding it."""
    _paused(registry, "r0010", [1.4287] * 8)          # the good one
    _paused(registry, "r0015", [1.4815] * 8)          # typical
    _paused(registry, "r0013", [1.5548] * 8)          # bad

    typical = registry.typical_new_region_elite()
    assert typical == pytest.approx(1.4815)           # median of the three

    got = agent4.reclaim_better_paused(registry, at_run=50)
    assert got is not None and got.region_id == "r0010"
    assert got.flag == ACTIVE
    assert registry.resumes_used("r0010") == 1


def test_a_paused_region_no_better_than_average_is_left_alone(agent4, registry):
    """Otherwise this just reinstates 'always go back', which is the behaviour
    that livelocked 60 runs."""
    _paused(registry, "r0013", [1.5548] * 8)
    _paused(registry, "r0015", [1.4815] * 8)
    _paused(registry, "r0014", [1.4479] * 8)
    # the median IS one of them, so only something better than the median wins
    assert agent4.reclaim_better_paused(registry, at_run=50).region_id == "r0014"
    # r0014 is now active; of what is left, nothing beats the median
    assert agent4.reclaim_better_paused(registry, at_run=51) is None


def test_the_resume_budget_is_what_keeps_the_livelock_dead(agent4, registry):
    """A region that pauses again immediately costs at most max_resumes extra
    iterations, not sixty."""
    from state.regions import PAUSED as _P
    good = _paused(registry, "r0010", [1.4287] * 8)
    _paused(registry, "r0015", [1.4815] * 8)
    _paused(registry, "r0013", [1.5548] * 8)

    for i in range(agent4.max_resumes):
        got = agent4.reclaim_better_paused(registry, at_run=50 + i)
        assert got is good, f"resume {i} should have been allowed"
        good.set_flag(_P, at_run=50 + i)   # it pauses again immediately

    assert registry.resumes_used("r0010") == agent4.max_resumes
    assert agent4.reclaim_better_paused(registry, at_run=99) is None
    assert good.flag == _P


def test_nothing_is_reclaimed_before_any_region_can_be_judged(agent4, registry):
    _paused(registry, "r0010", [1.20])   # below MIN_RUNS_FOR_ELITE_SCORE
    assert registry.typical_new_region_elite() is None
    assert agent4.reclaim_better_paused(registry, at_run=50) is None


def test_the_resume_budget_survives_a_restart(tmp_path, agent4):
    """Persisted, or a campaign that restarts mid-livelock silently starts the
    count over -- the same reasoning as overlap_streaks."""
    path = str(tmp_path / "state" / "regions.json")
    reg = RegionRegistry(path)
    _paused(reg, "r0010", [1.4287] * 8)
    _paused(reg, "r0015", [1.4815] * 8)
    _paused(reg, "r0013", [1.5548] * 8)
    agent4.reclaim_better_paused(reg, at_run=50)

    assert RegionRegistry(path).resumes_used("r0010") == 1


def _champion(registry, values=(1.40,) * 8):
    """A rankable region for the screen to measure against."""
    return _paused_active(registry, "r0001", values)


def _paused_active(registry, region_id, values):
    from state.regions import Region
    r = Region(region_id=region_id, anchor=dict(_hyperparams(0)),
               center=dict(_hyperparams(0)), flag=ACTIVE)
    for v in values:
        r.record(f"{region_id}_x", v)
    registry.regions.append(r)
    return r


def test_a_hopeless_opening_run_is_rejected_immediately(agent4, registry):
    """20 of 74 runs went to regions whose FIRST run already said no: r0007
    opened at 1.6027, r0012 at 1.6002, r0013 at 1.5632, r0016 at 1.4961, all
    against a champion elite of ~1.4325."""
    champ = _champion(registry)
    newborn = _paused_active(registry, "r0009", [1.60])

    bar = champ.elite_score() + agent4.reject_margin_sigma * agent4.sigma_region
    assert newborn.val_bpbs[0] > bar
    assert agent4.judge(newborn, registry, at_run=10) == NO_OPTIMUM
    assert newborn.flag == NO_OPTIMUM


def test_a_merely_mediocre_opening_run_survives(agent4, registry):
    """THE INVERSION THAT MATTERS. r0008 opened 4th-best of ten (1.4580) and
    finished 2nd-best (1.4325). "Discard the worst" would have thrown it away;
    the margin is wide precisely so it does not."""
    champ = _champion(registry)
    bar = champ.elite_score() + agent4.reject_margin_sigma * agent4.sigma_region
    survivor = _paused_active(registry, "r0009", [bar - 0.0005])

    assert agent4.judge(survivor, registry, at_run=10) == KEEP
    assert survivor.flag == ACTIVE


def test_the_screen_needs_a_champion_to_measure_against(agent4, registry):
    """No ranked region yet means no bar, and a rule that throws work away must
    not fire on a guess."""
    newborn = _paused_active(registry, "r0009", [1.90])
    assert registry.champion() is None
    assert agent4.judge(newborn, registry, at_run=10) == KEEP


def test_the_screen_never_rejects_the_champion_itself(agent4, registry):
    champ = _champion(registry)
    assert agent4._reject_on_first_run(champ, registry, at_run=10) is None


def test_a_region_inside_the_resolvable_gap_is_marked_tied(agent4, registry, monkeypatch):
    """The top three regions sat 0.0027 and 0.0065 apart against a 0.0093
    resolvable gap -- the order printed between them was partly a coin flip,
    and settling r0017 vs r0010 would have cost ~12 repeats of EACH."""
    monkeypatch.setattr(agent4, "_resolvable_gap", lambda: 0.0093)
    _champion(registry, values=(1.40,) * 8)
    rival = _paused_active(registry, "r0002", (1.4050,) * 8)  # 0.005 away

    assert agent4.judge(rival, registry, at_run=20) == TIED_FOR_BEST
    assert rival.flag == TIED_FOR_BEST
    assert not rival.schedulable


def test_the_champion_is_never_tied_with_itself(agent4, registry, monkeypatch):
    """A region is trivially indistinguishable from itself, so without a guard
    the champion is set aside the moment it becomes rankable and the campaign
    stops exploiting its best region for good. Observed on the first iteration
    of campaign 12: r0017 -> tied_for_best, champion r0017, difference 0.0."""
    monkeypatch.setattr(agent4, "_resolvable_gap", lambda: 0.0093)
    champ = _champion(registry, values=(1.40,) * 8)
    assert registry.champion() is champ

    assert agent4._tied_for_best(champ, registry, at_run=20) is None
    # It then falls through to the ORDINARY rules, which is the whole point:
    # the champion is governed like any region with no rival above it. This
    # fixture is eight identical values, so stagnation pauses it -- correctly.
    # What must never happen is tied_for_best.
    assert agent4.judge(champ, registry, at_run=20) != TIED_FOR_BEST
    assert champ.flag == PAUSED


def test_a_runner_up_inside_the_gap_is_still_tied(agent4, registry, monkeypatch):
    """The guard is only about self-comparison -- everything else still ties."""
    monkeypatch.setattr(agent4, "_resolvable_gap", lambda: 0.0093)
    _champion(registry, values=(1.40,) * 8)
    rival = _paused_active(registry, "r0002", (1.4050,) * 8)
    assert agent4.judge(rival, registry, at_run=20) == TIED_FOR_BEST


def test_a_separable_region_is_not_marked_tied(agent4, registry, monkeypatch):
    monkeypatch.setattr(agent4, "_resolvable_gap", lambda: 0.0093)
    _champion(registry, values=(1.40,) * 8)
    far = _paused_active(registry, "r0002", (1.50,) * 8)  # 0.10 away

    assert agent4.judge(far, registry, at_run=20) != TIED_FOR_BEST


def test_nothing_is_tied_before_it_can_be_ranked(agent4, registry, monkeypatch):
    """Below MIN_RUNS_FOR_ELITE_SCORE the score is one lucky draw, and 'we
    cannot tell these apart' would then be a statement about sample size."""
    monkeypatch.setattr(agent4, "_resolvable_gap", lambda: 0.0093)
    _champion(registry, values=(1.40,) * 8)
    young = _paused_active(registry, "r0002", (1.4050,) * 5)

    assert agent4.judge(young, registry, at_run=20) != TIED_FOR_BEST


def test_the_tie_rule_does_not_fire_on_an_unmeasured_gap(agent4, registry, monkeypatch):
    monkeypatch.setattr(agent4, "_resolvable_gap", lambda: None)
    _champion(registry, values=(1.40,) * 8)
    rival = _paused_active(registry, "r0002", (1.4050,) * 8)
    assert agent4.judge(rival, registry, at_run=20) != TIED_FOR_BEST


def test_a_tied_region_is_revived_when_nothing_else_is_left(agent4, registry, monkeypatch):
    """Set aside, not discarded. Once there is nowhere else to look, breaking
    the tie is the best remaining use of a run."""
    monkeypatch.setattr(agent4, "_resolvable_gap", lambda: 0.0093)
    a = _paused_active(registry, "r0002", (1.4050,) * 8)
    b = _paused_active(registry, "r0003", (1.4020,) * 8)
    for r in (a, b):
        r.set_flag(TIED_FOR_BEST, at_run=20)

    revived = agent4.revive_tied(registry, at_run=30)
    assert revived is b            # the better of the two
    assert revived.flag == ACTIVE
    assert registry.resumes_used("r0003") == 1


def test_reviving_a_tie_is_bounded(agent4, registry):
    """Same budget as reclaim_better_paused, for the same reason: a region
    re-tied immediately must cost iterations, not a livelock."""
    r = _paused_active(registry, "r0002", (1.4050,) * 8)
    for i in range(agent4.max_resumes):
        r.set_flag(TIED_FOR_BEST, at_run=20 + i)
        assert agent4.revive_tied(registry, at_run=30 + i) is r
    r.set_flag(TIED_FOR_BEST, at_run=99)
    assert agent4.revive_tied(registry, at_run=100) is None


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
        # sigma_region is deliberately absent: it is no longer a default at all
        # but B(r), read live from the geometry report at the fence radius,
        # because it moves with both the radius and the token budget. Its
        # fallback is checked separately below.
        ("retire_margin_sigma", "retire_margin_sigma"),
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


# --- a region too small to tune its way back is not worth opening -----------


def _stamp(state_dir, sweep_rungs, b_by_radius):
    """Write the two measurement reports the floor is derived from, stamped
    with the budget in force -- an unstamped or foreign-budget report is
    refused, and the floor then does not exist."""
    import json

    from prepare import TOKEN_BUDGET

    (state_dir / "size_sweep.json").write_text(json.dumps({
        "token_budget": int(TOKEN_BUDGET),
        "rungs": [{"params": p, "val_bpb": v} for p, v in sweep_rungs],
    }), encoding="utf-8")
    (state_dir / "region_geometry.json").write_text(json.dumps({
        "token_budget": int(TOKEN_BUDGET),
        "b_by_radius": {k: {"b_observed": v} for k, v in b_by_radius.items()},
    }), encoding="utf-8")


#: The real 4.19M ladder and geometry.
RUNGS = [(1.23e6, 1.796245), (8.60e6, 1.744126), (30.41e6, 1.724416),
         (68.81e6, 1.715182), (138.24e6, 1.708894), (232.24e6, 1.705331)]
B_BY_RADIUS = {"0.02": 0.010579, "0.05": 0.045045, "0.10": 0.078920}


def test_the_floor_is_the_smallest_size_tuning_can_still_rescue(agent4, tmp_path):
    """Derived from two measurements rather than chosen: the ladder gives each
    size's penalty, B(r) gives how far tuning inside a fence can move val_bpb.
    A penalty larger than B(r) is one the region can never search its way out
    of. At 4.19M that lands on 68.8M -- 0.0099 penalty against B(0.02)=0.0106
    -- while 30.4M, at 0.0191, is beyond rescue."""
    (tmp_path / "state").mkdir(exist_ok=True)
    _stamp(tmp_path / "state", RUNGS, B_BY_RADIUS)
    agent4.region_radius = 0.02

    assert agent4.minimum_region_size() == pytest.approx(68.81e6)


def test_the_floor_reads_b_at_the_fence_radius_not_the_largest_measured(agent4, tmp_path):
    """The geometry experiment characterises several radii, but a region's
    search is confined to region_radius. Taking the largest B instead reads
    B(0.10)=0.0789 against a 0.02 fence and puts the floor 8x too low, waving
    through the very regions this refuses."""
    (tmp_path / "state").mkdir(exist_ok=True)
    _stamp(tmp_path / "state", RUNGS, B_BY_RADIUS)

    agent4.region_radius = 0.02
    tight = agent4.minimum_region_size()
    agent4.region_radius = 0.10
    loose = agent4.minimum_region_size()

    assert tight > loose, "a tighter fence recovers less, so it must refuse more"
    assert tight == pytest.approx(68.81e6)


def test_a_hopeless_candidate_is_skipped_and_a_viable_one_is_not(agent4, tmp_path):
    (tmp_path / "state").mkdir(exist_ok=True)
    _stamp(tmp_path / "state", RUNGS, B_BY_RADIUS)
    agent4.region_radius = 0.02

    assert agent4._too_small_to_recover({"n_layer": 4, "n_embd": 320})     # 4.9M
    assert not agent4._too_small_to_recover({"n_layer": 15, "n_embd": 828})  # 123M


def test_without_the_measurements_nothing_is_refused(agent4, tmp_path):
    """A fresh checkout must never silently stop exploring. Same contract as
    every other rule here: no measurement, no verdict."""
    (tmp_path / "state").mkdir(exist_ok=True)

    assert agent4.minimum_region_size() is None
    assert not agent4._too_small_to_recover({"n_layer": 1, "n_embd": 128})


def test_a_floor_from_another_budget_is_refused(agent4, tmp_path):
    """How much a given size costs is a property of how much training it gets:
    at 12.5M the same ladder was still falling at 232M, at 4.19M it flattens by
    30M. A floor from the wrong budget would refuse regions that are fine."""
    import json

    from prepare import TOKEN_BUDGET

    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    _stamp(state, RUNGS, B_BY_RADIUS)
    stale = json.loads((state / "size_sweep.json").read_text())
    stale["token_budget"] = int(TOKEN_BUDGET) * 3
    (state / "size_sweep.json").write_text(json.dumps(stale), encoding="utf-8")

    assert agent4.minimum_region_size() is None


def test_sigma_region_is_measured_not_configured(tmp_path):
    """It is B(r) at the FENCE radius, read from the geometry report, because
    it moves with both the radius and the token budget. The shipped constant
    was 0.0028, measured at 12.5M inside a radius-0.05 fence; at 4.19M inside
    radius 0.02 it is 0.010579. Under the small value a region was retired for
    falling 0.0084 behind the field when 0.0318 is the real bar."""
    import json

    from prepare import TOKEN_BUDGET

    agent4 = Agent4LandscapeExplorer(
        config_path=str(_config(tmp_path, sigma_region=0.0028, region_radius=0.02)),
        root_dir=str(tmp_path), state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"))
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "region_geometry.json").write_text(json.dumps({
        "token_budget": int(TOKEN_BUDGET),
        "b_by_radius": {"0.02": {"b_observed": 0.010579},
                        "0.10": {"b_observed": 0.078920}},
    }), encoding="utf-8")

    assert agent4.sigma_region == pytest.approx(0.010579)
    # ...and B at the FENCE, not the largest measured, or a 0.02 fence would be
    # judged against a 0.10 neighbourhood.
    assert agent4.sigma_region != pytest.approx(0.078920)


def test_sigma_region_falls_back_to_the_configured_value(tmp_path):
    """No measurement at this budget, so the shipped constant stands in -- and
    the shipped constant must be the one the code would have used."""
    import yaml

    shipped = yaml.safe_load(open("agents_config.yaml", encoding="utf-8"))["agent4"]
    agent4 = Agent4LandscapeExplorer(config_path="does_not_exist.yaml")

    assert agent4._configured_sigma_region == pytest.approx(shipped["sigma_region"])


def test_a_stale_geometry_report_does_not_set_the_region_yardstick(tmp_path):
    import json

    from prepare import TOKEN_BUDGET

    agent4 = Agent4LandscapeExplorer(
        config_path=str(_config(tmp_path, sigma_region=0.0028, region_radius=0.02)),
        root_dir=str(tmp_path), state_dir=str(tmp_path / "state"),
        reports_dir=str(tmp_path / "reports"))
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "region_geometry.json").write_text(json.dumps({
        "token_budget": int(TOKEN_BUDGET) * 3,
        "b_by_radius": {"0.02": {"b_observed": 0.001345}},
    }), encoding="utf-8")

    assert agent4.sigma_region == pytest.approx(0.0028)
