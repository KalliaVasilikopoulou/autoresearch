"""Turn Agent 2's interpretability signals into a direction for Agent 4.

WHY THIS EXISTS. Every signal Agent 2 measures -- which attention heads can be
switched off at no cost, how much each layer contributes to the output, how far
attention reaches, the per-layer scalars -- is a statement about ARCHITECTURE.
None of them says anything about a learning rate or a weight decay. Architecture
is Agent 4's, so this is where those signals belong.

It also puts each method where it is strong. Statistics are good at continuous
knobs with hundreds of samples and bad at architecture, which is discrete,
expensive and thinly sampled; interpretability is the reverse. The surrogate
keeps the eight tunables, XAI steers the three architecture parameters, and
neither is asked to do the other's job.

WEIGHTING. XAI does not overrule the surrogate; it biases it, and by how much
depends on how well the surrogate is currently predicting. That is measurable
for free: `fit_surrogate` already computes out-of-bag predictions (each run
predicted only by the trees that never saw it). When the model predicts well,
trust it and leave XAI near zero. When it predicts badly -- early on, or
somewhere it has never looked -- lean on the direct observation instead.

The base weight is deliberately LOW. The thresholds these votes fire on are
uncalibrated by the code's own admission ("starting points, not calibrated
against real data yet" -- see the FINGERPRINT_* constants), so weighting them
heavily would amplify guesses. Raise it once they have been checked against
real fingerprint history.
"""

from typing import Any, Dict, List, Optional, Sequence

#: Fraction of probed heads that must be effectively dead before we call it
#: "too many heads". Head ablation is the DIRECT evidence for this -- a head
#: whose removal costs nothing is not doing anything -- where attention entropy
#: only infers it.
DEAD_HEAD_FRACTION = 0.5
#: An ablation impact this far below the largest observed one counts as dead.
#: Relative, never absolute: impacts scale with the model.
DEAD_HEAD_RATIO = 0.05


def latest_fingerprint(evidence: Optional[Sequence[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The most recent evidence entry carrying a behavioural fingerprint.
    None is the common case -- fingerprints are computed on a cadence, not
    every run, because they cost real GPU time."""
    for item in reversed(list(evidence or [])):
        if isinstance(item, dict) and item.get("token_fingerprint"):
            return item["token_fingerprint"]
    return None


def dead_head_vote(evidence: Optional[Sequence[Dict[str, Any]]]) -> Optional[int]:
    """-1 on n_head when most probed heads can be switched off for free.

    Not part of the fingerprint rules: head ablation is a separate, cheaper
    measurement (a few extra eval passes) and it is the most direct evidence
    of the three -- it does not infer that heads are redundant, it removes one
    and measures what happens.
    """
    for item in reversed(list(evidence or [])):
        impacts = item.get("head_ablation_impacts") if isinstance(item, dict) else None
        if not impacts:
            continue
        values = [abs(float(v)) for v in impacts.values()]
        if len(values) < 2:
            return None
        peak = max(values)
        if peak <= 0:
            return None
        dead = sum(1 for v in values if v < DEAD_HEAD_RATIO * peak)
        return -1 if dead >= DEAD_HEAD_FRACTION * len(values) else None
    return None


#: How many of the most recent fingerprints must AGREE on a direction before
#: it is allowed to steer anything. One fingerprint is one measurement of a
#: noisy model: the same run-to-run variation that makes val_bpb differences
#: below 0.0138 unreadable also moves dead-head counts and layer contributions,
#: so a single reading is an observation, not a finding.
MIN_AGREEING_FINGERPRINTS = 2


def _fingerprints(evidence: Optional[Sequence[Dict[str, Any]]],
                  limit: int) -> List[Dict[str, Any]]:
    """The most recent `limit` behavioural fingerprints, newest first."""
    out: List[Dict[str, Any]] = []
    for item in reversed(list(evidence or [])):
        if isinstance(item, dict) and item.get("token_fingerprint"):
            out.append(item["token_fingerprint"])
            if len(out) >= limit:
                break
    return out


def architecture_votes(evidence: Optional[Sequence[Dict[str, Any]]],
                       min_agreeing: int = MIN_AGREEING_FINGERPRINTS) -> Dict[str, int]:
    """Net +-1 votes on n_layer / n_embd / n_head, kept only where the last
    `min_agreeing` fingerprints AGREE on the sign.

    Reuses Agent 1's fingerprint rules rather than restating them, so there is
    one definition of what a signal means. Only the architecture half is
    returned -- window_s_fraction stays with Agent 1, because it changes no
    weights and so can vary safely inside a region.

    WHY AGREEMENT IS REQUIRED. This used to read the single latest fingerprint,
    so one run could move an architecture. But a fingerprint is a measurement
    of a noisy training run: the same variation that makes val_bpb differences
    under 0.0138 unreadable at one seed also moves dead-head counts and
    per-layer contributions. Acting on one reading is acting on noise, and
    architecture decisions are the expensive kind -- they open a whole new
    region rather than nudging a knob inside one.

    A parameter whose recent fingerprints DISAGREE contributes nothing, which
    is the honest answer: the evidence is telling us it does not know.
    """
    from agents.agent1_training_specialist import fingerprint_votes

    need = max(1, int(min_agreeing))
    recent = _fingerprints(evidence, need)

    votes: Dict[str, int] = {}
    if len(recent) >= need:
        per_reading: List[Dict[str, int]] = []
        for fingerprint in recent:
            cast_here: Dict[str, int] = {}
            for param, cast in fingerprint_votes(fingerprint).items():
                if param in ("n_layer", "n_embd", "n_head"):
                    cast_here[param] = cast_here.get(param, 0) + sum(cast)
            per_reading.append(cast_here)

        for param in ("n_layer", "n_embd", "n_head"):
            signs = {(1 if r.get(param, 0) > 0 else -1 if r.get(param, 0) < 0 else 0)
                     for r in per_reading}
            # Every reading must be non-zero AND point the same way.
            if len(signs) == 1 and 0 not in signs:
                votes[param] = sum(r.get(param, 0) for r in per_reading) // len(per_reading)

    # Head ablation is exempt: it does not INFER redundancy, it switches a head
    # off and measures what happens. dead_head_vote already requires a majority
    # of probed heads to be free, which is agreement within one measurement.
    head_vote = dead_head_vote(evidence)
    if head_vote:
        votes["n_head"] = votes.get("n_head", 0) + head_vote
    return {k: v for k, v in votes.items() if v}


def surrogate_accuracy(surrogate_model: Any) -> Optional[float]:
    """Out-of-bag R^2: how well the surrogate predicts runs it never trained
    on. Free -- the fit already computed these. None when unavailable.

    Clamped at 0 below: a negative R^2 means the model is worse than predicting
    the mean, and for the purpose here ("how much should we trust it") that is
    simply no trust, not negative trust.
    """
    actual = list(getattr(surrogate_model, "oob_actual", ()) or ())
    predicted = list(getattr(surrogate_model, "oob_predicted", ()) or ())
    if len(actual) < 3 or len(actual) != len(predicted):
        return None
    mean = sum(actual) / len(actual)
    ss_tot = sum((a - mean) ** 2 for a in actual)
    if ss_tot <= 0:
        return None
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return max(0.0, 1.0 - ss_res / ss_tot)


def weighted_step(votes: Dict[str, int], accuracy: Optional[float],
                  base_weight: float, max_step: int = 2) -> Dict[str, float]:
    """Scale the votes by how little the surrogate deserves trust.

    weight = base_weight * (1 - accuracy). A model predicting perfectly leaves
    XAI at zero; a useless one gives it the full base weight, which is itself
    small. With no accuracy figure yet (too few runs to have one) the full base
    weight applies -- that is the early phase, where the surrogate has nothing
    to say and the direct observation is all there is.
    """
    trust_gap = 1.0 if accuracy is None else max(0.0, 1.0 - accuracy)
    weight = base_weight * trust_gap
    out: Dict[str, float] = {}
    for param, vote in votes.items():
        capped = max(-max_step, min(max_step, vote))
        step = capped * weight
        if step:
            out[param] = step
    return out
