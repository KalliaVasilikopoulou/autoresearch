"""Deterministic pipeline validation (see dev/inpsect_workflow_ideas.txt).

Plain Python assertions, not an LLM agent: "is this NaN", "is this in
range", "did every parameter get an explicit disposition" are checks code
can do reliably and cheaply -- an LLM validator would add cost, latency,
and a new thing that can itself be wrong. Code finds the problem; an LLM
(later, optional, not built here) would explain it.

Severity model (deliberately not a blocking y/n prompt by default -- that
ends unattended runs on the first spurious warning):
  FATAL -- halt the orchestrator loop immediately, no prompt, ever.
  ERROR -- log, tag the iteration "suspect", continue.
  WARN  -- log only.
Orchestrator's --interactive flag additionally prompts on ERROR+, for when
a human is at the keyboard and wants the option to stop early -- never the
default, since a blocking prompt is exactly what kills overnight runs.
"""

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.agent1_training_specialist import SEARCH_SPACE

FATAL = "FATAL"
ERROR = "ERROR"
WARN = "WARN"


@dataclass
class Issue:
    severity: str  # FATAL | ERROR | WARN
    source: str    # "agent1" | "agent2" | "agent3" | "train"
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"severity": self.severity, "source": self.source, "message": self.message, "context": self.context}


def _is_bad_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isfinite(value)


def validate_agent1_decision(
    decision_log: Optional[Dict[str, Any]],
    evidence: Optional[list],
    latest_summary: Optional[str],
    decisions_dir: Optional[Path] = None,
    lookback: int = 3,
) -> List[Issue]:
    """FATAL if Agent 1 received real signal but produced no decision log at
    all (a regression guard on the decision log's own "total interface"
    invariant -- should be structurally impossible, so this is a canary).
    ERROR if any parameter's recorded value is NaN/inf. WARN if a parameter
    has sat at the exact same SEARCH_SPACE boundary for `lookback` straight
    iterations (reads recent decision_*.json files from disk) -- the "why
    is this extreme" signal for the architecture-param clamps that Part 2
    deliberately left untouched at the source.
    """
    issues: List[Issue] = []
    had_signal = bool(evidence) or bool(latest_summary)
    if had_signal and not decision_log:
        issues.append(Issue(FATAL, "agent1",
            "Agent 1 received evidence/summary but produced no decision log",
            {"evidence_count": len(evidence) if evidence else 0, "summary_present": bool(latest_summary)}))
        return issues

    if not decision_log:
        return issues

    for param, info in decision_log.get("params", {}).items():
        if _is_bad_number(info.get("after")):
            issues.append(Issue(ERROR, "agent1", f"{param} is NaN/inf after this decision",
                                 {"param": param, "value": info.get("after")}))

    issues.extend(_check_boundary_pinning(decisions_dir, decision_log.get("iteration", 0), lookback))
    issues.extend(_check_fingerprint_adjustment_thrashing(decisions_dir, decision_log.get("iteration", 0)))
    return issues


def _load_recent_decisions(decisions_dir: Optional[Path], iteration: int, lookback: int) -> List[Dict[str, Any]]:
    """Most-recent-first list of parsed decision_*.json files, walking back
    from `iteration`, stopping at the first missing/corrupt file. Shared by
    _check_boundary_pinning and _check_fingerprint_adjustment_thrashing so
    the "read recent decision logs off disk" logic exists in one place.
    """
    if not decisions_dir:
        return []
    decisions_dir = Path(decisions_dir)
    recent = []
    for i in range(iteration, max(-1, iteration - lookback), -1):
        p = decisions_dir / f"decision_{i:04d}.json"
        if not p.exists():
            break
        try:
            recent.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            break
    return recent


def _check_fingerprint_adjustment_thrashing(decisions_dir: Optional[Path], iteration: int, lookback: int = 6) -> List[Issue]:
    """Tier 4's fingerprint_adjustments (agents/agent1_training_specialist.py's
    _fingerprint_adjustment) was never validated before this. WARN when a
    param's fingerprint-driven votes have flipped sign at least twice across
    its last few real occurrences (e.g. +1, -1, +1) -- the rules pushing it
    back and forth rather than settling is a sign they're unstable for this
    param, not converging on anything.
    """
    recent = _load_recent_decisions(decisions_dir, iteration, lookback)
    if not recent:
        return []

    # recent[] is newest-first; walk it oldest-first per param so the sign
    # sequence reflects real chronological order.
    per_param_deltas: Dict[str, List[float]] = {}
    for log in reversed(recent):
        for entry in log.get("fingerprint_adjustments", []):
            param, delta = entry.get("param"), entry.get("delta")
            if param and isinstance(delta, (int, float)) and delta != 0:
                per_param_deltas.setdefault(param, []).append(delta)

    issues: List[Issue] = []
    for param, deltas in per_param_deltas.items():
        if len(deltas) < 3:
            continue
        signs = [1 if d > 0 else -1 for d in deltas]
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        if flips >= 2:
            issues.append(Issue(WARN, "agent1",
                f"{param} has been pushed in alternating directions by Tier 4 fingerprint rules "
                f"{flips} times across its last {len(deltas)} adjustments (deltas={deltas}) -- "
                f"these rules may be thrashing on this param rather than converging",
                {"param": param, "deltas": deltas, "flips": flips}))
    return issues


def _check_boundary_pinning(decisions_dir: Optional[Path], iteration: int, lookback: int) -> List[Issue]:
    recent = _load_recent_decisions(decisions_dir, iteration, lookback)
    if len(recent) < lookback:
        return []

    issues: List[Issue] = []
    for param, (lo, hi) in SEARCH_SPACE.items():
        values = []
        for log in recent:
            info = log.get("params", {}).get(param)
            if not info or not isinstance(info.get("after"), (int, float)):
                values = []
                break
            values.append(info["after"])
        if len(values) == lookback and all(v == values[0] for v in values) and values[0] in (lo, hi):
            issues.append(Issue(WARN, "agent1",
                f"{param} has been pinned at its boundary ({values[0]}) for {lookback} consecutive iterations",
                {"param": param, "value": values[0], "bound": "lo" if values[0] == lo else "hi", "lookback": lookback}))
    return issues


def validate_training_result(metrics: Dict[str, Any], requested_hyperparams: Optional[Dict[str, Any]] = None) -> List[Issue]:
    """ERROR per entry in metrics["hyperparam_clamps"] (see train.py/Part 2):
    "Agent 1 requested X, train.py actually used Y" -- structurally answers
    "what caused this extreme value" instead of leaving it a mystery. ERROR
    if val_bpb is NaN specifically (inf is the existing, legitimate "run
    failed" marker handled elsewhere in this codebase -- NaN is not).
    """
    issues: List[Issue] = []
    for param, info in (metrics.get("hyperparam_clamps") or {}).items():
        issues.append(Issue(ERROR, "train",
            f"train.py clamped {param}: requested={info.get('requested')}, actually used={info.get('clamped')}",
            {"param": param, **info}))
    val_bpb = metrics.get("val_bpb")
    if isinstance(val_bpb, float) and math.isnan(val_bpb):
        issues.append(Issue(ERROR, "train", "val_bpb is NaN", {"val_bpb": val_bpb}))
    return issues


def validate_agent2_report(evidence_dict: Dict[str, Any]) -> List[Issue]:
    """ERROR if any hyperparameter_importance value is non-finite or outside
    [0, 1]. WARN if a single-head ablation impact is implausibly large.
    Also validates the Tier 2 token_fingerprint, if present (see
    _check_token_fingerprint) -- it was never checked at all before this.
    """
    issues: List[Issue] = []
    for param, score in (evidence_dict.get("hyperparameter_importance") or {}).items():
        if _is_bad_number(score):
            issues.append(Issue(ERROR, "agent2", f"hyperparameter_importance[{param}] is NaN/inf",
                                 {"param": param, "value": score}))
        elif isinstance(score, (int, float)) and not (0.0 <= score <= 1.0):
            issues.append(Issue(ERROR, "agent2", f"hyperparameter_importance[{param}]={score} outside [0, 1]",
                                 {"param": param, "value": score}))
    for item in evidence_dict.get("important_heads") or []:
        impact = item.get("impact")
        if isinstance(impact, (int, float)) and math.isfinite(impact) and abs(impact) > 1.0:
            issues.append(Issue(WARN, "agent2",
                f"head {item.get('head')} impact={impact} implausibly large for a single-head ablation delta",
                {"head": item.get("head"), "impact": impact}))
    issues.extend(_check_token_fingerprint(evidence_dict.get("token_fingerprint") or {}))
    return issues


# Per-layer arrays that must contain only finite numbers -- a NaN/inf
# anywhere here would silently corrupt every Tier 2/3/4 consumer downstream
# (Agent 3's cluster fitting, Agent 1's fingerprint-driven architecture
# nudges) with no other check catching it first.
_FINGERPRINT_FINITE_ARRAYS = ("attn_entropy", "attn_distance", "dla", "x0_lambda", "pos_saliency")
# Arrays that are additionally mathematically non-negative by construction
# (softmax entropy, an absolute distance, |grad x input|) -- a negative
# value here isn't just implausible, it's a sign something upstream computed
# the wrong quantity entirely.
_FINGERPRINT_NONNEGATIVE_ARRAYS = ("attn_entropy", "attn_distance", "pos_saliency")


def _check_token_fingerprint(fingerprint: Dict[str, Any]) -> List[Issue]:
    """Tier 2's token_fingerprint (agents/xai_methods/token_methods.py) was
    never validated before this -- it flows straight from train.py's stdout
    through Agent 2 into every Tier 3/4 consumer with zero checks in
    between. ERROR on NaN/inf or a value that's mathematically impossible
    for what it represents (negative entropy, an induction "probability"
    outside [0,1]); WARN on a present-but-empty array, which suggests a
    partial/broken fingerprint rather than one that was simply never
    computed (an absent key is normal -- token_xai_enabled was off; an
    empty list for a key that IS present is not).
    """
    if not fingerprint:
        return []
    issues: List[Issue] = []

    for key in _FINGERPRINT_FINITE_ARRAYS:
        if key not in fingerprint:
            continue
        values = fingerprint[key]
        if not values:
            issues.append(Issue(WARN, "agent2", f"token_fingerprint[{key}] is present but empty",
                                 {"field": key}))
            continue
        for i, v in enumerate(values):
            if _is_bad_number(v):
                issues.append(Issue(ERROR, "agent2", f"token_fingerprint[{key}][{i}] is NaN/inf",
                                     {"field": key, "index": i, "value": v}))
        if key in _FINGERPRINT_NONNEGATIVE_ARRAYS:
            for i, v in enumerate(values):
                if isinstance(v, (int, float)) and math.isfinite(v) and v < 0:
                    issues.append(Issue(ERROR, "agent2", f"token_fingerprint[{key}][{i}]={v} is negative, "
                                         f"which should be mathematically impossible for this quantity",
                                         {"field": key, "index": i, "value": v}))

    induction_score = fingerprint.get("induction_score")
    if _is_bad_number(induction_score):
        issues.append(Issue(ERROR, "agent2", "token_fingerprint[induction_score] is NaN/inf",
                             {"value": induction_score}))
    elif isinstance(induction_score, (int, float)) and not (0.0 <= induction_score <= 1.0):
        issues.append(Issue(ERROR, "agent2", f"token_fingerprint[induction_score]={induction_score} outside [0, 1] "
                             f"(it's an attention weight, always a probability)",
                             {"value": induction_score}))

    attn_distance_slope = fingerprint.get("attn_distance_slope")
    if _is_bad_number(attn_distance_slope):
        issues.append(Issue(ERROR, "agent2", "token_fingerprint[attn_distance_slope] is NaN/inf",
                             {"value": attn_distance_slope}))

    return issues


def validate_agent3_summary(summary_dict: Dict[str, Any], total_reports: int = 0) -> List[Issue]:
    """ERROR if a recommended hyperparameter falls outside its SEARCH_SPACE
    bounds. WARN if the summary found nothing at all despite plenty of
    history (>= 10 reports) -- a summarizer producing no signal is itself
    a signal something upstream is broken. Also validates Tier 3's
    fingerprint_clusters, if present (see _check_fingerprint_clusters) --
    it was never checked at all before this.
    """
    issues: List[Issue] = []
    for param, value in (summary_dict.get("recommended_hyperparams") or {}).items():
        bounds = SEARCH_SPACE.get(param)
        if bounds and isinstance(value, (int, float)):
            lo, hi = bounds
            if not (lo <= value <= hi):
                issues.append(Issue(ERROR, "agent3",
                    f"recommended {param}={value} outside safe bounds [{lo}, {hi}]",
                    {"param": param, "value": value, "bounds": [lo, hi]}))
    if total_reports >= 10 and not summary_dict.get("stable_patterns") and not summary_dict.get("conflicting_signals"):
        issues.append(Issue(WARN, "agent3",
            f"summary found neither stable patterns nor conflicting signals despite {total_reports} reports of history",
            {"total_reports": total_reports}))
    issues.extend(_check_fingerprint_clusters(summary_dict.get("fingerprint_clusters") or {}))
    return issues


# Below this, a silhouette score is "weak but real structure" per the usual
# rule of thumb (same threshold this codebase's own Tier 3 tests use to
# separate "real, if noisy, separation" from "no separation" -- see
# tests/test_clustering.py) -- not proof the clusters are meaningless, but
# worth a WARN so a reader doesn't over-trust a weak split.
_WEAK_SILHOUETTE_THRESHOLD = 0.25


def _check_fingerprint_clusters(fingerprint_clusters: Dict[str, Any]) -> List[Issue]:
    """Tier 3's fingerprint_clusters (state/clustering.py) was never
    validated before this. WARN on a silhouette score that indicates weak
    or no real cluster structure (<=0 means "no better than a random
    split"); WARN on a cluster with fewer than 2 members -- should be
    structurally impossible (state/clustering.py's MIN_CLUSTER_SIZE=2 is
    enforced before a clustering is ever returned), so this is a canary for
    that invariant breaking, not an expected finding.
    """
    issues: List[Issue] = []
    for kind in ("overall", "trajectory"):
        result = fingerprint_clusters.get(kind)
        if not result:
            continue
        silhouette = result.get("silhouette")
        if isinstance(silhouette, (int, float)) and math.isfinite(silhouette):
            if silhouette <= 0:
                issues.append(Issue(WARN, "agent3",
                    f"Tier 3 {kind} clustering has silhouette={silhouette:.3f} (<=0 means no better than a "
                    f"random split) -- treat any pattern drawn from these clusters as unreliable",
                    {"kind": kind, "silhouette": silhouette}))
            elif silhouette < _WEAK_SILHOUETTE_THRESHOLD:
                issues.append(Issue(WARN, "agent3",
                    f"Tier 3 {kind} clustering has a weak silhouette={silhouette:.3f} (< {_WEAK_SILHOUETTE_THRESHOLD}) "
                    f"-- real but noisy separation, worth more data before trusting it strongly",
                    {"kind": kind, "silhouette": silhouette}))
        for cluster in result.get("clusters") or []:
            n = cluster.get("n")
            if isinstance(n, int) and n < 2:
                issues.append(Issue(WARN, "agent3",
                    f"Tier 3 {kind} cluster {cluster.get('cluster_id')} has only {n} member(s) -- "
                    f"should not be possible given clustering.py's own minimum-cluster-size guarantee",
                    {"kind": kind, "cluster_id": cluster.get("cluster_id"), "n": n}))
    return issues


def validate_batch_accumulation(report_batch_size: int, configured_batch_size: int, stall_multiplier: float = 3.0) -> List[Issue]:
    """WARN if the pending Agent 3 report batch has grown well past the
    configured trigger size without a summary ever firing.
    should_create_summary is a pure `report_count % batch_size == 0` check
    (agents/agent3_report_analyst.py) -- this should be structurally
    impossible unless batch_size changed mid-run or the batch counter
    itself is broken, so a stall here is a canary, not an expected finding.
    Orchestrator only has something meaningful to check here in the "batch
    not full yet" branch -- there's nothing to validate about an absent
    summary otherwise.
    """
    issues: List[Issue] = []
    if configured_batch_size > 0 and report_batch_size > stall_multiplier * configured_batch_size:
        issues.append(Issue(WARN, "agent3",
            f"pending report batch has grown to {report_batch_size}, well past the configured trigger "
            f"size of {configured_batch_size} ({stall_multiplier}x) without a summary firing -- "
            f"should_create_summary may be stuck",
            {"report_batch_size": report_batch_size, "configured_batch_size": configured_batch_size}))
    return issues


def render_issues(issues: List[Issue]) -> str:
    """Plain-text rendering for the terminal. Carries the same information
    as write_iteration_issues' JSON (message + context), not a subset of
    it -- context is appended inline rather than left JSON-only, so nothing
    a human would need is hidden in a file they'd have to go open.
    """
    if not issues:
        return "[pipeline_validator] OK -- no issues."
    lines = []
    for i in issues:
        line = f"[pipeline_validator] {i.severity} ({i.source}): {i.message}"
        if i.context:
            details = ", ".join(f"{k}={v}" for k, v in i.context.items())
            line += f" ({details})"
        lines.append(line)
    return "\n".join(lines)


def new_run_dir(validation_dir: Path) -> Path:
    """A fresh timestamped directory for this orchestrator run's validation
    logs -- never overwrites a previous run's history (that history is
    exactly what catches intermittent bugs; see prune_old_runs for the
    retention policy instead of clearing on startup).
    """
    validation_dir = Path(validation_dir)
    run_dir = validation_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prune_old_runs(validation_dir: Path, keep: int = 10) -> None:
    """Keep only the `keep` most recent run_* directories. Called once at
    Orchestrator startup, not per-iteration, and never clears the current
    run's own logs.
    """
    validation_dir = Path(validation_dir)
    if not validation_dir.exists():
        return
    run_dirs = sorted(p for p in validation_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    for stale in run_dirs[:-keep] if keep > 0 else run_dirs:
        shutil.rmtree(stale, ignore_errors=True)


def write_iteration_issues(run_dir: Path, iteration: int, issues: List[Issue], suspect: bool) -> None:
    path = Path(run_dir) / f"iteration_{iteration:04d}.json"
    existing: Dict[str, Any] = {"iteration": iteration, "issues": [], "suspect": False}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    existing["issues"] = existing.get("issues", []) + [i.to_dict() for i in issues]
    existing["suspect"] = existing.get("suspect", False) or suspect
    path.write_text(json.dumps(existing, indent=2))
