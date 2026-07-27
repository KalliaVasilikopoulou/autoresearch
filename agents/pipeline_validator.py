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
    return issues


def _check_boundary_pinning(decisions_dir: Optional[Path], iteration: int, lookback: int) -> List[Issue]:
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
    return issues


def validate_agent3_summary(summary_dict: Dict[str, Any], total_reports: int = 0) -> List[Issue]:
    """ERROR if a recommended hyperparameter falls outside its SEARCH_SPACE
    bounds. WARN if the summary found nothing at all despite plenty of
    history (>= 10 reports) -- a summarizer producing no signal is itself
    a signal something upstream is broken.
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
    return issues


def render_issues(issues: List[Issue]) -> str:
    if not issues:
        return "[pipeline_validator] OK -- no issues."
    return "\n".join(f"[pipeline_validator] {i.severity} ({i.source}): {i.message}" for i in issues)


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
