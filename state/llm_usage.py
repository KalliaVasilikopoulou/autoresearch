"""Shared campaign-wide LLM usage/budget tracking (dev/checks.txt item 4).

The Claude Code CLI doesn't expose your subscription's true remaining
usage %/reset date (confirmed by inspecting `claude --help` and `claude
auth status` -- neither surfaces it), so this tracks a self-logged,
locally-owned dollar-equivalent budget instead, using the same cost units
the CLI itself reports per call (`total_cost_usd`). One shared log/budget
across agent1/2/3 since they all draw on the same subscription.

Mirrors state/results_logger.py's append-one-record pattern.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_USAGE_PATH = "state/llm_usage.json"


def log_call(call_site: str, result: Dict[str, Any], usage_path: str = DEFAULT_USAGE_PATH) -> None:
    """Appends one record for a completed (successful) claude_cli.query()
    call. Never called for calls that returned None (claude_cli.query
    already means "nothing usable happened" -- there's no cost to log for
    a call that never produced a result)."""
    path = Path(usage_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        try:
            records = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            records = []

    records.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "call_site": call_site,
        "model": result.get("model"),
        "cost_usd": float(result.get("cost_usd", 0.0) or 0.0),
        "is_error": bool(result.get("is_error", False)),
    })
    path.write_text(json.dumps(records, indent=2))


def load_usage_log(usage_path: str = DEFAULT_USAGE_PATH) -> List[Dict[str, Any]]:
    path = Path(usage_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def cumulative_cost_usd(usage_path: str = DEFAULT_USAGE_PATH) -> float:
    return sum(r.get("cost_usd", 0.0) or 0.0 for r in load_usage_log(usage_path))


def remaining_budget_usd(campaign_budget_usd: float, usage_path: str = DEFAULT_USAGE_PATH) -> float:
    return max(0.0, campaign_budget_usd - cumulative_cost_usd(usage_path))
