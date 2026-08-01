"""Subprocess wrapper around the Claude Code CLI (`claude -p ...`) -- this
project's LLM backend (dev/checks.txt item 4). Deliberately NOT the
`anthropic` Python SDK / ANTHROPIC_API_KEY: that path bills a separate,
metered API key the user has never set up. This wraps the CLI you're
already authenticated with via your Claude Code subscription, so LLM calls
draw from that subscription's usage instead of a new bill.

NOT --bare: confirmed via a live test this session that --bare mode only
accepts ANTHROPIC_API_KEY/apiKeyHelper auth and explicitly never reads
OAuth/keychain ("Not logged in - Please run /login" even though `claude
auth status` shows a real logged-in Pro session) -- --bare is fundamentally
incompatible with subscription auth, so normal mode is used instead despite
its higher per-call overhead (CLAUDE.md/memory/hooks). Any task-specific
guidance is still passed inline via system_prompt (--append-system-prompt)
rather than new repo instruction files, since that's the more precise,
lower-overhead way to steer one scripted call regardless of bare/non-bare.

Every function here degrades to None/False on any failure (binary missing,
timeout, malformed output, budget exhausted) -- mirrors
agents/remote_runner.py's "never crash the orchestrator loop" convention.
"""

import json
import shutil
import subprocess
from typing import Any, Dict, Optional

from state import llm_usage

_cli_path_cache: Optional[str] = None
_availability_cache: Optional[bool] = None


def _cli_path() -> Optional[str]:
    global _cli_path_cache
    if _cli_path_cache is None:
        _cli_path_cache = shutil.which("claude") or ""
    return _cli_path_cache or None


def is_cli_available() -> bool:
    """True when the claude CLI is on PATH and logged in. Cached for the
    life of the process -- this doesn't change mid-run, and `claude auth
    status` is its own subprocess call not worth repeating per LLM call.
    """
    global _availability_cache
    if _availability_cache is not None:
        return _availability_cache

    path = _cli_path()
    if not path:
        _availability_cache = False
        return False

    try:
        result = subprocess.run(
            [path, "auth", "status"],
            capture_output=True, text=True, timeout=30,
        )
        payload = json.loads(result.stdout)
        _availability_cache = bool(payload.get("loggedIn"))
    except Exception:
        _availability_cache = False
    return _availability_cache


def query(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: str = "sonnet",
    max_budget_usd: float,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    """Runs one non-interactive `claude -p` call and returns
    {"text", "cost_usd", "model", "is_error"}, or None on any failure
    (missing binary, timeout, non-JSON output, or the CLI's own
    --max-budget-usd cap being hit before a result completed).
    """
    path = _cli_path()
    if not path:
        return None

    args = [
        path, "-p", prompt,
        "--output-format", "json",
        "--max-budget-usd", str(max_budget_usd),
        "--model", model,
    ]
    if system_prompt:
        args.extend(["--append-system-prompt", system_prompt])

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        payload = json.loads(result.stdout)
    except Exception:
        # Broad by design (matches is_cli_available()'s own convention just
        # above): this must degrade to None on ANY failure, never crash the
        # orchestrator loop. A narrower (TimeoutExpired, JSONDecodeError,
        # OSError, ValueError) tuple missed a real one in production --
        # result.stdout can come back None (not just malformed) on some
        # process-termination edge cases, and json.loads(None) raises
        # TypeError, which that tuple didn't cover.
        return None

    if payload.get("is_error") or payload.get("type") == "result" and payload.get("subtype", "").startswith("error"):
        return None

    text = payload.get("result")
    if not isinstance(text, str) or not text:
        return None

    return {
        "text": text,
        "cost_usd": float(payload.get("total_cost_usd", 0.0) or 0.0),
        "model": model,
        "is_error": False,
    }


def call_with_budget(
    prompt: str,
    call_site: str,
    *,
    system_prompt: Optional[str] = None,
    model: str,
    campaign_budget_usd: float,
    max_call_budget_usd: float,
    usage_path: str,
    backend: str = "cli",
    timeout: int = 120,
) -> Optional[str]:
    """The shared guard every LLM call site in agents/agent{1,2,3}_*.py
    uses: skip entirely when backend != "cli" (agents_config.yaml's
    llm.backend: "none" kill switch), the CLI isn't available, or the
    campaign budget is already exhausted; otherwise cap this call at
    whichever is smaller (the per-call cap or what's left of the campaign
    budget), log the result on success, and return just the response text
    (or None on any failure/skip -- callers already have an established
    "no LLM output" fallback path, so None is never a special case for
    them).
    """
    if backend != "cli":
        return None
    if not is_cli_available():
        return None

    remaining = llm_usage.remaining_budget_usd(campaign_budget_usd, usage_path)
    if remaining <= 0:
        print(f"[LLM] {call_site}: campaign budget (${campaign_budget_usd:.2f}) exhausted -- skipping")
        return None

    cap = min(max_call_budget_usd, remaining)
    result = query(prompt, system_prompt=system_prompt, model=model, max_budget_usd=cap, timeout=timeout)
    if result is None:
        print(f"[LLM] {call_site}: call failed or produced no usable result -- skipping")
        return None

    llm_usage.log_call(call_site, result, usage_path)
    new_cumulative = llm_usage.cumulative_cost_usd(usage_path)
    print(f"[LLM] {call_site}: cost=${result['cost_usd']:.4f}, "
          f"cumulative=${new_cumulative:.2f}/${campaign_budget_usd:.2f} budget")
    return result["text"]
