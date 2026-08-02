"""Synthetic-data tests for agents/claude_cli.py (dev/checks.txt item 4):
the Claude Code CLI subprocess wrapper. subprocess.run is mocked throughout
-- no real `claude` invocation happens in this suite, so it never spends
real usage against anyone's subscription.
"""

import json
import subprocess
from types import SimpleNamespace

import pytest

from agents import claude_cli


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    """is_cli_available() caches module-level state -- reset between tests
    so one test's monkeypatch doesn't leak into the next."""
    monkeypatch.setattr(claude_cli, "_cli_path_cache", None)
    monkeypatch.setattr(claude_cli, "_availability_cache", None)


def _fake_run(stdout: str = "", returncode: int = 0):
    def run(args, capture_output=True, text=True, timeout=None, **kwargs):
        return SimpleNamespace(args=args, stdout=stdout, stderr="", returncode=returncode)
    return run


# --- is_cli_available ----------------------------------------------------

def test_is_cli_available_false_when_binary_missing(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: None)
    assert claude_cli.is_cli_available() is False


def test_is_cli_available_true_when_logged_in(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run(json.dumps({"loggedIn": True})))
    assert claude_cli.is_cli_available() is True


def test_is_cli_available_false_when_not_logged_in(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run(json.dumps({"loggedIn": False})))
    assert claude_cli.is_cli_available() is False


def test_is_cli_available_false_on_malformed_output(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run("not json"))
    assert claude_cli.is_cli_available() is False


def test_is_cli_available_is_cached(monkeypatch):
    calls = {"n": 0}

    def which(name):
        calls["n"] += 1
        return "/fake/claude"

    monkeypatch.setattr(claude_cli.shutil, "which", which)
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run(json.dumps({"loggedIn": True})))
    assert claude_cli.is_cli_available() is True
    assert claude_cli.is_cli_available() is True
    assert calls["n"] == 1  # only checked once


# --- query -----------------------------------------------------------

def test_query_returns_none_when_binary_missing(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: None)
    assert claude_cli.query("hello", max_budget_usd=0.1) is None


def test_query_parses_successful_response(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    payload = {"result": "This run looks healthy.", "total_cost_usd": 0.0032, "is_error": False, "type": "result"}
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run(json.dumps(payload)))

    result = claude_cli.query("summarize this", max_budget_usd=0.2, model="sonnet")

    assert result == {"text": "This run looks healthy.", "cost_usd": 0.0032, "model": "sonnet", "is_error": False}


def test_query_returns_none_on_budget_exhausted_error(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    payload = {
        "is_error": True, "type": "result", "subtype": "error_max_budget_usd",
        "total_cost_usd": 0.065, "errors": ["Reached maximum budget ($0.05)"],
    }
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run(json.dumps(payload)))
    assert claude_cli.query("hello", max_budget_usd=0.05) is None


def test_query_returns_none_on_non_json_stdout(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run("garbled output, not json"))
    assert claude_cli.query("hello", max_budget_usd=0.1) is None


def test_query_returns_none_when_stdout_is_none(monkeypatch):
    """Regression: subprocess.run can return a CompletedProcess with
    stdout=None (seen in real production use) even without raising --
    json.loads(None) raises TypeError, which the original except tuple
    (TimeoutExpired, JSONDecodeError, OSError, ValueError) didn't cover,
    crashing the orchestrator loop instead of degrading to None."""
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    monkeypatch.setattr(claude_cli.subprocess, "run", _fake_run(stdout=None))
    assert claude_cli.query("hello", max_budget_usd=0.1) is None


def test_query_returns_none_on_timeout(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")

    def raise_timeout(args, capture_output=True, text=True, timeout=None, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(claude_cli.subprocess, "run", raise_timeout)
    assert claude_cli.query("hello", max_budget_usd=0.1, timeout=5) is None


def test_query_passes_budget_and_model_flags_but_not_bare(monkeypatch):
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    captured = {}

    def run(args, capture_output=True, text=True, timeout=None, **kwargs):
        captured["args"] = args
        return SimpleNamespace(stdout=json.dumps({"result": "ok", "total_cost_usd": 0.001, "type": "result"}), stderr="", returncode=0)

    monkeypatch.setattr(claude_cli.subprocess, "run", run)
    claude_cli.query("do the thing", max_budget_usd=0.15, model="opus", system_prompt="Be terse.")

    args = captured["args"]
    # --bare is deliberately never passed: it only accepts ANTHROPIC_API_KEY/
    # apiKeyHelper auth and never reads OAuth/keychain, so it's incompatible
    # with the subscription login this module relies on (confirmed live).
    assert "--bare" not in args
    assert "--output-format" in args and "json" in args
    assert "--max-budget-usd" in args and "0.15" in args
    assert "--model" in args and "opus" in args
    assert "--append-system-prompt" in args and "Be terse." in args


def test_query_forces_utf8_decoding_not_platform_default(monkeypatch):
    """Regression: subprocess.run(text=True) without an explicit encoding=
    decodes stdout using the platform's default codepage -- cp1252 on
    Windows, not UTF-8. Claude's generated prose routinely contains
    characters outside cp1252's range (smart quotes, em-dashes, etc.),
    which crashed subprocess.run's internal reader thread with
    UnicodeDecodeError in real production use."""
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: "/fake/claude")
    captured = {}

    def run(args, capture_output=True, text=True, timeout=None, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(stdout=json.dumps({"result": "ok", "total_cost_usd": 0.001, "type": "result"}), stderr="", returncode=0)

    monkeypatch.setattr(claude_cli.subprocess, "run", run)
    claude_cli.query("hello", max_budget_usd=0.1)

    assert captured.get("encoding") == "utf-8"
    assert captured.get("errors") == "replace"


# --- call_with_budget --------------------------------------------------

def test_call_with_budget_skips_when_backend_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_cli, "is_cli_available", lambda: True)
    result = claude_cli.call_with_budget(
        "hi", "some_call_site", model="sonnet", campaign_budget_usd=5.0,
        max_call_budget_usd=0.2, usage_path=str(tmp_path / "usage.json"), backend="none",
    )
    assert result is None


def test_call_with_budget_skips_when_cli_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_cli, "is_cli_available", lambda: False)
    result = claude_cli.call_with_budget(
        "hi", "some_call_site", model="sonnet", campaign_budget_usd=5.0,
        max_call_budget_usd=0.2, usage_path=str(tmp_path / "usage.json"),
    )
    assert result is None


def test_call_with_budget_skips_when_campaign_budget_exhausted(monkeypatch, tmp_path):
    usage_path = tmp_path / "usage.json"
    monkeypatch.setattr(claude_cli, "is_cli_available", lambda: True)

    def fake_query(*a, **k):
        raise AssertionError("query() must not be called once the budget is exhausted")

    monkeypatch.setattr(claude_cli, "query", fake_query)
    result = claude_cli.call_with_budget(
        "hi", "some_call_site", model="sonnet", campaign_budget_usd=0.0,
        max_call_budget_usd=0.2, usage_path=str(usage_path),
    )
    assert result is None


def test_call_with_budget_logs_successful_call_and_caps_at_remaining_budget(monkeypatch, tmp_path):
    usage_path = tmp_path / "usage.json"
    monkeypatch.setattr(claude_cli, "is_cli_available", lambda: True)
    captured_cap = {}

    def fake_query(prompt, *, system_prompt=None, model, max_budget_usd, timeout=120):
        captured_cap["cap"] = max_budget_usd
        return {"text": "a real answer", "cost_usd": 0.01, "model": model, "is_error": False}

    monkeypatch.setattr(claude_cli, "query", fake_query)

    result = claude_cli.call_with_budget(
        "hi", "some_call_site", model="sonnet", campaign_budget_usd=0.02,
        max_call_budget_usd=0.2, usage_path=str(usage_path),
    )

    assert result == "a real answer"
    assert captured_cap["cap"] == pytest.approx(0.02)  # capped by remaining budget, not max_call_budget_usd
    logged = json.loads(usage_path.read_text())
    assert len(logged) == 1
    assert logged[0]["call_site"] == "some_call_site"
    assert logged[0]["cost_usd"] == pytest.approx(0.01)


def test_call_with_budget_returns_none_and_logs_nothing_when_query_fails(monkeypatch, tmp_path):
    usage_path = tmp_path / "usage.json"
    monkeypatch.setattr(claude_cli, "is_cli_available", lambda: True)
    monkeypatch.setattr(claude_cli, "query", lambda *a, **k: None)

    result = claude_cli.call_with_budget(
        "hi", "some_call_site", model="sonnet", campaign_budget_usd=5.0,
        max_call_budget_usd=0.2, usage_path=str(usage_path),
    )
    assert result is None
    assert not usage_path.exists()
