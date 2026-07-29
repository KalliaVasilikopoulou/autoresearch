"""Synthetic-data tests for state/llm_usage.py's campaign-wide budget
tracking (dev/checks.txt item 4)."""

import json

import pytest

from state import llm_usage


def test_load_usage_log_empty_when_file_absent(tmp_path):
    assert llm_usage.load_usage_log(str(tmp_path / "usage.json")) == []


def test_log_call_appends_and_round_trips(tmp_path):
    path = tmp_path / "usage.json"
    llm_usage.log_call("site_a", {"model": "sonnet", "cost_usd": 0.01, "is_error": False}, str(path))
    llm_usage.log_call("site_b", {"model": "sonnet", "cost_usd": 0.02, "is_error": False}, str(path))

    records = llm_usage.load_usage_log(str(path))
    assert len(records) == 2
    assert [r["call_site"] for r in records] == ["site_a", "site_b"]
    assert all("timestamp" in r for r in records)


def test_cumulative_cost_usd_sums_all_records(tmp_path):
    path = tmp_path / "usage.json"
    llm_usage.log_call("site_a", {"cost_usd": 0.01, "model": "sonnet", "is_error": False}, str(path))
    llm_usage.log_call("site_a", {"cost_usd": 0.03, "model": "sonnet", "is_error": False}, str(path))
    assert llm_usage.cumulative_cost_usd(str(path)) == pytest.approx(0.04)


def test_remaining_budget_usd_never_goes_negative(tmp_path):
    path = tmp_path / "usage.json"
    llm_usage.log_call("site_a", {"cost_usd": 10.0, "model": "sonnet", "is_error": False}, str(path))
    assert llm_usage.remaining_budget_usd(5.0, str(path)) == 0.0


def test_remaining_budget_usd_reflects_spend(tmp_path):
    path = tmp_path / "usage.json"
    llm_usage.log_call("site_a", {"cost_usd": 1.5, "model": "sonnet", "is_error": False}, str(path))
    assert llm_usage.remaining_budget_usd(5.0, str(path)) == 3.5


def test_load_usage_log_tolerant_of_corrupt_file(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{not valid json")
    assert llm_usage.load_usage_log(str(path)) == []


def test_log_call_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "usage.json"
    llm_usage.log_call("site_a", {"cost_usd": 0.0, "model": "sonnet", "is_error": False}, str(path))
    assert path.exists()
    assert json.loads(path.read_text())[0]["call_site"] == "site_a"
