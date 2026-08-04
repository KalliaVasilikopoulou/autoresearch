"""A real 150-iteration campaign died on one transient SSH timeout:

    File "agents/orchestrator.py", line 598, in _run_parallel_wave
        remote_runner.sync_remote_code()
    TimeoutError: [WinError 10060] A connection attempt failed ...

sync_remote_code was the only connect site in remote_runner that let an
exception escape -- discover_available_gpus and run_training_remote both
already caught everything and degraded. Two SSH calls to the same host had
succeeded seconds earlier, so this was a blip, not an outage.

No real SSH here: paramiko.SSHClient is monkeypatched throughout.
"""

import socket

import pytest

from agents import remote_runner


class _FakeClient:
    """Minimal paramiko.SSHClient stand-in. `fail_times` connects raise a
    transient socket error before one finally succeeds."""

    def __init__(self, fail_times=0, exc=None):
        self.fail_times = fail_times
        self.exc = exc or socket.timeout("timed out")
        self.attempts = 0
        self.closed = False

    def set_missing_host_key_policy(self, policy):
        pass

    def connect(self, **kwargs):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise self.exc

    def get_transport(self):
        class _T:
            def set_keepalive(self, n): pass
        return _T()

    def exec_command(self, cmd, timeout=None):
        class _S:
            def read(self_inner): return b""
        return _S(), _S(), _S()

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry backoff is real seconds in production; not in tests."""
    monkeypatch.setattr(remote_runner.time, "sleep", lambda s: None)


# --- _connect_with_retry ---------------------------------------------------

def test_connect_retries_a_transient_failure_and_succeeds():
    client = _FakeClient(fail_times=2)
    remote_runner._connect_with_retry(client, {"host": "h", "port": 22, "user": "u", "password": "p"})
    assert client.attempts == 3


def test_connect_reraises_once_attempts_are_spent():
    """A genuinely-down host must still surface as an error rather than be
    retried forever."""
    client = _FakeClient(fail_times=99)
    with pytest.raises((socket.timeout, OSError)):
        remote_runner._connect_with_retry(
            client, {"host": "h", "port": 22, "user": "u", "password": "p"}, attempts=3)
    assert client.attempts == 3


def test_connect_succeeds_first_time_without_retrying():
    client = _FakeClient(fail_times=0)
    remote_runner._connect_with_retry(client, {"host": "h", "port": 22, "user": "u", "password": "p"})
    assert client.attempts == 1


def test_windows_10060_timeouterror_is_treated_as_transient():
    """The exact exception type from the production traceback."""
    client = _FakeClient(fail_times=1, exc=TimeoutError("[WinError 10060] connection attempt failed"))
    remote_runner._connect_with_retry(client, {"host": "h", "port": 22, "user": "u", "password": "p"})
    assert client.attempts == 2


# --- sync_remote_code degrades instead of raising --------------------------

def _patch_client(monkeypatch, client):
    monkeypatch.setattr(remote_runner, "_PARAMIKO_AVAILABLE", True)
    monkeypatch.setattr(remote_runner, "paramiko",
                        type("_P", (), {"SSHClient": lambda: client,
                                        "AutoAddPolicy": lambda: None}))
    monkeypatch.setattr(remote_runner, "_load_cfg",
                        lambda: {"host": "h", "port": 22, "user": "u",
                                 "password": "p", "repo": "/repo"})


def test_sync_returns_false_instead_of_raising_on_a_dead_host(monkeypatch):
    """The actual regression: this used to propagate and kill the campaign."""
    client = _FakeClient(fail_times=99)
    _patch_client(monkeypatch, client)
    monkeypatch.setattr(remote_runner, "_acquire_sync_lock", lambda *a, **k: True)
    monkeypatch.setattr(remote_runner, "_release_sync_lock", lambda *a, **k: None)

    assert remote_runner.sync_remote_code() is False  # must not raise
    assert client.closed


def test_sync_returns_true_on_success(monkeypatch):
    client = _FakeClient(fail_times=0)
    _patch_client(monkeypatch, client)
    monkeypatch.setattr(remote_runner, "_acquire_sync_lock", lambda *a, **k: True)
    monkeypatch.setattr(remote_runner, "_release_sync_lock", lambda *a, **k: None)

    assert remote_runner.sync_remote_code() is True


def test_sync_recovers_from_a_transient_blip(monkeypatch):
    """The production case: one dropped connect, server otherwise fine."""
    client = _FakeClient(fail_times=1)
    _patch_client(monkeypatch, client)
    monkeypatch.setattr(remote_runner, "_acquire_sync_lock", lambda *a, **k: True)
    monkeypatch.setattr(remote_runner, "_release_sync_lock", lambda *a, **k: None)

    assert remote_runner.sync_remote_code() is True
    assert client.attempts == 2
