"""Synthetic-data tests for agents/remote_runner.py's multi-GPU parallel
search additions (dev/checks.txt item 1): live GPU discovery, mkdir-based
locking with stale-lock recovery, and run_training_remote's new
gpu_index/hp_remote_name/run_label/skip_sync parameters. All SSH is mocked
-- no real connectivity is required or attempted.
"""

import time
from types import SimpleNamespace

import pytest

from agents import remote_runner


FAKE_CFG = {
    "host": "fake-host", "port": 22, "user": "fake-user", "password": "fake-pass",
    "repo": "/home/fake-user/autoresearch", "conda_env": "autoresearch",
    "conda_activate": "source /opt/anaconda3/bin/activate",
}


class FakeStream:
    """Mimics paramiko's stdout/stderr channel file objects."""

    def __init__(self, text: str = "", exit_status: int = 0):
        self._lines = text.splitlines(keepends=True) if text else []
        self._idx = 0
        self.channel = SimpleNamespace(recv_exit_status=lambda: exit_status)

    def readline(self):
        if self._idx >= len(self._lines):
            return ""
        line = self._lines[self._idx]
        self._idx += 1
        return line

    def read(self):
        remaining = "".join(self._lines[self._idx:])
        self._idx = len(self._lines)
        return remaining.encode("utf-8")


class FakeSFTP:
    def __init__(self):
        self.puts = []

    def put(self, local, remote):
        self.puts.append((local, remote))

    def close(self):
        pass


class FakeSSHClient:
    """responses: list of (substring_matcher, [(stdout_text, exit_status), ...]).
    Each matching exec_command call pops the next queued response; once a
    matcher's queue is down to one item, that item repeats for any further
    calls (so tests only need to spell out the responses that actually
    differ across repeated calls, e.g. a lock retry after reclaiming).
    """

    def __init__(self, responses=None, shared_commands=None):
        self._queues = [(matcher, list(queue)) for matcher, queue in (responses or [])]
        self.commands = shared_commands if shared_commands is not None else []
        self.sftp = FakeSFTP()
        self.closed = False

    def set_missing_host_key_policy(self, policy):
        pass

    def connect(self, **kwargs):
        pass

    def exec_command(self, cmd, timeout=None):
        self.commands.append(cmd)
        for matcher, queue in self._queues:
            if matcher in cmd:
                if not queue:
                    return None, FakeStream(""), FakeStream("")
                out, exit_status = queue[0] if len(queue) == 1 else queue.pop(0)
                return None, FakeStream(out, exit_status), FakeStream("")
        return None, FakeStream(""), FakeStream("")

    def open_sftp(self):
        return self.sftp

    def close(self):
        self.closed = True


def _patch_paramiko(monkeypatch, client_factory):
    fake_paramiko = SimpleNamespace(
        SSHClient=client_factory,
        AutoAddPolicy=lambda: None,
    )
    monkeypatch.setattr(remote_runner, "paramiko", fake_paramiko)
    monkeypatch.setattr(remote_runner, "_PARAMIKO_AVAILABLE", True)
    monkeypatch.setattr(remote_runner, "_load_cfg", lambda: dict(FAKE_CFG))


# --- discover_available_gpus -------------------------------------------

def test_discover_available_gpus_filters_by_live_thresholds_and_sorts_by_free_mem(monkeypatch):
    csv_output = (
        "0, 20000, 24000, 80\n"   # busy (util 80%) -- excluded
        "1, 500, 24000, 2\n"      # free=23500MB, util=2% -- available
        "2, 100, 24000, 1\n"      # free=23900MB, util=1% -- available, most free
    )
    client = FakeSSHClient(responses=[("nvidia-smi", [(csv_output, 0)])])
    _patch_paramiko(monkeypatch, lambda: client)

    gpus = remote_runner.discover_available_gpus()

    assert [g["index"] for g in gpus] == [2, 1]
    assert gpus[0]["free_mb"] > gpus[1]["free_mb"]


def test_discover_available_gpus_returns_empty_on_malformed_output(monkeypatch):
    client = FakeSSHClient(responses=[("nvidia-smi", [("not,a,valid,gpu,line,at,all\n", 0)])])
    _patch_paramiko(monkeypatch, lambda: client)

    assert remote_runner.discover_available_gpus() == []


def test_discover_available_gpus_returns_empty_when_paramiko_unavailable(monkeypatch):
    monkeypatch.setattr(remote_runner, "_PARAMIKO_AVAILABLE", False)
    assert remote_runner.discover_available_gpus() == []


def test_discover_available_gpus_returns_empty_on_connection_failure(monkeypatch):
    class RaisingClient(FakeSSHClient):
        def connect(self, **kwargs):
            raise OSError("connection refused")

    _patch_paramiko(monkeypatch, lambda: RaisingClient())
    assert remote_runner.discover_available_gpus() == []


# --- GPU lock acquire/release -------------------------------------------

def test_acquire_gpu_lock_succeeds_when_absent(monkeypatch):
    client = FakeSSHClient(responses=[
        ("mkdir /repo/.autoresearch_gpu_locks/gpu_3 2>&1", [("LOCK_OK\n", 0)]),
    ])
    assert remote_runner._acquire_gpu_lock(client, 3, "/repo", stale_after_seconds=600) is True


def test_acquire_gpu_lock_fails_when_present_and_fresh(monkeypatch):
    fresh_mtime = str(int(time.time()))
    client = FakeSSHClient(responses=[
        ("mkdir /repo/.autoresearch_gpu_locks/gpu_3 2>&1", [("LOCK_BUSY\n", 0)]),
        ("stat -c %Y /repo/.autoresearch_gpu_locks/gpu_3", [(fresh_mtime, 0)]),
    ])
    assert remote_runner._acquire_gpu_lock(client, 3, "/repo", stale_after_seconds=600) is False


def test_acquire_gpu_lock_reclaims_when_stale(monkeypatch):
    stale_mtime = str(int(time.time()) - 100_000)
    client = FakeSSHClient(responses=[
        ("mkdir /repo/.autoresearch_gpu_locks/gpu_3 2>&1", [("LOCK_BUSY\n", 0), ("LOCK_OK\n", 0)]),
        ("stat -c %Y /repo/.autoresearch_gpu_locks/gpu_3", [(stale_mtime, 0)]),
    ])
    assert remote_runner._acquire_gpu_lock(client, 3, "/repo", stale_after_seconds=600) is True
    assert any("rm -rf /repo/.autoresearch_gpu_locks/gpu_3" in c for c in client.commands)


def test_release_gpu_lock_issues_rm(monkeypatch):
    client = FakeSSHClient()
    remote_runner._release_gpu_lock(client, 3, "/repo")
    assert any("rm -rf /repo/.autoresearch_gpu_locks/gpu_3" in c for c in client.commands)


def test_release_gpu_lock_never_raises_on_failure(monkeypatch):
    class RaisingClient(FakeSSHClient):
        def exec_command(self, cmd, timeout=None):
            raise OSError("connection dropped")

    remote_runner._release_gpu_lock(RaisingClient(), 3, "/repo")  # must not raise


# --- run_training_remote regression + new-parameter behavior ------------

def test_run_training_remote_defaults_match_pre_multi_gpu_behavior(monkeypatch, tmp_path):
    """Default call (no gpu_index/hp_remote_name/run_label/skip_sync given)
    must still fall back to GPU 4 and model_hyperparams.yaml when discovery
    is unavailable -- the exact behavior every existing caller relied on
    before this feature existed.
    """
    shared_commands = []

    def client_factory():
        return FakeSSHClient(
            responses=[
                ("nvidia-smi", []),  # discovery: empty output -> []
                ("git stash", [("No local changes\n", 0)]),
                ("git pull", [("Already up to date.\n", 0)]),
                (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_4 2>&1", [("LOCK_OK\n", 0)]),
                ("python -u train.py", [(
                    "val_bpb:          1.234567\n"
                    "training_seconds: 12.0\n"
                    "status:           remote_ok\n", 0,
                )]),
            ],
            shared_commands=shared_commands,
        )

    _patch_paramiko(monkeypatch, client_factory)
    monkeypatch.delenv("REMOTE_DEFAULT_GPU", raising=False)

    hp_file = tmp_path / "model_hyperparams.yaml"
    hp_file.write_text("n_layer: 8\n")

    metrics = remote_runner.run_training_remote(str(hp_file), timeout=60)

    assert metrics["device"] == 4
    assert metrics["val_bpb"] == pytest.approx(1.234567)
    assert any("CUDA_VISIBLE_DEVICES=4" in c for c in shared_commands)
    assert any(f"{FAKE_CFG['repo']}/model_hyperparams.yaml" in c for c in shared_commands)
    assert any("git pull" in c for c in shared_commands)  # skip_sync=False by default


def test_run_training_remote_uses_explicit_gpu_index_and_hp_name_and_skips_sync(monkeypatch, tmp_path):
    shared_commands = []

    def client_factory():
        return FakeSSHClient(
            responses=[
                (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_7 2>&1", [("LOCK_OK\n", 0)]),
                ("python -u train.py", [("val_bpb:          0.5\nstatus:           remote_ok\n", 0)]),
            ],
            shared_commands=shared_commands,
        )

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")

    metrics = remote_runner.run_training_remote(
        str(hp_file), timeout=60, gpu_index=7, hp_remote_name="model_hyperparams_run0007.yaml",
        run_label="GPU7", skip_sync=True,
    )

    assert metrics["device"] == 7
    assert not any("git pull" in c for c in shared_commands)  # skip_sync=True honored
    assert any("model_hyperparams_run0007.yaml" in c for c in shared_commands)
    assert any("CUDA_VISIBLE_DEVICES=7" in c for c in shared_commands)


def test_run_training_remote_reports_error_status_on_nonzero_exit(monkeypatch, tmp_path):
    def client_factory():
        return FakeSSHClient(responses=[
            (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_2 2>&1", [("LOCK_OK\n", 0)]),
            ("python -u train.py", [("some crash output\n", 1)]),
        ])

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")

    metrics = remote_runner.run_training_remote(str(hp_file), timeout=60, gpu_index=2, skip_sync=True)

    assert metrics["status"] == "remote_error"
    assert metrics["val_bpb"] == float("inf")
    assert metrics["device"] == 2


def test_sync_remote_code_issues_stash_then_pull(monkeypatch):
    shared_commands = []
    _patch_paramiko(monkeypatch, lambda: FakeSSHClient(shared_commands=shared_commands))

    remote_runner.sync_remote_code()

    assert any("git stash" in c for c in shared_commands)
    assert any("git pull" in c for c in shared_commands)
