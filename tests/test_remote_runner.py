"""Synthetic-data tests for agents/remote_runner.py's multi-GPU parallel
search additions (dev/checks.txt item 1): live GPU discovery, mkdir-based
locking with stale-lock recovery, and run_training_remote's new
gpu_index/hp_remote_name/run_label/skip_sync parameters. All SSH is mocked
-- no real connectivity is required or attempted.
"""

import socket
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
    """Mimics paramiko's stdout/stderr channel file objects. raise_after/
    raise_exc let a test simulate a connection lost partway through
    reading -- readline() returns lines normally until raise_after lines
    have been yielded, then raises raise_exc instead of returning "" (EOF)."""

    def __init__(self, text: str = "", exit_status: int = 0, raise_after=None, raise_exc=None):
        self._lines = text.splitlines(keepends=True) if text else []
        self._idx = 0
        self._raise_after = raise_after
        self._raise_exc = raise_exc
        self.channel = SimpleNamespace(recv_exit_status=lambda: exit_status)

    def readline(self):
        if self._raise_after is not None and self._idx >= self._raise_after:
            raise self._raise_exc
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
    A response entry may also be a 4-tuple (stdout_text, exit_status,
    raise_after, raise_exc) to simulate a connection lost partway through
    reading -- see FakeStream.
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
                entry = queue[0] if len(queue) == 1 else queue.pop(0)
                if len(entry) == 4:
                    out, exit_status, raise_after, raise_exc = entry
                    return None, FakeStream(out, exit_status, raise_after, raise_exc), FakeStream("")
                out, exit_status = entry
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
    # Every train.py invocation carries this marker -- it's what lets
    # kill_stale_training_processes() identify a leftover process as ours
    # (see test_kill_stale_training_processes_* below) without ever
    # touching another user's/project's process.
    assert any("AUTORESEARCH_MANAGED=1" in c for c in shared_commands)


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


class FakeDisplay:
    """Stand-in for agents.live_progress.MultiGpuProgressDisplay -- records
    what run_training_remote routes to each method, without any real
    terminal/rich dependency."""

    def __init__(self):
        self.progress_calls = []
        self.printed_lines = []

    def update_progress(self, label, line):
        self.progress_calls.append((label, line))

    def print_line(self, text):
        self.printed_lines.append(text)


def test_run_training_remote_routes_progress_lines_to_display_when_provided(monkeypatch, tmp_path):
    def client_factory():
        return FakeSSHClient(responses=[
            (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_1 2>&1", [("LOCK_OK\n", 0)]),
            ("python -u train.py", [(
                "[--------------------]   0.0% | loss: 9.0 | remaining: 300.0s\n"
                "Time budget: 300s\n"
                "[==========----------]  50.0% | loss: 5.0 | remaining: 150.0s\n"
                "val_bpb:          0.9\n"
                "status:           remote_ok\n", 0,
            )]),
        ])

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")
    display = FakeDisplay()

    metrics = remote_runner.run_training_remote(
        str(hp_file), timeout=60, gpu_index=1, run_label="GPU1", skip_sync=True, display=display,
    )

    assert metrics["status"] == "remote_ok"
    progress_lines = [line for _label, line in display.progress_calls]
    assert any("0.0%" in line for line in progress_lines)
    assert any("50.0%" in line for line in progress_lines)
    assert all(label == "GPU1" for label, _line in display.progress_calls)
    assert any("Time budget: 300s" in text for text in display.printed_lines)
    # Non-progress status lines (connect/upload/execute/parsed metrics) also
    # go through the display, not a bare print(), so they can't clobber the
    # live-updating progress table.
    assert any("Connecting to" in text for text in display.printed_lines)
    assert any("Parsed metrics" in text for text in display.printed_lines)


def test_run_training_remote_salvages_val_bpb_when_connection_lost_after_training_completed(monkeypatch, tmp_path):
    """train.py prints a real val_bpb right after its own internal training
    budget expires -- well before the (silent, can run for minutes) head-
    ablation study that follows. A connection lost during that later, silent
    stretch must not discard the val_bpb that was already fully received."""
    captured_output = (
        "[--------------------]   0.0% | loss: 9.0 | remaining: 300.0s\n"
        "---\n"
        "val_bpb:          0.987654\n"
        "training_seconds: 300.1\n"
        "status:           remote_ok\n"
    )  # 5 lines -- all fully received before the simulated timeout below

    def client_factory():
        return FakeSSHClient(responses=[
            (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_1 2>&1", [("LOCK_OK\n", 0)]),
            ("python -u train.py", [(captured_output, 0, 5, socket.timeout("timed out"))]),
        ])

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")

    metrics = remote_runner.run_training_remote(str(hp_file), timeout=60, gpu_index=1, skip_sync=True)

    assert metrics["status"] == "remote_partial_timeout"
    assert metrics["val_bpb"] == pytest.approx(0.987654)
    assert metrics["device"] == 1
    # Post-training analysis fields never arrived -- honestly absent, not
    # fabricated, same convention as a run where that analysis never ran.
    assert "head_ablation_impacts" not in metrics


def test_run_training_remote_treats_timeout_as_error_when_nothing_usable_was_captured(monkeypatch, tmp_path):
    """A connection lost before val_bpb was ever printed (e.g. still mid
    training loop) has nothing to salvage -- must still report a failure,
    not fabricate a result."""
    partial_output = (
        "[--------------------]   0.0% | loss: 9.0 | remaining: 300.0s\n"
        "[--------------------]   5.0% | loss: 8.5 | remaining: 285.0s\n"
    )

    def client_factory():
        return FakeSSHClient(responses=[
            (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_1 2>&1", [("LOCK_OK\n", 0)]),
            ("python -u train.py", [(partial_output, 0, 2, socket.timeout("timed out"))]),
        ])

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")

    metrics = remote_runner.run_training_remote(str(hp_file), timeout=60, gpu_index=1, skip_sync=True)

    assert metrics["status"] == "remote_error"
    assert metrics["val_bpb"] == float("inf")
    assert metrics["device"] == 1


def test_run_training_remote_salvage_routes_through_display_when_provided(monkeypatch, tmp_path):
    captured_output = "---\nval_bpb:          1.1\nstatus:           remote_ok\n"

    def client_factory():
        return FakeSSHClient(responses=[
            (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_1 2>&1", [("LOCK_OK\n", 0)]),
            ("python -u train.py", [(captured_output, 0, 3, socket.timeout("timed out"))]),
        ])

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")
    display = FakeDisplay()

    metrics = remote_runner.run_training_remote(
        str(hp_file), timeout=60, gpu_index=1, run_label="GPU1", skip_sync=True, display=display,
    )

    assert metrics["status"] == "remote_partial_timeout"
    assert any("Lost connection" in text for text in display.printed_lines)
    assert any("Recovered a usable val_bpb" in text for text in display.printed_lines)


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


# --- kill_stale_training_processes ---------------------------------------

@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """kill_stale_training_processes sleeps a few real seconds between
    SIGTERM and its liveness check -- never let that actually happen in
    the test suite."""
    monkeypatch.setattr(remote_runner.time, "sleep", lambda seconds: None)


def test_kill_stale_training_processes_returns_empty_when_paramiko_unavailable(monkeypatch):
    monkeypatch.setattr(remote_runner, "_PARAMIKO_AVAILABLE", False)
    assert remote_runner.kill_stale_training_processes() == []


def test_kill_stale_training_processes_returns_empty_when_remote_not_configured(monkeypatch):
    monkeypatch.setattr(remote_runner, "_PARAMIKO_AVAILABLE", True)
    monkeypatch.setattr(remote_runner, "_load_cfg", lambda: {"host": "", "user": "", "repo": "", "password": ""})
    assert remote_runner.kill_stale_training_processes() == []


def test_kill_stale_training_processes_returns_empty_when_no_gpu_processes(monkeypatch):
    client = FakeSSHClient(responses=[("nvidia-smi --query-compute-apps", [("", 0)])])
    _patch_paramiko(monkeypatch, lambda: client)
    assert remote_runner.kill_stale_training_processes() == []


def test_kill_stale_training_processes_kills_only_the_matching_process(monkeypatch):
    """Two PIDs show up as GPU-attached: PID 111 is ours (owner, cmdline,
    cwd, and marker all match); PID 222 belongs to a different user and
    must never be touched -- not even inspected past the ownership check.
    """
    shared_commands = []

    def client_factory():
        return FakeSSHClient(
            responses=[
                ("nvidia-smi --query-compute-apps", [("111\n222\n", 0)]),
                ("ps -o ruser=,args= -p 111", [("fake-user python -u train.py", 0)]),
                ("ps -o ruser=,args= -p 222", [("some-other-user some_other_process.py", 0)]),
                ("readlink -f /proc/111/cwd", [(FAKE_CFG["repo"], 0)]),
                ('grep -zc "AUTORESEARCH_MANAGED=1" /proc/111/environ', [("1", 0)]),
                ("kill -0 111", [("DEAD\n", 0)]),
            ],
            shared_commands=shared_commands,
        )

    _patch_paramiko(monkeypatch, client_factory)

    killed = remote_runner.kill_stale_training_processes()

    assert len(killed) == 1
    assert killed[0]["pid"] == "111"
    assert killed[0]["escalated_to_sigkill"] is False
    assert any("kill -TERM 111" in c for c in shared_commands)
    assert not any("kill -TERM 222" in c for c in shared_commands)
    assert not any("/proc/222/cwd" in c for c in shared_commands)  # never got past the ownership check


def test_kill_stale_training_processes_skips_process_missing_the_marker(monkeypatch):
    shared_commands = []

    def client_factory():
        return FakeSSHClient(
            responses=[
                ("nvidia-smi --query-compute-apps", [("111\n", 0)]),
                ("ps -o ruser=,args= -p 111", [("fake-user python -u train.py", 0)]),
                ("readlink -f /proc/111/cwd", [(FAKE_CFG["repo"], 0)]),
                ('grep -zc "AUTORESEARCH_MANAGED=1" /proc/111/environ', [("0", 0)]),
            ],
            shared_commands=shared_commands,
        )

    _patch_paramiko(monkeypatch, client_factory)

    assert remote_runner.kill_stale_training_processes() == []
    assert not any("kill -TERM" in c for c in shared_commands)


def test_kill_stale_training_processes_skips_process_with_different_cwd(monkeypatch):
    """Same owner, same 'train.py' in the cmdline, but running from a
    different project entirely -- must not be treated as ours."""
    shared_commands = []

    def client_factory():
        return FakeSSHClient(
            responses=[
                ("nvidia-smi --query-compute-apps", [("111\n", 0)]),
                ("ps -o ruser=,args= -p 111", [("fake-user python -u train.py", 0)]),
                ("readlink -f /proc/111/cwd", [("/home/fake-user/some_other_project", 0)]),
            ],
            shared_commands=shared_commands,
        )

    _patch_paramiko(monkeypatch, client_factory)

    assert remote_runner.kill_stale_training_processes() == []
    assert not any("environ" in c for c in shared_commands)  # never even checked the marker


def test_kill_stale_training_processes_escalates_to_sigkill_when_still_alive(monkeypatch):
    shared_commands = []

    def client_factory():
        return FakeSSHClient(
            responses=[
                ("nvidia-smi --query-compute-apps", [("111\n", 0)]),
                ("ps -o ruser=,args= -p 111", [("fake-user python -u train.py", 0)]),
                ("readlink -f /proc/111/cwd", [(FAKE_CFG["repo"], 0)]),
                ('grep -zc "AUTORESEARCH_MANAGED=1" /proc/111/environ', [("1", 0)]),
                ("kill -0 111", [("ALIVE\n", 0)]),
            ],
            shared_commands=shared_commands,
        )

    _patch_paramiko(monkeypatch, client_factory)

    killed = remote_runner.kill_stale_training_processes()

    assert killed[0]["escalated_to_sigkill"] is True
    assert any("kill -TERM 111" in c for c in shared_commands)
    assert any("kill -KILL 111" in c for c in shared_commands)


def test_kill_stale_training_processes_skips_pid_that_already_exited(monkeypatch):
    """The ps lookup for a PID that's already gone (raced with its own
    natural exit) returns an empty line -- must be skipped, not crash."""
    client = FakeSSHClient(responses=[
        ("nvidia-smi --query-compute-apps", [("111\n", 0)]),
        ("ps -o ruser=,args= -p 111", [("", 0)]),
    ])
    _patch_paramiko(monkeypatch, lambda: client)
    assert remote_runner.kill_stale_training_processes() == []
