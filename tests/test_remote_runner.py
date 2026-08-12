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
SYNC_LOCK_DIR = f"{FAKE_CFG['repo']}/.autoresearch_sync_lock"


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
    def __init__(self, commands=None):
        self.puts = []
        self.written = {}
        self.commands = commands

    def put(self, local, remote):
        self.puts.append((local, remote))

    def mkdir(self, path):
        pass

    def file(self, path, mode="r"):
        """launch_detached writes the detached run's payload to a SCRIPT rather
        than building nested shell quoting -- the version that did nest broke on
        its own inner quotes and hung."""
        sftp = self

        class _Handle:
            def write(self, data):
                sftp.written[path] = data
                # Also recorded as a "command": it IS what gets run on the
                # remote, so assertions about the training invocation (which
                # GPU, which hyperparams file) still find it after the payload
                # moved out of exec_command and into a script.
                if sftp.commands is not None:
                    sftp.commands.append(data)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Handle()

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
        self.sftp = FakeSFTP(commands=self.commands)
        self.closed = False

    def set_missing_host_key_policy(self, policy):
        pass

    def connect(self, **kwargs):
        pass

    def get_transport(self):
        return SimpleNamespace(set_keepalive=lambda seconds: None)

    def exec_command(self, cmd, timeout=None):
        self.commands.append(cmd)

        # TRAINING IS DETACHED NOW: run_training_remote launches with
        # nohup/setsid, gets a PID, and polls a log file, so that a dropped
        # connection costs an observation rather than a run. A test still
        # declares what the run PRINTS -- via the "python -u train.py" matcher
        # -- and this translates that into the three calls the poller actually
        # makes, so the tests keep expressing intent instead of protocol.
        if "nohup setsid" in cmd:
            self._training_delivered = False
            return None, FakeStream("4242\n"), FakeStream("")
        if "tail -n +" in cmd and not any("tail -n +" in m for m, _ in self._queues):
            # fail_tails lets a test drop the connection mid-watch. Under
            # detached runs that is survivable, so it is now a property worth
            # asserting rather than a failure to salvage from.
            drops = getattr(self, "drops", None)
            if drops and drops.get("left", 0) > 0:
                drops["left"] -= 1
                raise socket.timeout("timed out")
            if getattr(self, "_training_delivered", False):
                return None, FakeStream(""), FakeStream("")
            for matcher, queue in self._queues:
                if "train.py" in matcher and queue:
                    entry = queue[0] if len(queue) == 1 else queue.pop(0)
                    out, status = entry[0], entry[1]
                    self._training_delivered = True
                    return None, FakeStream(
                        out + f"{remote_runner.EXIT_SENTINEL}{status}\n"), FakeStream("")
            return None, FakeStream(""), FakeStream("")
        # Only when the test has not declared its own -- kill -0 is also how
        # kill_stale_training_processes checks whether SIGTERM worked.
        if "kill -0" in cmd and not any("kill -0" in m for m, _ in self._queues):
            return None, FakeStream("yes\n"), FakeStream("")

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


# --- detached runs ----------------------------------------------------------
#
# run_training_remote no longer holds the training process on its own SSH
# channel. It launches with nohup/setsid, gets a PID back, and polls a log file
# -- so a dropped connection costs an OBSERVATION rather than a run. These
# helpers speak that protocol: a launch returns a pid, `tail -n +N` returns the
# log from line N, and `kill -0` reports liveness.


def _detached(log_text, exit_code=0, pid="4242", tail_entries=None):
    """Fake responses for one detached run that completes normally."""
    sentinel = f"{remote_runner.EXIT_SENTINEL}{exit_code}\n"
    return [
        ("nohup setsid", [(pid + "\n", 0)]),
        ("tail -n +", tail_entries or [(log_text + sentinel, 0)]),
        ("kill -0", [("yes\n", 0)]),
    ]


def _detached_then_gone(log_text, pid="4242"):
    """A run whose PID disappears without writing an exit status -- what being
    killed from outside looks like, which is how the one-GPU policy is
    enforced."""
    return [
        ("nohup setsid", [(pid + "\n", 0)]),
        ("tail -n +", [(log_text, 0), ("", 0)]),
        ("kill -0", [("", 0)]),
    ]


@pytest.fixture(autouse=True)
def _no_poll_delay(monkeypatch):
    """The detached watcher sleeps between polls. Real seconds in a unit test
    buy nothing -- the fake answers instantly."""
    monkeypatch.setattr(remote_runner, "POLL_INTERVAL_S", 0.0)


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
    # The one-GPU policy cap would truncate this list before the sort could be
    # observed. It is a separate property with its own test
    # (test_gpu_discovery_never_offers_more_than_the_policy_allows), so it is
    # lifted here to keep filtering and ordering testable on their own.
    monkeypatch.setattr(remote_runner, "MAX_CONCURRENT_GPUS", 8)

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
                (f"mkdir {SYNC_LOCK_DIR} 2>&1", [("LOCK_OK\n", 0)]),
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

    # Shared across reconnections, since each one builds a fresh client: the
    # first two polls drop, the third succeeds.
    drops = {"left": 2}

    def client_factory():
        client = FakeSSHClient(responses=[
            (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_1 2>&1", [("LOCK_OK\n", 0)]),
            ("python -u train.py", [(captured_output, 0)]),
        ])
        client.drops = drops
        return client

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")

    metrics = remote_runner.run_training_remote(str(hp_file), timeout=60, gpu_index=1, skip_sync=True)

    # A FULL result, not a salvaged partial. That is the point of detaching:
    # the run is not tied to the connection, so losing the connection costs an
    # observation and the watcher simply resumes from the last line it read.
    assert metrics["status"] == "remote_ok"
    assert metrics["val_bpb"] == pytest.approx(0.987654)
    assert metrics["device"] == 1
    # No salvage fields at all: nothing was lost, so there is nothing to
    # report about a loss.
    assert "connection_lost_line_count" not in metrics


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
            *_detached_then_gone(partial_output),
        ])

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")

    metrics = remote_runner.run_training_remote(str(hp_file), timeout=60, gpu_index=1, skip_sync=True)

    assert metrics["status"] == "remote_error"
    assert metrics["val_bpb"] == float("inf")
    assert metrics["device"] == 1
    # Everything printed before it died is still captured, because the log
    # outlives the process -- and the error names the real cause rather than
    # blaming the configuration.
    assert "5.0%" in metrics["error"] or "killed" in metrics["error"].lower()


def test_run_training_remote_salvage_routes_through_display_when_provided(monkeypatch, tmp_path):
    captured_output = "---\nval_bpb:          1.1\nstatus:           remote_ok\n"
    drops = {"left": 1}

    def client_factory():
        client = FakeSSHClient(responses=[
            (f"mkdir {FAKE_CFG['repo']}/.autoresearch_gpu_locks/gpu_1 2>&1", [("LOCK_OK\n", 0)]),
            ("python -u train.py", [(captured_output, 0)]),
        ])
        client.drops = drops
        return client

    _patch_paramiko(monkeypatch, client_factory)
    hp_file = tmp_path / "hp.yaml"
    hp_file.write_text("n_layer: 8\n")
    display = FakeDisplay()

    metrics = remote_runner.run_training_remote(
        str(hp_file), timeout=60, gpu_index=1, run_label="GPU1", skip_sync=True, display=display,
    )

    # A dropped poll is now survivable, so the display reports RECONNECTING
    # rather than a salvaged partial -- and the run completes normally.
    assert metrics["status"] == "remote_ok"
    assert metrics["val_bpb"] == pytest.approx(1.1)
    assert any("reconnecting" in text for text in display.printed_lines)


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

    def client_factory():
        return FakeSSHClient(
            responses=[(f"mkdir {SYNC_LOCK_DIR} 2>&1", [("LOCK_OK\n", 0)])],
            shared_commands=shared_commands,
        )

    _patch_paramiko(monkeypatch, client_factory)

    remote_runner.sync_remote_code()

    assert any("git stash" in c for c in shared_commands)
    assert any("git pull" in c for c in shared_commands)
    assert any(f"rm -rf {SYNC_LOCK_DIR}" in c for c in shared_commands)  # lock released


def test_sync_remote_code_waits_for_lock_then_proceeds(monkeypatch):
    """A second caller finds the lock held (e.g. a concurrent wave's own
    sync_remote_code call), waits, and proceeds once it's released --
    exactly the race that used to corrupt train.py mid-checkout."""
    shared_commands = []

    def client_factory():
        return FakeSSHClient(
            responses=[
                (f"mkdir {SYNC_LOCK_DIR} 2>&1", [("LOCK_BUSY\n", 0), ("LOCK_OK\n", 0)]),
                (f"stat -c %Y {SYNC_LOCK_DIR}", [(str(int(time.time())), 0)]),  # fresh, not stale
            ],
            shared_commands=shared_commands,
        )

    _patch_paramiko(monkeypatch, client_factory)

    remote_runner.sync_remote_code()

    assert any("git pull" in c for c in shared_commands)


def test_acquire_sync_lock_reclaims_when_stale(monkeypatch):
    stale_mtime = str(int(time.time()) - 100_000)
    client = FakeSSHClient(responses=[
        (f"mkdir {SYNC_LOCK_DIR} 2>&1", [("LOCK_BUSY\n", 0), ("LOCK_OK\n", 0)]),
        (f"stat -c %Y {SYNC_LOCK_DIR}", [(stale_mtime, 0)]),
    ])
    assert remote_runner._acquire_sync_lock(client, FAKE_CFG["repo"], max_wait_seconds=5) is True
    assert any(f"rm -rf {SYNC_LOCK_DIR}" in c for c in client.commands)


def test_acquire_sync_lock_gives_up_after_max_wait(monkeypatch):
    client = FakeSSHClient(responses=[
        (f"mkdir {SYNC_LOCK_DIR} 2>&1", [("LOCK_BUSY\n", 0)]),
        (f"stat -c %Y {SYNC_LOCK_DIR}", [(str(int(time.time())), 0)]),  # fresh -- never reclaimed
    ])
    assert remote_runner._acquire_sync_lock(client, FAKE_CFG["repo"], max_wait_seconds=0.05) is False


def test_release_sync_lock_never_raises_on_failure(monkeypatch):
    class RaisingClient(FakeSSHClient):
        def exec_command(self, cmd, timeout=None):
            raise OSError("connection dropped")

    remote_runner._release_sync_lock(RaisingClient(), FAKE_CFG["repo"])  # must not raise


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
                ("ps -o ruser:32=,args= -p 111", [("fake-user python -u train.py", 0)]),
                ("ps -o ruser:32=,args= -p 222", [("some-other-user some_other_process.py", 0)]),
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
                ("ps -o ruser:32=,args= -p 111", [("fake-user python -u train.py", 0)]),
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
                ("ps -o ruser:32=,args= -p 111", [("fake-user python -u train.py", 0)]),
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
                ("ps -o ruser:32=,args= -p 111", [("fake-user python -u train.py", 0)]),
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
        ("ps -o ruser:32=,args= -p 111", [("", 0)]),
    ])
    _patch_paramiko(monkeypatch, lambda: client)
    assert remote_runner.kill_stale_training_processes() == []


def test_the_log_is_truncated_before_the_run_is_forked(monkeypatch, tmp_path):
    """THE RACE THAT WOULD HAVE MISATTRIBUTED RESULTS. The watcher starts
    tailing within milliseconds, while the detached script does not reach its
    own redirect until conda activation finishes seconds later. In that window
    the file still holds the PREVIOUS run's output -- complete, with its exit
    sentinel -- so the watcher reads it, sees the sentinel, and reports the
    last run's val_bpb as this one's. A sequential campaign reuses one log
    name, so every run after the first was a coin flip on it.

    The truncation therefore has to happen in the FOREGROUND, before the fork,
    in the same command that launches."""
    client = FakeSSHClient()
    _patch_paramiko(monkeypatch, lambda: client)

    remote_runner.launch_detached(client, "python -u train.py", "/tmp/logs/run.log")

    launch = next(c for c in client.commands if "nohup setsid" in c)
    assert ": > /tmp/logs/run.log" in launch, "log not truncated at launch"
    assert launch.index(": > /tmp/logs/run.log") < launch.index("nohup setsid"), (
        "truncation must precede the fork, or the watcher can still read the "
        "previous run's log")


def test_the_whole_detached_script_is_redirected_not_just_its_last_command():
    """`a && b > f` redirects only b. With the redirect on the command chain,
    conda activation and cd wrote nowhere -- and the log was not truncated
    until python itself started, which is what opened the race above."""
    client = FakeSSHClient()
    remote_runner.launch_detached(client, "source activate && cd /r && python -u train.py",
                                  "/tmp/logs/run.log")

    script = next(iter(client.sftp.written.values()))
    assert "exec > /tmp/logs/run.log 2>&1" in script
    assert "python -u train.py > /tmp/logs/run.log" not in script


def test_the_owner_check_survives_a_username_longer_than_eight_characters(monkeypatch):
    """THE BUG THAT MADE THIS FUNCTION A NO-OP. `ps -o ruser=` pads that column
    to 8 characters and truncates the rest with a '+', so the real account
    "up1066590" came back as "up10665+" and never equalled cfg["user"]. Every
    process failed the owner check, so the orphan cleaner has never killed
    anything on this account.

    It stayed invisible while runs died with their SSH session. Detaching them
    surfaced it at once: stopping a campaign left two training processes alive
    on two GPUs, which is itself a breach of the one-GPU policy.
    """
    seen = []

    class _Recorder(FakeSSHClient):
        def exec_command(self, cmd, timeout=None):
            seen.append(cmd)
            return super().exec_command(cmd, timeout=timeout)

    client = _Recorder(responses=[("nvidia-smi", [("4242\n", 0)])])
    monkeypatch.setattr(remote_runner, "_PARAMIKO_AVAILABLE", True)
    monkeypatch.setattr(remote_runner, "_load_cfg",
                        lambda: dict(FAKE_CFG, user="up1066590"))

    remote_runner.kill_stale_training_processes(client=client)

    ps_call = next(c for c in seen if c.startswith("ps -o ruser"))
    assert "ruser:" in ps_call, (
        "ps must be given an explicit column width, or any username longer "
        "than 8 characters is truncated and the owner check can never match")
    width = int(ps_call.split("ruser:")[1].split("=")[0])
    assert width >= len("up1066590")
