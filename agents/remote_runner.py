"""SSH-based remote training runner.

Reads connection details from .env (via python-dotenv) and uses paramiko
to:
  1. Upload the local model_hyperparams.yaml to the remote server.
  2. Execute train.py inside the configured conda environment.
  3. Stream stdout/stderr back and parse the metrics.

Also supports multi-GPU parallel dispatch (see agents/orchestrator.py's
_run_parallel_wave): discover_available_gpus() finds live-free GPUs on the
remote server (no static exclusion list -- purely current
utilization/free-memory thresholds), and run_training_remote() accepts an
explicit gpu_index/hp_remote_name/run_label so multiple concurrent calls
(one per thread, one per GPU) never collide on the same remote file or
device.
"""

import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents import train_output
from agents.live_progress import MultiGpuProgressDisplay

try:
    import paramiko
    _PARAMIKO_AVAILABLE = True
except ImportError:
    _PARAMIKO_AVAILABLE = False

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


# A campaign runs for hours against a shared server over a network that
# occasionally hiccups. A single dropped TCP connect used to be fatal (see
# _connect_with_retry), so every connect goes through a short retry first --
# these are blips measured in seconds, not outages.
CONNECT_ATTEMPTS = 3
CONNECT_BACKOFF_S = 5

# How much of a failed run's stderr to show, taken from the END. See the use
# site: a traceback is the last thing written, a warning is the first, and
# prepare.py's torch.load FutureWarning alone is over 500 characters.
ERR_TAIL_CHARS = 2000

# ONE GPU AT A TIME. This is the shared cluster's stated policy, not a tuning
# knob, and the administrators had already sent two emails about it before a
# four-GPU wave from this project was killed on 2026-08-11.
#
# It is enforced in discover_available_gpus rather than in agents_config.yaml
# because config only governs the orchestrator, while scripts/region_geometry,
# scripts/seed_variance and scripts/size_sweep each dispatch their own waves
# straight off this function.
#
# It also explains an earlier misdiagnosis worth not repeating: the geometry
# experiment lost 13 of 39 cells and that was recorded as a dropped SSH
# session. The signature is identical -- exit code -1, no traceback, several
# runs dying at once while one survives -- because a process killed from
# outside never gets to write anything. Concurrency was the cause both times.
MAX_CONCURRENT_GPUS = 1
GPU_POLICY_REASON = ("the DGX allows each user one GPU at a time "
                     "(HPC notice, 2026-08-11)")

# THE SERVER RATE-LIMITS SSH CONNECTIONS. Measured 2026-08-05 against
# dgx.ceid.upatras.gr: a first TCP connect completes in 1.0s and returns a
# valid SSH banner; the next two time out at 15s; connections are accepted
# again after roughly 60-90s of quiet. It is a SYN-level block (WinError
# 10060, TCP handshake never completes), so it is a firewall/rate limiter in
# front of sshd rather than anything sshd itself reports.
#
# That is fatal to the obvious implementation of a multi-GPU wave, which used
# to open SEVEN connections within a few seconds: a campaign-start stale
# check, a per-wave stale check, GPU discovery, the code sync, and then one
# per concurrent training run. The first succeeded and the rest were dropped,
# so every wave lost most of its slots to `remote_error` -- two full 4-slot
# waves produced nothing but val_bpb=inf rows.
#
# The fix is to stop opening connections, not to retry harder into a closed
# window: one paramiko SSHClient carries many independent channels over a
# single TCP connection, so a whole wave now shares one. open_client() below
# creates it and every function here accepts it. Retrying was never going to
# work -- CONNECT_ATTEMPTS x CONNECT_BACKOFF_S spans ~90s, about the width of
# the block itself, and each retry restarts it.
CONNECT_BACKOFF_MULTIPLIER = 3.0


def open_client(cfg: Optional[Dict[str, Any]] = None, timeout: int = 30):
    """One connected, keepalive'd SSHClient the caller owns and must close.

    Pass it to discover_available_gpus / sync_remote_code /
    kill_stale_training_processes / run_training_remote so an entire wave
    costs ONE connection instead of one per call -- see the rate-limit note
    above for why that is not merely an optimization.
    """
    if not _PARAMIKO_AVAILABLE:
        raise RuntimeError("paramiko is not installed. Run: pip install paramiko python-dotenv")
    cfg = cfg or _load_cfg()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        _connect_with_retry(client, cfg, timeout=timeout)
    except Exception:
        # Close the half-built client before re-raising. Callers that let
        # this propagate never receive it and so cannot close it themselves,
        # and a campaign retries this every wave for hours.
        client.close()
        raise
    # A wave holds this open across several minutes of training; without a
    # keepalive an idle-connection timeout on the network path (NAT gateway
    # or firewall) silently drops it mid-run.
    client.get_transport().set_keepalive(30)
    return client


def _connect_with_retry(client, cfg: Dict[str, Any], timeout: int = 30,
                        attempts: int = CONNECT_ATTEMPTS,
                        backoff_s: float = CONNECT_BACKOFF_S) -> None:
    """client.connect(...) with a few retries on transient network failures.

    Exists because a real 150-iteration campaign died on one
    `TimeoutError: [WinError 10060]` from sync_remote_code, moments after two
    other SSH calls to the same host had succeeded -- i.e. a momentary blip,
    not an unreachable server. Re-raises the last exception once the attempts
    are spent, so a genuinely-down host still surfaces as an error rather
    than being retried forever.
    """
    last = None
    for attempt in range(1, attempts + 1):
        try:
            client.connect(
                hostname=cfg["host"], port=cfg["port"], username=cfg["user"],
                password=cfg["password"] or None, timeout=timeout,
            )
            return
        except (socket.timeout, socket.error, OSError, EOFError) as e:
            last = e
            if attempt < attempts:
                print(f"[RemoteRunner] Connection attempt {attempt}/{attempts} failed "
                      f"({type(e).__name__}: {e}) -- retrying in {backoff_s:.0f}s")
                time.sleep(backoff_s)
                # Exponential, not flat: the failure this actually hits is a
                # rate-limit window ~60-90s wide (see the note by
                # CONNECT_BACKOFF_MULTIPLIER), and three flat 5s retries all
                # land inside it. 5 -> 15 -> 45 spans it instead.
                backoff_s *= CONNECT_BACKOFF_MULTIPLIER
    raise last


def _load_cfg() -> Dict[str, Any]:
    """Load remote connection settings from .env (if present)."""
    if _DOTENV_AVAILABLE:
        # Search for .env from the repo root upward
        env_path = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)

    return {
        "host": os.getenv("REMOTE_HOST", ""),
        "port": int(os.getenv("REMOTE_PORT", "22")),
        "user": os.getenv("REMOTE_USER", ""),
        "password": os.getenv("REMOTE_PASSWORD", ""),
        "repo": os.getenv("REMOTE_REPO", ""),
        "conda_env": os.getenv("REMOTE_CONDA_ENV", "autoresearch"),
        "conda_activate": os.getenv(
            "REMOTE_CONDA_ACTIVATE", "source /opt/anaconda3/bin/activate"
        ),
    }


def is_remote_configured() -> bool:
    """Return True when the minimum required .env fields are filled in."""
    cfg = _load_cfg()
    return bool(cfg["host"] and cfg["user"] and cfg["repo"] and cfg["password"])


def discover_available_gpus(cfg: Optional[Dict[str, Any]] = None, client=None) -> List[Dict[str, Any]]:
    """Query the remote server's GPUs live via nvidia-smi and return the
    ones currently free enough to use. No static exclusion list -- purely
    driven by freshly-queried utilization/free-memory against
    REMOTE_GPU_UTIL_THRESHOLD_PCT (default 10) / REMOTE_GPU_MIN_FREE_MB
    (default 8000), so a GPU that's busy today but free tomorrow is used
    tomorrow without any code change. Returns [] on any connection/parse
    failure -- callers degrade to the single-GPU fallback, never crash.
    """
    if not _PARAMIKO_AVAILABLE:
        return []
    cfg = cfg or _load_cfg()
    if not (cfg["host"] and cfg["user"]):
        return []

    util_threshold = float(os.getenv("REMOTE_GPU_UTIL_THRESHOLD_PCT", "10"))
    min_free_mb = float(os.getenv("REMOTE_GPU_MIN_FREE_MB", "8000"))

    # An explicit client is reused and left open for its owner; None means
    # this call opens and closes its own. See CONNECT_BACKOFF_MULTIPLIER:
    # the server rate-limits connects, so callers doing several of these in
    # quick succession must share one.
    owns_client = client is None
    # The connect itself must be INSIDE the try: this function's contract is
    # "[] on any connection/parse failure, never raise" -- callers degrade to
    # the single-GPU fallback on the strength of that.
    try:
        if owns_client:
            client = open_client(cfg)
        _, stdout, _ = client.exec_command(
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
            "--format=csv,noheader,nounits",
            timeout=30,
        )
        output = stdout.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[RemoteRunner] GPU discovery failed: {e}")
        return []
    finally:
        if owns_client and client is not None:
            client.close()

    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
            mem_used, mem_total, util = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        gpus.append({
            "index": index, "mem_used_mb": mem_used, "mem_total_mb": mem_total,
            "util_pct": util, "free_mb": mem_total - mem_used,
        })

    available = [g for g in gpus if g["util_pct"] < util_threshold and g["free_mb"] > min_free_mb]
    available.sort(key=lambda g: -g["free_mb"])
    print(f"[RemoteRunner] GPU discovery: {len(gpus)} total, {len(available)} available "
          f"(util<{util_threshold:.0f}%, free>{min_free_mb:.0f}MB): {[g['index'] for g in available]}")

    # THE POLICY CAP IS APPLIED HERE, at the one place every dispatcher asks
    # what it may use -- the orchestrator's wave planner and each of the
    # scripts/ experiments. Capping in agents_config.yaml instead would leave
    # the scripts free to take four GPUs each, and they have.
    if len(available) > MAX_CONCURRENT_GPUS:
        print(f"[RemoteRunner] Capping to {MAX_CONCURRENT_GPUS} GPU(s): {GPU_POLICY_REASON}")
        available = available[:MAX_CONCURRENT_GPUS]
    return available


def _acquire_gpu_lock(
    client, gpu_index: int, remote_repo: str, stale_after_seconds: float,
    display: Optional[MultiGpuProgressDisplay] = None, label: str = "",
) -> bool:
    """mkdir is atomic over SSH (POSIX guarantee: fails if the dir already
    exists), so this is a real mutex with no extra dependency. A lock older
    than stale_after_seconds is assumed to belong to a crashed process and
    is reclaimed -- otherwise a single dead run would permanently strand a
    GPU. Best-effort: caller treats a failed acquire as "unavailable this
    round," never blocks waiting for it.

    display/label: when this runs as part of a concurrent multi-GPU wave,
    its one log line must go through the shared live display too, or it
    can clobber the live-updating progress table (see agents/live_progress.py).
    """
    lock_dir = f"{remote_repo}/.autoresearch_gpu_locks/gpu_{gpu_index}"
    parent = f"{remote_repo}/.autoresearch_gpu_locks"
    client.exec_command(f'mkdir -p {parent}')[1].read()

    def _try_mkdir() -> bool:
        _, stdout, _ = client.exec_command(f'mkdir {lock_dir} 2>&1 && echo LOCK_OK || echo LOCK_BUSY')
        result = stdout.read().decode("utf-8", errors="replace").strip()
        if result.endswith("LOCK_OK"):
            client.exec_command(f'echo "$(hostname) $(date +%s)" > {lock_dir}/owner')[1].read()
            return True
        return False

    if _try_mkdir():
        return True

    _, stdout, _ = client.exec_command(f'stat -c %Y {lock_dir} 2>/dev/null')
    mtime_str = stdout.read().decode("utf-8", errors="replace").strip()
    try:
        age = time.time() - float(mtime_str)
    except ValueError:
        age = 0.0
    if age > stale_after_seconds:
        message = f"GPU {gpu_index} lock stale ({age:.0f}s old) -- reclaiming"
        if display is not None:
            display.print_line(f"[{label}] {message}")
        else:
            print(f"[RemoteRunner] {message}")
        client.exec_command(f'rm -rf {lock_dir}')[1].read()
        return _try_mkdir()

    return False


def _release_gpu_lock(client, gpu_index: int, remote_repo: str) -> None:
    lock_dir = f"{remote_repo}/.autoresearch_gpu_locks/gpu_{gpu_index}"
    try:
        client.exec_command(f'rm -rf {lock_dir}')[1].read()
    except Exception:
        pass


#: Written to the log by the wrapper after train.py returns, so completion and
#: exit status are recoverable from the FILE. A detached run's exit code cannot
#: be read from the channel that started it -- that channel is long gone.
EXIT_SENTINEL = "AUTORESEARCH_EXIT:"

#: How long to wait between polls of a detached run's log.
POLL_INTERVAL_S = 5.0

#: Consecutive failed reconnections before a detached run is given up on. The
#: run itself survives a dropped connection now, so this only bounds how long
#: we keep trying to watch it.
MAX_POLL_RECONNECTS = 20


def launch_detached(client, remote_cmd: str, log_path: str) -> Optional[str]:
    """Start `remote_cmd` so it OUTLIVES this SSH session, and return its PID.

    `setsid` puts it in a new session with no controlling terminal and `nohup`
    detaches it from SIGHUP, so dropping the connection no longer kills the
    run. That mattered less when four ran at once and a lost one was cheap; at
    one GPU per user a dropped connection costs a whole 4.4-minute run, and the
    campaign has already lost 26 runs to exactly this.

    stdin comes from /dev/null and both streams go to `log_path`, because a
    detached process holding the session's pipes open would keep the channel
    from ever closing. The wrapper appends the exit status afterwards: a
    detached run's code cannot be read back from the channel that started it.
    """
    # THE PAYLOAD GOES IN A FILE, not into nested shell quoting. The first
    # version built `bash -lc "... bash -c '... echo "SENTINEL$?" ...'"`, whose
    # inner double quotes terminate the outer ones -- the command was malformed,
    # so the launch never printed a PID and the read blocked until it timed out.
    # Agent 1 then fell back to a SIMULATED result, which is a fabricated
    # val_bpb, and the campaign recorded it as a new best.
    #
    # A script file has no nesting to get wrong, and it is inspectable on the
    # remote afterwards, which the one-shot command never was.
    script_path = f"{log_path}.sh"
    script = (
        "#!/bin/bash\n"
        f"mkdir -p $(dirname {log_path})\n"
        f"{remote_cmd} > {log_path} 2>&1\n"
        f"echo {EXIT_SENTINEL}$? >> {log_path}\n"
    )
    sftp = client.open_sftp()
    try:
        sftp.mkdir(str(Path(log_path).parent).replace("\\", "/"))
    except OSError:
        pass                                   # already there
    try:
        with sftp.file(script_path, "w") as handle:
            handle.write(script)
    finally:
        sftp.close()

    # `bash -l` so the script still runs as a login shell: remote_cmd begins
    # with `source .../activate`, which needs the same environment the old
    # `bash -lc` gave it.
    _, stdout, _ = client.exec_command(
        f"nohup setsid bash -l {script_path} < /dev/null > /dev/null 2>&1 & echo $!",
        timeout=60)
    pid = stdout.read().decode("utf-8", errors="replace").strip().splitlines()
    return pid[-1].strip() if pid and pid[-1].strip().isdigit() else None


def _read_log_tail(client, log_path: str, from_line: int) -> List[str]:
    """Lines `from_line` onward (1-indexed), or [] if the file is not there
    yet -- a launch that has not created it is normal for the first poll."""
    _, stdout, _ = client.exec_command(
        f"tail -n +{from_line} {log_path} 2>/dev/null", timeout=60)
    text = stdout.read().decode("utf-8", errors="replace")
    return text.splitlines()


def _process_alive(client, pid: str) -> bool:
    """`kill -0` signals nothing; it only asks whether the PID can be signalled.

    Needed because the sentinel is written by the wrapper, and a run killed with
    SIGKILL -- which is how the one-GPU policy is enforced -- never gets to
    write it. Without this check such a run would be polled until the
    reconnect budget ran out.
    """
    _, stdout, _ = client.exec_command(f"kill -0 {pid} 2>/dev/null && echo yes", timeout=60)
    return stdout.read().decode("utf-8", errors="replace").strip() == "yes"


def follow_detached(client, cfg: Dict[str, Any], log_path: str, pid: str,
                    timeout: int, on_line, reconnect=None) -> Tuple[List[str], Optional[int], str]:
    """Watch a detached run to completion. Returns (lines, exit_code, outcome).

    `outcome` is one of "finished", "vanished", "timeout" or "unwatchable".
    Only the last means the RUN is in doubt -- the others all saw it end.

    A DROPPED CONNECTION IS NO LONGER FATAL, which is the whole point. The run
    is detached, so a failed poll is only a failed observation: this reconnects
    and resumes from the last line it read, and the log on the remote is the
    authority rather than a live pipe. Reading from `from_line` onward is what
    makes that safe -- no line is shown twice and none is skipped.

    "vanished" means the PID is gone with no exit sentinel, which is what an
    externally killed run looks like. That is not a hypothetical here: it is
    how the DGX's one-GPU-per-user policy is enforced.
    """
    lines: List[str] = []
    next_line = 1
    deadline = time.time() + timeout
    reconnects = 0
    reconnect = reconnect or (lambda: open_client(cfg))

    while True:
        try:
            new_lines = _read_log_tail(client, log_path, next_line)
            alive = _process_alive(client, pid) if not new_lines else True
            reconnects = 0
        except Exception as e:                       # noqa: BLE001 - any SSH fault
            reconnects += 1
            if reconnects > MAX_POLL_RECONNECTS:
                return lines, None, "unwatchable"
            on_line(f"[watcher] lost the connection ({e}); reconnecting "
                    f"({reconnects}/{MAX_POLL_RECONNECTS}) -- the run itself is "
                    f"detached and keeps going")
            time.sleep(POLL_INTERVAL_S)
            try:
                client = reconnect()
            except Exception:                        # noqa: BLE001
                pass
            continue

        for line in new_lines:
            if line.startswith(EXIT_SENTINEL):
                try:
                    return lines, int(line[len(EXIT_SENTINEL):].strip()), "finished"
                except ValueError:
                    return lines, None, "finished"
            lines.append(line)
            on_line(line)
        next_line += len(new_lines)

        if not new_lines:
            # Re-read once before believing it: the process can exit between
            # the tail and the liveness check, leaving its last lines -- and
            # the sentinel -- unread.
            if not alive:
                final = _read_log_tail(client, log_path, next_line)
                for line in final:
                    if line.startswith(EXIT_SENTINEL):
                        try:
                            return lines, int(line[len(EXIT_SENTINEL):].strip()), "finished"
                        except ValueError:
                            return lines, None, "finished"
                    lines.append(line)
                    on_line(line)
                return lines, None, "vanished"
            if time.time() > deadline:
                return lines, None, "timeout"
            time.sleep(POLL_INTERVAL_S)


def kill_stale_training_processes(cfg: Optional[Dict[str, Any]] = None, timeout: int = 30,
                                  client=None) -> List[Dict[str, Any]]:
    """Find and clean up any leftover train.py process from a previous,
    not-cleanly-stopped run of this project on the remote server, before a
    new campaign starts. Called once at Orchestrator startup, never
    per-iteration -- nothing new can go stale mid-campaign since this
    orchestrator process is the only thing dispatching trainings.

    A process is only ever touched when ALL of these hold:
      1. It's currently attached to a GPU (nvidia-smi --query-compute-apps).
      2. Its Linux owner is exactly the configured remote SSH user (`ps
         ruser`) -- and this isn't just an application-level filter: a
         non-root `kill` is refused by the kernel for any PID you don't
         own, so another user's process can never actually be signalled
         here even if this check had a bug.
      3. Its command line contains "train.py" (`ps args`).
      4. Its cwd is exactly this project's remote repo
         (`readlink /proc/<pid>/cwd`) -- rules out an unrelated train.py
         elsewhere on the same account.
      5. It carries the AUTORESEARCH_MANAGED=1 marker this project sets on
         every train.py invocation it launches (`/proc/<pid>/environ`) --
         the most specific signal; a coincidental match on 1-4 alone still
         wouldn't be enough.

    Sends SIGTERM first (lets it release its CUDA context cleanly), and
    only escalates to SIGKILL if it's still alive a few seconds later.
    Returns [] on any SSH/parse failure or when nothing matches all five
    checks -- never guesses, never touches anything ambiguous.
    """
    if not _PARAMIKO_AVAILABLE:
        return []
    cfg = cfg or _load_cfg()
    if not (cfg["host"] and cfg["user"] and cfg["repo"]):
        return []
    remote_repo = cfg["repo"].rstrip("/")

    owns_client = client is None
    try:
        if owns_client:
            client = open_client(cfg)
        _, stdout, _ = client.exec_command(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits", timeout=timeout,
        )
        gpu_pids = [p.strip() for p in stdout.read().decode("utf-8", errors="replace").splitlines() if p.strip().isdigit()]
        if not gpu_pids:
            return []

        killed: List[Dict[str, Any]] = []
        for pid in gpu_pids:
            _, stdout, _ = client.exec_command(f'ps -o ruser=,args= -p {pid} 2>/dev/null', timeout=timeout)
            ps_line = stdout.read().decode("utf-8", errors="replace").strip()
            if not ps_line:
                continue  # PID already gone, or not visible to us (not ours)
            parts = ps_line.split(None, 1)
            if len(parts) != 2:
                continue
            ruser, args = parts
            if ruser != cfg["user"] or "train.py" not in args:
                continue

            _, stdout, _ = client.exec_command(f'readlink -f /proc/{pid}/cwd 2>/dev/null', timeout=timeout)
            cwd = stdout.read().decode("utf-8", errors="replace").strip()
            if cwd != remote_repo:
                continue

            _, stdout, _ = client.exec_command(
                f'grep -zc "AUTORESEARCH_MANAGED=1" /proc/{pid}/environ 2>/dev/null', timeout=timeout,
            )
            marker_count = stdout.read().decode("utf-8", errors="replace").strip()
            if marker_count != "1":
                continue

            # All 5 checks passed -- unambiguously our own leftover process.
            client.exec_command(f'kill -TERM {pid} 2>/dev/null', timeout=timeout)
            time.sleep(3)
            _, stdout, _ = client.exec_command(f'kill -0 {pid} 2>/dev/null && echo ALIVE || echo DEAD', timeout=timeout)
            still_alive = stdout.read().decode("utf-8", errors="replace").strip() == "ALIVE"
            if still_alive:
                client.exec_command(f'kill -KILL {pid} 2>/dev/null', timeout=timeout)

            killed.append({"pid": pid, "cmd": args.strip(), "escalated_to_sigkill": still_alive})

        return killed
    except Exception as e:
        print(f"[RemoteRunner] Stale-process check failed: {e}")
        return []
    finally:
        if owns_client and client is not None:
            client.close()


def _acquire_sync_lock(client, remote_repo: str, max_wait_seconds: float = 60.0,
                        stale_after_seconds: float = 120.0) -> bool:
    """mkdir-based lock (same atomicity as _acquire_gpu_lock) around
    sync_remote_code's git stash/pull -- confirmed necessary the hard way:
    two uncoordinated callers git-pulling the same working tree at once
    (a parallel wave's sync racing a totally separate process's own
    sync_remote_code call, e.g. an unmocked test run) can leave train.py
    torn mid-write (a real IndentationError at one exact line) or briefly
    reading an old checked-out commit (a real run against code from before
    a since-merged refactor). Unlike a GPU lock, this is worth actually
    waiting for -- a git pull normally finishes in seconds -- so it polls
    briefly instead of giving up immediately.
    """
    lock_dir = f"{remote_repo}/.autoresearch_sync_lock"

    def _try_mkdir() -> bool:
        _, stdout, _ = client.exec_command(f'mkdir {lock_dir} 2>&1 && echo LOCK_OK || echo LOCK_BUSY')
        return stdout.read().decode("utf-8", errors="replace").strip().endswith("LOCK_OK")

    deadline = time.time() + max_wait_seconds
    while True:
        if _try_mkdir():
            return True

        _, stdout, _ = client.exec_command(f'stat -c %Y {lock_dir} 2>/dev/null')
        mtime_str = stdout.read().decode("utf-8", errors="replace").strip()
        try:
            age = time.time() - float(mtime_str)
        except ValueError:
            age = 0.0
        if age > stale_after_seconds:
            print(f"[RemoteRunner] Sync lock stale ({age:.0f}s old) -- reclaiming")
            client.exec_command(f'rm -rf {lock_dir}')[1].read()
            if _try_mkdir():
                return True

        if time.time() >= deadline:
            print("[RemoteRunner] Could not acquire sync lock in time -- proceeding without it (best effort)")
            return False
        time.sleep(2)


def _release_sync_lock(client, remote_repo: str) -> None:
    lock_dir = f"{remote_repo}/.autoresearch_sync_lock"
    try:
        client.exec_command(f'rm -rf {lock_dir}')[1].read()
    except Exception:
        pass


def sync_remote_code(cfg: Optional[Dict[str, Any]] = None, client=None) -> bool:
    """git stash + pull --ff-only on the remote clone, guarded by a remote
    mkdir-based lock (see _acquire_sync_lock) so two uncoordinated callers
    -- a parallel wave, a single sequential run, or an entirely separate
    process -- never race on the same working tree.

    Returns True when the remote tree was synced, False on any connection or
    command failure. This used to raise, and was the ONE connect site in this
    module that did -- discover_available_gpus and run_training_remote both
    already caught everything and degraded ("callers degrade to the
    single-GPU fallback, never crash"). That inconsistency killed a real
    150-iteration campaign on a single transient TimeoutError. Callers must
    still decide what a False means for them: it is emphatically NOT
    "carry on and train anyway", since an unreachable server cannot run
    anything.
    """
    if not _PARAMIKO_AVAILABLE:
        return False
    cfg = cfg or _load_cfg()
    remote_repo = cfg["repo"]
    owns_client = client is None
    # Inside the try for the same reason as discover_available_gpus: this
    # returns False on an unreachable host rather than raising.
    try:
        if owns_client:
            client = open_client(cfg)
        lock_acquired = _acquire_sync_lock(client, remote_repo)
        try:  # noqa: SIM105 -- inner lock release, see outer handler below
            print("[RemoteRunner] Pulling latest code on remote ...")
            stash_cmd = f'bash -lc "cd {remote_repo} && git stash 2>&1"'
            _, stash_o, _ = client.exec_command(stash_cmd)
            stash_out = stash_o.read().decode("utf-8", errors="replace").strip()
            if stash_out and "No local changes" not in stash_out:
                print(f"[RemoteRunner] git stash: {stash_out[:100]}")

            _, o, e = client.exec_command(
                f'bash -lc "cd {remote_repo} && git pull --ff-only 2>&1"'
            )
            pull_out = o.read().decode("utf-8", errors="replace").strip()
            print(f"[RemoteRunner] git pull: {pull_out[:120]}")
        finally:
            if lock_acquired:
                _release_sync_lock(client, remote_repo)
        return True
    except Exception as e:
        print(f"[RemoteRunner] Remote code sync failed: {type(e).__name__}: {e}")
        return False
    finally:
        if owns_client and client is not None:
            client.close()


def run_training_remote(
    hyperparams_local_path: str,
    timeout: int = 600,
    gpu_index: Optional[int] = None,
    hp_remote_name: str = "model_hyperparams.yaml",
    run_label: Optional[str] = None,
    skip_sync: bool = False,
    display: Optional[MultiGpuProgressDisplay] = None,
    client=None,
) -> Dict[str, Any]:
    """
    Upload hyperparams YAML to the remote server and run train.py there.

    display: when this call is one of several running concurrently (the
    multi-GPU wave dispatcher in agents/orchestrator.py), pass a shared
    MultiGpuProgressDisplay so this GPU's progress-bar lines land on their
    own pinned terminal line instead of every thread's old `\\r`-based
    update fighting over the same cursor position (see
    agents/live_progress.py). None (the default, single-GPU sequential
    path) keeps today's exact `\\r`-based in-place update, unchanged.

    gpu_index=None (the default) auto-discovers the best live-available GPU
    via discover_available_gpus(), falling back to REMOTE_DEFAULT_GPU (or 4,
    matching the old hardcoded behavior) if discovery is unavailable --
    single-run callers now automatically pick a genuinely free GPU instead
    of always targeting a fixed index. Pass an explicit gpu_index to skip
    discovery entirely (used by the parallel wave dispatcher, which already
    claimed a specific GPU).

    hp_remote_name lets concurrent callers upload to distinct remote
    filenames instead of racing on one shared model_hyperparams.yaml.

    run_label (default f"GPU{gpu_index}") prefixes every streamed stdout
    line so concurrent runs stay distinguishable in an interleaved
    terminal -- each print() call is one line, a single GIL-held syscall,
    so lines from different threads never splice mid-line.

    skip_sync=True skips the git stash/pull step -- set by the wave
    dispatcher, which already synced once for the whole wave via
    sync_remote_code() (concurrent syncs against one shared clone would
    themselves race).

    Steps:
      1. Connect via SSH.
      2. (unless skip_sync) Pull latest code on remote (git pull).
      3. Upload local hyperparams YAML via SFTP.
      4. Acquire this GPU's lock, execute train.py inside the conda env.
      5. Stream output back, parse metrics, release the lock.

    If the SSH connection is lost while reading output (most often during
    the head-ablation study's silent stretch, well after train.py's own
    time-budget-driven training loop already printed a real val_bpb --
    see train.py), whatever was already captured is parsed and, if it
    contains a usable val_bpb, returned with status "remote_partial_timeout"
    instead of being discarded as an error -- post-training analysis fields
    (head_ablation_impacts, etc.) may be absent from a partial result the
    same way they're absent whenever that analysis didn't run at all.

    Returns:
        Metrics dict with at least {"val_bpb": float, "status": str,
        "device": gpu_index}. status is "remote_ok" on a full completion,
        "remote_partial_timeout" on a salvaged partial result, or
        "remote_error" when nothing usable was recovered. On either of the
        latter two, also includes "connection_lost_after_seconds" (elapsed
        wall-clock time from exec_command to the dropped connection) and
        "connection_lost_line_count" (how many stdout lines were received
        first) -- diagnostic breadcrumbs for telling "died near connect"
        apart from "died mid/late in a real run" without guessing.
        "remote_error" additionally includes "connection_lost_last_progress"
        (the last progress-bar line seen, or None if training never even
        printed one).
    """
    if not _PARAMIKO_AVAILABLE:
        raise RuntimeError(
            "paramiko is not installed. "
            "Run: pip install paramiko python-dotenv"
        )

    cfg = _load_cfg()
    remote_repo = cfg["repo"]
    activate = cfg["conda_activate"]
    env = cfg["conda_env"]

    if gpu_index is None:
        candidates = discover_available_gpus(cfg)
        if candidates:
            gpu_index = candidates[0]["index"]
            print(f"[RemoteRunner] Auto-selected GPU {gpu_index} "
                  f"(free={candidates[0]['free_mb']:.0f}MB, util={candidates[0]['util_pct']:.0f}%)")
        else:
            gpu_index = int(os.getenv("REMOTE_DEFAULT_GPU", "4"))
            print(f"[RemoteRunner] GPU discovery unavailable -- falling back to GPU {gpu_index}")

    label = run_label or f"GPU{gpu_index}"
    remote_hyperparams = f"{remote_repo}/{hp_remote_name}"
    stale_after = float(os.getenv("REMOTE_GPU_LOCK_STALE_SECONDS", str(timeout + 600)))

    def _print(message: str) -> None:
        if display is not None:
            display.print_line(f"[{label}] {message}")
        else:
            print(f"[{label}] {message}")

    if not skip_sync:
        sync_remote_code(cfg)

    # This used to call client.connect() raw -- no retry at all, unlike every
    # other connect in this module. Under the server's connect rate limit that
    # meant a wave's concurrent slots failed instantly and permanently. They
    # now share the caller's connection when given one, and otherwise go
    # through open_client's retry.
    owns_client = client is None
    lock_acquired = False

    try:
        if owns_client:
            _print(f"Connecting to {cfg['user']}@{cfg['host']}:{cfg['port']} ...")
            client = open_client(cfg)
        # A training session can hold this connection open for several
        # minutes; without a keepalive, an idle-connection timeout anywhere
        # between here and the remote server (a NAT gateway or firewall on
        # the network path is the usual culprit) can silently drop it
        # before any output ever comes back -- exactly the failure mode
        # behind a large fraction of this project's real remote_error runs.
        client.get_transport().set_keepalive(30)

        lock_acquired = _acquire_gpu_lock(client, gpu_index, remote_repo, stale_after, display=display, label=label)
        if not lock_acquired:
            _print(f"GPU {gpu_index}'s lock is held by another run -- proceeding anyway "
                   f"(best-effort lock, not a hard reservation)")

        # --- Upload hyperparameters ---
        _print(f"Uploading hyperparams -> {remote_hyperparams}")
        sftp = client.open_sftp()
        sftp.put(hyperparams_local_path, remote_hyperparams)
        sftp.close()

        # --- Run training on the selected GPU ---
        inner_cmd = (
            f'{activate} {env} && cd {remote_repo} && '
            f'CUDA_VISIBLE_DEVICES={gpu_index} AUTORESEARCH_HP_PATH={remote_hyperparams} '
            # AUTORESEARCH_MANAGED=1 marks this process as one this project
            # launched -- kill_stale_training_processes() requires it (along
            # with owner/cmdline/cwd matches) before ever signalling a PID,
            # so a leftover process from a previous run can be identified
            # unambiguously and other users'/projects' processes never can.
            f'AUTORESEARCH_MANAGED=1 python -u train.py'
        )
        remote_cmd = inner_cmd
        _print(f"Executing on GPU {gpu_index}: {remote_cmd}")
        exec_start = time.time()

        # DETACHED, so a dropped connection costs an observation rather than a
        # run. Under the one-GPU-per-user limit each run is 4.4 minutes of the
        # only slot there is, and this project has already lost 26 runs to
        # session-bound training.
        log_path = f"{remote_repo}/logs/{hp_remote_name}.log"
        pid = launch_detached(client, inner_cmd, log_path)
        if not pid:
            _print("Could not start a detached run (no PID returned)")
            return {"val_bpb": float("inf"), "status": "remote_error",
                    "error": "detached launch returned no pid", "device": gpu_index}
        _print(f"Detached as PID {pid}, log {log_path}")

        output_lines = []
        last_progress_bar = None
        last_progress_line = None
        lost_connection = False
        connection_lost_elapsed = None

        def _emit(line: str) -> None:
            """One line of the run's log, routed exactly as the live stream
            used to be so the multi-GPU display and the sequential progress bar
            both behave unchanged."""
            nonlocal last_progress_bar, last_progress_line
            is_progress = line.startswith('[') and ']' in line and '%' in line
            if is_progress:
                last_progress_line = line
            if display is not None:
                if is_progress:
                    display.update_progress(label, line)
                elif line.strip():
                    display.print_line(f"[{label}] {line}")
            else:
                if is_progress:
                    last_progress_bar = line
                    print(f"\r  [{label}] {line}", end="", flush=True)
                elif line.strip():
                    if last_progress_bar:
                        print()
                        last_progress_bar = None
                    print(f"  [{label}] {line}", flush=True)

        output_lines, exit_code, outcome = follow_detached(
            client, cfg, log_path, pid, timeout=timeout, on_line=_emit)

        if outcome == "unwatchable":
            # The RUN may well be fine -- it is detached. What failed is our
            # ability to watch it, so say that rather than blaming the config.
            lost_connection = True
            connection_lost_elapsed = round(time.time() - exec_start, 1)
            _print(f"Gave up watching after {MAX_POLL_RECONNECTS} failed reconnections "
                   f"({connection_lost_elapsed}s, {len(output_lines)} line(s) read). The run "
                   f"is detached and may still be going -- its log is {log_path}")
        elif outcome == "timeout":
            lost_connection = True
            connection_lost_elapsed = round(time.time() - exec_start, 1)
            _print(f"Still running after the {timeout}s watch budget "
                   f"({len(output_lines)} line(s) read); log is {log_path}")
        elif outcome == "vanished":
            # No exit sentinel and the PID is gone: killed from outside. That
            # is not hypothetical here -- it is how the one-GPU-per-user policy
            # is enforced, and it is why nothing useful ever reached stderr.
            _print(f"Process {pid} disappeared without writing an exit status -- "
                   f"killed from outside. {GPU_POLICY_REASON}")
            exit_code = exit_code if exit_code is not None else -1

        if display is None and last_progress_bar:
            print()  # final newline after last progress bar

        if lost_connection:
            partial_metrics = _parse_output("\n".join(output_lines))
            if math.isfinite(partial_metrics.get("val_bpb", float("inf"))):
                partial_metrics["status"] = "remote_partial_timeout"
                partial_metrics["device"] = gpu_index
                partial_metrics["connection_lost_after_seconds"] = connection_lost_elapsed
                partial_metrics["connection_lost_line_count"] = len(output_lines)
                _print(f"Recovered a usable val_bpb before the connection was lost -- treating as "
                       f"a partial success (post-training analysis, e.g. head ablation, may be "
                       f"incomplete or missing): {partial_metrics}")
                return partial_metrics
            _print("No usable val_bpb was captured before the connection was lost -- treating as a failure")
            return {
                "val_bpb": float("inf"),
                "error": "connection lost while reading output, before any usable result was captured",
                "status": "remote_error",
                "device": gpu_index,
                "connection_lost_after_seconds": connection_lost_elapsed,
                "connection_lost_line_count": len(output_lines),
                "connection_lost_last_progress": last_progress_line,
            }

        # Both streams were merged into the log by the detached wrapper, so
        # the tail of what we read IS the stderr that used to be separate.
        err_output = "\n".join(output_lines[-40:]).strip()

        if exit_code:
            _print(f"Remote process exited with code {exit_code}")
            if err_output:
                # THE TAIL, NOT THE HEAD. A Python traceback is the last thing
                # written to stderr, while the first thing is routinely a
                # warning -- prepare.py's torch.load FutureWarning alone is
                # over 500 characters. Truncating from the front therefore
                # showed that warning and nothing else on every failure, which
                # is exactly as useful as printing nothing. A whole campaign's
                # worth of failures was undiagnosable because of it.
                _print(f"stderr ({len(err_output)} chars, last {ERR_TAIL_CHARS}): "
                       f"{err_output[-ERR_TAIL_CHARS:]}")
            return {
                "val_bpb": float("inf"),
                "error": err_output or f"exit code {exit_code}",
                "status": "remote_error",
                "device": gpu_index,
            }

        metrics = _parse_output("\n".join(output_lines))
        metrics["device"] = gpu_index
        _print(f"Parsed metrics: {metrics}")
        return metrics

    finally:
        if lock_acquired:
            _release_gpu_lock(client, gpu_index, remote_repo)
        if owns_client and client is not None:
            client.close()


# Mapping from train.py output key -> metrics dict key
def _parse_output(stdout: str) -> Dict[str, Any]:
    """Extract all metrics from train.py's final summary block.

    The field map and the scanning itself live in agents/train_output.py, shared
    with Agent1TrainingSpecialist._parse_training_output -- see that module for
    why they must not be two hand-synced copies.

    Only the defaults below are specific to this path: a remote run carries a
    transport `status`, and `training_time` is seeded to None because this
    function is also called on PARTIAL output when an SSH channel drops
    mid-summary, where the key may legitimately never appear.
    """
    metrics: Dict[str, Any] = {
        "val_bpb": float("inf"),
        "training_time": None,
        "status": "remote_ok",
    }
    metrics.update(train_output.parse(stdout))
    return metrics
