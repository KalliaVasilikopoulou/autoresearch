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

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def discover_available_gpus(cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=cfg["host"], port=cfg["port"], username=cfg["user"],
            password=cfg["password"] or None, timeout=30,
        )
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
    return available


def _acquire_gpu_lock(client, gpu_index: int, remote_repo: str, stale_after_seconds: float) -> bool:
    """mkdir is atomic over SSH (POSIX guarantee: fails if the dir already
    exists), so this is a real mutex with no extra dependency. A lock older
    than stale_after_seconds is assumed to belong to a crashed process and
    is reclaimed -- otherwise a single dead run would permanently strand a
    GPU. Best-effort: caller treats a failed acquire as "unavailable this
    round," never blocks waiting for it.
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
        print(f"[RemoteRunner] GPU {gpu_index} lock stale ({age:.0f}s old) -- reclaiming")
        client.exec_command(f'rm -rf {lock_dir}')[1].read()
        return _try_mkdir()

    return False


def _release_gpu_lock(client, gpu_index: int, remote_repo: str) -> None:
    lock_dir = f"{remote_repo}/.autoresearch_gpu_locks/gpu_{gpu_index}"
    try:
        client.exec_command(f'rm -rf {lock_dir}')[1].read()
    except Exception:
        pass


def sync_remote_code(cfg: Optional[Dict[str, Any]] = None) -> None:
    """git stash + pull --ff-only on the remote clone. Call this once per
    dispatch round (a single run, or once for a whole parallel wave) --
    never concurrently from multiple SSH sessions against the same working
    tree, which would itself be a race between two git processes.
    """
    if not _PARAMIKO_AVAILABLE:
        return
    cfg = cfg or _load_cfg()
    remote_repo = cfg["repo"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=cfg["host"], port=cfg["port"], username=cfg["user"],
            password=cfg["password"] or None, timeout=30,
        )
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
        client.close()


def run_training_remote(
    hyperparams_local_path: str,
    timeout: int = 600,
    gpu_index: Optional[int] = None,
    hp_remote_name: str = "model_hyperparams.yaml",
    run_label: Optional[str] = None,
    skip_sync: bool = False,
) -> Dict[str, Any]:
    """
    Upload hyperparams YAML to the remote server and run train.py there.

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

    Returns:
        Metrics dict with at least {"val_bpb": float, "status": str,
        "device": gpu_index}.
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

    if not skip_sync:
        sync_remote_code(cfg)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    lock_acquired = False

    try:
        print(f"[{label}] Connecting to {cfg['user']}@{cfg['host']}:{cfg['port']} ...")
        client.connect(
            hostname=cfg["host"],
            port=cfg["port"],
            username=cfg["user"],
            password=cfg["password"] or None,
            timeout=30,
        )

        lock_acquired = _acquire_gpu_lock(client, gpu_index, remote_repo, stale_after)
        if not lock_acquired:
            print(f"[{label}] GPU {gpu_index}'s lock is held by another run -- proceeding anyway "
                  f"(best-effort lock, not a hard reservation)")

        # --- Upload hyperparameters ---
        print(f"[{label}] Uploading hyperparams -> {remote_hyperparams}")
        sftp = client.open_sftp()
        sftp.put(hyperparams_local_path, remote_hyperparams)
        sftp.close()

        # --- Run training on the selected GPU ---
        remote_cmd = (
            f'bash -lc "{activate} {env} && cd {remote_repo} && '
            f'CUDA_VISIBLE_DEVICES={gpu_index} AUTORESEARCH_HP_PATH={remote_hyperparams} python -u train.py"'
        )
        print(f"[{label}] Executing on GPU {gpu_index}: {remote_cmd}")
        _stdin, stdout, stderr = client.exec_command(remote_cmd, timeout=timeout)

        output_lines = []
        last_progress_bar = None
        for line in iter(stdout.readline, ""):
            line = line.rstrip("\n")
            output_lines.append(line)
            # Progress bar lines: update in-place locally with \r
            if line.startswith('[') and ']' in line and '%' in line:
                last_progress_bar = line
                print(f"\r  [{label}] {line}", end="", flush=True)
            # Other non-empty lines: print normally with newline
            elif line.strip():
                if last_progress_bar:
                    print()  # newline to finish the progress bar line
                    last_progress_bar = None
                print(f"  [{label}] {line}", flush=True)

        if last_progress_bar:
            print()  # final newline after last progress bar

        err_output = stderr.read().decode("utf-8", errors="replace").strip()
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            print(f"[{label}] Remote process exited with code {exit_code}")
            if err_output:
                print(f"[{label}] stderr: {err_output[:500]}")
            return {
                "val_bpb": float("inf"),
                "error": err_output or f"exit code {exit_code}",
                "status": "remote_error",
                "device": gpu_index,
            }

        metrics = _parse_output("\n".join(output_lines))
        metrics["device"] = gpu_index
        print(f"[{label}] Parsed metrics: {metrics}")
        return metrics

    finally:
        if lock_acquired:
            _release_gpu_lock(client, gpu_index, remote_repo)
        client.close()


# Mapping from train.py output key -> metrics dict key
_OUTPUT_FIELDS = {
    "val_bpb:": ("val_bpb", float),
    "training_seconds:": ("training_time", float),
    "total_seconds:": ("total_seconds", float),
    "peak_vram_mb:": ("peak_vram_mb", float),
    "mfu_percent:": ("mfu_percent", float),
    "total_tokens_m:": ("total_tokens_M", float),
    "num_steps:": ("num_steps", int),
    "num_params_m:": ("num_params_M", float),
    "depth:": ("depth", int),
    "holdout_val_bpb:": ("holdout_val_bpb", float),
}


# Lines like `interpretable_scalars: {...}` / `head_ablation_impacts: {...}`
# carry real per-run evidence as a JSON blob rather than a single scalar.
_JSON_OUTPUT_KEYS = {"interpretable_scalars", "head_ablation_impacts", "hyperparam_clamps", "token_fingerprint"}


def _parse_output(stdout: str) -> Dict[str, Any]:
    """Extract all metrics from train.py's final summary block."""
    metrics: Dict[str, Any] = {
        "val_bpb": float("inf"),
        "training_time": None,
        "status": "remote_ok",
    }
    for line in stdout.splitlines():
        if ":" in line:
            prefix, _, rest = line.partition(":")
            key = prefix.strip()
            if key in _JSON_OUTPUT_KEYS:
                try:
                    metrics[key] = json.loads(rest.strip())
                except (json.JSONDecodeError, ValueError):
                    pass
                continue

        key = line.split()[0].lower() if line.split() else ""
        if key in _OUTPUT_FIELDS:
            dest, cast = _OUTPUT_FIELDS[key]
            parts = line.split()
            if len(parts) >= 2:
                try:
                    metrics[dest] = cast(parts[1])
                except (ValueError, IndexError):
                    pass
    return metrics
