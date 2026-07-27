"""SSH-based remote training runner.

Reads connection details from .env (via python-dotenv) and uses paramiko
to:
  1. Upload the local model_hyperparams.yaml to the remote server.
  2. Execute train.py inside the configured conda environment.
  3. Stream stdout/stderr back and parse the metrics.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

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


def run_training_remote(
    hyperparams_local_path: str,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    Upload hyperparams YAML to the remote server and run train.py there.

    Steps:
      1. Connect via SSH.
      2. Pull latest code on remote (git pull).
      3. Upload local model_hyperparams.yaml via SFTP.
      4. Execute train.py inside the conda env.
      5. Stream output back and parse metrics.

    Returns:
        Metrics dict with at least {"val_bpb": float, "status": str}.
    """
    if not _PARAMIKO_AVAILABLE:
        raise RuntimeError(
            "paramiko is not installed. "
            "Run: pip install paramiko python-dotenv"
        )

    cfg = _load_cfg()
    remote_repo = cfg["repo"]
    remote_hyperparams = f"{remote_repo}/model_hyperparams.yaml"
    activate = cfg["conda_activate"]
    env = cfg["conda_env"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(f"[RemoteRunner] Connecting to {cfg['user']}@{cfg['host']}:{cfg['port']} ...")
        client.connect(
            hostname=cfg["host"],
            port=cfg["port"],
            username=cfg["user"],
            password=cfg["password"] or None,
            timeout=30,
        )

        # --- 1. Pull latest code on remote so train.py is always up to date ---
        print("[RemoteRunner] Pulling latest code on remote ...")
        # Stash local changes (from previous SFTP uploads) before pulling
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

        # --- 2. Upload hyperparameters ---
        print(f"[RemoteRunner] Uploading hyperparams -> {remote_hyperparams}")
        sftp = client.open_sftp()
        sftp.put(hyperparams_local_path, remote_hyperparams)
        sftp.close()

        # --- 3. Run training on GPU 4 (free A100) ---
        # Use CUDA_VISIBLE_DEVICES to avoid GPU 0 (which has running processes)
        remote_cmd = (
            f'bash -lc "{activate} {env} && cd {remote_repo} && CUDA_VISIBLE_DEVICES=4 python -u train.py"'
        )
        print(f"[RemoteRunner] Executing on GPU 4: {remote_cmd}")
        _stdin, stdout, stderr = client.exec_command(remote_cmd, timeout=timeout)

        output_lines = []
        last_progress_bar = None
        for line in iter(stdout.readline, ""):
            line = line.rstrip("\n")
            output_lines.append(line)
            # Progress bar lines: update in-place locally with \r
            if line.startswith('[') and ']' in line and '%' in line:
                last_progress_bar = line
                print(f"\r  [remote] {line}", end="", flush=True)
            # Other non-empty lines: print normally with newline
            elif line.strip():
                if last_progress_bar:
                    print()  # newline to finish the progress bar line
                    last_progress_bar = None
                print(f"  [remote] {line}", flush=True)
        
        if last_progress_bar:
            print()  # final newline after last progress bar

        err_output = stderr.read().decode("utf-8", errors="replace").strip()
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            print(f"[RemoteRunner] Remote process exited with code {exit_code}")
            if err_output:
                print(f"[RemoteRunner] stderr: {err_output[:500]}")
            return {
                "val_bpb": float("inf"),
                "error": err_output or f"exit code {exit_code}",
                "status": "remote_error",
            }

        metrics = _parse_output("\n".join(output_lines))
        print(f"[RemoteRunner] Parsed metrics: {metrics}")
        return metrics

    finally:
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
_JSON_OUTPUT_KEYS = {"interpretable_scalars", "head_ablation_impacts"}


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
