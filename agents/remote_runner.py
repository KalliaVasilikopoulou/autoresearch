"""SSH-based remote training runner.

Reads connection details from .env (via python-dotenv) and uses paramiko
to:
  1. Upload the local model_hyperparams.yaml to the remote server.
  2. Execute train.py inside the configured conda environment.
  3. Stream stdout/stderr back and parse the metrics.
"""

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
    timeout: int = 360,
) -> Dict[str, Any]:
    """
    Upload hyperparams YAML to the remote server and run train.py there.

    Args:
        hyperparams_local_path: local path to model_hyperparams.yaml.
        timeout: seconds to wait for the remote training process.

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

    client = paramiko.SSHClient()
    # AutoAddPolicy accepts unknown host keys automatically.
    # On a trusted university network this is acceptable.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        print(
            f"[RemoteRunner] Connecting to {cfg['user']}@{cfg['host']}:{cfg['port']} …"
        )
        client.connect(
            hostname=cfg["host"],
            port=cfg["port"],
            username=cfg["user"],
            password=cfg["password"] or None,
            timeout=30,
        )

        # --- 1. Upload hyperparameters ---
        print(f"[RemoteRunner] Uploading hyperparams → {remote_hyperparams}")
        sftp = client.open_sftp()
        sftp.put(hyperparams_local_path, remote_hyperparams)
        sftp.close()

        # --- 2. Run training remotely ---
        activate = cfg["conda_activate"]
        env = cfg["conda_env"]
        # The semicolons make this a single login-like shell so conda works.
        remote_cmd = (
            f"bash -lc \"{activate} {env} && cd {remote_repo} && python train.py\""
        )
        print(f"[RemoteRunner] Executing: {remote_cmd}")

        _stdin, stdout, stderr = client.exec_command(remote_cmd, timeout=timeout)

        # Stream output lines as they arrive
        output_lines = []
        for line in iter(stdout.readline, ""):
            line = line.rstrip("\n")
            output_lines.append(line)
            print(f"  [remote] {line}")

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


def _parse_output(stdout: str) -> Dict[str, Any]:
    """Extract val_bpb and other metrics from train.py stdout."""
    metrics: Dict[str, Any] = {
        "val_bpb": float("inf"),
        "train_loss": None,
        "training_time": None,
        "status": "remote_ok",
    }
    for line in stdout.splitlines():
        ll = line.lower()
        if "val_bpb" in ll:
            parts = line.split()
            for i, part in enumerate(parts):
                if "bpb" in part.lower() and i + 1 < len(parts):
                    try:
                        metrics["val_bpb"] = float(parts[i + 1])
                    except ValueError:
                        pass
        if "train_loss" in ll:
            parts = line.split()
            for i, part in enumerate(parts):
                if "loss" in part.lower() and i + 1 < len(parts):
                    try:
                        metrics["train_loss"] = float(parts[i + 1])
                    except ValueError:
                        pass
    return metrics
