"""
Live connectivity test for the remote GPU server.

Run with:
    python tests/test_remote_connection.py

Checks (in order):
  1. .env is fully configured
  2. SSH connection succeeds
  3. Python is available on the remote server
  4. The configured conda environment exists
  5. The remote repo directory exists
  6. PyTorch sees a CUDA GPU on the remote server
"""

import sys
from pathlib import Path

# Make sure repo root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.remote_runner import _load_cfg, is_remote_configured

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
INFO = "\033[94m INFO\033[0m"


def _ssh_client():
    import paramiko
    cfg = _load_cfg()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=cfg["host"],
        port=cfg["port"],
        username=cfg["user"],
        password=cfg["password"] or None,
        timeout=15,
    )
    return client, cfg


def run(label: str, cmd: str, client) -> tuple[bool, str]:
    """Run a remote command and return (success, output)."""
    _, stdout, stderr = client.exec_command(cmd, timeout=60)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    exit_code = stdout.channel.recv_exit_status()
    ok = exit_code == 0
    tag = PASS if ok else FAIL
    print(f"{tag}  {label}")
    if out:
        for line in out.splitlines()[:5]:   # show first 5 lines max
            print(f"       {line}")
    if not ok and err:
        print(f"       stderr: {err[:200]}")
    return ok, out


def main():
    print("\n=== Remote GPU server connectivity test ===\n")

    # 1. .env check
    if is_remote_configured():
        print(f"{PASS}  .env is fully configured")
    else:
        print(f"{FAIL}  .env is missing required fields (REMOTE_HOST / REMOTE_USER / REMOTE_REPO / REMOTE_PASSWORD)")
        sys.exit(1)

    cfg = _load_cfg()
    print(f"{INFO}  Connecting to {cfg['user']}@{cfg['host']}:{cfg['port']} …\n")

    # 2. SSH connection
    try:
        client, cfg = _ssh_client()
        print(f"{PASS}  SSH connection established")
    except Exception as e:
        print(f"{FAIL}  SSH connection failed: {e}")
        sys.exit(1)

    try:
        # 3. Python available
        run("Python version", "python --version 2>&1 || python3 --version 2>&1", client)

        # 4. Conda env exists
        activate = cfg["conda_activate"]
        env_name = cfg["conda_env"]
        run(
            f"Conda env '{env_name}' activates",
            f"bash -lc \"{activate} {env_name} && echo activated && python --version\"",
            client,
        )

        # 5. Remote repo directory exists
        repo = cfg["repo"]
        run(f"Repo directory exists  ({repo})", f"test -d {repo} && echo exists", client)

        # 6. PyTorch + CUDA
        run(
            "PyTorch sees a CUDA GPU",
            (
                f"bash -lc \"{activate} {env_name} && "
                "python -c 'import torch; "
                "print(\\\"cuda:\\\", torch.cuda.is_available(), "
                "torch.cuda.device_count(), \\\"devices\\\"); "
                "[print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'\""
            ),
            client,
        )

        # 7. train.py exists in repo
        run(f"train.py found in repo", f"test -f {repo}/train.py && echo found", client)

    finally:
        client.close()

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
