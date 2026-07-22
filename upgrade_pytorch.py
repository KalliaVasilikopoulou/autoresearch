"""Upgrade PyTorch on remote to FA3-compatible version."""
import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(
    os.getenv("REMOTE_HOST"),
    int(os.getenv("REMOTE_PORT", 22)),
    os.getenv("REMOTE_USER"),
    os.getenv("REMOTE_PASSWORD"),
    timeout=30,
)

# Upgrade PyTorch to 2.6.x with CUDA 12.2
print("Upgrading PyTorch on remote to 2.6.x...")
cmd = "bash -lc 'source /opt/anaconda3/bin/activate autoresearch && conda install -y pytorch::pytorch pytorch::pytorch-cuda=12.2 -c pytorch'"
_, o, _ = c.exec_command(cmd, timeout=600)
for line in iter(o.readline, ""):
    if line:
        print(line.rstrip())

# Verify version
print("\nVerifying PyTorch version...")
_, o, _ = c.exec_command(
    "bash -lc 'source /opt/anaconda3/bin/activate autoresearch && python -c \"import torch; print(torch.__version__)\"'"
)
print(o.read().decode("utf-8", errors="replace"))

c.close()
print("Done!")
