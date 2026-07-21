"""Start data preparation on the remote server in the background."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(
    os.getenv("REMOTE_HOST"),
    int(os.getenv("REMOTE_PORT", 22)),
    os.getenv("REMOTE_USER"),
    os.getenv("REMOTE_PASSWORD"),
    timeout=15,
)

activate = os.getenv("REMOTE_CONDA_ACTIVATE", "source /opt/anaconda3/bin/activate")
env = os.getenv("REMOTE_CONDA_ENV", "autoresearch")
repo = os.getenv("REMOTE_REPO")

cmd = (
    f'bash -lc "{activate} {env} && cd {repo} && '
    f'nohup python prepare.py --num-shards 20 > /tmp/prepare_log.txt 2>&1 & echo PID:$!"'
)
_, stdout, stderr = c.exec_command(cmd)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print("stdout:", out)
if err:
    print("stderr:", err)
c.close()
print("Data preparation started in background on the remote server.")
print("Watch progress: ssh up1066590@dgx.ceid.upatras.gr 'tail -f /tmp/prepare_log.txt'")
