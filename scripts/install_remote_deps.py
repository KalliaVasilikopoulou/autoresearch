"""Install missing dependencies on the remote server and restart data prep."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.getenv("REMOTE_HOST"), int(os.getenv("REMOTE_PORT", 22)),
          os.getenv("REMOTE_USER"), os.getenv("REMOTE_PASSWORD"), timeout=15)

activate = os.getenv("REMOTE_CONDA_ACTIVATE", "source /opt/anaconda3/bin/activate")
env = os.getenv("REMOTE_CONDA_ENV", "autoresearch")
repo = os.getenv("REMOTE_REPO")

# Install missing packages (pyproject.toml deps minus torch which is already installed)
pip_cmd = (
    f'bash -lc "{activate} {env} && pip install '
    f'rustbpe tiktoken requests pyarrow numpy pandas pyyaml paramiko python-dotenv '
    f'kernels 2>&1 | tail -20"'
)
print("Installing missing packages …")
_, o, e = c.exec_command(pip_cmd, timeout=300)
print(o.read().decode())

# Restart prepare.py in the background
run_cmd = (
    f'bash -lc "{activate} {env} && cd {repo} && '
    f'nohup python prepare.py --num-shards 20 > /tmp/prepare_log.txt 2>&1 & echo PID:$!"'
)
_, o, _ = c.exec_command(run_cmd)
print("prepare.py restart:", o.read().decode().strip())
c.close()
