"""Verify the updated train.py can read model_hyperparams.yaml on the remote."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.getenv("REMOTE_HOST"), int(os.getenv("REMOTE_PORT", 22)),
          os.getenv("REMOTE_USER"), os.getenv("REMOTE_PASSWORD"), timeout=10)

activate = os.getenv("REMOTE_CONDA_ACTIVATE", "source /opt/anaconda3/bin/activate")
env      = os.getenv("REMOTE_CONDA_ENV", "autoresearch")
repo     = os.getenv("REMOTE_REPO")

# Run a tiny Python snippet on remote that mimics the YAML-reading block in train.py
snippet = (
    "import yaml, math; "
    "ASPECT_RATIO=64; HEAD_DIM=128; DEPTH=8; WEIGHT_DECAY=0.2; WARMUP_RATIO=0.0; "
    "hp=yaml.safe_load(open('model_hyperparams.yaml')); "
    "DEPTH=int(hp.get('n_layer', DEPTH)); "
    "lcm=DEPTH*HEAD_DIM//math.gcd(DEPTH,HEAD_DIM); "
    "t=int(hp.get('n_embd', DEPTH*ASPECT_RATIO)); "
    "snapped=max(lcm, round(t/lcm)*lcm); ASPECT_RATIO=snapped//DEPTH; "
    "print(f'DEPTH={DEPTH} ASPECT_RATIO={ASPECT_RATIO} "
    "model_dim={DEPTH*ASPECT_RATIO} heads={DEPTH*ASPECT_RATIO//HEAD_DIM}')"
)

cmd = f'bash -lc "{activate} {env} && cd {repo} && python -c \\"{snippet}\\""'
_, o, e = c.exec_command(cmd, timeout=15)
print("stdout:", o.read().decode().strip())
err = e.read().decode().strip()
if err:
    print("stderr:", err[:300])
c.close()
