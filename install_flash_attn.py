"""Install flash-attn directly on remote."""
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

# Install flash-attn 2.x
print("Installing flash-attn on remote...")
cmd = "bash -lc 'source /opt/anaconda3/bin/activate autoresearch && pip install flash-attn --no-build-isolation -U'"
_, o, e = c.exec_command(cmd, timeout=600)

# Stream output
for line in iter(o.readline, ""):
    if line:
        print(line.rstrip())

err_output = e.read().decode("utf-8", errors="replace")
if err_output:
    print("STDERR:", err_output[:500])

# Verify installation
print("\n" + "="*60)
print("Verifying flash-attn installation...")
print("="*60)
_, o, _ = c.exec_command(
    "bash -lc 'source /opt/anaconda3/bin/activate autoresearch && python -c \"import flash_attn; print(f\\\"flash-attn: {flash_attn.__version__}\\\")\"'"
)
out = o.read().decode("utf-8", errors="replace")
print(out)

c.close()
print("Done!")
