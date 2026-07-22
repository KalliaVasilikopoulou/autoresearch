"""Install flash-attn from pre-built wheel."""
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

# Try installing flash-attn 2.6.3 pre-built wheel
print("Installing flash-attn from pre-built wheel...")
cmd = "bash -lc 'source /opt/anaconda3/bin/activate autoresearch && pip install flash-attn==2.6.3 --only-binary flash-attn 2>&1 || echo WHEEL_FAILED'"
_, o, e = c.exec_command(cmd, timeout=300)

# Stream output
for line in iter(o.readline, ""):
    if line:
        print(line.rstrip())

# Verify installation
print("\n" + "="*60)
print("Checking flash-attn...")
print("="*60)
_, o, _ = c.exec_command(
    "bash -lc 'source /opt/anaconda3/bin/activate autoresearch && python -c \"try:\\n    import flash_attn\\n    print(f\\\"flash-attn: {flash_attn.__version__}\\\")\\nexcept:\\n    print(\\\"flash-attn: NOT INSTALLED\\\")\" 2>&1'"
)
out = o.read().decode("utf-8", errors="replace")
print(out)

c.close()
print("Done!")
