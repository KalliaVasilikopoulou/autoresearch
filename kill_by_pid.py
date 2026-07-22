"""Force kill training processes on remote by PID."""
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

# List processes using GPU 0
print("Processes on GPU 0:")
_, o, _ = c.exec_command("nvidia-smi | grep -A 100 'Processes'")
print(o.read().decode("utf-8", errors="replace"))

# Kill processes on GPU 0
print("\nKilling GPU 0 processes...")
cmd = "bash -lc 'kill -9 1672592 2>/dev/null; echo killed'"
_, o, _ = c.exec_command(cmd)
print(o.read().decode("utf-8", errors="replace"))

# Wait a bit and check again
import time
time.sleep(2)

print("\nGPU status after kill:")
_, o, _ = c.exec_command("nvidia-smi | head -40")
print(o.read().decode("utf-8", errors="replace"))

c.close()
