"""Kill zombie training processes on remote GPU 0."""
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

# Kill any stray python processes from previous training runs
print("Killing zombie python processes on remote...")
cmd = "bash -lc 'pkill -9 -f \"python.*train.py\" || echo no processes found'"
_, o, _ = c.exec_command(cmd)
out = o.read().decode("utf-8", errors="replace")
print(out)

print("\nChecking GPU status...")
_, o, _ = c.exec_command("nvidia-smi | head -30")
print(o.read().decode("utf-8", errors="replace"))

c.close()
print("Done!")
