import paramiko
from dotenv import load_dotenv
import os

load_dotenv()
ssh_host = os.getenv("REMOTE_HOST", "dgx.ceid.upatras.gr")
ssh_port = int(os.getenv("REMOTE_PORT", 22))
ssh_user = os.getenv("REMOTE_USER", "up1066590")
ssh_password = os.getenv("REMOTE_PASSWORD")
remote_repo = os.getenv("REMOTE_REPO", "/home/up1066590/autoresearch")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(ssh_host, port=ssh_port, username=ssh_user, password=ssh_password, timeout=10)

sftp = client.open_sftp()
local_file = r"c:\Users\Kallia\Desktop\Multi-agent autoresearch\autoresearch\train.py"
remote_file = f"{remote_repo}/train.py"
sftp.put(local_file, remote_file)
print(f"✓ Uploaded train.py to {remote_file}")
sftp.close()
client.close()
