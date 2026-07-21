"""Check data preparation progress on the remote server."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paramiko
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.getenv("REMOTE_HOST"), int(os.getenv("REMOTE_PORT", 22)),
          os.getenv("REMOTE_USER"), os.getenv("REMOTE_PASSWORD"), timeout=15)

# Count downloaded shards and check tokenizer
_, o, _ = c.exec_command(
    "ls ~/.cache/autoresearch/data/*.parquet 2>/dev/null | wc -l; "
    "ls ~/.cache/autoresearch/tokenizer/ 2>/dev/null && echo tokenizer_ok || echo tokenizer_missing; "
    "tail -5 /tmp/prepare_log.txt 2>/dev/null"
)
print(o.read().decode())
c.close()
