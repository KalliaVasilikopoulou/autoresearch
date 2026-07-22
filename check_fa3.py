import os
import sys
import paramiko
from dotenv import load_dotenv

load_dotenv()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.getenv("REMOTE_HOST"), int(os.getenv("REMOTE_PORT", 22)), 
          os.getenv("REMOTE_USER"), os.getenv("REMOTE_PASSWORD"), timeout=10)

# Check PyTorch and FA versions
cmd = "bash -lc 'source /opt/anaconda3/bin/activate autoresearch && python << EOF\nimport torch\ntry:\n    import flash_attn\n    print(f'Flash Attention: {flash_attn.__version__}')\nexcept Exception as e:\n    print(f'Flash Attention: ERROR - {e}')\nprint(f'PyTorch: {torch.__version__}')\nEOF\n'"

_, o, e = c.exec_command(cmd)
out = o.read().decode("utf-8", errors="replace")
err = e.read().decode("utf-8", errors="replace")
print(out)
if err:
    print("STDERR:", err)

c.close()
