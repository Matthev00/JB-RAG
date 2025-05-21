import os
from pyngrok import ngrok
import time
import subprocess

ngrok.kill()

public_url = ngrok.connect(8501)
print(f"✅ Streamlit is live at: {public_url}")

process = subprocess.Popen(
    ["streamlit", "run", "src/main.py", "--server.fileWatcherType=none"]
)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
    process.terminate()
    ngrok.kill()
