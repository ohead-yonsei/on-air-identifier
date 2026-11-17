import subprocess,sys
from pathlib import Path
r=Path("/home/ohead/ai-workspace/projects/onair_overnight_20261116")
for task in ["view","localize"]:
 args=[sys.executable,str(r/"code/run_vlm.py"),"--task",task,"--core-only","--model","qwen3.8:27b-q4_K_M","--digest","25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e"]
 subprocess.run(args,check=True)
