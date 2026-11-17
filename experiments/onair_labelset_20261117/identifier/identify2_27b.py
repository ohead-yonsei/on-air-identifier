"""identify2.py with the cap-reading model switched to Qwen 27B. Nothing else changes. See models/DESIGN_IDENTIFIER_KO.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import identify
import identify2

identify.MODEL = identify2.MODEL = "qwen3.8:27b-q4_K_M"   # ask_qwen() reads identify.MODEL; identify2.MODEL is used to unload at the end
if __name__ == "__main__":
    identify2.main()
