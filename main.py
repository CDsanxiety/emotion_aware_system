# main.py
import sys
import os

# 将 src 加入路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.core.orchestrator import EmotionSystemOrchestrator

if __name__ == "__main__":
    orchestrator = EmotionSystemOrchestrator()
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        orchestrator.stop()
        sys.exit(0)
