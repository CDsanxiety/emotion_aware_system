# main.py
import sys
import os
import signal
import time

# 将当前目录加入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.orchestrator import EmotionSystemOrchestrator
from src.utils.logger import logger

# 全局 orchestrator 引用，用于信号处理时清理资源
_orchestrator = None

def _shutdown(signum, frame):
    """
    统一关机处理器：捕获 SIGINT/SIGTERM/SIGHUP
    确保无论是 Ctrl+C、kill 命令还是 SSH 断开，LED 和硬件都能被正确清理
    """
    logger.info(f"[Main] 收到信号 {signum}，正在安全关机...")
    if _orchestrator:
        _orchestrator.stop()
    sys.exit(0)

def main():
    global _orchestrator
    _orchestrator = EmotionSystemOrchestrator()

    # 注册所有退出信号，确保任何情况下都能清理硬件
    signal.signal(signal.SIGINT, _shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, _shutdown)  # kill 命令
    signal.signal(signal.SIGHUP, _shutdown)   # SSH 断开时发送的信号

    # 播放启动音
    logger.info("--- 🚀 系统启动中 ---")
    _orchestrator.hw.play_sound("music/startup.mp3", wait=False)

    print("\n" + "="*40)
    print("  情感感知机器人 (直连模式) 已启动")
    print("  按下 Ctrl+C 退出系统")
    print("="*40 + "\n")

    try:
        _orchestrator.run()
    except Exception as e:
        logger.error(f"[Main] 运行时错误: {e}")
    finally:
        print("\n[System] 正在执行安全关机程序...")
        _orchestrator.hw.play_sound("music/shutdown.mp3", wait=False)
        _orchestrator.stop()
        sys.exit(0)

if __name__ == "__main__":
    if not os.path.exists("music"):
        os.makedirs("music")

    main()
