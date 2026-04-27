# main.py
import sys
import os
import signal
import time

# 将当前目录加入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.orchestrator import EmotionSystemOrchestrator
from src.utils.logger import logger

def main():
    orchestrator = EmotionSystemOrchestrator()
    
    def signal_handler(sig, frame):
        print("\n[System] 正在执行安全关机程序...")
        orchestrator.hw.play_sound("music/shutdown.mp3", wait=True)
        orchestrator.stop()
        sys.exit(0)

    # 注册退出信号
    signal.signal(signal.SIGINT, signal_handler)

    # 1. 播放启动音 (改为非阻塞，防止音箱冲突导致系统卡死)
    logger.info("--- 🚀 系统启动中 ---")
    orchestrator.hw.play_sound("music/startup.mp3", wait=False)

    # 2. 运行主循环
    print("\n" + "="*40)
    print("  情感感知机器人 (直连模式) 已启动")
    print("  按下 Ctrl+C 退出系统")
    print("="*40 + "\n")
    
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"[Main] 运行时错误: {e}")
    finally:
        print("\n[System] 正在执行安全关机程序...")
        orchestrator.hw.play_sound("music/shutdown.mp3", wait=False)
        orchestrator.stop()
        sys.exit(0)

if __name__ == "__main__":
    # 确保 music 文件夹存在
    if not os.path.exists("music"):
        os.makedirs("music")
        
    main()
