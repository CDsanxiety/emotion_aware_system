import time
import os
import sys

def print_step(message):
    print(f"\033[1;34m[*]\033[0m {message}")
    time.sleep(1.5)

def run_demo():
    print("\n" + "="*40)
    print("      情感感知系统 - 演示模式 (Demo)      ")
    print("="*40 + "\n")

    # 1. 启动滴声
    print_step("系统初始化中...")
    # 尝试播放滴声，如果没有文件则用系统蜂鸣
    if os.path.exists("assets/startup.mp3"):
        os.system("mpg123 assets/startup.mp3 > /dev/null 2>&1")
    else:
        print("\a") # 系统蜂鸣
    print("\033[1;32m[OK]\033[0m 扬声器测试正常：滴！")

    # 2. 模拟捕捉画面
    print_step("正在开启摄像头，捕捉面部画面...")
    for i in range(3):
        sys.stdout.write(f"\r   分析画面帧 {i+1}/3 ...")
        sys.stdout.flush()
        time.sleep(1)
    print("\n\033[1;32m[OK]\033[0m 画面捕捉完成。")

    # 3. 模拟录音
    print_step("正在开启麦克风，开始录音...")
    for i in range(5, 0, -1):
        sys.stdout.write(f"\r   正在监听中... {i}s ")
        sys.stdout.flush()
        time.sleep(1)
    print("\n\033[1;32m[OK]\033[0m 录音已保存并上传云端。")

    # 4. 模拟大模型分析
    print_step("调用大模型分析情绪中...")
    time.sleep(2)
    
    # 5. 核心回复
    print("\n" + "-"*40)
    reply = "看起来你很紧张呀，没关系有我陪着你呢，我来播放一点舒缓的音乐来减少你紧张的情绪吧。"
    print(f"\033[1;35m[机器人回复]:\033[0m {reply}")
    print("-"*40 + "\n")

    # 6. 播放音乐
    print_step("正在从 music 目录检索舒缓音乐...")
    music_path = "music/happy.mp3" # 假设你的音乐文件叫这个
    if os.path.exists(music_path):
        print(f"\033[1;32m[PLAY]\033[0m 正在播放: {music_path}")
        os.system(f"mpg123 {music_path}")
    else:
        print(f"\033[1;31m[Error]\033[0m 未找到音乐文件: {music_path}，请检查路径。")

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n演示已手动停止。")
