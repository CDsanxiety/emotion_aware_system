import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 测试音频系统
print("测试音频系统初始化...")
from audio_manager import play_system_audio, play_background

print("\n测试1: 播放开机音")
play_system_audio("startup")
time.sleep(3)

print("\n测试2: 播放背景音")
play_background()
time.sleep(2)

print("\n测试3: 播放唤醒音（应该停止背景音）")
play_system_audio("wakeup")
time.sleep(2)

print("\n测试4: 恢复背景音")
play_background()
time.sleep(2)

print("\n测试完成！")
input("按回车退出")