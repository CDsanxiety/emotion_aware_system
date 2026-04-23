import os
import time
from config import AUDIO_OUTPUT_DEVICE, ROS_ACTION_TOPIC
from ros_bridge import get_ros_bridge
from utils import logger

def play_audio(file_path):
    """使用系统播放器播放音频"""
    if not os.path.exists(file_path):
        logger.warning(f"[语音驱动] 文件不存在: {file_path}")
        return

    # 使用 config.py 中定义的声卡设备 (例如 hw:0,0)
    cmd = f"mpg123 -a {AUDIO_OUTPUT_DEVICE} {file_path}"
    logger.info(f"[语音驱动] 执行播放: {cmd}")
    os.system(cmd)

def on_status_received(msg):
    """处理来自 ROS 的状态消息"""
    try:
        # 统一读取 last_robot_audio_path 字段
        audio_path = msg.get("last_robot_audio_path")
        if audio_path:
            play_audio(audio_path)
    except Exception as e:
        logger.error(f"[语音驱动] 处理出错: {e}")

if __name__ == "__main__":
    logger.info("--- 暖暖机器人 语音播放驱动已启动 ---")
    ros_bridge = get_ros_bridge()
    # 订阅动作话题，获取音频播放指令
    ros_bridge.subscribe(ROS_ACTION_TOPIC, on_status_received)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("语音驱动已停止")
