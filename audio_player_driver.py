import os
import time
import json
import roslibpy
from config import AUDIO_OUTPUT_DEVICE, ROS_ACTION_TOPIC
from ros_client import global_ros_manager
from utils import logger


def play_audio(file_path):
    if not os.path.exists(file_path): return
    cmd = f"mpg123 -a {AUDIO_OUTPUT_DEVICE} {file_path}"
    logger.info(f"[语音驱动] 播放: {cmd}")
    os.system(cmd)


def on_action_received(msg_data):
    try:
        msg = json.loads(msg_data['data'])
        audio_path = msg.get("last_robot_audio_path")
        if audio_path: play_audio(audio_path)
    except Exception as e:
        logger.error(f"[语音驱动] 错误: {e}")


if __name__ == "__main__":
    logger.info("--- 语音播放驱动已启动 ---")
    global_ros_manager.connect()
    while not global_ros_manager.is_connected: time.sleep(0.5)

    listener = roslibpy.Topic(global_ros_manager.client, ROS_ACTION_TOPIC, 'std_msgs/String')
    listener.subscribe(on_action_received)

    while True: time.sleep(1)
