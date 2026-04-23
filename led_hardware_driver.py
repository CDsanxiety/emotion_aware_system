import time
import json
import board
import neopixel
import roslibpy
from config import LED_COUNT, LED_PIN, LED_BRIGHTNESS, ROS_ACTION_TOPIC
from ros_client import global_ros_manager
from utils import logger

pixels = neopixel.NeoPixel(getattr(board, f"D{LED_PIN}"), LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False)


def on_action_received(msg_data):
    try:
        msg = json.loads(msg_data['data'])
        execution = msg.get("execution", {})
        emotion = execution.get("emotion") or msg.get("emotion", "neutral")

        colors = {"happy": (255, 255, 0), "sad": (0, 0, 255), "angry": (255, 0, 0), "neutral": (0, 255, 0)}
        color = colors.get(emotion, (0, 255, 0))
        pixels.fill(color)
        pixels.show()
    except Exception as e:
        logger.error(f"[灯光驱动] 错误: {e}")


if __name__ == "__main__":
    logger.info("--- 灯带驱动已启动 ---")
    global_ros_manager.connect()
    while not global_ros_manager.is_connected: time.sleep(0.5)

    listener = roslibpy.Topic(global_ros_manager.client, ROS_ACTION_TOPIC, 'std_msgs/String')
    listener.subscribe(on_action_received)

    while True: time.sleep(1)
