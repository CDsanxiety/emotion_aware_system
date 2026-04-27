# src/hardware/drivers/led_driver.py
import time
import json
import board
import neopixel
import roslibpy
from src.core.config import LED_COUNT, LED_PIN, LED_BRIGHTNESS, ROS_BRIDGE_URI, TOPIC_ACTION
from src.utils.logger import logger

# 初始化灯带
pixels = neopixel.NeoPixel(getattr(board, f"D{LED_PIN}"), LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False)

def on_action_received(msg_data):
    try:
        msg = json.loads(msg_data['data'])
        execution = msg.get("execution", {})
        emotion = execution.get("emotion") or msg.get("emotion", "neutral")
        action = execution.get("action") or "none"

        # 映射情绪到颜色
        colors = {
            "happy": (255, 255, 0),     # 黄色
            "sad": (0, 0, 255),       # 蓝色
            "angry": (255, 0, 0),      # 红色
            "neutral": (0, 255, 0),    # 绿色
            "light_warm": (255, 147, 41), # 暖橘色
            "light_bright": (255, 255, 255) # 纯白色
        }
        
        # 优先考虑 action 指定的灯光
        color_key = action if action.startswith("light_") else emotion
        color = colors.get(color_key, (0, 255, 0))
        
        logger.info(f"[LED Driver] 设置颜色: {color_key} -> {color}")
        pixels.fill(color)
        pixels.show()
    except Exception as e:
        logger.error(f"[LED Driver] 错误: {e}")

def main():
    logger.info("--- LED 灯带驱动 (ROS版) 启动 ---")
    host_port = ROS_BRIDGE_URI.replace("ws://", "").split(":")
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else 9090
    
    client = roslibpy.Ros(host=host, port=port)
    listener = roslibpy.Topic(client, TOPIC_ACTION, 'std_msgs/String')
    
    client.on('ready', lambda: logger.info("[LED Driver] 已连接到 ROSBridge"))
    listener.subscribe(on_action_received)
    
    client.run_forever()

if __name__ == "__main__":
    main()
