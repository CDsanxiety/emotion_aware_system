import time
import board
import neopixel
from config import LED_COUNT, LED_BRIGHTNESS, ROS_ACTION_TOPIC
from ros_bridge import get_ros_bridge
from utils import logger

# 从 config.py 读取硬件配置
LED_PIN = board.D18

pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False)

def set_emotion_effect(emotion):
    """根据情绪设置灯光效果"""
    if emotion == "happy":
        # 暖橘色渐变
        pixels.fill((255, 100, 0))
    elif emotion == "sad":
        # 忧郁蓝
        pixels.fill((0, 0, 150))
    elif emotion == "angry":
        # 警示红
        pixels.fill((200, 0, 0))
    elif emotion == "thinking":
        # 亮蓝色呼吸 (简化为静态显示)
        pixels.fill((0, 150, 255))
    else:
        # 默认温和白光
        pixels.fill((50, 50, 50))
    pixels.show()

def on_status_received(msg):
    """处理来自 ROS 的状态消息"""
    try:
        # 这里的 msg 结构应与 agent_loop 发送的一致
        emotion = msg.get("emotion", "neutral")
        logger.info(f"[LED驱动] 收到情绪指令: {emotion}")
        set_emotion_effect(emotion)
    except Exception as e:
        logger.error(f"[LED驱动] 处理出错: {e}")

if __name__ == "__main__":
    logger.info("--- 暖暖机器人 LED 硬件驱动已启动 ---")
    ros_bridge = get_ros_bridge()
    # 订阅动作话题，获取机器人当前情绪
    ros_bridge.subscribe(ROS_ACTION_TOPIC, on_status_received)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pixels.fill((0, 0, 0))
        pixels.show()
        logger.info("LED 驱动已停止")
