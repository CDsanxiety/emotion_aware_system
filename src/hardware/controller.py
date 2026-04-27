# src/hardware/controller.py
import json
import roslibpy
from src.core.config import ROS_BRIDGE_URI, TOPIC_ACTION
from src.utils.logger import logger

class HardwareController:
    def __init__(self):
        self.client = None
        self.action_topic = None
        self.is_ready = False

    def connect(self):
        try:
            logger.info(f"[Hardware] 正在连接 ROSBridge: {ROS_BRIDGE_URI}")
            host_port = ROS_BRIDGE_URI.replace("ws://", "").split(":")
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 9090
            
            self.client = roslibpy.Ros(host=host, port=port)
            self.action_topic = roslibpy.Topic(self.client, TOPIC_ACTION, 'std_msgs/String')
            
            self.client.on('ready', self._on_ready)
            self.client.run()
        except Exception as e:
            logger.error(f"[Hardware] ROS 连接失败: {e}")

    def _on_ready(self):
        logger.info("[Hardware] ROS 连接已就绪")
        self.is_ready = True

    def execute(self, action, emotion):
        """下发指令到硬件驱动"""
        if not self.is_ready:
            logger.warning("[Hardware] ROS 未就绪，跳过指令下发")
            return

        payload = {
            "execution": {
                "action": action,
                "emotion": emotion
            }
        }
        msg = roslibpy.Message({'data': json.dumps(payload)})
        self.action_topic.publish(msg)
        logger.info(f"[Hardware] 已下发动作指令: {action} ({emotion})")

# 全局单例
hw_controller = HardwareController()
