# ros_client.py
import roslibpy
import json
import threading
from utils import logger

class RosBridgeClient:
    def __init__(self, host='localhost', port=9090):
        self.host = host
        self.port = port
        self.client = None
        self.talker = None
        self.is_connected = False

    def connect(self):
        """异步连接 ROSBridge"""
        def _connect_thread():
            try:
                self.client = roslibpy.Ros(host=self.host, port=self.port)
                self.client.on_ready(self._on_ready)
                self.client.run()
            except Exception as e:
                logger.error(f"ROSBridge 连接异常: {e}")
                self.is_connected = False

        thread = threading.Thread(target=_connect_thread, daemon=True)
        thread.start()

    def _on_ready(self):
        logger.info(f"成功连接至 ROSBridge: {self.host}:{self.port}")
        self.is_connected = True
        # 初始化发布者，Topic 名为 /nuannuan/command，消息类型为 std_msgs/String
        self.talker = roslibpy.Topic(self.client, '/nuannuan/command', 'std_msgs/String')

    def publish_action(self, action_data: dict):
        """发布动作指令"""
        if not self.is_connected or self.talker is None:
            logger.warning("ROS 未连接，取消发送指令")
            return
        
        try:
            # 将 dict 转为字符串发布
            msg_str = json.dumps(action_data, ensure_ascii=False)
            self.talker.publish(roslibpy.Message({'data': msg_str}))
            logger.info(f"已向 ROS 发布指令: {msg_str}")
        except Exception as e:
            logger.error(f"指令发布失败: {e}")

    def close(self):
        if self.client:
            self.client.terminate()

# 全局单例
ros_manager = RosBridgeClient()
