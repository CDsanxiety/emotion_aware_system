# ros_client.py
import json
import threading
import roslibpy
from config import ROS_BRIDGE_URI, ROS_ACTION_TOPIC, ROS_STATUS_TOPIC
from utils import logger

class ROSManager:
    """
    高可靠性 ROSBridge 客户端。
    实现“闭环反馈”机制：不仅能下发指令 (Action)，还能监听硬件执行状态 (Status)。
    """
    def __init__(self, uri=ROS_BRIDGE_URI):
        self.uri = uri
        self.client = None
        self.action_topic = None
        self.status_listener = None
        self.is_connected = False
        self.last_hardware_status = {}  # 存储硬件反馈的最新状态

    def connect(self):
        """异步初始化连接，不阻塞主程序启动。"""
        def _run():
            try:
                # 解析 URI 中的 host 和 port
                clean_uri = self.uri.replace("ws://", "").split(":")
                host = clean_uri[0]
                port = int(clean_uri[1]) if len(clean_uri) > 1 else 9090
                
                self.client = roslibpy.Ros(host=host, port=port)
                self.client.on_ready(self._on_ready)
                self.client.on_error(lambda err: logger.error(f"ROSBridge 错误: {err}"))
                self.client.run()
            except Exception as e:
                logger.error(f"无法建立 ROSBridge 连接 ({self.uri}): {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _on_ready(self):
        logger.info(f"成功挂载 ROSBridge 控制链路: {self.uri}")
        self.is_connected = True
        
        # 1. 初始化指令下发通道 (Talker)
        self.action_topic = roslibpy.Topic(self.client, ROS_ACTION_TOPIC, 'std_msgs/String')
        
        # 2. 初始化状态反馈监听通道 (Listener) - 实现闭环控制的关键
        self.status_listener = roslibpy.Topic(self.client, ROS_STATUS_TOPIC, 'std_msgs/String')
        self.status_listener.subscribe(self._on_status_receive)

    def _on_status_receive(self, message):
        """处理来自树莓派/ROS 的反馈信息。"""
        try:
            data = json.loads(message['data'])
            self.last_hardware_status = data
            logger.debug(f"接收到硬件状态反馈: {data}")
        except Exception as e:
            logger.warning(f"状态反馈解析失败: {e}")

    def publish_action(self, action_payload: dict):
        """
        向硬件发布指令。
        action_payload: 包含 emotion, action, reply 等决策信息的字典。
        """
        if not self.is_connected or not self.action_topic:
            # 静默失败，不影响主对话逻辑，但记录日志
            logger.debug("ROS 未就绪，指令仅在 UI 展示，不发送硬件。")
            return
        
        try:
            msg_str = json.dumps(action_payload, ensure_ascii=False)
            self.action_topic.publish(roslibpy.Message({'data': msg_str}))
            logger.info(f">>> 硬件指令已下发: {msg_str}")
        except Exception as e:
            logger.error(f"ROS 指令发布异常: {e}")

    def get_status(self) -> dict:
        """获取最近一次监测到的硬件状态（用于 LLM 决策参考）。"""
        return self.last_hardware_status

# 全局单例管理器
global_ros_manager = ROSManager()

if __name__ == "__main__":
    # 本地模拟测试
    mgr = ROSManager()
    mgr.connect()
    import time
    time.sleep(2)
    if mgr.is_connected:
        print("连接成功，按 Ctrl+C 退出测试。")
        while True:
            time.sleep(1)
