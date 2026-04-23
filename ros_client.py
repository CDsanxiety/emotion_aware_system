# ros_client.py
import json
import threading
import roslibpy
from config import ROS_BRIDGE_URI, ROS_ACTION_TOPIC, ROS_STATUS_TOPIC, ROS_STATE_TOPIC
from utils import logger


class ROSManager:
    """
    高可靠性 ROSBridge 客户端，符合 ROS-LLM 框架设计。
    """

    def __init__(self, uri=ROS_BRIDGE_URI):
        self.uri = uri
        self.client = None
        self.action_topic = None
        self.status_listener = None
        self.state_publisher = None
        self.is_connected = False
        self.last_hardware_status = {}
        self.node_name = "llm_ros_manager"

    def connect(self):
        """异步初始化连接，不阻塞主程序启动。"""

        def _run():
            try:
                # 解析 URI
                clean_uri = self.uri.replace("ws://", "").split(":")
                host = clean_uri[0]
                port = int(clean_uri[1]) if len(clean_uri) > 1 else 9090

                self.client = roslibpy.Ros(host=host, port=port)
                # 使用标准的 .on('event', callback) 语法，确保兼容性
                self.client.on('ready', self._on_ready)
                self.client.on('error', lambda err: logger.error(f"ROSBridge 错误: {err}"))
                self.client.on('close', lambda reason: logger.info(f"ROSBridge 连接已关闭{reason}"))
                self.client.run()
            except Exception as e:
                logger.error(f"无法建立 ROSBridge 连接 ({self.uri}): {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _on_ready(self,_=None):
        logger.info(f"成功挂载 ROSBridge 控制链路: {self.uri}")
        self.is_connected = True
        # 初始化话题
        self.action_topic = roslibpy.Topic(self.client, ROS_ACTION_TOPIC, 'std_msgs/String')
        self.status_listener = roslibpy.Topic(self.client, ROS_STATUS_TOPIC, 'std_msgs/String')
        self.status_listener.subscribe(self._on_status_receive)
        self.state_publisher = roslibpy.Topic(self.client, ROS_STATE_TOPIC, 'std_msgs/String')
        # 发布就绪信号
        self.publish_state("ready")

    def _on_status_receive(self, message):
        try:
            self.last_hardware_status = json.loads(message['data'])
        except:
            pass

    def publish_action(self, action_data):
        if self.is_connected and self.action_topic:
            msg = roslibpy.Message({'data': json.dumps(action_data)})
            self.action_topic.publish(msg)

    def publish_state(self, state):
        if self.is_connected and self.state_publisher:
            msg = roslibpy.Message({'data': state})
            self.state_publisher.publish(msg)


# 全局单例
global_ros_manager = ROSManager()
