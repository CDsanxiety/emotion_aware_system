# ros_client.py
import json
import threading
import roslibpy
from config import ROS_BRIDGE_URI, ROS_ACTION_TOPIC, ROS_STATUS_TOPIC, ROS_STATE_TOPIC
from utils import logger

class ROSManager:
    """
    高可靠性 ROSBridge 客户端，符合 ROS-LLM 框架设计。
    实现“闭环反馈”机制：不仅能下发指令 (Action)，还能监听硬件执行状态 (Status)。
    支持将大模型的自然语言指令转化为 ROS 标准消息，并支持系统状态发布。
    """
    def __init__(self, uri=ROS_BRIDGE_URI):
        self.uri = uri
        self.client = None
        self.action_topic = None
        self.status_listener = None
        self.state_publisher = None
        self.is_connected = False
        self.last_hardware_status = {}  # 存储硬件反馈的最新状态
        self.node_name = "llm_ros_manager"

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
                self.client.on_close(lambda: logger.info("ROSBridge 连接已关闭"))
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
        
        # 3. 初始化系统状态发布通道
        self.state_publisher = roslibpy.Topic(self.client, ROS_STATE_TOPIC, 'std_msgs/String')
        
        # 发布系统就绪状态
        self.publish_state("ready")

    def _on_status_receive(self, message):
        """处理来自树莓派/ROS 的反馈信息。"""
        try:
            data = json.loads(message['data'])
            self.last_hardware_status = data
            logger.debug(f"接收到硬件状态反馈: {data}")
        except Exception as e:
            logger.warning(f"状态反馈解析失败: {e}")

    def publish_state(self, state: str):
        """发布系统状态"""
        if not self.is_connected or not self.state_publisher:
            return
        
        try:
            msg = roslibpy.Message({'data': state})
            self.state_publisher.publish(msg)
            logger.debug(f"发布系统状态: {state}")
        except Exception as e:
            logger.error(f"状态发布异常: {e}")

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
            # 将大模型指令转化为 ROS 标准消息
            msg_str = json.dumps(action_payload, ensure_ascii=False)
            msg = roslibpy.Message({'data': msg_str})
            self.action_topic.publish(msg)
            logger.info(f">>> 硬件指令已下发: {msg_str}")
        except Exception as e:
            logger.error(f"ROS 指令发布异常: {e}")

    def get_status(self) -> dict:
        """获取最近一次监测到的硬件状态（用于 LLM 决策参考）。"""
        return self.last_hardware_status

    def shutdown(self):
        """关闭 ROS 连接，处理节点生命周期"""
        if self.client:
            self.publish_state("shutdown")
            self.client.terminate()
            logger.info("ROSBridge 连接已关闭")

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
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            mgr.shutdown()
            print("测试结束。")
