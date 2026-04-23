"""ros_bridge.py
ROS 与微影听镜项目的桥梁：实现 ROS 话题订阅与发布，以及函数调用服务。
与 ROS-LLM 框架的架构思想结合，提供统一的 ROS 接口。
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

from utils import logger


class ROSBridge:
    """ROS 桥梁类：提供 ROS 通信接口"""

    def __init__(self, node: Optional[Any] = None):
        """
        初始化 ROS 桥梁
        
        Args:
            node: ROS 节点实例（如果为 None 则使用模拟模式）
        """
        self.node = node
        self._subscribers = {}
        self._publishers = {}
        self._services = {}
        self._is_connected = False
        self._lock = threading.RLock()
        
        if node is not None:
            self._is_connected = True
            self._setup_ros_communication()
        else:
            logger.info("ROS 节点未提供，使用模拟模式")

    def _setup_ros_communication(self) -> None:
        """设置 ROS 通信"""
        if not self.node:
            return

        # 订阅话题
        self._subscribers["/llm_state"] = self.node.create_subscription(
            String, "/llm_state", self._state_callback, 0
        )
        
        self._subscribers["/llm_input_audio_to_text"] = self.node.create_subscription(
            String, "/llm_input_audio_to_text", self._audio_to_text_callback, 0
        )

        # 发布话题
        self._publishers["/llm_feedback_to_user"] = self.node.create_publisher(
            String, "/llm_feedback_to_user", 0
        )

        # 提供服务
        if hasattr(self.node, "create_service"):
            try:
                from llm_interfaces.srv import ChatGPT
                self._services["/ChatGPT_function_call_service"] = self.node.create_service(
                    ChatGPT, "/ChatGPT_function_call_service", self._function_call_service
                )
                logger.info("ROS 函数调用服务已注册")
            except ImportError:
                logger.warning("llm_interfaces 未找到，跳过服务注册")

    def _state_callback(self, msg: Any) -> None:
        """状态回调"""
        if msg and hasattr(msg, "data"):
            logger.info(f"ROS 状态更新: {msg.data}")
            if msg.data == "listening":
                logger.info("开始监听语音...")

    def _audio_to_text_callback(self, msg: Any) -> None:
        """语音转文本回调"""
        if msg and hasattr(msg, "data"):
            text = msg.data
            logger.info(f"收到语音转文本: {text}")
            # 这里可以调用 llm_api.get_response

    def _function_call_service(self, request: Any, response: Any) -> Any:
        """函数调用服务"""
        if not request or not hasattr(request, "request_text"):
            response.response_text = "请求格式错误"
            return response

        try:
            from robot_functions import execute_robot_action
            
            request_data = json.loads(request.request_text)
            action = request_data.get("action", "无动作")
            result = execute_robot_action(action)
            
            response.response_text = json.dumps({
                "success": result.success,
                "message": result.message,
                "action_type": result.action_type.value
            })
        except Exception as e:
            logger.error(f"函数调用服务失败: {e}")
            response.response_text = f"执行失败: {str(e)}"

        return response

    def publish_message(self, topic: str, message: Dict[str, Any]) -> bool:
        """发布消息到 ROS 话题"""
        if not self._is_connected:
            logger.info(f"[模拟] 发布到 {topic}: {message}")
            return True

        try:
            if topic not in self._publishers:
                from std_msgs.msg import String
                self._publishers[topic] = self.node.create_publisher(String, topic, 0)
                time.sleep(0.1)  # 等待发布者准备

            publisher = self._publishers[topic]
            msg = String()
            msg.data = json.dumps(message)
            publisher.publish(msg)
            logger.info(f"发布到 {topic}: {message}")
            return True
        except Exception as e:
            logger.error(f"发布消息失败: {e}")
            return False

    def subscribe_topic(self, topic: str, callback: callable) -> bool:
        """订阅 ROS 话题"""
        if not self._is_connected:
            logger.info(f"[模拟] 订阅 {topic}")
            return True

        try:
            from std_msgs.msg import String
            subscriber = self.node.create_subscription(
                String, topic, callback, 0
            )
            self._subscribers[topic] = subscriber
            logger.info(f"已订阅 {topic}")
            return True
        except Exception as e:
            logger.error(f"订阅话题失败: {e}")
            return False

    def call_service(self, service_name: str, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """调用 ROS 服务"""
        if not self._is_connected:
            logger.info(f"[模拟] 调用服务 {service_name}: {request_data}")
            return {"success": True, "message": "模拟调用成功"}

        try:
            from llm_interfaces.srv import ChatGPT
            
            client = self.node.create_client(ChatGPT, service_name)
            if not client.wait_for_service(timeout_sec=2.0):
                logger.error(f"服务 {service_name} 不可用")
                return None

            request = ChatGPT.Request()
            request.request_text = json.dumps(request_data)
            
            future = client.call_async(request)
            future.add_done_callback(lambda f: self._service_callback(f, service_name))
            
            logger.info(f"调用服务 {service_name}")
            return {"success": True, "message": "服务调用中"}
        except Exception as e:
            logger.error(f"调用服务失败: {e}")
            return None

    def _service_callback(self, future: Any, service_name: str) -> None:
        """服务调用回调"""
        try:
            result = future.result()
            if result and hasattr(result, "response_text"):
                logger.info(f"服务 {service_name} 响应: {result.response_text}")
        except Exception as e:
            logger.error(f"服务回调失败: {e}")

    def is_connected(self) -> bool:
        """检查是否连接到 ROS"""
        return self._is_connected

    def shutdown(self) -> None:
        """关闭 ROS 桥梁"""
        with self._lock:
            for name, subscriber in self._subscribers.items():
                try:
                    if hasattr(subscriber, "destroy"):
                        subscriber.destroy()
                except Exception:
                    pass
            self._subscribers.clear()
            
            for name, publisher in self._publishers.items():
                try:
                    if hasattr(publisher, "destroy"):
                        publisher.destroy()
                except Exception:
                    pass
            self._publishers.clear()
            
            for name, service in self._services.items():
                try:
                    if hasattr(service, "destroy"):
                        service.destroy()
                except Exception:
                    pass
            self._services.clear()

        logger.info("ROS 桥梁已关闭")


class MockROSNode:
    """模拟 ROS 节点，用于无 ROS 环境时"""

    def __init__(self):
        self.name = "mock_ros_node"
        self.published_messages = []
        self.subscribed_topics = []

    def create_publisher(self, msg_type, topic, qos):
        self.subscribed_topics.append(topic)
        return MockPublisher(topic)

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscribed_topics.append(topic)
        return MockSubscriber(topic, callback)

    def create_service(self, srv_type, service_name, callback):
        return MockService(service_name, callback)

    def create_client(self, srv_type, service_name):
        return MockClient(service_name)


class MockPublisher:
    """模拟发布者"""
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        logger.info(f"[MockPublisher] 发布到 {self.topic}: {msg.data if hasattr(msg, 'data') else msg}")


class MockSubscriber:
    """模拟订阅者"""
    def __init__(self, topic, callback):
        self.topic = topic
        self.callback = callback

    def destroy(self):
        pass


class MockService:
    """模拟服务"""
    def __init__(self, service_name, callback):
        self.service_name = service_name
        self.callback = callback

    def destroy(self):
        pass


class MockClient:
    """模拟客户端"""
    def __init__(self, service_name):
        self.service_name = service_name

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, request):
        return MockFuture()


class MockFuture:
    """模拟未来对象"""
    def result(self):
        class MockResponse:
            response_text = "模拟服务响应"
        return MockResponse()

    def add_done_callback(self, callback):
        pass


# 兼容导入
try:
    from std_msgs.msg import String
    _has_ros = True
except ImportError:
    _has_ros = False
    class String:
        pass


def create_ros_bridge(ros_node: Optional[Any] = None) -> ROSBridge:
    """创建 ROS 桥梁实例"""
    if ros_node is None and not _has_ros:
        ros_node = MockROSNode()
    return ROSBridge(ros_node)


def get_ros_bridge() -> ROSBridge:
    """获取全局 ROS 桥梁实例"""
    global _ros_bridge_instance
    if _ros_bridge_instance is None:
        _ros_bridge_instance = create_ros_bridge()
    return _ros_bridge_instance


_ros_bridge_instance: Optional[ROSBridge] = None