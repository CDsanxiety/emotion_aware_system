"""robot_functions.py
机器人函数调用执行器：将 LLM 输出的 action 转化为实际机器人动作。
与 ROS-LLM 框架的 function_call 思想结合，支持多种执行后端。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from utils import logger


class ActionType(Enum):
    PLAY_MUSIC = "播放音乐"
    ADJUST_LIGHT = "调节灯光"
    NO_ACTION = "无动作"
    CUSTOM = "自定义"


@dataclass
class ActionResult:
    success: bool
    action_type: ActionType
    message: str
    details: Optional[Dict[str, Any]] = None


class ActionExecutor(ABC):
    """动作执行器抽象基类"""

    @abstractmethod
    def execute(self, action: str, **kwargs) -> ActionResult:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError


class LightController(ActionExecutor):
    """灯光控制器"""

    def __init__(self, ros_node: Optional[Any] = None):
        self.ros_node = ros_node
        self._current_brightness = 100
        self._current_color = "warm"

    def execute(self, action: str, **kwargs) -> ActionResult:
        if not self.is_available():
            return ActionResult(False, ActionType.ADJUST_LIGHT, "灯光控制器不可用")

        try:
            brightness = kwargs.get("brightness", self._infer_brightness(action))
            color = kwargs.get("color", self._infer_color(action))

            self._current_brightness = brightness
            self._current_color = color

            if self.ros_node is not None:
                self._publish_light_command(brightness, color)
                return ActionResult(True, ActionType.ADJUST_LIGHT, f"灯光已调整为 {color}色、亮度{brightness}%")

            return ActionResult(True, ActionType.ADJUST_LIGHT, f"灯光调整为 {color}色、亮度{brightness}%")

        except Exception as e:
            logger.error(f"灯光调节失败: {e}")
            return ActionResult(False, ActionType.ADJUST_LIGHT, f"灯光调节失败: {e}")

    def _infer_brightness(self, action: str) -> int:
        if any(k in action for k in ["亮", "明亮", "提亮", "开灯"]):
            return 100
        elif any(k in action for k in ["暗", "暗淡", "调暗", "关灯"]):
            return 30
        elif any(k in action for k in ["柔", "柔和", "温暖"]):
            return 60
        return 70

    def _infer_color(self, action: str) -> str:
        if any(k in action for k in ["暖", "暖黄", "温暖", "暖色"]):
            return "暖黄"
        elif any(k in action for k in ["冷", "冷白", "清爽"]):
            return "冷白"
        elif any(k in action for k in ["柔", "柔和"]):
            return "暖黄"
        return "暖黄"

    def _publish_light_command(self, brightness: int, color: str) -> None:
        if self.ros_node is not None:
            msg = {"command": "set_light", "brightness": brightness, "color": color}
            self.ros_node.publish_message("/robot/light_control", msg)

    def is_available(self) -> bool:
        return True


class MusicPlayer(ActionExecutor):
    """音乐播放器"""

    def __init__(self, ros_node: Optional[Any] = None):
        self.ros_node = ros_node
        self._current_state = "stopped"
        self._current_volume = 70

    def execute(self, action: str, **kwargs) -> ActionResult:
        if not self.is_available():
            return ActionResult(False, ActionType.PLAY_MUSIC, "音乐播放器不可用")

        try:
            music_type = kwargs.get("music_type", self._infer_music_type(action))
            volume = kwargs.get("volume", self._current_volume)

            if self.ros_node is not None:
                self._publish_music_command(music_type, volume)
                return ActionResult(True, ActionType.PLAY_MUSIC, f"正在播放{music_type}音乐")

            return ActionResult(True, ActionType.PLAY_MUSIC, f"播放{music_type}，音量{volume}%")

        except Exception as e:
            logger.error(f"音乐播放失败: {e}")
            return ActionResult(False, ActionType.PLAY_MUSIC, f"音乐播放失败: {e}")

    def _infer_music_type(self, action: str) -> str:
        if any(k in action for k in ["欢快", "开心", "快乐", "庆祝"]):
            return "欢快治愈"
        elif any(k in action for k in ["平静", "安静", "舒缓", "放松"]):
            return "平静舒缓"
        elif any(k in action for k in ["悲伤", "难过", "低沉"]):
            return "温暖舒缓"
        elif any(k in action for k in ["舒缓", "柔和"]):
            return "轻柔"
        return "轻音乐"

    def _publish_music_command(self, music_type: str, volume: int) -> None:
        if self.ros_node is not None:
            msg = {"command": "play_music", "type": music_type, "volume": volume}
            self.ros_node.publish_message("/robot/music_control", msg)

    def is_available(self) -> bool:
        return True


class RobotFunctions:
    """机器人函数调用管理器"""

    def __init__(self, ros_node: Optional[Any] = None):
        self.ros_node = ros_node
        self.light = LightController(ros_node)
        self.music = MusicPlayer(ros_node)

    def execute_action(self, action_str: str, emotion: str = "neutral", **kwargs) -> ActionResult:
        """根据动作字符串执行相应操作"""
        action_str = action_str.strip()

        if action_str in [ActionType.PLAY_MUSIC.value, "播放音乐"]:
            return self.music.execute(action_str, **kwargs)
        elif action_str in [ActionType.ADJUST_LIGHT.value, "调节灯光"]:
            return self.light.execute(action_str, **kwargs)
        elif action_str in [ActionType.NO_ACTION.value, "无动作"]:
            return ActionResult(True, ActionType.NO_ACTION, "无需执行动作")
        else:
            return ActionResult(False, ActionType.CUSTOM, f"未知动作类型: {action_str}")

    def execute_from_llm_result(self, llm_result: Dict[str, Any]) -> List[ActionResult]:
        """从 LLM 返回结果中提取并执行动作"""
        results = []

        if not isinstance(llm_result, dict):
            return results

        execution = llm_result.get("execution", {})
        if isinstance(execution, dict):
            action = execution.get("action", "无动作")
            emotion = execution.get("emotion", "neutral")
            result = self.execute_action(action, emotion=emotion)
            results.append(result)

        return results


_robot_functions_instance: Optional[RobotFunctions] = None


def get_robot_functions(ros_node: Optional[Any] = None) -> RobotFunctions:
    global _robot_functions_instance
    if _robot_functions_instance is None:
        _robot_functions_instance = RobotFunctions(ros_node)
    return _robot_functions_instance


def execute_robot_action(action: str, **kwargs) -> ActionResult:
    rf = get_robot_functions()
    return rf.execute_action(action, **kwargs)