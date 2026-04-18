import time
import threading
from typing import Optional, Callable, Dict, Any, List

class Blackboard:
    """全局状态黑板：所有感知模块只管往这里写，决策模块只管读"""
    def __init__(self):
        self.lock = threading.Lock()
        self.last_speech_text = ""
        self.last_speech_time = 0
        self.current_vision_desc = "无画面"
        self.vision_update_time = 0
        self.user_presence = False
        self.robot_status = "idle"
        self._vision_history: List[str] = []
        self._max_history = 10
        self._change_callbacks: List[Callable[[str, Any], None]] = []
        self._last_vision_desc = ""
        self._last_presence = False
        self._scene_change_threshold = 0.6

    def register_change_callback(self, callback: Callable[[str, Any], None]) -> None:
        with self.lock:
            self._change_callbacks.append(callback)

    def _notify_change(self, change_type: str, value: Any) -> None:
        for callback in self._change_callbacks:
            try:
                callback(change_type, value)
            except Exception as e:
                print(f"[Blackboard] 变化通知失败: {e}")

    def update_vision(self, desc: str, presence: bool) -> None:
        with self.lock:
            desc_changed = desc != self._last_vision_desc
            presence_changed = presence != self._last_presence

            self._last_vision_desc = self.current_vision_desc
            self.current_vision_desc = desc
            self.user_presence = presence
            self.vision_update_time = time.time()

            self._vision_history.append(desc)
            if len(self._vision_history) > self._max_history:
                self._vision_history.pop(0)

            if desc_changed:
                self._notify_change("vision_desc_changed", desc)
            if presence_changed:
                self._notify_change("presence_changed", presence)

    def update_speech(self, text: str) -> None:
        with self.lock:
            text_changed = text != self.last_speech_text
            self.last_speech_text = text
            self.last_speech_time = time.time()
            if text_changed:
                self._notify_change("speech_text_changed", text)

    def get_vision_data(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "description": self.current_vision_desc,
                "presence": self.user_presence,
                "update_time": self.vision_update_time
            }

    def get_speech_data(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "text": self.last_speech_text,
                "update_time": self.last_speech_time
            }

    def get_robot_status(self) -> str:
        with self.lock:
            return self.robot_status

    def set_robot_status(self, status: str) -> None:
        with self.lock:
            old_status = self.robot_status
            self.robot_status = status
            if old_status != status:
                self._notify_change("robot_status_changed", status)

    def detect_scene_change(self, keywords: List[str]) -> Optional[str]:
        with self.lock:
            if not self.current_vision_desc or self.current_vision_desc == "无画面":
                return None

            current_lower = self.current_vision_desc.lower()
            for keyword in keywords:
                if keyword.lower() in current_lower:
                    return keyword
            return None

    def get_vision_variance(self) -> float:
        with self.lock:
            if len(self._vision_history) < 2:
                return 0.0
            changes = sum(
                1 for i in range(1, len(self._vision_history))
                if self._vision_history[i] != self._vision_history[i-1]
            )
            return changes / (len(self._vision_history) - 1)

    def is_user_active(self, idle_threshold: float = 30.0) -> bool:
        with self.lock:
            if not self.user_presence:
                return False
            idle_time = time.time() - self.last_speech_time
            return idle_time < idle_threshold

    def get_idle_time(self) -> float:
        with self.lock:
            return time.time() - self.last_speech_time

    def check_proactive_conditions(
        self,
        scene_keywords: List[str],
        idle_threshold: float = 20.0,
        scene_change_threshold: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        with self.lock:
            if not self.user_presence:
                return None

            idle_time = time.time() - self.last_speech_time
            if idle_time < idle_threshold:
                return None

            scene_variance = self.get_vision_variance()
            if scene_variance < scene_change_threshold:
                return None

            scene_trigger = self.detect_scene_change(scene_keywords)
            if scene_trigger:
                return {
                    "type": "scene_trigger",
                    "trigger": scene_trigger,
                    "idle_time": idle_time,
                    "scene_variance": scene_variance
                }

            return None