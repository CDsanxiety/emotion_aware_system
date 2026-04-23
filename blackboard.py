# blackboard.py
"""
分布式事件总线架构
从单例黑板进化为三层数据模型 + 发布-订阅架构
"""
import time
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import copy


class EventType(Enum):
    RAW_VISION_UPDATE = "raw_vision_update"
    RAW_AUDIO_UPDATE = "raw_audio_update"
    SEMANTIC_SCENE_DETECTED = "semantic_scene_detected"
    SEMANTIC_EMOTION_UPDATE = "semantic_emotion_update"
    INTENT_USER_QUERY = "intent_user_query"
    INTENT_ROBOT_ACTION = "intent_robot_action"
    PRESENCE_CHANGED = "presence_changed"
    STATUS_CHANGED = "status_changed"


@dataclass
class Event:
    """事件对象"""
    event_type: EventType
    timestamp: float
    data: Dict[str, Any]
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "source": self.source
        }


class Subscriber:
    """订阅者"""
    def __init__(self, callback: Callable[[Event], None], event_types: List[EventType] = None):
        self.callback = callback
        self.event_types = event_types or []
        self.id = id(callback)

    def handle_event(self, event: Event) -> None:
        if not self.event_types or event.event_type in self.event_types:
            try:
                self.callback(event)
            except Exception as e:
                print(f"[EventBus] 事件处理失败: {e}")


class LayerMetadata:
    """元数据层：原始传感器数据"""
    def __init__(self):
        self.lock = threading.Lock()
        self.vision_raw: str = "无画面"
        self.audio_raw: str = ""
        self.presence: bool = False
        self.vision_timestamp: float = 0.0
        self.audio_timestamp: float = 0.0
        self._vision_history: List[Dict[str, Any]] = []
        self._max_history = 10

    def update_vision(self, desc: str, presence: bool) -> Dict[str, Any]:
        with self.lock:
            old_presence = self.presence
            self.vision_raw = desc
            self.presence = presence
            self.vision_timestamp = time.time()

            self._vision_history.append({
                "description": desc,
                "presence": presence,
                "timestamp": self.vision_timestamp
            })
            if len(self._vision_history) > self._max_history:
                self._vision_history.pop(0)

            return {
                "old_presence": old_presence,
                "new_presence": presence,
                "description": desc
            }

    def update_audio(self, text: str) -> bool:
        with self.lock:
            old_text = self.audio_raw
            self.audio_raw = text
            self.audio_timestamp = time.time()
            return text != old_text

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "vision_raw": self.vision_raw,
                "audio_raw": self.audio_raw,
                "presence": self.presence,
                "vision_timestamp": self.vision_timestamp,
                "audio_timestamp": self.audio_timestamp,
                "vision_history": copy.deepcopy(self._vision_history)
            }


class LayerSemantic:
    """语义层：对原始数据的语义解释"""
    def __init__(self):
        self.lock = threading.Lock()
        self.scene_type: str = ""
        self.emotion: str = "neutral"
        self.emotion_confidence: float = 0.5
        self.scene_confidence: float = 0.0
        self.decision_mode: str = "confident"
        self.extracted_entities: Dict[str, Any] = {}
        self.context_summary: str = ""
        self.update_timestamp: float = 0.0

    def update_scene(self, scene_type: str, confidence: float = 1.0) -> None:
        with self.lock:
            self.scene_type = scene_type
            self.scene_confidence = confidence
            self.update_timestamp = time.time()

    def update_emotion(self, emotion: str, confidence: float) -> None:
        with self.lock:
            self.emotion = emotion
            self.emotion_confidence = confidence
            self.update_timestamp = time.time()

    def update_decision_mode(self, mode: str) -> None:
        with self.lock:
            self.decision_mode = mode
            self.update_timestamp = time.time()

    def set_entities(self, entities: Dict[str, Any]) -> None:
        with self.lock:
            self.extracted_entities = entities
            self.update_timestamp = time.time()

    def set_context_summary(self, summary: str) -> None:
        with self.lock:
            self.context_summary = summary
            self.update_timestamp = time.time()

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "scene_type": self.scene_type,
                "scene_confidence": self.scene_confidence,
                "emotion": self.emotion,
                "emotion_confidence": self.emotion_confidence,
                "decision_mode": self.decision_mode,
                "extracted_entities": copy.deepcopy(self.extracted_entities),
                "context_summary": self.context_summary,
                "update_timestamp": self.update_timestamp
            }


class LayerIntent:
    """意图层：用户意图和机器人决策"""
    def __init__(self):
        self.lock = threading.Lock()
        self.user_intent: str = ""
        self.robot_action: str = "无动作"
        self.action_confidence: float = 0.0
        self.pending_inquiry: str = ""
        self.execution_status: str = "idle"
        self.last_action_timestamp: float = 0.0
        self.intent_history: List[Dict[str, Any]] = []
        self._max_history = 5

    def set_user_intent(self, intent: str) -> None:
        with self.lock:
            self.user_intent = intent
            self._append_history("intent", intent)

    def set_robot_action(self, action: str, confidence: float = 1.0) -> None:
        with self.lock:
            self.robot_action = action
            self.action_confidence = confidence
            self.last_action_timestamp = time.time()
            self._append_history("action", action)

    def set_pending_inquiry(self, inquiry: str) -> None:
        with self.lock:
            self.pending_inquiry = inquiry
            self._append_history("inquiry", inquiry)

    def set_execution_status(self, status: str) -> None:
        with self.lock:
            self.execution_status = status

    def _append_history(self, intent_type: str, value: str) -> None:
        self.intent_history.append({
            "type": intent_type,
            "value": value,
            "timestamp": time.time()
        })
        if len(self.intent_history) > self._max_history:
            self.intent_history.pop(0)

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "user_intent": self.user_intent,
                "robot_action": self.robot_action,
                "action_confidence": self.action_confidence,
                "pending_inquiry": self.pending_inquiry,
                "execution_status": self.execution_status,
                "last_action_timestamp": self.last_action_timestamp,
                "intent_history": copy.deepcopy(self.intent_history)
            }


class EventBus:
    """
    分布式事件总线
    实现发布-订阅架构，替代传统的轮询机制
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._subscribers: Dict[EventType, List[Subscriber]] = defaultdict(list)
        self._global_subscribers: List[Subscriber] = []
        self._lock_subscriber = threading.Lock()
        self._event_history: List[Event] = []
        self._max_history = 100
        self._layer_metadata = LayerMetadata()
        self._layer_semantic = LayerSemantic()
        self._layer_intent = LayerIntent()
        self._polling_mode = False

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def metadata(self) -> LayerMetadata:
        return self._layer_metadata

    @property
    def semantic(self) -> LayerSemantic:
        return self._layer_semantic

    @property
    def intent(self) -> LayerIntent:
        return self._layer_intent

    def subscribe(self, callback: Callable[[Event], None], event_types: List[EventType] = None) -> int:
        """订阅事件"""
        subscriber = Subscriber(callback, event_types)
        with self._lock_subscriber:
            if event_types is None:
                self._global_subscribers.append(subscriber)
            else:
                for et in event_types:
                    if subscriber not in self._subscribers[et]:
                        self._subscribers[et].append(subscriber)
        return subscriber.id

    def unsubscribe(self, subscriber_id: int) -> None:
        """取消订阅"""
        with self._lock_subscriber:
            self._global_subscribers = [s for s in self._global_subscribers if s.id != subscriber_id]
            for et in self._subscribers:
                self._subscribers[et] = [s for s in self._subscribers[et] if s.id != subscriber_id]

    def publish(self, event: Event) -> None:
        """发布事件"""
        with self._lock_subscriber:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            notified = set()
            for subscriber in self._global_subscribers:
                subscriber.handle_event(event)
                notified.add(subscriber.id)

            for subscriber in self._subscribers.get(event.event_type, []):
                if subscriber.id not in notified:
                    subscriber.handle_event(event)

    def emit_vision_update(self, desc: str, presence: bool, source: str = "vision") -> None:
        """发布视觉更新事件"""
        changes = self._layer_metadata.update_vision(desc, presence)

        event = Event(
            event_type=EventType.RAW_VISION_UPDATE,
            timestamp=time.time(),
            data={
                "description": desc,
                "presence": presence,
                "changes": changes
            },
            source=source
        )
        self.publish(event)

        if changes.get("new_presence") != changes.get("old_presence"):
            self.emit_presence_changed(presence, source)

    def emit_audio_update(self, text: str, source: str = "audio") -> None:
        """发布音频更新事件"""
        if self._layer_metadata.update_audio(text):
            event = Event(
                event_type=EventType.RAW_AUDIO_UPDATE,
                timestamp=time.time(),
                data={"text": text},
                source=source
            )
            self.publish(event)

    def emit_scene_detected(self, scene_type: str, confidence: float = 1.0) -> None:
        """发布场景检测事件"""
        self._layer_semantic.update_scene(scene_type, confidence)

        event = Event(
            event_type=EventType.SEMANTIC_SCENE_DETECTED,
            timestamp=time.time(),
            data={
                "scene_type": scene_type,
                "confidence": confidence
            },
            source="scene_detector"
        )
        self.publish(event)

    def emit_emotion_update(self, emotion: str, confidence: float) -> None:
        """发布情绪更新事件"""
        self._layer_semantic.update_emotion(emotion, confidence)

        event = Event(
            event_type=EventType.SEMANTIC_EMOTION_UPDATE,
            timestamp=time.time(),
            data={
                "emotion": emotion,
                "confidence": confidence
            },
            source="emotion_detector"
        )
        self.publish(event)

    def emit_user_query(self, query: str) -> None:
        """发布用户查询意图事件"""
        self._layer_intent.set_user_intent(query)

        event = Event(
            event_type=EventType.INTENT_USER_QUERY,
            timestamp=time.time(),
            data={"query": query},
            source="nlu"
        )
        self.publish(event)

    def emit_robot_action(self, action: str, confidence: float = 1.0) -> None:
        """发布机器人动作意图事件"""
        self._layer_intent.set_robot_action(action, confidence)

        event = Event(
            event_type=EventType.INTENT_ROBOT_ACTION,
            timestamp=time.time(),
            data={
                "action": action,
                "confidence": confidence
            },
            source="decision_module"
        )
        self.publish(event)

    def emit_presence_changed(self, presence: bool, source: str = "unknown") -> None:
        """发布用户在场状态变化事件"""
        event = Event(
            event_type=EventType.PRESENCE_CHANGED,
            timestamp=time.time(),
            data={"presence": presence},
            source=source
        )
        self.publish(event)

    def emit_status_changed(self, status: str) -> None:
        """发布状态变化事件"""
        self._layer_intent.set_execution_status(status)

        event = Event(
            event_type=EventType.STATUS_CHANGED,
            timestamp=time.time(),
            data={"status": status},
            source="system"
        )
        self.publish(event)

    def get_full_state(self) -> Dict[str, Any]:
        """获取完整的三层状态"""
        return {
            "metadata": self._layer_metadata.get_snapshot(),
            "semantic": self._layer_semantic.get_snapshot(),
            "intent": self._layer_intent.get_snapshot()
        }

    def enable_polling_mode(self) -> None:
        """启用轮询模式（兼容旧代码）"""
        self._polling_mode = True
        logger.warning("[EventBus] 轮询模式已启用，优先使用事件驱动")

    def get_event_history(self, event_type: EventType = None, limit: int = 10) -> List[Event]:
        """获取事件历史"""
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type][-limit:]
        return self._event_history[-limit:]


class Blackboard:
    """
    兼容层：保留旧接口，内部调用 EventBus
    """
    def __init__(self):
        self._event_bus = EventBus.get_instance()

    def register_change_callback(self, callback: Callable[[str, Any], None]) -> None:
        event_type_map = {
            "vision_desc_changed": EventType.RAW_VISION_UPDATE,
            "presence_changed": EventType.PRESENCE_CHANGED,
            "speech_text_changed": EventType.RAW_AUDIO_UPDATE,
            "robot_status_changed": EventType.STATUS_CHANGED,
        }

        def wrapped_callback(event: Event):
            callback(event.event_type.value, event.data)

        for old_type, new_type in event_type_map.items():
            if f"{old_type}" in str(callback):
                self._event_bus.subscribe(wrapped_callback, [new_type])
                break
        else:
            self._event_bus.subscribe(wrapped_callback)

    def update_vision(self, desc: str, presence: bool) -> None:
        self._event_bus.emit_vision_update(desc, presence, source="blackboard_compat")

    def update_speech(self, text: str) -> None:
        self._event_bus.emit_audio_update(text, source="blackboard_compat")

    def get_vision_data(self) -> Dict[str, Any]:
        return self._event_bus.metadata.get_snapshot()

    def get_speech_data(self) -> Dict[str, Any]:
        return {
            "text": self._event_bus.metadata.audio_raw,
            "update_time": self._event_bus.metadata.audio_timestamp
        }

    def set_robot_status(self, status: str) -> None:
        self._event_bus.emit_status_changed(status)

    def get_robot_status(self) -> str:
        return self._event_bus.intent.execution_status

    @property
    def user_presence(self) -> bool:
        return self._event_bus.metadata.presence

    @property
    def last_speech_text(self) -> str:
        return self._event_bus.metadata.audio_raw

    @property
    def last_speech_time(self) -> float:
        return self._event_bus.metadata.audio_timestamp


from utils import logger


_global_event_bus = EventBus.get_instance()
