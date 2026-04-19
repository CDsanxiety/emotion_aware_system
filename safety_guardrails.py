# safety_guardrails.py
"""
安全护栏模块 (Safety Guardrails)
为机器人决策引入确定性状态机安全过滤
防止 LLM 幻觉导致的危险动作
"""
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import threading


class SafetyLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGER = "danger"
    BLOCKED = "blocked"


class RobotMode(Enum):
    IDLE = "idle"
    MOVING = "moving"
    SPEAKING = "speaking"
    EXECUTING = "executing"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    passed: bool
    safety_level: SafetyLevel
    risk_factors: List[str]
    modified_action: Optional[str]
    blocked_reason: Optional[str]
    timestamp: float

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PhysicalState:
    """物理状态：传感器数据"""
    def __init__(self):
        self.lock = threading.Lock()
        self.obstacle_detected: bool = False
        self.obstacle_distance: float = float('inf')
        self.obstacle_direction: str = ""
        self.near_stairs: bool = False
        self.on_slope: bool = False
        self.battery_level: float = 100.0
        self.temperature: float = 25.0
        self.in_motion: bool = False
        self.current_position: str = "unknown"
        self.ground_clearance: float = 0.0
        self.last_sensor_update: float = 0.0

    def update_from_sensors(self, sensor_data: Dict[str, Any]) -> None:
        with self.lock:
            self.obstacle_detected = sensor_data.get("obstacle_detected", self.obstacle_detected)
            self.obstacle_distance = sensor_data.get("obstacle_distance", self.obstacle_distance)
            self.obstacle_direction = sensor_data.get("obstacle_direction", self.obstacle_direction)
            self.near_stairs = sensor_data.get("near_stairs", self.near_stairs)
            self.on_slope = sensor_data.get("on_slope", self.on_slope)
            self.battery_level = sensor_data.get("battery_level", self.battery_level)
            self.temperature = sensor_data.get("temperature", self.temperature)
            self.in_motion = sensor_data.get("in_motion", self.in_motion)
            self.current_position = sensor_data.get("position", self.current_position)
            self.ground_clearance = sensor_data.get("ground_clearance", self.ground_clearance)
            self.last_sensor_update = time.time()

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "obstacle_detected": self.obstacle_detected,
                "obstacle_distance": self.obstacle_distance,
                "obstacle_direction": self.obstacle_direction,
                "near_stairs": self.near_stairs,
                "on_slope": self.on_slope,
                "battery_level": self.battery_level,
                "temperature": self.temperature,
                "in_motion": self.in_motion,
                "position": self.current_position,
                "ground_clearance": self.ground_clearance,
                "last_update": self.last_sensor_update
            }


class SafetyRule:
    """安全规则"""
    def __init__(self, name: str, condition: Callable[[PhysicalState, Dict], bool],
                 risk_level: SafetyLevel, risk_factors: List[str],
                 action_modifier: Callable[[str], str] = None,
                 block_action: bool = False):
        self.name = name
        self.condition = condition
        self.risk_level = risk_level
        self.risk_factors = risk_factors
        self.action_modifier = action_modifier
        self.block_action = block_action

    def evaluate(self, physical_state: PhysicalState, action_context: Dict[str, Any]) -> Optional[SafetyCheckResult]:
        if self.condition(physical_state, action_context):
            modified_action = None
            blocked_reason = None

            if self.block_action:
                blocked_reason = f"安全规则 '{self.name}' 阻止了动作"
                return SafetyCheckResult(
                    passed=False,
                    safety_level=self.risk_level,
                    risk_factors=self.risk_factors,
                    modified_action=None,
                    blocked_reason=blocked_reason
                )

            if self.action_modifier:
                modified_action = self.action_modifier(action_context.get("action", ""))
            else:
                modified_action = action_context.get("action")

            return SafetyCheckResult(
                passed=True,
                safety_level=self.risk_level,
                risk_factors=self.risk_factors,
                modified_action=modified_action,
                blocked_reason=None
            )

        return None


class SafetyGuardrails:
    """
    安全护栏状态机
    确定性过滤所有 LLM 输出的动作
    """
    def __init__(self):
        self.physical_state = PhysicalState()
        self.current_mode = RobotMode.IDLE
        self.emergency_stop_active = False
        self._rules: List[SafetyRule] = []
        self._lock = threading.Lock()
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        self._rules = [
            SafetyRule(
                name="obstacle_collision_prevention",
                condition=lambda s, ctx: s.obstacle_detected and s.obstacle_distance < 0.5,
                risk_level=SafetyLevel.DANGER,
                risk_factors=["障碍物距离过近", f"距离: {0}米"],
                block_action=True
            ),

            SafetyRule(
                name="stairs_fall_prevention",
                condition=lambda s, ctx: s.near_stairs and "move" in ctx.get("action", "").lower(),
                risk_level=SafetyLevel.BLOCKED,
                risk_factors=["检测到楼梯/台阶", "移动动作被阻止"],
                block_action=True
            ),

            SafetyRule(
                name="slope_motion_caution",
                condition=lambda s, ctx: s.on_slope and s.in_motion,
                risk_level=SafetyLevel.CAUTION,
                risk_factors=["机器人位于斜坡上", "运动可能不稳定"]
            ),

            SafetyRule(
                name="low_battery_warning",
                condition=lambda s, ctx: s.battery_level < 20.0,
                risk_level=SafetyLevel.CAUTION,
                risk_factors=[f"电池电量过低: {0}%"]
            ),

            SafetyRule(
                name="high_temperature_warning",
                condition=lambda s, ctx: s.temperature > 45.0,
                risk_level=SafetyLevel.DANGER,
                risk_factors=[f"温度过高: {0}°C", "可能过热"]
            ),

            SafetyRule(
                name="speaking_while_moving",
                condition=lambda s, ctx: s.in_motion and "speak" in ctx.get("action", "").lower(),
                risk_level=SafetyLevel.CAUTION,
                risk_factors=["移动时说话可能影响语音识别"]
            ),

            SafetyRule(
                name="emergency_stop_overrides_all",
                condition=lambda s, ctx: self.emergency_stop_active,
                risk_level=SafetyLevel.BLOCKED,
                risk_factors=["紧急停止已激活"],
                block_action=True
            ),

            SafetyRule(
                name="dark_environment_caution",
                condition=lambda s, ctx: ctx.get("light_level", 1.0) < 0.2 and "move" in ctx.get("action", "").lower(),
                risk_level=SafetyLevel.CAUTION,
                risk_factors=["光线过暗", "移动可能不安全"]
            ),

            SafetyRule(
                name="music_volume_limit",
                condition=lambda s, ctx: ctx.get("volume", 70) > 85,
                risk_level=SafetyLevel.CAUTION,
                risk_factors=["音量过大可能损伤听力"],
                action_modifier=lambda a: a.replace("volume=100", "volume=70")
            ),
        ]

    def update_physical_state(self, sensor_data: Dict[str, Any]) -> None:
        """更新物理状态（从传感器）"""
        self.physical_state.update_from_sensors(sensor_data)

    def set_robot_mode(self, mode: RobotMode) -> None:
        """设置机器人模式"""
        with self._lock:
            self.current_mode = mode

    def activate_emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """激活紧急停止"""
        with self._lock:
            self.emergency_stop_active = True
            self.current_mode = RobotMode.EMERGENCY_STOP
            from utils import logger
            logger.warning(f"[Safety] 紧急停止激活: {reason}")

    def deactivate_emergency_stop(self) -> None:
        """取消紧急停止"""
        with self._lock:
            self.emergency_stop_active = False
            self.current_mode = RobotMode.IDLE

    def validate_action(self, action: str, context: Dict[str, Any] = None) -> SafetyCheckResult:
        """
        验证动作安全性
        所有 LLM 输出的动作都必须经过此函数过滤
        """
        if context is None:
            context = {}

        context["action"] = action

        if self.emergency_stop_active:
            return SafetyCheckResult(
                passed=False,
                safety_level=SafetyLevel.BLOCKED,
                risk_factors=["紧急停止已激活"],
                modified_action=None,
                blocked_reason="紧急停止已激活，所有动作被阻止"
            )

        highest_risk = SafetyLevel.SAFE
        all_risk_factors = []
        modified_action = action
        blocked = False
        blocked_reason = None

        for rule in self._rules:
            result = rule.evaluate(self.physical_state, context)
            if result:
                if result.safety_level.value > highest_risk.value:
                    highest_risk = result.safety_level

                all_risk_factors.extend(result.risk_factors)

                if result.blocked_reason:
                    blocked = True
                    blocked_reason = result.blocked_reason

                if result.modified_action:
                    modified_action = result.modified_action

        return SafetyCheckResult(
            passed=not blocked,
            safety_level=highest_risk,
            risk_factors=list(set(all_risk_factors)),
            modified_action=modified_action if not blocked else None,
            blocked_reason=blocked_reason
        )

    def get_safe_action(self, proposed_action: str, context: Dict[str, Any] = None) -> tuple[str, SafetyCheckResult]:
        """
        获取安全动作
        返回 (安全动作, 检查结果)
        """
        result = self.validate_action(proposed_action, context)

        if result.passed:
            return result.modified_action or proposed_action, result
        else:
            fallback = self._get_fallback_action(proposed_action, result)
            return fallback, result

    def _get_fallback_action(self, original_action: str, result: SafetyCheckResult) -> str:
        """根据安全检查结果获取后备动作"""
        if result.blocked_reason:
            if "move" in original_action.lower():
                return "stop_moving"
            elif "play" in original_action.lower():
                return "pause"
            else:
                return "无动作"

        return "无动作"

    def get_current_state(self) -> Dict[str, Any]:
        """获取当前安全状态"""
        return {
            "emergency_stop_active": self.emergency_stop_active,
            "current_mode": self.current_mode.value,
            "physical_state": self.physical_state.get_snapshot(),
            "active_rules_count": len([r for r in self._rules if r.condition(self.physical_state, {})]),
            "timestamp": time.time()
        }

    def add_custom_rule(self, rule: SafetyRule) -> None:
        """添加自定义安全规则"""
        with self._lock:
            self._rules.append(rule)


_global_safety_guardrails: Optional[SafetyGuardrails] = None
_safety_lock = threading.Lock()


def get_safety_guardrails() -> SafetyGuardrails:
    """获取全局安全护栏实例"""
    global _global_safety_guardrails
    if _global_safety_guardrails is None:
        with _safety_lock:
            if _global_safety_guardrails is None:
                _global_safety_guardrails = SafetyGuardrails()
    return _global_safety_guardrails


def validate_llm_action(action: str, context: Dict[str, Any] = None) -> SafetyCheckResult:
    """
    便捷函数：验证 LLM 输出的动作
    所有 LLM 建议的动作都应该调用此函数进行安全过滤
    """
    guardrails = get_safety_guardrails()
    return guardrails.validate_action(action, context)


def get_safe_action_from_llm(proposed_action: str, context: Dict[str, Any] = None) -> tuple[str, SafetyCheckResult]:
    """便捷函数：获取安全的动作"""
    guardrails = get_safety_guardrails()
    return guardrails.get_safe_action(proposed_action, context)
