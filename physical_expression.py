"""
physical_expression.py
PAD 物理表达系统：将 Pleasure-Arousal-Dominance 三维情绪模型映射到机器人"肢体语言"
包括：LED 灯带颜色/亮度、头部动作速度/角度、表情显示等
"""
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from utils import logger


class LEDColor(Enum):
    """预设 LED 颜色"""
    WARM_WHITE = (255, 200, 150)    # 暖白色 - 愉悦
    SOFT_PINK = (255, 150, 180)     # 柔粉色 - 温馨
    CALM_BLUE = (100, 150, 255)     # 平静蓝 - 安抚
    ENERGY_GREEN = (100, 255, 150)  # 能量绿 - 活力
    ALERT_ORANGE = (255, 150, 50)   # 警示橙 - 激活
    SAD_PURPLE = (150, 100, 200)     # 忧郁紫 - 悲伤
    CALM_WARM = (255, 180, 120)     # 温和暖色 - 关怀
    NEUTRAL = (200, 200, 200)       # 中性灰白


@dataclass
class HeadPose:
    """头部姿态"""
    yaw_angle: float = 0.0      # 水平旋转角度 (度) - 左负右正
    pitch_angle: float = 0.0    # 俯仰角度 (度) - 上负下正
    speed: float = 1.0           # 动作速度系数 (0.5-2.0)
    nod_frequency: float = 0.0  # 点头频率 (Hz, 0=不点头)


@dataclass
class PhysicalExpression:
    """物理表达参数"""
    led_color: Tuple[int, int, int] = (200, 200, 200)
    led_brightness: float = 0.5   # 0.0-1.0
    led_animation: str = "steady" # steady, pulse, breathe, rainbow
    head_pose: HeadPose = None
    body_language: str = "neutral" # neutral, welcoming, caring, alert, subdued
    facial_expression: str = "neutral" # neutral, happy, concerned, curious, sad
    # 运动动力学参数
    acceleration: float = 1.0      # 加速度系数 (0.1-2.0)
    deceleration: float = 1.0      # 减速度系数 (0.1-2.0)
    max_speed: float = 1.0         # 最大速度系数 (0.5-1.5)
    jerk: float = 0.5              # 抖动率 (0.0-1.0)
    motion_smoothness: float = 0.8 # 运动平滑度 (0.0-1.0)
    deceleration_profile: str = "linear" # linear, exponential, polynomial


class PADToPhysicalMapper:
    """
    PAD 到物理表达的映射器
    将三维情绪空间映射到具体的物理动作参数
    """

    # PAD 阈值定义
    P_HIGH_THRESHOLD = 0.3      # 愉悦度高
    P_LOW_THRESHOLD = -0.3       # 愉悦度低
    A_HIGH_THRESHOLD = 0.3       # 激活度高
    A_LOW_THRESHOLD = -0.3       # 激活度低
    D_HIGH_THRESHOLD = 0.3       # 优势度高
    D_LOW_THRESHOLD = -0.3       # 优势度低

    def __init__(self):
        self.current_expression = PhysicalExpression()
        self.target_expression = PhysicalExpression()
        self.transition_speed = 0.1  # 平滑过渡速度
        self._lock = threading.Lock()

    def map_pad_to_expression(self, p: float, a: float, d: float) -> PhysicalExpression:
        """
        将 PAD 值映射到物理表达

        参数:
            p: Pleasure (-1.0 到 1.0) - 愉悦度
            a: Arousal (-1.0 到 1.0) - 激活度
            d: Dominance (-1.0 到 1.0) - 优势度

        返回:
            PhysicalExpression: 物理表达参数
        """
        with self._lock:
            expression = PhysicalExpression()

            # 1. 映射 LED 颜色和亮度 (基于 P 和 A)
            expression.led_color, expression.led_brightness = self._map_led(p, a)

            # 2. 映射 LED 动画效果 (基于 A)
            expression.led_animation = self._map_led_animation(a)

            # 3. 映射头部姿态 (基于 P, A, D)
            expression.head_pose = self._map_head_pose(p, a, d)

            # 4. 映射肢体语言 (基于 D)
            expression.body_language = self._map_body_language(p, a, d)

            # 5. 映射表情 (基于 P)
            expression.facial_expression = self._map_facial_expression(p, a)

            # 6. 映射运动动力学参数 (基于 P, A, D)
            self._map_motion_dynamics(expression, p, a, d)

            self.target_expression = expression
            return expression

    def _map_motion_dynamics(self, expression: PhysicalExpression, p: float, a: float, d: float):
        """
        映射运动动力学参数

        规则:
        - P 高: 运动更流畅，减速度更平滑
        - P 低: 运动更缓慢，减速度更大（悲伤时）
        - A 高: 加速度更大，抖动率更高
        - A 低: 加速度更小，运动更平稳
        - D 高: 最大速度更高，运动更直接
        - D 低: 最大速度更低，运动更谨慎
        """
        # 加速度: 激活度影响，高激活度时加速度更大
        expression.acceleration = 1.0 + (a * 0.5)

        # 减速度: 愉悦度影响，悲伤时减速度更大
        if p < self.P_LOW_THRESHOLD:
            expression.deceleration = 1.5 + abs(p) * 0.5  # 悲伤时更快减速
        else:
            expression.deceleration = 1.0 + (a * 0.3)  # 激活度影响减速度

        # 最大速度: 优势度和激活度共同影响
        expression.max_speed = 1.0 + (d * 0.3) + (a * 0.2)

        # 抖动率: 激活度影响，高激活度时抖动更多
        expression.jerk = 0.5 + (a * 0.3)

        # 运动平滑度: 愉悦度影响，高愉悦度时更平滑
        expression.motion_smoothness = 0.8 + (p * 0.15)

        # 减速曲线: 基于情绪状态选择
        if p < self.P_LOW_THRESHOLD:
            # 悲伤时使用指数减速，更平滑
            expression.deceleration_profile = "exponential"
        elif a > self.A_HIGH_THRESHOLD:
            # 高激活度时使用线性减速，更直接
            expression.deceleration_profile = "linear"
        else:
            # 其他情况使用多项式减速
            expression.deceleration_profile = "polynomial"

        # 限制范围
        expression.acceleration = max(0.1, min(2.0, expression.acceleration))
        expression.deceleration = max(0.1, min(2.0, expression.deceleration))
        expression.max_speed = max(0.5, min(1.5, expression.max_speed))
        expression.jerk = max(0.0, min(1.0, expression.jerk))
        expression.motion_smoothness = max(0.0, min(1.0, expression.motion_smoothness))

    def calculate_deceleration_profile(self, initial_speed: float, target_speed: float, 
                                     profile_type: str, deceleration_factor: float) -> List[float]:
        """
        计算减速曲线

        参数:
            initial_speed: 初始速度
            target_speed: 目标速度
            profile_type: 减速曲线类型 (linear, exponential, polynomial)
            deceleration_factor: 减速系数

        返回:
            List[float]: 减速过程中的速度序列
        """
        speed_diff = initial_speed - target_speed
        if speed_diff <= 0:
            return [initial_speed]

        # 生成10个采样点的减速曲线
        steps = 10
        speed_profile = []

        for i in range(steps + 1):
            t = i / steps  # 归一化时间

            if profile_type == "linear":
                # 线性减速
                current_speed = initial_speed - (speed_diff * t)

            elif profile_type == "exponential":
                # 指数减速（更平滑）
                current_speed = target_speed + (speed_diff * (1 - t) ** (2 * deceleration_factor))

            elif profile_type == "polynomial":
                # 多项式减速（平滑过渡）
                current_speed = target_speed + (speed_diff * (1 - t) ** 3)

            else:
                # 默认线性减速
                current_speed = initial_speed - (speed_diff * t)

            current_speed = max(target_speed, current_speed)
            speed_profile.append(current_speed)

        return speed_profile

    def _map_led(self, p: float, a: float) -> Tuple[Tuple[int, int, int], float]:
        """
        映射 LED 颜色和亮度

        规则:
        - P 高 + A 高: 暖白色，高亮度 (开心兴奋)
        - P 高 + A 低: 柔粉色，中亮度 (放松愉悦)
        - P 低 + A 高: 警示橙，中亮度 (紧张不安)
        - P 低 + A 低: 忧郁紫，低亮度 (悲伤疲惫)
        - D 高: 能量绿 (自信)
        - D 低: 温和暖色 (顺从关怀)
        """
        brightness = 0.5 + (a * 0.3)  # A 影响基础亮度

        if p > self.P_HIGH_THRESHOLD and a > self.A_HIGH_THRESHOLD:
            color = LEDColor.WARM_WHITE.value
            brightness = 0.8
        elif p > self.P_HIGH_THRESHOLD and a <= self.A_HIGH_THRESHOLD:
            color = LEDColor.SOFT_PINK.value
            brightness = 0.6
        elif p <= self.P_LOW_THRESHOLD and a > self.A_HIGH_THRESHOLD:
            color = LEDColor.ALERT_ORANGE.value
            brightness = 0.7
        elif p <= self.P_LOW_THRESHOLD and a <= self.A_HIGH_THRESHOLD:
            color = LEDColor.SAD_PURPLE.value
            brightness = 0.4
        elif d > self.D_HIGH_THRESHOLD if 'd' in dir() else False:
            color = LEDColor.ENERGY_GREEN.value
            brightness = 0.7
        elif d < self.D_LOW_THRESHOLD if 'd' in dir() else False:
            color = LEDColor.CALM_WARM.value
            brightness = 0.5
        else:
            color = LEDColor.NEUTRAL.value
            brightness = 0.5

        brightness = max(0.2, min(1.0, brightness))
        return color, brightness

    def _map_led_animation(self, a: float) -> str:
        """
        映射 LED 动画效果

        规则:
        - A 高: pulse (充满能量)
        - A 中: steady (稳定)
        - A 低: breathe (疲惫低沉)
        """
        if a > self.A_HIGH_THRESHOLD:
            return "pulse"
        elif a < self.A_LOW_THRESHOLD:
            return "breathe"
        else:
            return "steady"

    def _map_head_pose(self, p: float, a: float, d: float) -> HeadPose:
        """
        映射头部姿态

        规则:
        - A 高: 转头速度变快 (充满能量或焦急)
        - A 低: 转头速度变慢 (疲惫)
        - P 高: 头部上扬，轻快动作 (开心)
        - P 低: 头部下垂 (悲伤)
        - D 高: 头部挺直 (自信)
        - D 低: 头部微微下垂 (顺从关怀)
        """
        pose = HeadPose()

        # 激活度影响动作速度
        pose.speed = 1.0 + (a * 0.5)  # 0.5 到 1.5

        # 愉悦度影响俯仰角度
        if p > self.P_HIGH_THRESHOLD:
            pose.pitch_angle = -5.0  # 头部微微上扬
        elif p < self.P_LOW_THRESHOLD:
            pose.pitch_angle = 8.0   # 头部下垂
        else:
            pose.pitch_angle = 0.0

        # 优势度影响水平角度和头部挺直程度
        if d > self.D_HIGH_THRESHOLD:
            pose.yaw_angle = 0.0     # 挺直，自信
        elif d < self.D_LOW_THRESHOLD:
            pose.yaw_angle = -10.0  # 微微偏向一侧，表示顺从
            pose.pitch_angle += 3.0 # 头再低一点
        else:
            pose.yaw_angle = 0.0

        # 激活度影响点头频率 (高激活度时点头更频繁表示关注)
        if a > self.A_HIGH_THRESHOLD:
            pose.nod_frequency = 1.5  # Hz
        elif a < self.A_LOW_THRESHOLD:
            pose.nod_frequency = 0.0
        else:
            pose.nod_frequency = 0.5

        # 限制范围
        pose.yaw_angle = max(-30.0, min(30.0, pose.yaw_angle))
        pose.pitch_angle = max(-15.0, min(15.0, pose.pitch_angle))
        pose.speed = max(0.5, min(2.0, pose.speed))
        pose.nod_frequency = max(0.0, min(3.0, pose.nod_frequency))

        return pose

    def _map_body_language(self, p: float, a: float, d: float) -> str:
        """
        映射肢体语言类型
        """
        if p > self.P_HIGH_THRESHOLD and d > self.D_HIGH_THRESHOLD:
            return "welcoming"  # 开心自信 - 欢迎姿态
        elif p > self.P_HIGH_THRESHOLD:
            return "happy"       # 开心愉悦
        elif d < self.D_LOW_THRESHOLD:
            return "caring"      # 顺从关怀
        elif a > self.A_HIGH_THRESHOLD:
            return "alert"       # 高激活 - 警觉
        elif p < self.P_LOW_THRESHOLD:
            return "subdued"     # 悲伤低沉
        else:
            return "neutral"

    def _map_facial_expression(self, p: float, a: float) -> str:
        """
        映射面部表情
        """
        if p > self.P_HIGH_THRESHOLD:
            if a > self.A_HIGH_THRESHOLD:
                return "happy"      # 开心兴奋
            else:
                return "content"    # 满足平静
        elif p < self.P_LOW_THRESHOLD:
            if a > self.A_HIGH_THRESHOLD:
                return "concerned"  # 担忧不安
            else:
                return "sad"        # 悲伤
        elif a > self.A_HIGH_THRESHOLD:
            return "curious"       # 好奇警觉
        else:
            return "neutral"


class PhysicalExpressionController:
    """
    物理表达控制器
    负责执行具体的物理动作，发布到 ROS
    """
    def __init__(self, ros_manager=None):
        self.ros_manager = ros_manager
        self.mapper = PADToPhysicalMapper()
        self.current_expression = PhysicalExpression()
        self.is_active = False
        self._lock = threading.Lock()
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        """启动物理表达控制器"""
        if self.is_active:
            return

        self.is_active = True
        self._stop_event.clear()
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="PhysicalExpression"
        )
        self._update_thread.start()
        logger.info("[物理表达] 控制器已启动")

    def stop(self):
        """停止物理表达控制器"""
        if not self.is_active:
            return

        self.is_active = False
        self._stop_event.set()
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        logger.info("[物理表达] 控制器已停止")

    def update_emotion(self, p: float, a: float, d: float):
        """
        更新情绪状态，触发物理表达变化

        参数:
            p: Pleasure (-1.0 到 1.0)
            a: Arousal (-1.0 到 1.0)
            d: Dominance (-1.0 到 1.0)
        """
        with self._lock:
            self.current_expression = self.mapper.map_pad_to_expression(p, a, d)

    def _update_loop(self):
        """更新循环：平滑过渡并发布到 ROS"""
        while not self._stop_event.is_set():
            try:
                self._smooth_transition()
                self._apply_expression()
            except Exception as e:
                logger.error(f"[物理表达] 更新异常: {e}")

            self._stop_event.wait(0.1)  # 10Hz 更新频率

    def _smooth_transition(self):
        """平滑过渡到目标表达式"""
        with self._lock:
            # 获取目标表达式
            target = self.mapper.target_expression
            current = self.current_expression

            if not target or not current:
                return

            # 平滑过渡 LED 参数
            current.led_brightness = self._smooth_value(current.led_brightness, target.led_brightness)

            # 平滑过渡头部姿态
            if target.head_pose and current.head_pose:
                current.head_pose.yaw_angle = self._smooth_value(current.head_pose.yaw_angle, target.head_pose.yaw_angle)
                current.head_pose.pitch_angle = self._smooth_value(current.head_pose.pitch_angle, target.head_pose.pitch_angle)
                current.head_pose.speed = self._smooth_value(current.head_pose.speed, target.head_pose.speed)
                current.head_pose.nod_frequency = self._smooth_value(current.head_pose.nod_frequency, target.head_pose.nod_frequency)

            # 平滑过渡运动动力学参数
            current.acceleration = self._smooth_value(current.acceleration, target.acceleration)
            current.deceleration = self._smooth_value(current.deceleration, target.deceleration)
            current.max_speed = self._smooth_value(current.max_speed, target.max_speed)
            current.jerk = self._smooth_value(current.jerk, target.jerk)
            current.motion_smoothness = self._smooth_value(current.motion_smoothness, target.motion_smoothness)

            # 直接切换减速曲线类型
            current.deceleration_profile = target.deceleration_profile

            # 直接切换其他离散参数
            current.led_color = target.led_color
            current.led_animation = target.led_animation
            current.body_language = target.body_language
            current.facial_expression = target.facial_expression

    def _smooth_value(self, current: float, target: float, factor: float = 0.1) -> float:
        """平滑过渡数值"""
        return current + (target - current) * factor

    def _apply_expression(self):
        """应用当前表达式到硬件"""
        if not self.ros_manager or not self.ros_manager.is_connected:
            return

        expr = self.current_expression

        # 构建 ROS 消息
        expression_msg = {
            "type": "physical_expression",
            "led": {
                "color": list(expr.led_color),
                "brightness": expr.led_brightness,
                "animation": expr.led_animation
            },
            "head": {
                "yaw_angle": expr.head_pose.yaw_angle if expr.head_pose else 0.0,
                "pitch_angle": expr.head_pose.pitch_angle if expr.head_pose else 0.0,
                "speed": expr.head_pose.speed if expr.head_pose else 1.0,
                "nod_frequency": expr.head_pose.nod_frequency if expr.head_pose else 0.0
            },
            "body_language": expr.body_language,
            "facial_expression": expr.facial_expression,
            "motion": {
                "acceleration": expr.acceleration,
                "deceleration": expr.deceleration,
                "max_speed": expr.max_speed,
                "jerk": expr.jerk,
                "motion_smoothness": expr.motion_smoothness,
                "deceleration_profile": expr.deceleration_profile
            }
        }

        try:
            self.ros_manager.publish_action(expression_msg)
            logger.debug(f"[物理表达] 已发布: LED={expr.led_color}, 动画={expr.led_animation}, "
                        f"加速度={expr.acceleration:.2f}, 减速度={expr.deceleration:.2f}, "
                        f"减速曲线={expr.deceleration_profile}")
        except Exception as e:
            logger.error(f"[物理表达] 发布失败: {e}")

    def get_expression_description(self) -> str:
        """获取当前物理表达的描述（用于调试）"""
        expr = self.current_expression
        head = expr.head_pose

        desc_parts = []

        # LED 描述
        color_names = {
            LEDColor.WARM_WHITE.value: "暖白",
            LEDColor.SOFT_PINK.value: "柔粉",
            LEDColor.CALM_BLUE.value: "平静蓝",
            LEDColor.ENERGY_GREEN.value: "能量绿",
            LEDColor.ALERT_ORANGE.value: "警示橙",
            LEDColor.SAD_PURPLE.value: "忧郁紫",
            LEDColor.CALM_WARM.value: "温和暖",
            LEDColor.NEUTRAL.value: "中性"
        }
        color_name = color_names.get(expr.led_color, "未知")
        brightness_pct = int(expr.led_brightness * 100)
        desc_parts.append(f"LED: {color_name} {brightness_pct}%")

        # 动画描述
        animation_names = {
            "steady": "稳定",
            "pulse": "脉动",
            "breathe": "呼吸",
            "rainbow": "彩虹"
        }
        anim_name = animation_names.get(expr.led_animation, "未知")
        desc_parts.append(f"动画: {anim_name}")

        # 头部描述
        if head:
            desc_parts.append(f"头部: yaw={head.yaw_angle:.1f}° pitch={head.pitch_angle:.1f}° 速度={head.speed:.1f}")

        # 肢体语言
        body_names = {
            "neutral": "中性",
            "welcoming": "欢迎",
            "happy": "开心",
            "caring": "关怀",
            "alert": "警觉",
            "subdued": "低沉"
        }
        body_name = body_names.get(expr.body_language, "未知")
        desc_parts.append(f"肢体: {body_name}")

        # 运动动力学描述
        desc_parts.append(f"运动: 加速度={expr.acceleration:.2f} 减速度={expr.deceleration:.2f} "
                        f"最大速度={expr.max_speed:.2f} 抖动={expr.jerk:.2f} "
                        f"平滑度={expr.motion_smoothness:.2f} 减速曲线={expr.deceleration_profile}")

        return " | ".join(desc_parts)


# 全局物理表达控制器
_global_expression_controller: Optional[PhysicalExpressionController] = None


def get_expression_controller(ros_manager=None) -> PhysicalExpressionController:
    """获取全局物理表达控制器单例"""
    global _global_expression_controller
    if _global_expression_controller is None:
        _global_expression_controller = PhysicalExpressionController(ros_manager)
    return _global_expression_controller


def create_expression_from_pad(p: float, a: float, d: float) -> PhysicalExpression:
    """工厂函数：从 PAD 值创建物理表达"""
    mapper = PADToPhysicalMapper()
    return mapper.map_pad_to_expression(p, a, d)
