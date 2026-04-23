# social_norms.py
"""
社交规范模块
实现多人在场时的社交互动
- 礼貌插话策略
- 根据 PAD 状态进行社交博弈
- 社交距离管理
"""
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from utils import logger


class SocialDistance(Enum):
    """社交距离等级"""
    INTIMATE = "intimate"      # 亲密距离 (0-0.5m)
    PERSONAL = "personal"      # 个人距离 (0.5-1.2m)
    SOCIAL = "social"          # 社交距离 (1.2-3.6m)
    PUBLIC = "public"          # 公共距离 (3.6m+)


class ConversationRole(Enum):
    """对话角色"""
    SPEAKER = "speaker"        # 发言者
    LISTENER = "listener"      # 倾听者
    OBSERVER = "observer"      # 观察者
    INTERRUPTER = "interrupter"  # 插话者


class InterruptionType(Enum):
    """插话类型"""
    POLITE = "polite"          # 礼貌插话
    URGENT = "urgent"          # 紧急插话
    SOCIAL = "social"          # 社交插话
    ERROR = "error"            # 错误纠正


@dataclass
class SocialContext:
    """社交上下文"""
    people_present: List[str]  # 在场人员
    current_speaker: Optional[str]  # 当前发言者
    conversation_topic: Optional[str]  # 对话主题
    conversation_duration: float  # 对话持续时间
    social_distance: SocialDistance  # 社交距离
    environment: str  # 环境类型


@dataclass
class PersonState:
    """人员状态"""
    user_id: str
    pad_state: Dict[str, float]  # PAD 状态
    emotional_state: str  # 情绪状态
    engagement_level: float  # 参与度 (0-1)
    last_speaking_time: float  # 最后发言时间


class SocialNorms:
    """
    社交规范管理器
    处理多人在场时的社交互动
    """
    def __init__(self):
        self._person_states: Dict[str, PersonState] = {}
        self._conversation_history: List[Dict[str, Any]] = []
        self._last_update_time = time.time()
        
        # 社交规则配置
        self._interruption_thresholds = {
            InterruptionType.POLITE: 0.6,  # 礼貌插话阈值
            InterruptionType.URGENT: 0.9,  # 紧急插话阈值
            InterruptionType.SOCIAL: 0.4,  # 社交插话阈值
            InterruptionType.ERROR: 0.8    # 错误纠正阈值
        }
        
        # 对话轮次配置
        self._max_turn_duration = 60.0  # 最大发言时长
        self._min_turn_duration = 2.0   # 最小发言时长
        self._turn_transition_time = 1.0  # 轮次转换时间
    
    def update_person_state(self, user_id: str, pad_state: Dict[str, float],
                          emotional_state: str, engagement_level: float) -> None:
        """更新人员状态
        
        Args:
            user_id: 用户ID
            pad_state: PAD状态
            emotional_state: 情绪状态
            engagement_level: 参与度
        """
        self._person_states[user_id] = PersonState(
            user_id=user_id,
            pad_state=pad_state,
            emotional_state=emotional_state,
            engagement_level=engagement_level,
            last_speaking_time=time.time()
        )
        self._last_update_time = time.time()
    
    def should_interrupt(self, context: SocialContext, robot_purpose: str,
                        target_user: Optional[str] = None) -> Tuple[bool, InterruptionType]:
        """判断是否应该插话
        
        Args:
            context: 社交上下文
            robot_purpose: 机器人目的
            target_user: 目标用户
            
        Returns:
            (是否应该插话, 插话类型)
        """
        # 紧急情况
        if robot_purpose in ["emergency", "safety", "urgent"]:
            return True, InterruptionType.URGENT
        
        # 错误纠正
        if robot_purpose == "correction":
            return True, InterruptionType.ERROR
        
        # 检查对话状态
        if context.current_speaker:
            speaker_state = self._person_states.get(context.current_speaker)
            if speaker_state:
                # 发言时间过长
                if context.conversation_duration > self._max_turn_duration:
                    return True, InterruptionType.POLITE
                
                # 发言时间过短（避免打断刚开始的发言）
                if context.conversation_duration < self._min_turn_duration:
                    return False, InterruptionType.POLITE
        
        # 社交插话
        if robot_purpose in ["greeting", "farewell", "offer"]:
            # 检查是否有合适的插话时机
            if self._is_good_interruption_moment(context):
                return True, InterruptionType.SOCIAL
        
        # 目标用户相关
        if target_user:
            user_state = self._person_states.get(target_user)
            if user_state:
                # 目标用户参与度低
                if user_state.engagement_level < 0.3:
                    return True, InterruptionType.SOCIAL
                
                # 目标用户情绪激动
                if self._is_emotionally_engaged(user_state):
                    return True, InterruptionType.POLITE
        
        return False, InterruptionType.POLITE
    
    def _is_good_interruption_moment(self, context: SocialContext) -> bool:
        """判断是否是合适的插话时机
        
        Args:
            context: 社交上下文
            
        Returns:
            是否是合适的插话时机
        """
        # 对话暂停
        if not context.current_speaker:
            return True
        
        # 对话轮次转换
        if context.conversation_duration > self._min_turn_duration:
            # 检查对话历史，看是否有自然停顿
            if self._conversation_history:
                last_entry = self._conversation_history[-1]
                if time.time() - last_entry.get("timestamp", 0) > self._turn_transition_time:
                    return True
        
        return False
    
    def _is_emotionally_engaged(self, person_state: PersonState) -> bool:
        """判断是否情绪激动
        
        Args:
            person_state: 人员状态
            
        Returns:
            是否情绪激动
        """
        pad = person_state.pad_state
        # 高唤醒度或极端愉悦/不愉悦
        if pad.get("A", 0) > 0.5 or abs(pad.get("P", 0)) > 0.6:
            return True
        return False
    
    def generate_interruption_script(self, context: SocialContext, 
                                   interruption_type: InterruptionType,
                                   target_user: Optional[str] = None) -> str:
        """生成插话脚本
        
        Args:
            context: 社交上下文
            interruption_type: 插话类型
            target_user: 目标用户
            
        Returns:
            插话脚本
        """
        scripts = {
            InterruptionType.POLITE: [
                "抱歉打断一下，",
                "不好意思，我想补充一点，",
                "打扰了，关于这个问题，",
                "请允许我插句话，"
            ],
            InterruptionType.URGENT: [
                "紧急情况，",
                "注意，",
                "重要提醒，",
                "需要立即处理，"
            ],
            InterruptionType.SOCIAL: [
                "你好，",
                "大家好，",
                "打扰一下，",
                "嗨，"
            ],
            InterruptionType.ERROR: [
                "对不起，我觉得可能有误，",
                "抱歉，我需要纠正一下，",
                "等一下，我认为，",
                "打扰了，这里可能需要调整，"
            ]
        }
        
        # 选择合适的脚本
        script_options = scripts.get(interruption_type, scripts[InterruptionType.POLITE])
        script = script_options[0]  # 简单起见，选择第一个
        
        # 针对特定用户
        if target_user:
            user_state = self._person_states.get(target_user)
            if user_state:
                # 根据用户情绪调整语气
                if user_state.emotional_state == "happy":
                    script = f"嗨 {target_user}，"
                elif user_state.emotional_state == "sad":
                    script = f"你好 {target_user}，我注意到你看起来有点难过，"
                elif user_state.emotional_state == "angry":
                    script = f"{target_user}，冷静一下，"
        
        return script
    
    def calculate_social_distance(self, user_id: str, robot_position: Dict[str, float],
                                 user_position: Dict[str, float]) -> SocialDistance:
        """计算社交距离
        
        Args:
            user_id: 用户ID
            robot_position: 机器人位置
            user_position: 用户位置
            
        Returns:
            社交距离等级
        """
        import math
        
        # 计算欧氏距离
        distance = math.sqrt(
            (robot_position.get("x", 0) - user_position.get("x", 0)) ** 2 +
            (robot_position.get("y", 0) - user_position.get("y", 0)) ** 2
        )
        
        # 映射到社交距离等级
        if distance < 0.5:
            return SocialDistance.INTIMATE
        elif distance < 1.2:
            return SocialDistance.PERSONAL
        elif distance < 3.6:
            return SocialDistance.SOCIAL
        else:
            return SocialDistance.PUBLIC
    
    def get_social_strategy(self, context: SocialContext) -> Dict[str, Any]:
        """获取社交策略
        
        Args:
            context: 社交上下文
            
        Returns:
            社交策略
        """
        strategy = {
            "engagement_priority": [],
            "conversation_style": "neutral",
            "distance_preference": SocialDistance.SOCIAL,
            "interaction_mode": "polite"
        }
        
        # 根据在场人员和状态确定优先级
        for user_id, state in self._person_states.items():
            # 参与度高的用户优先
            priority_score = state.engagement_level
            # 情绪激动的用户优先
            if self._is_emotionally_engaged(state):
                priority_score += 0.3
            # 长时间未发言的用户优先
            time_since_speaking = time.time() - state.last_speaking_time
            if time_since_speaking > 60:  # 超过1分钟未发言
                priority_score += 0.2
            
            strategy["engagement_priority"].append((user_id, priority_score))
        
        # 排序优先级
        strategy["engagement_priority"].sort(key=lambda x: x[1], reverse=True)
        strategy["engagement_priority"] = [user_id for user_id, _ in strategy["engagement_priority"]]
        
        # 根据环境调整对话风格
        if context.environment == "formal":
            strategy["conversation_style"] = "formal"
            strategy["interaction_mode"] = "reserved"
        elif context.environment == "casual":
            strategy["conversation_style"] = "casual"
            strategy["interaction_mode"] = "friendly"
        
        # 根据社交距离调整策略
        if context.social_distance == SocialDistance.INTIMATE:
            strategy["interaction_mode"] = "personal"
        elif context.social_distance == SocialDistance.PUBLIC:
            strategy["interaction_mode"] = "formal"
        
        return strategy
    
    def resolve_social_conflict(self, conflicts: List[Tuple[str, str, float]]) -> str:
        """解决社交冲突
        
        Args:
            conflicts: 冲突列表，每个元素为 (用户A, 用户B, 冲突程度)
            
        Returns:
            冲突解决策略
        """
        # 找到冲突程度最高的
        if not conflicts:
            return "维持现状"
        
        conflicts.sort(key=lambda x: x[2], reverse=True)
        highest_conflict = conflicts[0]
        user_a, user_b, conflict_level = highest_conflict
        
        # 根据冲突程度和用户状态制定策略
        if conflict_level > 0.7:
            # 高冲突：分开处理
            return f"分别与 {user_a} 和 {user_b} 单独交流"
        elif conflict_level > 0.4:
            # 中等冲突：中立调解
            return f"邀请 {user_a} 和 {user_b} 共同参与讨论"
        else:
            # 低冲突：忽视
            return "维持现状"
    
    def record_conversation(self, speaker: str, content: str, duration: float) -> None:
        """记录对话
        
        Args:
            speaker: 发言者
            content: 内容
            duration: 时长
        """
        entry = {
            "speaker": speaker,
            "content": content,
            "duration": duration,
            "timestamp": time.time()
        }
        self._conversation_history.append(entry)
        
        # 限制历史记录长度
        if len(self._conversation_history) > 50:
            self._conversation_history.pop(0)
        
        # 更新发言者状态
        if speaker in self._person_states:
            state = self._person_states[speaker]
            state.last_speaking_time = time.time()
    
    def get_conversation_summary(self) -> str:
        """获取对话摘要
        
        Returns:
            对话摘要
        """
        if not self._conversation_history:
            return "无对话记录"
        
        # 统计发言次数
        speaker_counts = {}
        for entry in self._conversation_history:
            speaker = entry.get("speaker")
            if speaker:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        # 生成摘要
        summary = "对话摘要：\n"
        for speaker, count in speaker_counts.items():
            summary += f"- {speaker}: {count}次发言\n"
        
        # 最近的发言
        if self._conversation_history:
            last_entry = self._conversation_history[-1]
            summary += f"\n最近发言：{last_entry.get('speaker')} - {last_entry.get('content')[:50]}..."
        
        return summary


# 全局社交规范管理器实例
_global_social_norms = None


def get_social_norms() -> SocialNorms:
    """获取社交规范管理器实例"""
    global _global_social_norms
    if _global_social_norms is None:
        _global_social_norms = SocialNorms()
    return _global_social_norms


def should_interrupt_safely(context: SocialContext, robot_purpose: str,
                          target_user: Optional[str] = None) -> Tuple[bool, InterruptionType]:
    """安全判断是否应该插话"""
    norms = get_social_norms()
    return norms.should_interrupt(context, robot_purpose, target_user)


def generate_polite_interruption(context: SocialContext, 
                               interruption_type: InterruptionType,
                               target_user: Optional[str] = None) -> str:
    """生成礼貌插话脚本"""
    norms = get_social_norms()
    return norms.generate_interruption_script(context, interruption_type, target_user)


def get_social_strategy_for_context(context: SocialContext) -> Dict[str, Any]:
    """获取社交策略"""
    norms = get_social_norms()
    return norms.get_social_strategy(context)