"""
multi_agent.py
多代理协作决策系统
实现 Agentic Reasoning：感知代理、记忆代理、执行代理分工协作
"""
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from utils import logger
#from llm_api import call_llm


@dataclass
class PerceptionResult:
    """感知代理结果"""
    raw_vision: str
    raw_audio: str
    emotion: str
    extracted_details: List[str] = field(default_factory=list)
    scene_context: str = ""
    potential_issues: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class MemoryResult:
    """记忆代理结果"""
    relevant_memories: List[str] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    recent_emotions: List[str] = field(default_factory=list)
    special_dates: List[str] = field(default_factory=list)
    health_status: str = ""
    reasoning: str = ""


@dataclass
class ExecutionResult:
    """执行代理结果"""
    reply: str
    emotion: str
    action: str
    music_type: Optional[str] = None
    reasoning_chain: List[str] = field(default_factory=list)
    should_suppress: List[str] = field(default_factory=list)
    final_decision: str = ""


class PerceptionAgent:
    """
    感知代理：解析 VLM 细节，理解场景上下文
    职责：提取关键物体、人物状态、环境细节
    """
    def __init__(self):
        self.name = "PerceptionAgent"

    def analyze(self, vision_desc: str, audio_text: str, current_emotion: str) -> PerceptionResult:
        """
        分析视觉和音频输入，提取关键细节
        """
        result = PerceptionResult(
            raw_vision=vision_desc,
            raw_audio=audio_text,
            emotion=current_emotion
        )

        if not vision_desc or vision_desc == "无画面":
            result.scene_context = "暂无视觉信息"
            result.reasoning = "视觉信息缺失，保持简单回应"
            return result

        vision_lower = vision_desc.lower()
        audio_lower = audio_text.lower() if audio_text else ""

        self._extract_details(result, vision_lower, audio_lower)
        self._analyze_scene(result, vision_lower)
        self._detect_potential_issues(result, vision_lower, audio_lower)

        result.reasoning = (
            f"感知分析完成：发现{len(result.extracted_details)}个细节，"
            f"识别到{len(result.potential_issues)}个潜在关注点"
        )

        logger.info(f"[{self.name}] {result.reasoning}")
        return result

    def _extract_details(self, result: PerceptionResult, vision: str, audio: str) -> None:
        """提取关键细节"""
        detail_keywords = {
            "杯子": ["杯子", "水杯", "茶杯", "咖啡杯", "马克杯"],
            "桌子": ["桌子", "桌面", "茶几", "书桌"],
            "环境": ["乱", "脏", "整洁", "干净", "凌乱"],
            "人物": ["主人", "用户", "人", "孩子", "家人"],
            "表情": ["笑", "开心", "高兴", "难过", "悲伤", "疲惫", "累"],
            "光线": ["暗", "黑", "亮", "光线", "灯光"],
            "物体": ["手机", "电脑", "书", "文件", "包裹", "外卖"]
        }

        for category, keywords in detail_keywords.items():
            for keyword in keywords:
                if keyword in vision or keyword in audio:
                    detail = f"[{category}] {keyword}"
                    if detail not in result.extracted_details:
                        result.extracted_details.append(detail)

    def _analyze_scene(self, result: PerceptionResult, vision: str) -> None:
        """分析场景类型"""
        if any(word in vision for word in ["回家", "进门", "到家", "回来"]):
            result.scene_context = "主人刚回家"
        elif any(word in vision for word in ["客厅", "沙发", "电视"]):
            result.scene_context = "客斤放松"
        elif any(word in vision for word in ["厨房", "餐厅", "餐桌"]):
            result.scene_context = "餐斤区域"
        elif any(word in vision for word in ["书房", "工作", "电脑"]):
            result.scene_context = "工作状态"
        elif any(word in vision for word in ["卧室", "床", "休息"]):
            result.scene_context = "休息状态"
        else:
            result.scene_context = "日常场景"

    def _detect_potential_issues(self, result: PerceptionResult, vision: str, audio: str) -> None:
        """检测潜在关注点"""
        issue_mapping = {
            "未洗的杯子": ["杯子", "脏", "乱"],
            "主人看起来累": ["累", "疲惫", "无力", "困"],
            "环境光线不足": ["暗", "黑", "光线不足"],
            "主人情绪低落": ["难过", "悲伤", "哭", "沮丧"],
            "主人工作忙": ["电脑", "文件", "忙", "工作"]
        }

        for issue, keywords in issue_mapping.items():
            if any(kw in vision or kw in audio for kw in keywords):
                if issue not in result.potential_issues:
                    result.potential_issues.append(issue)


class MemoryAgent:
    """
    记忆代理：检索历史上下文，理解用户画像
    职责：查询长期记忆、用户健康状态、特殊日期等
    """
    def __init__(self, memory_system=None):
        self.name = "MemoryAgent"
        self.memory_system = memory_system

    def retrieve(self, current_context: str, perception: PerceptionResult) -> MemoryResult:
        """
        检索相关记忆，返回用户画像和历史上下文
        """
        result = MemoryResult()

        if not self.memory_system:
            result.reasoning = "记忆系统未初始化，使用默认上下文"
            return result

        try:
            combined_context = f"{current_context} {' '.join(perception.extracted_details)}"

            recall_result = self.memory_system.recall(combined_context)
            if recall_result:
                result.relevant_memories = [recall_result]
                result.reasoning = f"检索到{len(result.relevant_memories)}条相关记忆"
            else:
                result.reasoning = "未检索到相关记忆"

            self._extract_user_profile(result, perception)
            self._check_special_dates(result)
            self._assess_health_status(result, perception)

            logger.info(f"[{self.name}] {result.reasoning}")
            logger.info(f"[{self.name}] 用户画像: 健康状态={result.health_status}, 特殊日期={result.special_dates}")

        except Exception as e:
            logger.error(f"[{self.name}] 记忆检索失败: {e}")
            result.reasoning = f"记忆检索出错: {str(e)}"

        return result

    def _extract_user_profile(self, result: MemoryResult, perception: PerceptionResult) -> None:
        """提取用户画像"""
        user_profile_keywords = {
            "工作繁忙": ["忙", "工作", "加班", "压力"],
            "感冒中": ["感冒", "发烧", "咳嗽", "不舒服", "医院"],
            "心情不好": ["难过", "伤心", "生气", "沮丧"],
            "健身中": ["锻炼", "健身", "跑步", "运动"],
            "学习紧张": ["考试", "学习", "复习", "作业"]
        }

        for profile, keywords in user_profile_keywords.items():
            if any(kw in perception.raw_audio.lower() for kw in keywords):
                result.user_profile[profile] = True

        emotion_keywords = {
            "happy": ["开心", "高兴", "快乐", "棒"],
            "sad": ["难过", "伤心", "沮丧"],
            "angry": ["生气", "愤怒", "烦躁"],
            "fear": ["害怕", "担心", "紧张"],
            "neutral": ["一般", "正常", "还好"]
        }

        for emotion, keywords in emotion_keywords.items():
            if any(kw in perception.raw_audio.lower() for kw in keywords):
                if emotion not in result.recent_emotions:
                    result.recent_emotions.append(emotion)

    def _check_special_dates(self, result: MemoryResult) -> None:
        """检查特殊日期"""
        from memory_rag import global_memory

        today = time.strftime("%m-%d")

        special_date_mapping = {
            "01-01": "元旦",
            "02-14": "情人节",
            "05-01": "劳动节",
            "06-01": "儿童节",
            "10-01": "国庆节",
            "12-25": "圣诞节"
        }

        user_birthday = global_memory.get_user_birthday()
        if user_birthday:
            birthday_month_day = user_birthday[5:] if len(user_birthday) >= 5 else user_birthday
            special_date_mapping[birthday_month_day] = "生日"

        if today in special_date_mapping:
            result.special_dates.append(f"今天是{special_date_mapping[today]}")
            result.reasoning += f"，检测到特殊日期: {special_date_mapping[today]}"

    def _assess_health_status(self, result: MemoryResult, perception: PerceptionResult) -> None:
        """评估健康状态"""
        health_keywords_positive = ["好", "恢复", "痊愈", "没事"]
        health_keywords_negative = ["感冒", "发烧", "咳嗽", "不舒服", "难受", "累", "疲惫"]

        audio_lower = perception.raw_audio.lower()

        if any(kw in audio_lower for kw in health_keywords_positive):
            result.health_status = "恢复中"
        elif any(kw in audio_lower for kw in health_keywords_negative):
            result.health_status = "需要关心"
        else:
            result.health_status = "正常"


class ExecutionAgent:
    """
    执行代理：综合感知和记忆，生成最优决策
    职责：权衡多方信息，做出"老友式"回应
    """
    def __init__(self):
        self.name = "ExecutionAgent"

    def decide(
        self,
        perception: PerceptionResult,
        memory: MemoryResult,
        original_emotion: str
    ) -> ExecutionResult:
        """
        综合分析，生成最终决策
        """
        result = ExecutionResult(
            reply="",
            emotion=original_emotion,
            action="none"
        )

        reasoning_chain = []
        should_suppress = []

        reasoning_chain.append(f"原始情绪识别: {original_emotion}")
        reasoning_chain.append(f"感知细节: {', '.join(perception.extracted_details) or '无'}")

        if memory.relevant_memories:
            reasoning_chain.append(f"相关记忆: {memory.relevant_memories[0][:50]}...")

        reasoning_chain.append(f"用户健康状态: {memory.health_status}")
        reasoning_chain.append(f"用户画像: {list(memory.user_profile.keys()) or '普通'}")

        self._apply_suppression_rules(perception, memory, should_suppress)

        if should_suppress:
            reasoning_chain.append(f"触发抑制规则: {', '.join(should_suppress)}")

        final_response, emotion, action, music_type = self._generate_response(
            perception, memory, original_emotion, should_suppress
        )

        result.reply = final_response
        result.emotion = emotion
        result.action = action
        result.music_type = music_type
        result.reasoning_chain = reasoning_chain
        result.should_suppress = should_suppress
        result.final_decision = f"基于{len(reasoning_chain)}步推理，选择情绪{emotion}和动作{action}"

        logger.info(f"[{self.name}] {result.final_decision}")
        for step in reasoning_chain:
            logger.info(f"[{self.name}] 推理链: {step}")

        return result

    def _apply_suppression_rules(
        self,
        perception: PerceptionResult,
        memory: MemoryResult,
        should_suppress: List[str]
    ) -> None:
        """
        应用抑制规则：避免在某些情况下提及特定事项
        核心：不要增加用户压力
        """
        is_tired = "主人看起来累" in perception.potential_issues or "累" in perception.raw_audio.lower()
        is_sick = memory.health_status == "需要关心" or memory.health_status == "恢复中"
        is_busy = "工作繁忙" in memory.user_profile

        if "未洗的杯子" in perception.potential_issues:
            if is_tired or is_sick:
                should_suppress.append("不要提醒洗杯子（会增加压力）")
                reasoning = "因为主人累了/身体不适，抑制提醒洗杯子"
                logger.info(f"[{self.name}] {reasoning}")

        if is_busy and "未洗的杯子" in perception.potential_issues:
            should_suppress.append("不要提醒洗杯子（主人在忙）")

        if is_sick:
            should_suppress.append("不要提及敏感话题")
            should_suppress.append("优先关心健康状态")

        if memory.health_status == "恢复中" and "感冒" in perception.extracted_details:
            should_suppress.append("不要过度提及感冒不适")
            reasoning = "主人正在恢复，温和关怀即可"
            logger.info(f"[{self.name}] {reasoning}")

    def _generate_response(
        self,
        perception: PerceptionResult,
        memory: MemoryResult,
        original_emotion: str,
        should_suppress: List[str]
    ) -> tuple:
        """
        生成最终回复
        返回: (reply, emotion, action, music_type)
        """
        # 构建上下文
        context = f"""
        感知信息:
        - 场景: {perception.scene_context}
        - 提取的细节: {', '.join(perception.extracted_details) or '无'}
        - 潜在问题: {', '.join(perception.potential_issues) or '无'}
        - 原始情绪: {original_emotion}
        - 原始音频: {perception.raw_audio}
        - 原始视觉: {perception.raw_vision}
        
        记忆信息:
        - 相关记忆: {memory.relevant_memories[0][:100] if memory.relevant_memories else '无'}
        - 用户健康状态: {memory.health_status}
        - 用户画像: {list(memory.user_profile.keys()) or '普通'}
        - 最近情绪: {', '.join(memory.recent_emotions) or '无'}
        - 特殊日期: {', '.join(memory.special_dates) or '无'}
        
        抑制规则:
        - {', '.join(should_suppress) or '无'}
        """

        # 构建用户文本，用于调用 LLM
        user_text = perception.raw_audio or "用户没有说话"
        vision_desc = perception.raw_vision or "无视觉信息"

        try:
            # 调用 LLM 生成回复
            llm_result = call_llm(original_emotion, user_text, vision_desc)
            
            # 解析 LLM 回复
            if "execution" in llm_result:
                reply = llm_result["execution"].get("reply", "你好呀！有什么我可以帮你的吗？")
                emotion = llm_result["execution"].get("emotion", original_emotion if original_emotion != "neutral" else "friendly")
                action = llm_result["execution"].get("action", "none")
                music_type = llm_result["execution"].get("music_type", None)
            else:
                reply = llm_result.get("reply", "你好呀！有什么我可以帮你的吗？")
                emotion = llm_result.get("emotion", original_emotion if original_emotion != "neutral" else "friendly")
                action = llm_result.get("action", "none")
                music_type = None
        except Exception as e:
            logger.error(f"[{self.name}] LLM 调用失败: {e}")
            # 降级到默认回复
            reply = "你好呀！有什么我可以帮你的吗？"
            emotion = original_emotion if original_emotion != "neutral" else "friendly"
            action = "interact"
            music_type = None

        return reply, emotion, action, music_type


class MultiAgentCoordinator:
    """
    多代理协调器：整合三个代理，实现 Agentic Reasoning
    """
    def __init__(self, memory_system=None):
        self.name = "MultiAgentCoordinator"
        self.perception_agent = PerceptionAgent()
        self.memory_agent = MemoryAgent(memory_system)
        self.execution_agent = ExecutionAgent()

        self._lock = threading.Lock()

    def think(
        self,
        vision_desc: str,
        audio_text: str,
        current_emotion: str,
        context: str = ""
    ) -> ExecutionResult:
        """
        完整的 Agentic Reasoning 流程
        1. 感知代理：解析 VLM 细节
        2. 记忆代理：检索历史上下文
        3. 执行代理：综合决策生成
        """
        with self._lock:
            logger.info(f"[{self.name}] 开始 Agentic Reasoning...")
            logger.info(f"[{self.name}] 输入: vision={vision_desc[:50]}..., audio={audio_text[:30]}...")

            start_time = time.time()

            perception_result = self.perception_agent.analyze(vision_desc, audio_text, current_emotion)

            current_context = f"视觉:{vision_desc} 语音:{audio_text} 情绪:{current_emotion}"
            memory_result = self.memory_agent.retrieve(current_context, perception_result)

            execution_result = self.execution_agent.decide(perception_result, memory_result, current_emotion)

            elapsed = time.time() - start_time
            logger.info(f"[{self.name}] Agentic Reasoning 完成，耗时: {elapsed:.3f}s")

            return execution_result


def create_multi_agent_coordinator(memory_system=None) -> MultiAgentCoordinator:
    """工厂函数：创建多代理协调器"""
    return MultiAgentCoordinator(memory_system)
