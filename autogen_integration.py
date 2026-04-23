# autogen_integration.py
"""
AutoGen 集成模块
实现多代理系统，增强机器人决策能力
"""
import os
import time
from config import API_KEY

try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTO_GEN_AVAILABLE = True
except ImportError:
    AUTO_GEN_AVAILABLE = False

from typing import Dict, Any, Optional, List
from utils import logger


class AutoGenManager:
    def __init__(self):
        self.agents = {}
        self.group_chat = None
        self.group_chat_manager = None
        self._initialized = False
        self._last_analysis_time = 0.0
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 5.0

        if AUTO_GEN_AVAILABLE:
            self.config_list = [
                {
                    "model": "gpt-4",
                    "api_key": API_KEY,
                }
            ]
            self.llm_config = {
                "config_list": self.config_list,
                "seed": 42,
            }
            print("AutoGen 已安装，启用真实多代理协作")
        else:
            self.llm_config = None
            print("AutoGen 未安装，使用增强的模拟实现")

    def initialize_agents(self) -> bool:
        if self._initialized:
            return True

        if not AUTO_GEN_AVAILABLE or self.llm_config is None:
            return False

        try:
            self.agents["user"] = UserProxyAgent(
                name="User",
                system_message="用户代理，负责与用户交互并协调多代理讨论",
                code_execution_config=False,
                human_input_mode="NEVER",
            )

            self.agents["assistant"] = AssistantAgent(
                name="Assistant",
                system_message="助手代理，负责提供专业建议和综合分析",
                llm_config=self.llm_config,
            )

            self.agents["emotion_analyzer"] = AssistantAgent(
                name="EmotionAnalyzer",
                system_message="""你是情感分析专家。根据用户的话语和视觉描述，分析用户的情绪状态。
返回JSON格式：{"emotion": "happy|sad|angry|fear|surprise|neutral", "intensity": 0.0-1.0, "reasoning": "分析理由"}""",
                llm_config=self.llm_config,
            )

            self.agents["robot_controller"] = AssistantAgent(
                name="RobotController",
                system_message="""你是机器人动作规划专家。根据用户情绪和场景，决定机器人应该执行的动作。
可选动作：播放音乐(music_happy|music_calm)、调节灯光(light_warm|light_bright)、无动作(none)
返回JSON格式：{"action": "具体动作", "reasoning": "决策理由"}""",
                llm_config=self.llm_config,
            )

            self.group_chat = GroupChat(
                agents=list(self.agents.values()),
                messages=[],
                max_round=3,
                speaker_selection_method="round_robin",
            )

            self.group_chat_manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config=self.llm_config,
            )

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"AutoGen 代理初始化失败: {e}")
            return False

    def _generate_cache_key(self, user_text: str, vision_desc: str) -> str:
        return f"{hash(user_text + vision_desc) % 10000}"

    def _get_cached_result(self, user_text: str, vision_desc: str) -> Optional[Dict[str, Any]]:
        cache_key = self._generate_cache_key(user_text, vision_desc)
        if cache_key in self._analysis_cache:
            cached = self._analysis_cache[cache_key]
            if time.time() - cached.get("timestamp", 0) < self._cache_ttl:
                return cached.get("result")
        return None

    def _cache_result(self, user_text: str, vision_desc: str, result: Dict[str, Any]) -> None:
        cache_key = self._generate_cache_key(user_text, vision_desc)
        self._analysis_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

    def analyze_emotion_and_plan_action(self, user_text: str, vision_desc: str) -> Dict[str, Any]:
        cached = self._get_cached_result(user_text, vision_desc)
        if cached:
            return cached

        if AUTO_GEN_AVAILABLE and self._initialized:
            return self._real_multibot_analysis(user_text, vision_desc)
        else:
            return self._enhanced_fallback_analysis(user_text, vision_desc)

    def _real_multibot_analysis(self, user_text: str, vision_desc: str) -> Dict[str, Any]:
        context = f"用户输入: {user_text}\n视觉描述: {vision_desc}"

        try:
            analysis_prompt = f"""请分析以下情境并给出建议：

{context}

请依次进行：
1. 情感分析：分析用户当前的情绪状态
2. 动作规划：根据情绪状态，推荐机器人应执行的动作

请用JSON格式返回：
{{"emotion": "情绪类型", "intensity": 强度, "action": "推荐动作", "reasoning": "分析理由"}}
"""

            response = self.agents["user"].initiate_chat(
                self.group_chat_manager,
                message=analysis_prompt,
            )

            result = self._parse_agent_response(response)
            self._cache_result(user_text, vision_desc, result)
            return result

        except Exception as e:
            logger.error(f"AutoGen 多代理分析失败: {e}")
            return self._enhanced_fallback_analysis(user_text, vision_desc)

    def _parse_agent_response(self, response: Any) -> Dict[str, Any]:
        try:
            if hasattr(response, 'chat_history'):
                last_message = response.chat_history[-1] if response.chat_history else {}
                content = last_message.get('content', '')
                if 'emotion' in content.lower():
                    return self._extract_json_from_text(content)
        except:
            pass
        return {"emotion": "neutral", "action": "无动作", "reasoning": "分析完成"}

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        import re
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            try:
                import json
                return json.loads(json_match.group())
            except:
                pass
        return {"emotion": "neutral", "action": "无动作", "reasoning": text[:100]}

    def _enhanced_fallback_analysis(self, user_text: str, vision_desc: str) -> Dict[str, Any]:
        from uncertainty import get_uncertainty_manager
        from decision_tracer import get_decision_tracer

        tracer = get_decision_tracer()
        tracer.start_tracing()

        tracer.record_raw_sensor("vision", vision_desc)
        tracer.record_perception(vision_desc, confidence=1.0)

        uncertainty_mgr = get_uncertainty_manager()
        evidence, pad_state, decision_mode = uncertainty_mgr.analyze_and_update(
            user_text, vision_desc
        )

        tracer.record_emotion_detection(evidence.label, evidence.confidence, pad_state)
        tracer.record_uncertainty_reasoning(decision_mode.value, evidence.confidence, evidence.reasoning)

        action = self._rule_based_action_planning(evidence.label, user_text, vision_desc)

        tracer.record_action_selection(action, action, modified=False)

        if decision_mode.value == "query":
            inquiry_prompt = uncertainty_mgr.generate_inquiry_prompt(evidence)
            tracer.record_final_action("询问用户", True, {"inquiry": inquiry_prompt})
            tracer.end_tracing()
            return {
                "emotion": evidence.label,
                "confidence": evidence.confidence,
                "action": inquiry_prompt,
                "decision_mode": decision_mode.value,
                "pad_state": pad_state,
                "reasoning": evidence.reasoning,
                "needs_query": True
            }

        tracer.record_final_action(action, True)
        tracer.end_tracing()

        return {
            "emotion": evidence.label,
            "confidence": evidence.confidence,
            "action": action,
            "decision_mode": decision_mode.value,
            "pad_state": pad_state,
            "reasoning": evidence.reasoning,
            "needs_query": False
        }

    def _rule_based_emotion_detection(self, user_text: str, vision_desc: str) -> str:
        combined = (user_text + " " + vision_desc).lower()

        emotion_keywords = {
            "happy": ["开心", "高兴", "快乐", "欢快", "棒", "好开心", "太好了", "happy", "joy"],
            "sad": ["难过", "伤心", "悲伤", "哭", "沮丧", "郁闷", "sad"],
            "angry": ["生气", "愤怒", "恼火", "烦躁", "angry"],
            "fear": ["害怕", "恐惧", "担心", "紧张", "怕", "fear"],
            "surprise": ["惊讶", "吃惊", "意外", "惊奇", "surprise"],
        }

        for emotion, keywords in emotion_keywords.items():
            if any(kw in combined for kw in keywords):
                return emotion

        return "neutral"

    def _rule_based_action_planning(self, emotion: str, user_text: str, vision_desc: str) -> str:
        combined = (user_text + " " + vision_desc).lower()

        if emotion == "happy":
            return "music_happy"
        elif emotion == "sad":
            return "music_calm"
        elif any(kw in combined for kw in ["累", "疲惫", "困", "休息", "睡觉"]):
            return "music_calm"
        elif any(kw in combined for kw in ["暗", "黑", "关灯"]):
            return "light_warm"
        elif any(kw in combined for kw in ["亮", "开灯", "光线"]):
            return "light_bright"
        return "无动作"

    def chat(self, message: str, context: str = None) -> str:
        if not AUTO_GEN_AVAILABLE or not self._initialized:
            return f"[模拟响应] {message}"

        try:
            full_message = message
            if context:
                full_message = f"上下文: {context}\n\n消息: {message}"

            response = self.agents["user"].initiate_chat(
                self.group_chat_manager,
                message=full_message,
            )
            return str(response)
        except Exception as e:
            logger.error(f"AutoGen chat 失败: {e}")
            return f"[模拟响应] {message}"

    def get_agents(self) -> Dict[str, Any]:
        return self.agents

    def is_available(self) -> bool:
        return AUTO_GEN_AVAILABLE and self._initialized


global_autogen_manager = AutoGenManager()
