# llm_api.py
import os
import json
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tts import speak_sync
from utils import logger
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
import config  # 使用集中化配置
from memory_rag import LongTermMemory
from pad_model import PADEmotionEngine
from multi_agent import create_multi_agent_coordinator, MultiAgentCoordinator
from physical_expression import get_expression_controller
from decision_tracer import DecisionTracer, NodeType, ModelType

load_dotenv()

# ================== 配置加载 ==================
API_KEY = config.API_KEY
USE_MOCK_LLM = not bool(API_KEY)
USE_MULTI_AGENT = True  # 启用多代理协作

if USE_MOCK_LLM:
    logger.info("LLM_API_KEY 未设置：使用本地模拟回复。")

# ---------- Memory 缓存 ----------
_SESSION_MAX_TURNS = config.SESSION_HISTORY_LEN
_session_memory: Deque[Dict[str, Any]] = deque(maxlen=_SESSION_MAX_TURNS * 2)

_client: Optional[OpenAI] = None

# 初始化新模块
_memory = LongTermMemory()
_pad_engine = PADEmotionEngine()
_multi_agent_coordinator: Optional[MultiAgentCoordinator] = None
_expression_controller = None

def _get_expression_controller():
    """获取并初始化物理表达控制器"""
    global _expression_controller
    if _expression_controller is None:
        from ros_client import global_ros_manager
        _expression_controller = get_expression_controller(global_ros_manager)
        _expression_controller.start()
        _pad_engine.bind_expression_controller(_expression_controller)
        logger.info("[物理表达] 控制器已绑定到 PAD 引擎")
    return _expression_controller

def _get_multi_agent_coordinator() -> MultiAgentCoordinator:
    """获取多代理协调器单例"""
    global _multi_agent_coordinator
    if _multi_agent_coordinator is None:
        _multi_agent_coordinator = create_multi_agent_coordinator(_memory)
        logger.info("[多代理] Agentic Reasoning 协调器已初始化")
    return _multi_agent_coordinator

# 优化后的后缀提示词：更强调简洁性以节省 Token 并降低解析失败率
_EMPATHY_VISION_SUFFIX = """
## 约束 (Constraints):
- 必须综合 VLM 画面与 STT 语义。
- 严禁空回复。若由于网络或视觉缺失无法判断，请用温柔的话术引导。
- **必须**仅输出 JSON 格式。
- 保持 reply 在 45 字以内，节省 Token 消耗。
"""

def clear_memory() -> None:
    _session_memory.clear()

def _get_client() -> Optional[OpenAI]:
    global _client
    if USE_MOCK_LLM:
        return None
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=config.BASE_URL)
    return _client

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    稳健的 JSON 提取器：支持 Markdown 代码块包裹或文本中嵌入的 JSON。
    """
    text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试正则提取第一个 { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    return None

def _build_user_content(emotion: str, user_text: str, vision_desc: Optional[str]) -> str:
    vd = (vision_desc or "").strip() or "（暂无有效画面描述）"
    stt = (user_text or "").strip() or "（暂无语音识别内容）"
    return f"【VLM】: {vd}\n【STT】: {stt}\n【EMO】: {emotion}"

def _append_turn(user_content: str, assistant_reply: str) -> None:
    _session_memory.append({"role": "user", "content": user_content})
    _session_memory.append({"role": "assistant", "content": assistant_reply})

_cached_system_prompt = None

def call_llm(emotion: str, user_text: str, vision_desc: str = "", prompt_file: str = "prompt.txt") -> dict:
    user_content = _build_user_content(emotion, user_text, vision_desc)
    client = _get_client()

    if USE_MOCK_LLM or not client:
        return {"emotion": emotion, "action": "none", "reply": "（本地模式）我收到啦。"}

    # 初始化决策追踪器
    decision_tracer = DecisionTracer.get_instance()

    try:
        # 开始延迟追踪
        decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id="unknown")
        
        # 1. 记忆检索 (RAG)
        current_context = f"视觉: {vision_desc}\n语音: {user_text}\n情绪: {emotion}"
        memory_recall = _memory.recall(current_context)
        
        # 缓存 prompt 文件内容，避免每次调用都重新读取
        global _cached_system_prompt
        if _cached_system_prompt is None:
            with open(prompt_file, "r", encoding="utf-8") as f:
                _cached_system_prompt = f.read().strip()
        
        # 构建系统提示词
        system_prompt = _cached_system_prompt + "\n" + _EMPATHY_VISION_SUFFIX
        if memory_recall:
            system_prompt += f"\n\n## 历史记忆参考 (Memory Recall):\n{memory_recall}"

        messages = [{"role": "system", "content": system_prompt}] + list(_session_memory) + [{"role": "user", "content": user_content}]

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT,
        )
        
        raw_reply = response.choices[0].message.content or ""
        result = _extract_json(raw_reply)

        if not result:
            logger.error(f"JSON 解析彻底失败，原始输出: {raw_reply}")
            result = {"execution": {"emotion": emotion, "action": "none", "reply": raw_reply[:60] if raw_reply else "我在听哦。"}}

        # 确保关键字段存在
        reply_text = result.get("execution", {}).get("reply", "...") if "execution" in result else result.get("reply", "...")
        _append_turn(user_content, reply_text)
        
        # 结束延迟追踪
        decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id="unknown")
        
        return result

    except Exception as e:
        # 结束延迟追踪（异常情况）
        decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id="unknown")
        logger.error(f"LLM 链路异常: {e}")
        return {"emotion": "neutral", "action": "none", "reply": "抱歉，我刚才走神了，能再说一遍吗？"}

def get_response(face_emotion: str, voice_text: str, enable_tts: bool = True, vision_desc: str = "") -> tuple:
    # 核心调用
    result = call_llm(face_emotion, voice_text, vision_desc=vision_desc)

    # 逻辑提取与兼容性处理
    if "execution" in result:
        reply_text = result["execution"].get("reply", "")
        emotion_tag = result["execution"].get("emotion", face_emotion)
    else:
        reply_text = result.get("reply", "")
        emotion_tag = result.get("emotion", face_emotion)

    # 2. 长期记忆存储
    if voice_text:
        _memory.save_memory(voice_text, reply_text, emotion_tag)

    # 3. 更新 PAD 情绪引擎
    sentiment_score = 0
    emotion_map = {"happy": 1.0, "sad": -1.0, "angry": -0.8, "fear": -0.6, "surprise": 0.5, "disgust": -0.7}
    sentiment_score = emotion_map.get(face_emotion, 0)

    # 简单模拟音量评分
    volume_score = 0.5 if voice_text else 0.0
    _pad_engine.update(sentiment_score, volume_score, face_emotion)

    # 获取根据情绪状态调整的 TTS 参数
    tts_params = _pad_engine.get_tts_params()

    audio_path = None
    if enable_tts and reply_text:
        # speak_sync 现在支持传入情绪调节参数
        audio_path = speak_sync(reply_text, tts_params=tts_params)

    return result, audio_path


def get_response_with_multi_agent(
    face_emotion: str,
    voice_text: str,
    enable_tts: bool = True,
    vision_desc: str = ""
) -> tuple:
    """
    使用多代理协作系统生成回复 (Agentic Reasoning)
    流程：感知代理 -> 记忆代理 -> 执行代理 -> 最终决策
    """
    if not USE_MULTI_AGENT:
        return get_response(face_emotion, voice_text, enable_tts, vision_desc)

    # 初始化决策追踪器
    decision_tracer = DecisionTracer.get_instance()

    try:
        # 开始延迟追踪
        decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id="unknown")
        
        # 确保物理表达控制器已初始化
        _get_expression_controller()

        coordinator = _get_multi_agent_coordinator()

        current_context = f"视觉: {vision_desc} 语音: {voice_text} 情绪: {face_emotion}"
        execution_result = coordinator.think(
            vision_desc=vision_desc,
            audio_text=voice_text,
            current_emotion=face_emotion,
            context=current_context
        )

        reply_text = execution_result.reply
        emotion_tag = execution_result.emotion
        action = execution_result.action
        music_type = execution_result.music_type

        logger.info(f"[多代理] 最终决策: {execution_result.final_decision}")
        logger.info(f"[多代理] 回复内容: {reply_text}")

        result = {
            "execution": {
                "reply": reply_text,
                "emotion": emotion_tag,
                "action": action,
                "music_type": music_type,
                "reasoning_chain": execution_result.reasoning_chain,
                "should_suppress": execution_result.should_suppress
            }
        }

        # 更新会话记忆（修复会话失忆问题）
        user_content = _build_user_content(face_emotion, voice_text, vision_desc)
        _append_turn(user_content, reply_text)

        # 长期记忆存储
        if voice_text:
            _memory.save_memory(voice_text, reply_text, emotion_tag)

        # 更新 PAD 情绪引擎
        sentiment_score = 0
        emotion_map = {"happy": 1.0, "sad": -1.0, "angry": -0.8, "fear": -0.6, "surprise": 0.5, "disgust": -0.7, "caring": 0.3, "supportive": 0.2, "empathetic": 0.1, "helpful": 0.1}
        sentiment_score = emotion_map.get(emotion_tag, 0)

        volume_score = 0.5 if voice_text else 0.0
        _pad_engine.update(sentiment_score, volume_score, emotion_tag)

        # 获取 PAD 物理表达描述（用于日志和调试）
        pad_values = _pad_engine.get_pad_values()
        emotion_state = _pad_engine.get_emotion_state_name()
        expression_desc = _expression_controller.get_expression_description() if _expression_controller else ""

        logger.info(f"[PAD] P={pad_values['P']:.2f} A={pad_values['A']:.2f} D={pad_values['D']:.2f} | 状态: {emotion_state}")
        if expression_desc:
            logger.info(f"[物理表达] {expression_desc}")

        tts_params = _pad_engine.get_tts_params()

        audio_path = None
        if enable_tts and reply_text:
            audio_path = speak_sync(reply_text, tts_params=tts_params)

            if music_type:
                logger.info(f"[多代理] 建议播放音乐: {music_type}")

        # 结束延迟追踪
        decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id="unknown")
        
        return result, audio_path

    except Exception as e:
        # 结束延迟追踪（异常情况）
        decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id="unknown")
        logger.error(f"[多代理] Agentic Reasoning 出错: {e}")
        import traceback
        traceback.print_exc()
        return get_response(face_emotion, voice_text, enable_tts, vision_desc)
