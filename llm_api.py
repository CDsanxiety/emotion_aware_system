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

load_dotenv()

# ================== 配置加载 ==================
API_KEY = config.API_KEY
USE_MOCK_LLM = not bool(API_KEY)

if USE_MOCK_LLM:
    logger.info("LLM_API_KEY 未设置：使用本地模拟回复。")

# ---------- Memory 缓存 ----------
_SESSION_MAX_TURNS = config.SESSION_HISTORY_LEN
_session_memory: Deque[Dict[str, Any]] = deque(maxlen=_SESSION_MAX_TURNS * 2)

_client: Optional[OpenAI] = None

# 初始化新模块
_memory = LongTermMemory()
_pad_engine = PADEmotionEngine()

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

def call_llm(emotion: str, user_text: str, vision_desc: str = "", prompt_file: str = "prompt.txt") -> dict:
    user_content = _build_user_content(emotion, user_text, vision_desc)
    client = _get_client()

    if USE_MOCK_LLM or not client:
        # 这里可以调用之前定义的 _mock_llm_result
        return {"emotion": emotion, "action": "none", "reply": "（本地模式）我收到啦。"}

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip() + "\n" + _EMPATHY_VISION_SUFFIX

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
            # 这里的兜底结构应当根据 prompt.txt 要求微调
            result = {"execution": {"emotion": emotion, "action": "none", "reply": raw_reply[:60] if raw_reply else "我在听哦。"}}

        # 确保关键字段存在
        _append_turn(user_content, result.get("execution", {}).get("reply", "...") if "execution" in result else result.get("reply", "..."))
        return result

    except Exception as e:
        logger.error(f"LLM 链路异常: {e}")
        return {"emotion": "neutral", "action": "none", "reply": "抱歉，我刚才走神了，能再说一遍吗？"}

def get_response(face_emotion: str, voice_text: str, enable_tts: bool = True, vision_desc: str = "") -> tuple:
    # 构建当前上下文
    current_context = f"【VLM】: {vision_desc}\n【STT】: {voice_text}\n【EMO】: {face_emotion}"
    
    # 记忆检索
    memory_recall = _memory.recall(current_context)
    if memory_recall:
        current_context += f"\n{memory_recall}"
    
    # 调用LLM
    result = call_llm(face_emotion, voice_text, vision_desc=vision_desc)
    
    # 兼容旧版的逻辑提取
    if "execution" in result:
        reply_text = result["execution"].get("reply", "")
        emotion_tag = result["execution"].get("emotion", face_emotion)
    else:
        reply_text = result.get("reply", "")
        emotion_tag = result.get("emotion", face_emotion)
    
    # 记忆存储
    if voice_text:
        _memory.save_memory(voice_text, reply_text, emotion_tag)
    
    # 更新PAD情绪状态
    # 简单的情绪评分转换：positive=1, negative=-1, neutral=0
    sentiment_score = 0
    if face_emotion == "happy":
        sentiment_score = 1.0
    elif face_emotion == "sad":
        sentiment_score = -1.0
    elif face_emotion == "angry":
        sentiment_score = -0.8
    elif face_emotion == "fear":
        sentiment_score = -0.6
    elif face_emotion == "surprise":
        sentiment_score = 0.5
    elif face_emotion == "disgust":
        sentiment_score = -0.7
    
    # 语音音量评分（这里简化处理，实际应从音频分析获取）
    volume_score = 0.5 if voice_text else 0.0
    
    _pad_engine.update(sentiment_score, volume_score, face_emotion)
    
    # 获取TTS参数
    tts_params = _pad_engine.get_tts_params()
    
    audio_path = None
    if enable_tts and reply_text:
        audio_path = speak_sync(reply_text, tts_params)

    return result, audio_path
