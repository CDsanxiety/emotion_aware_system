# llm_api.py
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tts import speak_sync
from utils import logger
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)

load_dotenv()

# ================== 配置区 ==================
MODEL = "qwen-turbo"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.getenv("LLM_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============================================

def call_llm(emotion: str, user_text: str, prompt_file: str = "prompt.txt") -> dict:
    """
    核心函数：接收表情 + 用户语音文本 → 返回 JSON 格式响应
    返回格式: {"emotion": "...", "action": "...", "reply": "..."}
    """
    # 读取 Prompt
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        logger.error(f"Prompt 文件不存在: {prompt_file}")
        system_prompt = "你是一个温柔的助手。"

    # 拼接上下文
    full_prompt = f"""
当前用户表情：{emotion}
用户说的话：{user_text}

请严格按照系统提示词要求的 JSON 格式输出。
"""

    system_msg: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": system_prompt
    }
    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": full_prompt
    }

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[system_msg, user_msg],
            temperature=0.7,
            max_tokens=200
        )
        raw_reply = response.choices[0].message.content.strip()
        logger.debug(f"LLM 原始回复: {raw_reply}")

        # 清理 markdown 代码块
        if raw_reply.startswith("```json"):
            raw_reply = raw_reply[7:]
        if raw_reply.startswith("```"):
            raw_reply = raw_reply[3:]
        if raw_reply.endswith("```"):
            raw_reply = raw_reply[:-3]
        raw_reply = raw_reply.strip()

        # 解析 JSON
        result = json.loads(raw_reply)

        # 确保必要字段存在
        if "emotion" not in result:
            result["emotion"] = emotion
        if "action" not in result:
            result["action"] = "无动作"
        if "reply" not in result:
            result["reply"] = "嗯嗯，我在这里陪着你呢～"

        return result

    except json.JSONDecodeError:
        logger.warning(f"JSON 解析失败，原始回复: {raw_reply}")
        return {
            "emotion": emotion,
            "action": "无动作",
            "reply": "嗯嗯，我在这里陪着你呢～"
        }
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return {
            "emotion": emotion,
            "action": "无动作",
            "reply": "稍微等一下哦，我在整理思绪～"
        }


def get_response(face_emotion: str, voice_text: str, enable_tts: bool = True) -> dict:
    """
    统一接口：接收表情和语音文本，返回完整 JSON 响应
    """
    if not voice_text or voice_text.strip() == "":
        result = {
            "emotion": face_emotion,
            "action": "无动作",
            "reply": "我在听呢，你想说什么呀～"
        }
    else:
        result = call_llm(face_emotion, voice_text)

    reply_text = result.get("reply", "")
    logger.info(f"表情: {face_emotion} | 动作: {result.get('action')} | 回复: {reply_text[:30]}...")

    if enable_tts and reply_text:
        speak_sync(reply_text)

    return result


# 测试用
if __name__ == "__main__":
    print("测试 LLM + TTS 联动（JSON 模式）...")
    result = get_response("happy", "今天天气真好", enable_tts=True)
    print(f"返回结果: {json.dumps(result, ensure_ascii=False, indent=2)}")