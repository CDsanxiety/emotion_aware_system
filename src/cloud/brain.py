# src/cloud/brain.py
import json
import base64
import cv2
from openai import OpenAI
from src.core.config import QWEN_API_KEY, QWEN_API_BASE, BRAIN_MODEL
from src.utils.logger import logger

client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_BASE)

SYSTEM_PROMPT = """你是一个情感陪伴机器人。
你需要综合分析用户的【视觉画面】和【语音内容】，给出回应。
输出必须是严格的 JSON 格式：
{
  "emotion": "happy|sad|angry|neutral",
  "reply": "你的回应话术",
  "action": "light_warm|light_bright|music_happy|music_calm|none"
}
回应要简短温暖，像老朋友一样。"""

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def think(frame, text):
    """
    多模态思考逻辑
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    user_content = []
    if text:
        user_content.append({"type": "text", "text": f"用户说: {text}"})
    
    if frame is not None:
        base64_image = encode_image(frame)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    
    if not user_content:
        return None

    messages.append({"role": "user", "content": user_content})

    try:
        logger.info("[Brain] 正在调用云端大脑推理...")
        response = client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.info(f"[Brain] 推理结果: {result}")
        return result
    except Exception as e:
        logger.error(f"[Brain] 异常: {e}")
        return None
