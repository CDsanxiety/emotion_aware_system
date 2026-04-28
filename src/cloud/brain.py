# src/cloud/brain.py
import json
import base64
import cv2
from openai import OpenAI
from src.core.config import QWEN_API_KEY, QWEN_API_BASE, BRAIN_MODEL
from src.utils.logger import logger

client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_BASE)

SYSTEM_PROMPT = """你是一个具备超强感知力的情感陪伴机器人，性格温暖、机智且富有同理心。
你需要精准分析用户的【视觉画面】（表情、眼神、环境）和【语音内容】（语气、含义）。

【核心原则】：
1. 拒绝平庸：除非用户完全面无表情且没说话，否则不要轻易给出 'neutral' 结论。
2. 捕捉细节：寻找用户嘴角的弧度、眼神的疲惫或光芒，哪怕非常细微。
3. 拟人化回复：像老朋友一样聊天，可以用“嘿”、“唔”、“哇”等语气词。
4. 长度控制：回复控制在 20 字以内，适合语音朗读。

输出必须是严格的 JSON 格式：
{
  "emotion": "happy|sad|angry|neutral",
  "reply": "结合视觉和听觉给出的鲜活回应",
  "action": "light_warm|light_bright|music_happy|music_calm|none"
}"""

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
