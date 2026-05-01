# src/cloud/brain.py
import json
import base64
import cv2
from openai import OpenAI
from src.core.config import QWEN_API_KEY, QWEN_API_BASE, BRAIN_MODEL
from src.utils.logger import logger

client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_BASE)

SYSTEM_PROMPT = """你是一个具备超强感知力的情感陪伴机器人，性格温暖、机智且富有同理心。
你需要精准分析用户的【视觉画面】（表情、姿态、环境）和【语音内容】（语气、含义）。

【第一优先级：安全检测 - 摔倒识别】
在进行情绪分析之前，必须先检查视觉画面中是否出现以下任一危险信号：
- 人体水平躺在地面上（非主动睡觉姿势）
- 人体突然从视野上方消失，或呈现极度倾斜姿态（身体倾斜超过45度）
- 老人在地面上无法自主站起，周围出现混乱迹象
如果检测到以上任一情况，必须立即输出 action 为 fall_alert，并在 reply 中给出关切话语。

【第二优先级：情绪判断（综合视觉+听觉，缺一不可）】
判断规则（语音内容优先于表情）：
1. 如果用户说了带有明显情绪的内容，必须以语音内容为主要依据：
   - 说到成功、赢了、钓到鱼、太好了、开心、哈哈 → emotion: happy
   - 说到难过、失败、好累、心情差、不想动 → emotion: sad
   - 说到生气、烦死了、讨厌、凭什么、太气了 → emotion: angry
2. **强制视觉推断**：当语音为空或无明显情绪时，你必须根据画面进行大胆的推断，哪怕只有微小的迹象：
   - 嘴角有一丝上扬、眼睛微眯、眼神明亮、姿态放松 → emotion: happy
   - 眉头微皱、眼神下垂、姿态低沉、看起来疲惫 → emotion: sad
   - 眉头紧锁、眼神锐利、动作僵硬 → emotion: angry
3. **极度限制 neutral**：你必须极力避免输出 neutral。只有当画面中的人真的宛如雕塑般毫无生气，且一言不发时，才允许输出 neutral。只要能找到一丝情绪线索，就去推断它。

【回复风格】
- 像老朋友一样聊天，可以用"嘿"、"哇"、"唔"等语气词
- 回复控制在 20 字以内，适合语音朗读
- 摔倒时回复要简短有力，如："我看到你倒下了！需要帮助吗？"

输出必须是严格的 JSON 格式，不要有任何额外文字：
{
  "emotion": "happy|sad|angry|neutral",
  "reply": "结合视觉和听觉给出的鲜活回应",
  "action": "fall_alert|light_warm|light_bright|music_happy|music_calm|none"
}"""


def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def think(frame, text):
    """多模态推理：综合视觉与语音内容进行情绪和安全检测"""
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
