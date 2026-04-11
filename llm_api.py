# llm_api.py
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from tts import speak_sync
from utils import logger
from vision import analyze_scene

load_dotenv()

# ================== 配置区 ==================
MODEL = "qwen-max"  # 用户指定的大语言模型
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.getenv("LLM_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============================================

def call_llm_structured(semantic_context: str, user_text: str, prompt_file: str = "prompt.txt") -> dict:
    """
    第二阶段：接收 VLM 语义描述 + 用户语音文本 → 返回结构化 JSON
    """
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        logger.error(f"Prompt 文件不存在: {prompt_file}")
        system_prompt = "你是一个暖心的助手。请按 JSON 格式回复。"

    # 拼接上下文
    full_prompt = f"""
【视觉与环境感知】：
{semantic_context}

【用户语音】：
"{user_text}"

请基于以上多模态信息，判断用户真实情感，并输出 JSON 动作指令。
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        raw_reply = response.choices[0].message.content.strip()
        logger.debug(f"LLM 原始回复: {raw_reply}")

        # 清理可能存在的 markdown 标记
        if "```json" in raw_reply:
            raw_reply = raw_reply.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_reply:
            raw_reply = raw_reply.split("```")[1].split("```")[0].strip()

        result = json.loads(raw_reply)
        return result

    except Exception as e:
        logger.error(f"决策层调用失败: {e}")
        return {
            "emotion": "感知模糊",
            "action": "none",
            "reply": "抱歉，我现在有点累，能再说一遍吗？",
            "confidence": 0.0
        }

def get_response(frame, voice_text: str, enable_tts: bool = True) -> dict:
    """
    V2.0 统一接口：端到端图文融合感知
    1. 调用 VLM 获取语义描述
    2. 调用 LLM 进行结构化决策
    3. 可选执行 TTS
    """
    # 第一阶段：VLM 场景分析
    logger.info("开始第一阶段：VLM 语义扫描...")
    semantic_context = analyze_scene(frame, voice_text)
    logger.info(f"VLM 结果: {semantic_context[:50]}...")

    # 第二阶段：LLM 结构化决策
    logger.info("开始第二阶段：LLM 逻辑决策...")
    result = call_llm_structured(semantic_context, voice_text)
    
    # 将 VLM 的原始描述也存入结果，方便 UI 调试显示
    result["vlm_description"] = semantic_context

    reply_text = result.get("reply", "")
    if enable_tts and reply_text:
        speak_sync(reply_text)

    return result

# 测试用
if __name__ == "__main__":
    import cv2
    print("测试 V2.0 两阶段推理...")
    # 模拟一张空图像或读取本地测试图
    test_frame = cv2.imread("test.jpg") # 需确保目录下有此文件或处理 None
    res = get_response(test_frame, "我今天很开心")
    print(f"最终决策: {json.dumps(res, ensure_ascii=False, indent=2)}")