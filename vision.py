# vision.py
"""
摄像头画面 → 阿里云百炼 qwen-vl-max 场景理解（含杂乱室内、多物体、非人物）；
失败时用 OpenCV 启发式 + 可选 FER，保证 description 非空、便于 LLM 做社交共情。
"""
import base64
import os
from typing import Any, Dict, List, Optional

import cv2
from dotenv import load_dotenv
from fer.fer import FER
from openai import OpenAI

from utils import logger

load_dotenv()

    client = _get_client()
    if not client:
        return ""
    resp = client.chat.completions.create(
        model=VL_MODEL,
        messages=[
            {"role": "system", "content": VL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        temperature=0.25,
        max_tokens=512,
        timeout=VL_TIMEOUT_SEC,
    )
    return (resp.choices[0].message.content or "").strip()


def process_image(frame) -> Dict[str, Any]:
    """
    输入：Gradio / OpenCV 图像帧（numpy BGR）。
    输出：success / description / is_fallback；description 始终非空以利 LLM 联动。
    """
    if frame is None:
        return _make_result(False, "未获取到画面，无法描述场景。", True)

    data_url = _frame_to_data_url(frame)
    if not data_url:
        return _make_result(False, _fallback_visual_description(frame), True)

    client = _get_client()
    if client is None:
        logger.warning("未配置 LLM_API_KEY，视觉降级为本地启发式 + FER")
        return _make_result(False, _fallback_visual_description(frame), True)

    try:
        raw = _call_qwen_vl(data_url, VL_USER_PROMPT)
        if not raw:
            raw = _call_qwen_vl(data_url, VL_USER_PROMPT_RETRY)
        if raw:
            return _make_result(True, raw, False)
        logger.warning("qwen-vl-plus 返回空内容，降级本地 + 可选 FER")
        return _make_result(False, _fallback_visual_description(frame), True)
    except Exception as e:
        logger.warning(f"qwen-vl-plus 不可用，降级: {e}")
        return _make_result(False, _fallback_visual_description(frame), True)


def no_input_vision() -> Dict[str, Any]:
    return _make_result(False, "未获取到画面。", True)


def get_current_emotion() -> Dict[str, Any]:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return _make_result(False, "无法读取摄像头画面。", True)
    return process_image(frame)


if __name__ == "__main__":
    print("测试视觉：qwen-vl-plus（Ctrl+C 退出）")
    try:
        while True:
            out = get_current_emotion()
            print(
                f"[success={out['success']} fallback={out['is_fallback']}] "
                f"{out['description'][:120]}...",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\n结束")
>>>>>>> 483d2714ed6e5219c69a9876fb45ccb0e51adfc6
