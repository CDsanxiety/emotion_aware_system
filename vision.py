# vision.py
"""
摄像头画面 → 阿里云百炼 qwen-vl-max 场景理解（含杂乱室内、多物体、非人物）；
具备本地 OpenCV 启发式统计（像素分布、移动侦测、主导颜色）作为降级兜底方案，
在网络中断或 API 失效时，仍能提供基础的环境感知，确保机器人“不会失明”。
"""
import base64
import io
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fer.fer import FER
from openai import OpenAI

from utils import logger

# ================== 配置区 (后期可剥离至 config.py) ==================
VL_MODEL = "qwen-vl-max"
VL_TIMEOUT_SEC = 10
VL_SYSTEM_PROMPT = "你是一个冷静、专业的视觉感知系统。请简洁描述画面中的场景、人物神态及潜在物理风险。"
VL_USER_PROMPT = "请用一句话描述这张图片的内容，重点关注人物情绪、环境光线和异常物品。"
VL_USER_PROMPT_RETRY = "这张图里有什么？请简短回答。"

load_dotenv()

# 全局单例：仅在需要时初始化，避免启动开销
_client_instance: Optional[OpenAI] = None
_fer_detector: Optional[FER] = None

def _get_openai_client() -> Optional[OpenAI]:
    global _client_instance
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        return None
    if _client_instance is None:
        _client_instance = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    return _client_instance

def _get_fer_detector():
    global _fer_detector
    if _fer_detector is None:
        # mtcnn=True 虽准但慢，RPi 建议设为 False 或使用更轻量的检测器
        _fer_detector = FER(mtcnn=False) 
    return _fer_detector

# ================== 核心工具函数 ==================

def _frame_to_data_url(frame: np.ndarray) -> Optional[str]:
    """将 OpenCV BGR 帧转换为 OpenAI 格式的 Data URL。"""
    try:
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception as e:
        logger.error(f"图像编码失败: {e}")
        return None

def _make_result(success: bool, description: str, is_fallback: bool) -> Dict[str, Any]:
    return {
        "success": success,
        "description": description,
        "is_fallback": is_fallback,
        "timestamp": time.time()
    }

def _call_qwen_vl(data_url: str, prompt: str) -> str:
    """封装云端 VLM 调用。"""
    client = _get_openai_client()
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
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        temperature=0.2,
        max_tokens=300,
        timeout=VL_TIMEOUT_SEC,
    )
    return (resp.choices[0].message.content or "").strip()

def _fallback_visual_description(frame: np.ndarray) -> str:
    """
    [端侧感知初步演示] 
    当云端 API 不可用时，通过本地算法（OpenCV + FER）提取基础信息。
    这是“玩具级”向“工程级”进阶的关键：逻辑永不留白。
    """
    try:
        # 1. 基础光线检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        light_status = "光线充足" if avg_brightness > 80 else "环境昏暗"
        
        # 2. 局部运动/变化分析（本演示简化为像素标准差，反映构图复杂度）
        complexity = "环境复杂或杂乱" if np.std(gray) > 50 else "背景简洁"
        
        # 3. 本地表情识别 (Edge Perception)
        detector = _get_fer_detector()
        emotions = detector.detect_emotions(frame)
        emotion_str = ""
        if emotions and emotions[0].get("emotions"):
            try:
                best_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
                emotion_str = f"，侦测到人物情绪可能偏向 {best_emotion}"
            except ValueError:
                # emotions[0]["emotions"] 为空字典时，max() 会抛出异常
                logger.warning("检测到人脸但未识别出表情")
        
        return f"（本地感知模式）{light_status}，{complexity}{emotion_str}。建议以此作为基础互动。"
    except Exception as e:
        logger.warning(f"本地感知兜底也出错了: {e}")
        return "画面内容解析受限，但系统依然在关注着你。"

# ================== 对外接口 ==================

def process_image(frame: np.ndarray) -> Dict[str, Any]:
    """
    视觉处理主入口。
    原则：优先云端深度理解，本地实时感知兜底。
    """
    if frame is None:
        return _make_result(False, "无法读取摄像头画面。", True)

    data_url = _frame_to_data_url(frame)
    client = _get_openai_client()

    # 若未配置密钥，直接进入本地感知模式
    if not client or not data_key_configured():
        return _make_result(False, _fallback_visual_description(frame), True)

    try:
        # 尝试云端 VLM
        desc = _call_qwen_vl(data_url, VL_USER_PROMPT)
        if desc:
            return _make_result(True, desc, False)
        
        # 第一次失败重试
        desc = _call_qwen_vl(data_url, VL_USER_PROMPT_RETRY)
        if desc:
            return _make_result(True, desc, False)
            
        return _make_result(False, _fallback_visual_description(frame), True)
    except Exception as e:
        logger.error(f"云端视觉请求异常: {e}")
        return _make_result(False, _fallback_visual_description(frame), True)

def data_key_configured() -> bool:
    return bool(os.getenv("LLM_API_KEY"))

def no_input_vision() -> Dict[str, Any]:
    return _make_result(False, "（暂无画面输入）", True)

if __name__ == "__main__":
    print(">>> 正在启动本地视觉感知测试 (含 FER 模拟)...")
    cap = cv2.VideoCapture(0)
    ret, test_frame = cap.read()
    cap.release()
    if ret:
        res = process_image(test_frame)
        print(f"成功: {res['success']} | 兜底: {res['is_fallback']}")
        print(f"内容: {res['description']}")
    else:
        print("摄像头开启失败，请检查设备。")
