# vision.py
import cv2
import numpy as np
from utils import logger
from config import CAMERA_INDEX

# 尝试导入 fer
try:
    from fer import FER

    # 树莓派 3B 建议禁用 MTCNN 以节省资源
    emotion_detector = FER(mtcnn=False)
    HAS_FER = True
except ImportError:
    logger.warning("未检测到 fer 模块，跳过本地情绪识别")
    HAS_FER = False


import cv2
import time
from utils import logger
from config import CAMERA_INDEX

_global_cap = None

def get_global_camera():
    """获取单例全局摄像头，避免频繁开启引发树莓派死锁"""
    global _global_cap
    if _global_cap is None:
        logger.info(f"[Vision] 正在初始化全局摄像头 (Index: {CAMERA_INDEX})...")
        _global_cap = cv2.VideoCapture(CAMERA_INDEX)
        _global_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        _global_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        _global_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(3): _global_cap.grab() # 预热
    return _global_cap

def release_global_camera():
    global _global_cap
    if _global_cap is not None:
        _global_cap.release()
        _global_cap = None
        logger.info("[Vision] 已释放全局摄像头。")

def grab_frame():
    """对外提供统一的图像抓取接口"""
    cap = get_global_camera()
    ret, frame = cap.read()
    if not ret:
        logger.error("[Vision] 摄像头帧抓取失败。")
        return None
    return frame

def process_image(frame):
    """
    处理图像：
    1. 如果 frame 为 None，尝试从本地摄像头抓取
    2. 识别面部情绪
    3. 返回描述
    """
    is_fallback = False

    # 如果 Gradio 没传画面，尝试直接打开本地摄像头
    if frame is None:
        frame = grab_frame()
        
        if frame is None:
            return "摄像头连接超时或权限不足", True
        is_fallback = True

    local_emotion = "平静"
    if HAS_FER:
        try:
            results = emotion_detector.detect_emotions(frame)
            if results:
                # 获取概率最高的情绪
                emotions = results[0]["emotions"]
                local_emotion_en = max(emotions, key=emotions.get)
                # 简单翻译
                emo_map = {"happy": "开心", "sad": "悲伤", "angry": "生气", "neutral": "平静", "surprise": "惊讶"}
                local_emotion = emo_map.get(local_emotion_en, local_emotion_en)
        except Exception as e:
            logger.error(f"FER 识别出错: {e}")

    description = f"本地视觉识别：检测到用户，表情呈现【{local_emotion}】状态。"
    return description, is_fallback