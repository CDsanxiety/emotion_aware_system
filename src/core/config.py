# src/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ================== 1. Cloud API 配置 ==================
# 推荐使用 Gemini 1.5 Flash，响应快且有免费配额
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_BASE = os.getenv("QWEN_API_BASE")
BRAIN_MODEL = os.getenv("QWEN_API_MODEL", "qwen-max")
STT_MODEL = "whisper-1"
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# ================== 2. 硬件配置 ==================
CAMERA_INDEX = 0
AUDIO_INPUT_INDEX = 2  # USB 声卡输入
AUDIO_OUTPUT_DEVICE = "hw:2,0" # USB 声卡输出

# ROS 配置
ROS_BRIDGE_URI = os.getenv("ROS_BRIDGE_URI", "ws://localhost:9090")
TOPIC_ACTION = "/robot/action"
TOPIC_STATUS = "/robot/status"

# ================== 3. 策略配置 ==================
IDLE_THRESHOLD = 30.0 # 闲置多久触发主动关心
VISION_INTERVAL = 5.0  # 视觉采样间隔
STT_TIMEOUT = 8.0     # 语音录制超时
