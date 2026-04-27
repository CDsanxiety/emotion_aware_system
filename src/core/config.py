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
LED_PIN = 18          # 灯带连接的第 12 针 (BCM 18)
LED_COUNT = 60        # 灯珠数量
LED_BRIGHTNESS = 0.2  # 功率限制：20% 亮度，保护树莓派供电安全

CAMERA_INDEX = 0
AUDIO_INPUT_INDEX = 1  # 摄像头自带麦克风 (Card 1)
AUDIO_OUTPUT_DEVICE = "plughw:2,0" # USB 播放设备 (Card 2)

# ROS 配置
ROS_BRIDGE_URI = os.getenv("ROS_BRIDGE_URI", "ws://localhost:9090")
TOPIC_ACTION = "/robot/action"
TOPIC_STATUS = "/robot/status"

# ================== 3. 策略配置 ==================
IDLE_THRESHOLD = 30.0 # 闲置多久触发主动关心
VISION_INTERVAL = 5.0  # 视觉采样间隔
STT_TIMEOUT = 8.0     # 语音录制超时
