# config.py 全功能补全版
import os
from dotenv import load_dotenv

load_dotenv()

# --- 基础配置 ---
API_KEY = os.getenv("API_KEY", "your-api-key")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")

# --- ROS 配置 ---
# 建议把 localhost 换成 127.0.0.1，更稳定
ROS_BRIDGE_URI = os.getenv("ROS_BRIDGE_URI", "ws://127.0.0.1:9090")
ROS_ACTION_TOPIC = "/robot/action"
ROS_STATUS_TOPIC = "/robot/status"
ROS_STATE_TOPIC = "/robot/state"

# --- 硬件与语音参数 (补齐缺失项) ---
LED_COUNT = 60
LED_PIN = 18
LED_BRIGHTNESS = 0.1
AUDIO_OUTPUT_DEVICE = "hw:2,0"
AUDIO_INPUT_INDEX = 1        # audio.py 需要这个名字
MIC_INDEX = 1                # 兼容旧代码
STT_TIMEOUT = 5              # 补齐：语音识别超时
STT_PHRASE_LIMIT = 10        # 补齐：单句时长限制
STT_LANGUAGE = "zh-CN"       # 补齐：语言

# --- 情绪引擎 ---
EMOTION_UPDATE_INTERVAL = 2.0
