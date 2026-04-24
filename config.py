# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ================== 1. 全局 API 配置 ==================
API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ================== 2. 视觉 (Vision) 配置 ==================
CAMERA_INDEX = 0
VL_MODEL = "qwen-vl-max"
VL_TIMEOUT = 15
VL_PROMPT_MAIN = "请用一句话描述这张图片的内容，重点关注人物情绪、环境光线和异常物品。"

# ================== 3. 听觉 (Audio STT) 配置 ==================
STT_TIMEOUT = 5
STT_PHRASE_LIMIT = 8
STT_LANGUAGE = "zh-CN"
# 💡 强制指定外接 USB 声卡麦克风
AUDIO_INPUT_INDEX = 2

# ================== 4. 语音播放 配置 ==================
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
# 💡 强制指定外接 USB 声卡扬声器
AUDIO_OUTPUT_DEVICE = "hw:2,0"

# ================== 5. 灯带 (LED) 配置 ==================
LED_PIN = 18
LED_COUNT = 60
# ⚠️ 物理安全红线：直接取电亮度锁死在 0.1
LED_BRIGHTNESS = 0.1

# ================== 6. 硬件通信 (ROS) 话题 ==================
ROS_BRIDGE_URI = os.getenv("ROS_BRIDGE_URI", "ws://localhost:9090")
ROS_ACTION_TOPIC = "/robot/action"
ROS_STATUS_TOPIC = "/robot/status"
ROS_STATE_TOPIC = "/robot/state"

# ================== 7. 认知决策 (LLM) 配置 ==================
LLM_MODEL = "qwen-max"
LLM_TEMPERATURE = 0.6
SESSION_HISTORY_LEN = 5