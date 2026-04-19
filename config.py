# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ================== 全局 API 配置 ==================
API_KEY = (os.getenv("LLM_API_KEY") or "").strip()
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ================== 视觉 (Vision) 配置 ==================
VL_MODEL = "qwen-vl-max"
VL_TIMEOUT = 12
VL_PROMPT_MAIN = "请用一句话描述这张图片的内容，重点关注人物情绪、环境光线和异常物品。"
VL_PROMPT_RETRY = "请简短描述画面中的核心内容。"

# ================== 语言 (LLM) 配置 ==================
LLM_MODEL = "qwen-max"
LLM_TIMEOUT = 15
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 512
SESSION_HISTORY_LEN = 5  # 记忆轮数

# ================== 听觉 (Audio) 配置 ==================
STT_TIMEOUT = 5              # 等待说话的最长时间 (秒)
STT_PHRASE_LIMIT = 8         # 单次说话的最长时间 (秒)
STT_LANGUAGE = "zh-CN"
STT_RETRY_COUNT = 1          # 网络失败重试次数

# ================== 语音 (TTS) 配置 ==================
TTS_VOICE = "zh-CN-XiaoxiaoNeural"  # 温馨女声
TTS_TIMEOUT = 10

# ================== 硬件 (ROS) 配置 ==================
ROS_BRIDGE_URI = os.getenv("ROS_BRIDGE_URI", "ws://localhost:9090")
ROS_ACTION_TOPIC = "/nuannuan/action"
ROS_STATUS_TOPIC = "/nuannuan/status"
ROS_STATE_TOPIC = "/nuannuan/state"

# ================== 不确定性 (Uncertainty) 配置 ==================
CONFIDENCE_THRESHOLD_HIGH = 0.75
CONFIDENCE_THRESHOLD_LOW = 0.45
BAYESIAN_DECAY_RATE = 0.02  # 降低衰减率，使机器人情绪保持更久
