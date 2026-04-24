# audio_manager.py
import os
from utils import logger

def play_system_audio(event_type):
    audio_map = {
        "startup": "music/startup.mp3",
        "shutdown": "music/shutdown.mp3",
        "wake": "music/wake.mp3"
    }
    file_path = audio_map.get(event_type)
    if file_path and os.path.exists(file_path):
        try:
            from config import AUDIO_OUTPUT_DEVICE
        except ImportError:
            AUDIO_OUTPUT_DEVICE = "hw:2,0"
            
        logger.info(f"[Audio] 正在播放音效: {file_path} -> {AUDIO_OUTPUT_DEVICE}")
        os.system(f"mpg123 -a {AUDIO_OUTPUT_DEVICE} {file_path} > /dev/null 2>&1")
    else:
        logger.warning(f"[Audio] 音效文件不存在: {file_path}")
