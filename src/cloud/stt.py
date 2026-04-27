# src/cloud/stt.py
import os
import subprocess
import time
from openai import OpenAI
from src.core.config import QWEN_API_KEY, QWEN_API_BASE, AUDIO_INPUT_INDEX, STT_TIMEOUT
from src.utils.logger import logger

client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_BASE)

def capture_and_transcribe():
    """录音并转写"""
    temp_audio = "temp_stt.wav"
    try:
        # 1. 录音 (使用 arecord)
        logger.info(f"[STT] 正在录音 (最多 {STT_TIMEOUT}s)...")
        cmd = [
            "arecord", "-D", f"plughw:{AUDIO_INPUT_INDEX}", 
            "-d", str(int(STT_TIMEOUT)), "-f", "cd", "-q", temp_audio
        ]
        # 增加 1s 缓冲
        subprocess.run(cmd, timeout=STT_TIMEOUT + 1, check=False)
        
        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
            return ""

        # 2. 调用 Whisper API (或兼容接口)
        logger.info("[STT] 正在上传云端转写...")
        with open(temp_audio, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
        text = transcript.text.strip()
        logger.info(f"[STT] 识别结果: {text}")
        return text

    except Exception as e:
        logger.error(f"[STT] 异常: {e}")
        return ""
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
