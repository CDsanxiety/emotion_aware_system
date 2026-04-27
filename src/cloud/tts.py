# src/cloud/tts.py
import asyncio
import edge_tts
import os
import subprocess
from src.core.config import TTS_VOICE, AUDIO_OUTPUT_DEVICE
from src.utils.logger import logger

async def _generate_audio(text, output_file):
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(output_file)

def speak(text):
    """合成并播放语音"""
    if not text: return
    
    output_file = "temp_tts.mp3"
    try:
        # 1. 合成
        logger.info(f"[TTS] 正在合成: {text[:20]}...")
        asyncio.run(_generate_audio(text, output_file))
        
        # 2. 播放 (使用 mpg123)
        # 树莓派需安装: sudo apt-get install mpg123
        logger.info(f"[TTS] 正在播放...")
        cmd = ["mpg123", "-a", AUDIO_OUTPUT_DEVICE, "-q", output_file]
        subprocess.run(cmd, check=False)
        
    except Exception as e:
        logger.error(f"[TTS] 异常: {e}")
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)
