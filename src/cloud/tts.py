# src/cloud/tts.py
import os
import subprocess
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
from src.core.config import QWEN_API_KEY, AUDIO_OUTPUT_DEVICE
from src.utils.logger import logger

# 配置阿里云 API KEY
dashscope.api_key = QWEN_API_KEY

def speak(text):
    """使用阿里云原生 TTS 进行语音合成并播放"""
    if not text: return
    
    output_file = "temp_tts.mp3"
    try:
        logger.info(f"[TTS] 正在合成: {text[:20]}...")
        
        # 使用阿里云 CosyVoice/Sambert 模型
        result = SpeechSynthesizer.call(
            model='sambert-zhichu-v1', # 这是一个非常自然的女声
            text=text,
            sample_rate=16000,
            format='mp3'
        )
        
        if result.get_audio_data() is not None:
            with open(output_file, 'wb') as f:
                f.write(result.get_audio_data())
            
            # 播放 (强制指定 ALSA 驱动)
            logger.info("[TTS] 合成成功，正在播放...")
            cmd = ["mpg123", "-o", "alsa", "-a", AUDIO_OUTPUT_DEVICE, "-q", output_file]
            subprocess.run(cmd, check=False)
        else:
            logger.error(f"[TTS] 合成失败: Code: {result.get_status_code()}, Msg: {result.get_error_message()}")
            
    except Exception as e:
        logger.error(f"[TTS] 异常: {e}")
    finally:
        if os.path.exists(output_file):
            os.remove(output_file)
