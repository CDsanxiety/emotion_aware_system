# src/cloud/stt.py
import os
import subprocess
import dashscope
from dashscope.audio.asr import Recognition
from src.core.config import QWEN_API_KEY, AUDIO_INPUT_INDEX, STT_TIMEOUT
from src.utils.logger import logger

# 配置阿里云 API KEY
dashscope.api_key = QWEN_API_KEY

def capture_and_transcribe():
    """使用阿里云原生 Paraformer 进行录音并转写"""
    temp_audio = "temp_stt.wav"
    try:
        # 1. 录音 (采样率 16000 符合 Paraformer 要求)
        logger.info(f"[STT] 正在录音 (最多 {STT_TIMEOUT}s)...")
        cmd = [
            "arecord", "-D", f"plughw:{AUDIO_INPUT_INDEX}", 
            "-d", str(int(STT_TIMEOUT)), "-f", "S16_LE", "-r", "16000", "-c", "1", "-q", temp_audio
        ]
        subprocess.run(cmd, timeout=STT_TIMEOUT + 2)
        
        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
            return ""

        # 2. 调用阿里云原生 Paraformer 接口
        logger.info("[STT] 正在上传阿里云 Paraformer 进行识别...")
        recognition = Recognition(model='paraformer-v1', format='wav', sample_rate=16000)
        result = recognition.call(temp_audio)
        
        if result.status_code == 200:
            # 提取转写文本
            sentences = result.output.get('sentence', [])
            if sentences:
                text = sentences[0].get('text', "").strip()
                logger.info(f"[STT] 识别结果: {text}")
                return text
        else:
            logger.error(f"[STT] 识别失败: Code: {result.status_code}, Msg: {result.message}")
            
        return ""

    except Exception as e:
        logger.error(f"[STT] 异常: {e}")
        return ""
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
