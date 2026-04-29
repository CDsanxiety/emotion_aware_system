# src/cloud/stt.py
import os
import subprocess
import dashscope
import re
from dashscope.audio.asr import Transcription
from src.core.config import QWEN_API_KEY, AUDIO_INPUT_INDEX, STT_TIMEOUT
from src.utils.logger import logger

# 配置阿里云 API KEY
dashscope.api_key = QWEN_API_KEY

def capture_and_transcribe():
    """诊断版 STT：增加了音频标准化处理 (-3dB)"""
    temp_audio = "temp_stt.wav"
    norm_audio = "norm_stt.wav"
    
    try:
        # 1. 使用 FFmpeg 录音
        logger.info(f"[STT] 正在以 16000Hz 录音 (时长: {STT_TIMEOUT}s)...")
        cmd_record = [
            "ffmpeg", "-y", "-f", "alsa", "-thread_queue_size", "1024", 
            "-ar", "16000", "-i", f"plughw:{AUDIO_INPUT_INDEX},0",
            "-t", str(int(STT_TIMEOUT)), 
            "-af", "highpass=f=200, lowpass=f=3500", 
            "-ac", "1", temp_audio
        ]
        subprocess.run(cmd_record, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
            logger.error("[STT] 录音文件不存在或太小")
            return ""

        # 2. 音频标准化处理 (-3dB)
        logger.info("[STT] 正在对音频进行标准化处理 (-3dB)...")
        cmd_norm = [
            "ffmpeg", "-y", "-i", temp_audio, 
            "-af", "loudnorm=I=-16:TP=-3:LRA=11", 
            "-ar", "16000", norm_audio
        ]
        subprocess.run(cmd_norm, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 3. 诊断音量
        cmd_detect = ["ffmpeg", "-i", norm_audio, "-af", "volumedetect", "-f", "null", "-"]
        result = subprocess.run(cmd_detect, capture_output=True, text=True)
        match = re.search(r"max_volume: ([\-\d\.]+) dB", result.stderr)
        if match:
            logger.info(f"[STT] 标准化后最大音量: {match.group(1)} dB")

        # 4. 调用阿里云原生 Transcription 接口
        logger.info("[STT] 正在请求阿里云 Paraformer...")
        audio_file_path = 'file://' + os.path.abspath(norm_audio)
        
        task_response = Transcription.async_call(
            model='paraformer-v1',
            file_urls=[audio_file_path]
        )
        
        status = Transcription.wait(task_response)
        
        if status.status_code == 200:
            results = status.output.get('results', [])
            if results:
                text = results[0].get('transcription', "").strip()
                if not text:
                    logger.warning(f"[STT] 识别成功但文字为空。原始响应内容: {status.output}")
                else:
                    logger.info(f"[STT] 最终识别结果: {text}")
                return text
            else:
                logger.warning(f"[STT] 响应中没有 results 字段。完整响应: {status}")
        else:
            logger.error(f"[STT] 转写失败: {status.message} (Code: {status.status_code})")
            
        return ""

    except Exception as e:
        logger.error(f"[STT] 异常: {e}")
        return ""
    # 暂时不删除，以便检查
