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
    """诊断版 STT：使用 FFmpeg 录音并进行音量检测"""
    temp_audio = "temp_stt.wav"
    
    try:
        # 1. 使用 FFmpeg 录音 (使用 44100 硬件通用采样率录制，防止爆音)
        logger.info(f"[STT] 正在以 44100Hz 录音 (时长: {STT_TIMEOUT}s)...")
        # 移除 15dB 增益，改用 44100 录制，之后再降采样
        cmd_record = [
            "ffmpeg", "-y", "-f", "alsa", "-ar", "44100", "-i", f"plughw:{AUDIO_INPUT_INDEX},0",
            "-t", str(int(STT_TIMEOUT)), 
            "-af", "highpass=f=200, lowpass=f=3500", # 仅保留人声频率
            "-ar", "16000", "-ac", "1", temp_audio
        ]
        # 运行录音，捕获输出以备诊断
        subprocess.run(cmd_record, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
            logger.error("[STT] 录音文件不存在或太小")
            return ""

        # 2. 诊断音量：检测最大电平 (max_volume)
        cmd_detect = [
            "ffmpeg", "-i", temp_audio, "-af", "volumedetect", "-f", "null", "-"
        ]
        result = subprocess.run(cmd_detect, capture_output=True, text=True)
        # 匹配日志中的 max_volume: -20.5 dB 这种格式
        match = re.search(r"max_volume: ([\-\d\.]+) dB", result.stderr)
        if match:
            max_vol = float(match.group(1))
            logger.info(f"[STT] 录音诊断 - 最大音量: {max_vol} dB")
            if max_vol < -40:
                logger.warning("[STT] 警告：录音音量极低，可能麦克风没接好或被静音！")
        
        # 3. 调用阿里云原生 Transcription 接口
        logger.info("[STT] 正在请求阿里云 Paraformer...")
        audio_file_path = 'file://' + os.path.abspath(temp_audio)
        
        task_response = Transcription.async_call(
            model='paraformer-v1',
            file_urls=[audio_file_path]
        )
        
        status = Transcription.wait(task_response)
        
        if status.status_code == 200:
            results = status.output.get('results', [])
            if results:
                text = results[0].get('transcription', "").strip()
                logger.info(f"[STT] 最终识别结果: {text}")
                return text
        else:
            logger.error(f"[STT] 转写失败: {status.message}")
            
        return ""

    except Exception as e:
        logger.error(f"[STT] 异常: {e}")
        return ""
    # 注意：为了诊断，我们暂时【不删除】temp_stt.wav，让你能手动听
    # finally:
    #     if os.path.exists(temp_audio):
    #         os.remove(temp_audio)
