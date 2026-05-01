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
    """
    录音并识别：
    1. 使用 FFmpeg 录音并标准化为 PCM 16bit WAV
    2. 直接将文件二进制数据发送给阿里云 Recognition 接口
       (避免使用 file:// 路径，该路径阿里云服务器无法访问)
    """
    temp_audio = "temp_stt.wav"
    norm_audio = "norm_stt.wav"

    try:
        # 1. 使用 FFmpeg 录音 (16000Hz 减轻树莓派 3B USB 总线压力)
        logger.info(f"[STT] 正在录音 (时长: {STT_TIMEOUT}s)...")
        # thread_queue_size: 防止缓冲区溢出; ar 16000: 减少数据量
        cmd_record = [
            "ffmpeg", "-y", "-f", "alsa", "-thread_queue_size", "1024",
            "-ar", "16000", "-i", f"plughw:{AUDIO_INPUT_INDEX},0",
            "-t", str(int(STT_TIMEOUT)),
            "-af", "highpass=f=200, lowpass=f=3500",
            "-ac", "1", temp_audio
        ]
        subprocess.run(cmd_record, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
            logger.error("[STT] 录音文件不存在或太小，跳过识别")
            return ""

        # 2. 标准化为阿里云兼容格式：PCM 16bit 小端序
        logger.info("[STT] 正在转换为标准 PCM 16bit 格式...")
        cmd_norm = [
            "ffmpeg", "-y", "-i", temp_audio,
            "-af", "loudnorm=I=-16:TP=-3:LRA=11",
            "-c:a", "pcm_s16le",   # 强制使用阿里云最兼容的格式
            "-ar", "16000", "-ac", "1",
            norm_audio
        ]
        subprocess.run(cmd_norm, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        target_file = norm_audio if os.path.exists(norm_audio) else temp_audio

        # 3. 读取文件二进制内容，直接发送给阿里云（不传路径）
        logger.info(f"[STT] 正在上传音频并识别（文件大小: {os.path.getsize(target_file)} bytes）...")
        with open(target_file, 'rb') as f:
            audio_data = f.read()

        # 正确姿势：先实例化 Recognition，再调用实例的 .call() 方法
        recognizer = Recognition(
            model='paraformer-realtime-v2',
            format='wav',
            sample_rate=16000,
            callback=None  # 非流式模式，不需要回调
        )
        response = recognizer.call(target_file)

        if response.status_code == 200:
            # 提取识别出的文字
            sentence_list = response.output.get('sentence', [])
            text = "".join([s.get('text', '') for s in sentence_list]).strip()
            if text:
                logger.info(f"[STT] 识别结果: {text}")
            else:
                logger.warning(f"[STT] 识别成功但内容为空。原始输出: {response.output}")
            return text
        else:
            logger.error(f"[STT] 识别失败: {response.message} (Code: {response.status_code})")
            return ""

    except Exception as e:
        logger.error(f"[STT] 异常: {e}")
        return ""
    finally:
        # 清理临时文件
        for f in [temp_audio, norm_audio]:
            if os.path.exists(f):
                os.remove(f)
