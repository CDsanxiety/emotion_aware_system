import socket
import os
import subprocess
import speech_recognition as sr
from config import STT_TIMEOUT, STT_PHRASE_LIMIT, STT_LANGUAGE, AUDIO_INPUT_INDEX
from utils import logger

recognizer = sr.Recognizer()
_system_shutting_down = False


def _check_network(host="www.google.com", port=80, timeout=2):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


def set_system_shutting_down(status: bool):
    """设置系统关机状态"""
    global _system_shutting_down
    _system_shutting_down = status


def recognize_speech(timeout: int = STT_TIMEOUT, phrase_time_limit: int = STT_PHRASE_LIMIT) -> str:
    if _system_shutting_down:
        return ""
    try:
        from config import AUDIO_INPUT_INDEX
        temp_audio = "temp_record.wav"
        
        logger.info(f">>> 正在聆听 (使用 arecord plughw:{AUDIO_INPUT_INDEX})...")
        cmd = ["arecord", "-D", f"plughw:{AUDIO_INPUT_INDEX}", "-d", str(phrase_time_limit), "-f", "cd", "-q", temp_audio]
        
        # 使用 subprocess，并比限定时长多给 2 秒缓冲时间，超时立刻操作系统级击杀
        try:
            subprocess.run(cmd, timeout=phrase_time_limit + 2, check=False)
        except subprocess.TimeoutExpired:
            logger.error(f"[Audio] 录音底层驱动无响应卡死，已强制熔断 (超时 {phrase_time_limit+2}s)")
            return ""
        
        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 100:
            logger.debug("录音文件为空或未生成")
            return ""

        if not _check_network():
            logger.error("网络连接失败，无法使用 STT")
            return ""

        # 读取生成的音频文件进行识别
        with sr.AudioFile(temp_audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=STT_LANGUAGE)
            logger.info(f"你说: {text}")
            return text.strip()
            
    except sr.UnknownValueError:
        # 没有识别到有效语音的正常情况，不需要打印堆栈
        logger.debug("未识别到清晰语音")
        return ""
    except Exception as e:
        logger.debug(f"语音捕获未触发或识别错误: {e}")
        return ""


def transcribe_file(audio_file_path: str) -> str:
    if not audio_file_path or _system_shutting_down: return ""
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            if not _check_network(): return ""
            return recognizer.recognize_google(audio_data, language=STT_LANGUAGE).strip()
    except:
        return ""


if __name__ == "__main__":
    print("测试麦克风...")
    print(f"结果: {recognize_speech()}")