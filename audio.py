# audio.py
import socket
import speech_recognition as sr
from config import STT_TIMEOUT, STT_PHRASE_LIMIT, STT_LANGUAGE, AUDIO_INPUT_INDEX
from utils import logger

recognizer = sr.Recognizer()

def _check_network(host="www.google.com", port=80, timeout=2):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def recognize_speech(timeout: int = STT_TIMEOUT, phrase_time_limit: int = STT_PHRASE_LIMIT) -> str:
    try:
        # 💡 关键：指定摄像头麦克风设备
        with sr.Microphone(device_index=AUDIO_INPUT_INDEX) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            logger.info(">>> 正在聆听 (摄像头麦克风)...")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            if not _check_network():
                logger.error("网络连接失败，无法使用 STT")
                return ""

            text = recognizer.recognize_google(audio, language=STT_LANGUAGE)
            logger.info(f"你说: {text}")
            return text.strip()
    except Exception as e:
        logger.debug(f"语音捕获未触发: {e}")
        return ""

def transcribe_file(audio_file_path: str) -> str:
    if not audio_file_path: return ""
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            if not _check_network(): return ""
            return recognizer.recognize_google(audio_data, language=STT_LANGUAGE).strip()
    except: return ""

if __name__ == "__main__":
    print("测试麦克风...")
    print(f"结果: {recognize_speech()}")