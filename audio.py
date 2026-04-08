# audio.py
import speech_recognition as sr

recognizer = sr.Recognizer()

def recognize_speech(timeout: int = 5, phrase_time_limit: int = 8) -> str:
    """
    从麦克风录音并返回识别的文字（中文）
    失败时返回空字符串 ""
    """
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio, language="zh-CN")
            return text.strip()
        except:
            return ""   # 任何错误都返回空字符串

def transcribe_file(audio_file_path: str) -> str:
    """
    将给定的音频文件转为文字
    """
    if not audio_file_path:
        return ""
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="zh-CN")
            return text.strip()
    except Exception as e:
        print(f"语音文件转写失败: {e}")
        return ""