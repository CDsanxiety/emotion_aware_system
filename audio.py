# audio.py
import socket
import speech_recognition as sr
from config import STT_TIMEOUT, STT_PHRASE_LIMIT, STT_LANGUAGE
from utils import logger

recognizer = sr.Recognizer()

def _check_network(host="www.google.com", port=80, timeout=2):
    """
    快速检测网络连通性，避免 recognizer.recognize_google 导致的长时间阻塞。
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        logger.warning(f"外部网络连接检测失败 (Google STT 可能不可用): {ex}")
        return False

def recognize_speech(timeout: int = STT_TIMEOUT, phrase_time_limit: int = STT_PHRASE_LIMIT) -> str:
    """
    从麦克风录音并返回识别的文字（中文）。
    使用了详细的异常处理，避免“静默失败”。
    """
    try:
        with sr.Microphone() as source:
            # 动态调整环境噪音，建议由于实时性要求，duration 设小
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            logger.info(">>> 麦克风已就绪，正在聆听...")
            
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            # 在发起 Google 请求前进行网络预检
            if not _check_network():
                logger.error("网络连接异常，跳过 Google STT 识别以节省等待时间。")
                return ""

            text = recognizer.recognize_google(audio, language=STT_LANGUAGE)
            logger.info(f"语音识别成功: {text}")
            return text.strip()

    except sr.WaitTimeoutError:
        logger.info("未检测到语音输入（超时）。")
        return ""
    except sr.UnknownValueError:
        logger.warning("Google STT 无法理解当前音频内容。")
        return ""
    except sr.RequestError as e:
        logger.error(f"Google STT 服务请求失败; {e}")
        return ""
    except Exception as e:
        logger.error(f"语音捕获过程中发生非预期异常: {e}")
        return ""

def transcribe_file(audio_file_path: str) -> str:
    """
    将本地音频文件转为文字，增加鲁棒性校验。
    """
    if not audio_file_path:
        return ""
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            
            if not _check_network():
                return ""

            text = recognizer.recognize_google(audio_data, language=STT_LANGUAGE)
            return text.strip()
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        logger.error(f"本地语音文件转写失败: {e}")
        return ""

if __name__ == "__main__":
    print(">>> 正在启动麦克风识别测试 (请对着麦克风说话)...")
    result = recognize_speech()
    if result:
        print(f"识别结果: {result}")
    else:
        print("识别失败或无语音输入。")