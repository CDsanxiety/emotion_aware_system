# audio.py
import socket
import base64
import json
import time
import speech_recognition as sr
from config import STT_TIMEOUT, STT_PHRASE_LIMIT, STT_LANGUAGE
from utils import logger

# 阿里云智能语音交互配置
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

ALI_ACCESS_KEY_ID = os.getenv("ALIYUN_ACCESS_KEY_ID", "")
ALI_ACCESS_KEY_SECRET = os.getenv("ALIYUN_ACCESS_KEY_SECRET", "")
ALI_APPKEY = os.getenv("ALIYUN_APP_KEY", "")

recognizer = sr.Recognizer()

# 全局变量，用于标识系统是否正在关闭
_system_shutting_down = False

def set_system_shutting_down(shutting_down: bool) -> None:
    """
    设置系统关闭状态
    """
    global _system_shutting_down
    _system_shutting_down = shutting_down

def _check_network(host="www.aliyun.com", port=80, timeout=5):
    """
    快速检测网络连通性，避免语音识别服务导致的长时间阻塞。
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        logger.warning(f"外部网络连接检测失败 (阿里云 STT 可能不可用): {ex}")
        return False

def _get_aliyun_signature(access_key_secret, timestamp):
    """
    生成阿里云 API 签名
    """
    import hmac
    import hashlib
    import base64
    string_to_sign = timestamp + "&"
    h = hmac.new(access_key_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha1)
    return base64.b64encode(h.digest()).decode('utf-8')

def _aliyun_stt(audio_data):
    """
    调用阿里云智能语音交互 STT API
    """
    import requests
    
    # 检查配置是否完整
    if not ALI_ACCESS_KEY_ID or not ALI_ACCESS_KEY_SECRET or not ALI_APPKEY:
        logger.error("阿里云 STT 配置不完整，请填入 AccessKey 和 AppKey")
        return ""
    
    try:
        # 将音频数据转换为 WAV 格式并编码为 base64
        import io
        import wave
        
        # 创建 WAV 格式的内存文件
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data.get_raw_data())
        wav_io.seek(0)
        audio_base64 = base64.b64encode(wav_io.read()).decode('utf-8')
        
        # 构建请求参数
        timestamp = str(int(time.time() * 1000))
        signature = _get_aliyun_signature(ALI_ACCESS_KEY_SECRET, timestamp)
        
        url = "https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "appkey": ALI_APPKEY,
            "format": "wav",
            "sample_rate": 16000,
            "signature": signature,
            "access_key_id": ALI_ACCESS_KEY_ID,
            "timestamp": timestamp,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": True,
            "audio": audio_base64
        }
        
        # 发送请求
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        if result.get("status") == 20000000:
            return result.get("result", "")
        else:
            logger.error(f"阿里云 STT 识别失败: {result.get('message', '未知错误')}")
            return ""
    except Exception as e:
        logger.error(f"阿里云 STT 服务请求失败: {e}")
        return ""

def recognize_speech(timeout: int = STT_TIMEOUT, phrase_time_limit: int = STT_PHRASE_LIMIT) -> str:
    """
    从麦克风录音并返回识别的文字（中文）。
    使用了详细的异常处理，避免“静默失败”。
    """
    # 检查系统是否正在关闭，如果是，直接返回
    if _system_shutting_down:
        logger.info("系统正在关闭，跳过语音识别")
        return ""
    
    try:
        with sr.Microphone() as source:
            # 调整环境噪音，增加时间以更好地适应环境
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            logger.info(">>> 麦克风已就绪，正在聆听...")
            
            # 再次检查系统是否正在关闭，避免在获取麦克风后系统开始关闭
            if _system_shutting_down:
                logger.info("系统正在关闭，停止语音识别")
                return ""
            
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            # 再次检查系统是否正在关闭，避免在录音过程中系统开始关闭
            if _system_shutting_down:
                logger.info("系统正在关闭，停止语音识别")
                return ""
            
            # 在发起阿里云请求前进行网络预检
            if not _check_network():
                logger.error("网络连接异常，跳过阿里云 STT 识别以节省等待时间。")
                return ""

            text = _aliyun_stt(audio)
            if text:
                logger.info(f"语音识别成功: {text}")
                return text.strip()
            else:
                logger.warning("阿里云 STT 无法理解当前音频内容。")
                return ""

    except sr.WaitTimeoutError:
        logger.info("未检测到语音输入（超时）。")
        return ""
    except sr.UnknownValueError:
        logger.warning("音频数据无效。")
        return ""
    except sr.RequestError as e:
        logger.error(f"语音捕获设备错误: {e}")
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

            text = _aliyun_stt(audio_data)
            if text:
                return text.strip()
            else:
                return ""
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