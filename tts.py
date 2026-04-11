# tts.py
import asyncio
import edge_tts
import os
import platform
from typing import Optional

# 可选音色（Product 同学可以挑选）
VOICES = {
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",      # 活泼女声（默认推荐）
    "yunxi": "zh-CN-YunxiNeural",            # 温柔男声
    "xiaoyi": "zh-CN-XiaoyiNeural",          # 知性女声
    "yunjian": "zh-CN-YunjianNeural",        # 阳光男声
    "xiaohan": "zh-CN-XiaohanNeural",        # 软萌妹子
    "xiaomo": "zh-CN-XiaomoNeural",          # 御姐
}

async def speak(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> Optional[str]:
    """
    将文字转为语音并播放
    参数:
        text: 要播放的文字
        voice: 音色名称
    返回:
        生成的音频文件名，失败返回 None
    """
    if not text or text.strip() == "":
        return None
    
    import uuid
    output_file = f"response_{uuid.uuid4().hex[:8]}.mp3"
    
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
        
        # 音频全权交由 Gradio Web 组件播放，彻底移除物理级播放
        return output_file
    except Exception as e:
        print(f"TTS 出错: {e}")
        return None

def speak_sync(text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> Optional[str]:
    """
    同步版本，方便在普通函数里调用
    """
    return asyncio.run(speak(text, voice))

# 测试
if __name__ == "__main__":
    print("测试 TTS...")
    speak_sync("你好呀，我是暖暖，今天过得怎么样？")
    print("播放完成")