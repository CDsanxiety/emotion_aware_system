# debug_mode.py
"""
赛场防死机后门 - Debug Mode
当摄像头/麦克风/网络出问题时，手动输入情绪和文本，绕过硬件直接展示 LLM 能力
"""

# 预设情绪选项（供下拉框使用）
EMOTION_OPTIONS = ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"]

# 预设测试用例（一键填充）
PRESET_TESTS = {
    "😊 开心场景": ("happy", "我今天中奖了！太开心了！"),
    "😢 难过场景": ("sad", "我失恋了，好难过..."),
    "😠 愤怒场景": ("angry", "今天被老板骂了，气死我了！"),
    "😨 害怕场景": ("fear", "刚才路上看到一条蛇，吓死我了"),
    "😲 惊讶场景": ("surprise", "哇！外面下雪了！"),
    "😐 平静场景": ("neutral", "今天午饭吃什么好呢"),
    "😖 厌恶场景": ("disgust", "这菜太难吃了"),
}

def get_emotion_options() -> list:
    """返回可选表情列表"""
    return EMOTION_OPTIONS

def get_preset_tests() -> dict:
    """返回预设测试用例"""
    return PRESET_TESTS

def apply_preset(test_name: str):
    """根据预设名称返回 (emotion, text)"""
    return PRESET_TESTS.get(test_name, ("neutral", ""))