import time
import threading

class Blackboard:
    """全局状态黑板：所有感知模块只管往这里写，决策模块只管读"""
    def __init__(self):
        self.lock = threading.Lock()
        self.last_speech_text = ""
        self.last_speech_time = 0
        self.current_vision_desc = "无画面"
        self.vision_update_time = 0
        self.user_presence = False  # 用户是否在场
        self.robot_status = "idle"  # 机器人状态：idle, listening, thinking, acting
        
    def update_vision(self, desc, presence):
        with self.lock:
            self.current_vision_desc = desc
            self.user_presence = presence
            self.vision_update_time = time.time()

    def update_speech(self, text):
        with self.lock:
            self.last_speech_text = text
            self.last_speech_time = time.time()
