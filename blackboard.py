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
    
    def get_vision_data(self):
        with self.lock:
            return {
                "description": self.current_vision_desc,
                "presence": self.user_presence,
                "update_time": self.vision_update_time
            }
    
    def get_speech_data(self):
        with self.lock:
            return {
                "text": self.last_speech_text,
                "update_time": self.last_speech_time
            }
    
    def get_robot_status(self):
        with self.lock:
            return self.robot_status
    
    def set_robot_status(self, status):
        with self.lock:
            self.robot_status = status