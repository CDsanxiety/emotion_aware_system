# src/core/orchestrator.py
import threading
import cv2
from src.cloud import brain, stt, tts
from src.hardware.physical_interface import PhysicalInterface
from src.core.config import VISION_INTERVAL, CAMERA_INDEX
from src.utils.logger import logger

class EmotionSystemOrchestrator:
    def __init__(self):
        self.hw = PhysicalInterface()
        self.running = False
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        logger.info("[Orchestrator] 初始化完成，准备进入情感循环")

    def capture_vision(self):
        """抓拍一帧画面"""
        ret, frame = self.cap.read()
        if ret:
            logger.info("[Vision] 抓拍成功")
            return frame
        logger.warning("[Vision] 抓拍失败")
        return None

    def step(self):
        """单次交互循环"""
        logger.info("\n--- 🧠 启动新一轮感知 ---")
        
        # 1. 并发感知：拍照和录音同时进行
        vision_result = []
        audio_result = []

        def get_vision():
            vision_result.append(self.capture_vision())
        
        def get_audio():
            audio_result.append(stt.capture_and_transcribe())

        t1 = threading.Thread(target=get_vision)
        t2 = threading.Thread(target=get_audio)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()

        frame = vision_result[0]
        text = audio_result[0]

        if not text and frame is None:
            return

        # 2. 云端思考
        response = brain.think(frame, text)
        if not response:
            return

        emotion = response.get("emotion", "neutral")
        reply = response.get("reply", "我还在学习中...")
        action = response.get("action", "none")

        # 3. 屏幕显示结果
        print(f"\n>>>> [智能体检测到情绪]: {emotion.upper()}")
        print(f">>>> [智能体回复]: {reply}")

        # 4. 硬件响应：灯光 + 语音 + 音乐
        # 先切灯光
        self.hw.set_led_emotion(emotion)
        
        # 播放语音 (TTS)
        tts.speak(reply)
        
        # 如果需要播放背景音乐 (test.mp3)
        if action.startswith("music"):
            self.hw.play_sound("music/test.mp3")

    def run(self):
        self.running = True
        while self.running:
            try:
                self.step()
            except Exception as e:
                logger.error(f"[Orchestrator] 循环异常: {e}")
                break

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.hw.clear_led()
        logger.info("[Orchestrator] 系统已停止")
