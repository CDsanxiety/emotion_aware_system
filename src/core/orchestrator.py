# src/core/orchestrator.py
import threading
import cv2
from src.cloud import brain, stt, tts
from src.hardware.physical_interface import PhysicalInterface
from src.core.config import CAMERA_INDEX
from src.utils.logger import logger

class EmotionSystemOrchestrator:
    def __init__(self):
        self.hw = PhysicalInterface()
        self.running = False
        # 不再保持摄像头开启，改为用时开启，抓完即放
        logger.info("[Orchestrator] 初始化完成，准备进入情感循环 (错峰模式)")

    def capture_vision(self):
        """抓拍一帧画面 (用完即关，为录音腾出总线带宽)"""
        import time
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("[Vision] 无法开启摄像头")
            return None
            
        try:
            # 连续抓取 5 帧旧数据并丢弃，解决硬件缓冲区延迟问题
            for _ in range(5):
                cap.grab()
            
            ret, frame = cap.read()
            if ret:
                logger.info("[Vision] 抓拍成功，已释放摄像头资源")
                return frame
        finally:
            cap.release()
            
        return None

    def step(self):
        """单次交互循环 (串行模式以保证 3B 稳定性)"""
        # 1. 抓拍照片 (之后会自动释放摄像头)
        frame = self.capture_vision()
        
        # 2. 关键：等待 0.5 秒，让 USB 总线带宽和电压恢复平稳
        import time
        time.sleep(0.5)
        
        # 3. 开始录音
        text = stt.capture_and_transcribe()

        # 即使文字为空（没说话），只要有画面，我们也让大脑分析表情
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
        self.hw.clear_led()
        logger.info("[Orchestrator] 系统已停止")
