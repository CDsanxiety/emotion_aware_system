# src/core/orchestrator.py
import time
import cv2
import threading
from src.cloud import stt, tts, brain
from src.hardware.controller import hw_controller
from src.core.config import CAMERA_INDEX, VISION_INTERVAL, IDLE_THRESHOLD
from src.utils.logger import logger

class EmotionSystemOrchestrator:
    def __init__(self):
        self.cap = None
        self.running = False
        self.last_interaction_time = time.time()

    def start(self):
        logger.info("================ 系统重构版启动 ================")
        
        # 1. 初始化硬件控制
        hw_controller.connect()
        
        # 2. 初始化摄像头
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        
        # 3. 启动主循环
        self._main_loop()

    def _main_loop(self):
        while self.running:
            try:
                # [阶段 1: 语音采集]
                # 阻塞式监听（可改为触发式）
                voice_text = stt.capture_and_transcribe()
                
                # [阶段 2: 视觉采样]
                ret, frame = self.cap.read()
                if not ret: frame = None
                
                # [阶段 3: 云端推理]
                if voice_text or (time.time() - self.last_interaction_time > IDLE_THRESHOLD):
                    if not voice_text:
                        logger.info("[Orchestrator] 触发主动关心模式...")
                    
                    # 思考
                    result = brain.think(frame, voice_text)
                    
                    if result:
                        self.last_interaction_time = time.time()
                        
                        # [阶段 4: 硬件执行]
                        # 4.1 动作下发 (LED等)
                        hw_controller.execute(result.get("action"), result.get("emotion"))
                        
                        # 4.2 语音回复 (TTS)
                        tts.speak(result.get("reply"))
                
                # 采样间隔
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                logger.error(f"[Orchestrator] 循环异常: {e}")
                time.sleep(2)

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()
        logger.info("系统已安全关闭")

if __name__ == "__main__":
    orchestrator = EmotionSystemOrchestrator()
    orchestrator.start()
