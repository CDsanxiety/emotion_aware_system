# src/hardware/physical_interface.py
import time
import subprocess
import board
import neopixel
from src.core.config import LED_COUNT, LED_PIN, LED_BRIGHTNESS, AUDIO_OUTPUT_DEVICE
from src.utils.logger import logger

class PhysicalInterface:
    def __init__(self):
        try:
            self.pixels = neopixel.NeoPixel(
                getattr(board, f"D{LED_PIN}"), 
                LED_COUNT, 
                brightness=LED_BRIGHTNESS, 
                auto_write=False
            )
            logger.info(f"[Hardware] LED 接口初始化成功 (Pin: {LED_PIN})")
        except Exception as e:
            logger.error(f"[Hardware] LED 初始化失败: {e}")
            self.pixels = None

    def play_sound(self, file_path, wait=False):
        """直接调用 mpg123 播放音频"""
        if not os.path.exists(file_path):
            logger.warning(f"[Hardware] 找不到音效文件: {file_path}")
            return
        
        cmd = ["mpg123", "-a", AUDIO_OUTPUT_DEVICE, "-q", file_path]
        try:
            if wait:
                subprocess.run(cmd)
            else:
                subprocess.Popen(cmd)
        except Exception as e:
            logger.error(f"[Hardware] 播放音效失败: {e}")

    def set_led_emotion(self, emotion):
        """根据情绪切换灯光效果"""
        if not self.pixels: return
        
        colors = {
            "happy": (0, 255, 0),      # 绿
            "sad": (0, 0, 255),        # 蓝
            "angry": (255, 0, 0),      # 红
            "neutral": (255, 100, 0),  # 暖黄
        }
        
        color = colors.get(emotion, (255, 255, 255)) # 默认白
        self.pixels.fill(color)
        self.pixels.show()
        logger.info(f"[Hardware] 灯光切换为: {emotion} 模式")

    def clear_led(self):
        if self.pixels:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()

import os
