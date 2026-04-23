import os
import pygame
import logging

# 配置日志
logger = logging.getLogger(__name__)

class AudioManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.music_folder = os.path.join(os.path.dirname(__file__), "music")
        self._init_audio()

    def _init_audio(self):
        try:
            # 确保 pygame 只初始化一次
            if not pygame.get_init():
                pygame.init()
            # 初始化音频 mixer
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._mixer = pygame.mixer
            # 设置音量为最大
            self._mixer.music.set_volume(1.0)
        except Exception as e:
            self._mixer = None
            logger.error(f"音频系统初始化失败: {e}")

    def play(self, name):
        if not self._mixer:
            logger.error("音频系统未初始化，无法播放音频")
            return
        try:
            path = os.path.join(self.music_folder, f"{name}.mp3")
            if not os.path.exists(path):
                logger.error(f"音频文件不存在: {path}")
                return
            
            snd = self._mixer.Sound(path)
            channel = snd.play()
            logger.info(f"播放音频: {name}")
        except Exception as e:
            logger.error(f"播放音频失败: {e}")

audio = AudioManager()

def play_system_audio(name):
    audio.play(name)