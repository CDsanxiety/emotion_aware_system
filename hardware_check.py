# hardware_check.py
import time
import os
import cv2
import board
import neopixel
import subprocess
from src.core.config import LED_COUNT, LED_PIN, LED_BRIGHTNESS, CAMERA_INDEX, AUDIO_INPUT_INDEX, AUDIO_OUTPUT_DEVICE
from src.utils.logger import logger

def test_led():
    logger.info("--- 正在测试灯带 (GPIO 12) ---")
    try:
        pixels = neopixel.NeoPixel(getattr(board, f"D{LED_PIN}"), LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False)
        for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
            pixels.fill(color)
            pixels.show()
            time.sleep(1)
        pixels.fill((0, 0, 0))
        pixels.show()
        logger.info("✅ 灯带测试完成")
    except Exception as e:
        logger.error(f"❌ 灯带测试失败: {e}")

def test_camera():
    logger.info("--- 正在测试摄像头 (Index 0) ---")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("❌ 无法打开摄像头")
        return
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_cam.jpg", frame)
        logger.info("✅ 摄像头拍照成功，请检查项目目录下的 test_cam.jpg")
    else:
        logger.error("❌ 摄像头读取画面失败")
    cap.release()

def test_audio():
    logger.info("--- 正在测试音频 (录音 -> 播放) ---")
    temp_file = "test_audio.wav"
    try:
        # 录音
        logger.info(f"请对着麦克风说话 (录制 3 秒)...")
        cmd_rec = ["arecord", "-D", f"plughw:{AUDIO_INPUT_INDEX}", "-d", "3", "-f", "cd", "-q", temp_file]
        subprocess.run(cmd_rec, check=True)
        
        # 播放
        logger.info(f"正在播放刚才录制的声音...")
        cmd_play = ["aplay", "-D", AUDIO_OUTPUT_DEVICE, "-q", temp_file]
        subprocess.run(cmd_play, check=True)
        logger.info("✅ 音频链路测试完成")
    except Exception as e:
        logger.error(f"❌ 音频测试失败: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    logger.info("🚀 开始硬件全链路自检...")
    test_led()
    test_camera()
    test_audio()
    logger.info("🏁 自检结束")
