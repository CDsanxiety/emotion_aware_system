# hardware_check.py
import cv2
import time
import os
import board
import neopixel
import asyncio
from edge_tts import Communicate
from config import LED_COUNT, LED_BRIGHTNESS, AUDIO_INPUT_INDEX, AUDIO_OUTPUT_DEVICE, CAMERA_INDEX

print("--- 🛠️ 暖暖机器人硬件自检程序启动 ---")

# 1. 测试 LED 灯带
print("\n[1/4] 正在测试 LED 灯带 (60 颗)...")
try:
    pixels = neopixel.NeoPixel(board.D18, LED_COUNT, brightness=LED_BRIGHTNESS, auto_write=False)
    for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        pixels.fill(color)
        pixels.show()
        time.sleep(1)
    pixels.fill((0, 0, 0))
    pixels.show()
    print("✅ LED 灯带测试完成 (红/绿/蓝 循环)。")
except Exception as e:
    print(f"❌ LED 测试失败: {e}")

# 2. 测试摄像头
print("\n[2/4] 正在测试摄像头...")
try:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_camera.jpg", frame)
        print("✅ 摄像头抓图成功，已保存至 test_camera.jpg")
    else:
        print("❌ 摄像头开启失败，请检查排线。")
    cap.release()
except Exception as e:
    print(f"❌ 摄像头测试异常: {e}")

# 3. 测试音频输出 (TTS + 扬声器)
print("\n[3/4] 正在测试扬声器 (使用 mpg123)...")
async def test_audio():
    try:
        # 测试 TTS
        text = "系统自检开始，当前硬件连接正常。"
        communicate = Communicate(text, "zh-CN-XiaoxiaoNeural")
        await communicate.save("check_audio.mp3")
        os.system(f"mpg123 -a {AUDIO_OUTPUT_DEVICE} check_audio.mp3")
        print(f"✅ 语音合成 (TTS) 测试通过。")
        
        # 测试背景音乐
        print("正在尝试试听背景音乐 happy.mp3...")
        if os.path.exists("music/happy.mp3"):
            os.system(f"mpg123 -a {AUDIO_OUTPUT_DEVICE} music/happy.mp3 --frames 150") # 只播几秒
            print("✅ 背景音乐文件读取正常。")
        else:
            print("❌ 未发现背景音乐文件，请检查 music/ 目录。")
    except Exception as e:
        print(f"❌ 扬声器测试失败: {e}")

asyncio.run(test_audio())

# 4. 测试麦克风录音
print("\n[4/4] 正在测试摄像头麦克风 (录音 3 秒)...")
try:
    # 使用 arecord 录音测试
    cmd = f"arecord -D plughw:{AUDIO_INPUT_INDEX} -d 3 -f cd test_mic.wav"
    print(f"执行录音指令: {cmd}")
    os.system(cmd)
    if os.path.exists("test_mic.wav"):
        print("✅ 录音成功，已保存至 test_mic.wav")
        print("正在回放录音...")
        os.system(f"mpg123 -a {AUDIO_OUTPUT_DEVICE} test_mic.wav")
    else:
        print("❌ 录音文件未生成，请检查 arecord -l 确认索引。")
except Exception as e:
    print(f"❌ 麦克风测试异常: {e}")

print("\n--- 🏁 自检结束 ---")
print("提示：如果所有步骤都显示 OK，你就可以放心启动 app.py 了！")
