#!/usr/bin/env python3
# test_aliyun_stt.py
# 测试阿里云智能语音交互 STT 功能

import sys
import os

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio import recognize_speech, transcribe_file

print("=== 阿里云智能语音交互 STT 测试 ===")
print("1. 测试麦克风语音识别")
print("2. 测试音频文件转写")
print("3. 退出")

while True:
    choice = input("请选择测试选项 (1-3): ")
    
    if choice == "1":
        print("\n=== 麦克风语音识别测试 ===")
        print("请对着麦克风说话，系统将识别您的语音...")
        print("按 Ctrl+C 停止测试")
        
        try:
            result = recognize_speech()
            if result:
                print(f"识别结果: {result}")
            else:
                print("未检测到语音或识别失败")
        except KeyboardInterrupt:
            print("\n测试已取消")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            
    elif choice == "2":
        print("\n=== 音频文件转写测试 ===")
        audio_path = input("请输入音频文件路径: ")
        
        if not os.path.exists(audio_path):
            print(f"文件不存在: {audio_path}")
            continue
        
        try:
            result = transcribe_file(audio_path)
            if result:
                print(f"转写结果: {result}")
            else:
                print("转写失败")
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            
    elif choice == "3":
        print("\n测试结束")
        break
    else:
        print("无效选项，请重新选择")
    
    print("\n" + "-" * 50 + "\n")
