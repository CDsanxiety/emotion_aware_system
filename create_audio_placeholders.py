#!/usr/bin/env python3
# create_audio_placeholders.py
"""
创建音频文件占位符
"""
import os

# 音频文件列表
audio_files = [
    "startup.mp3",
    "shutdown.mp3",
    "alert.mp3",
    "background.mp3",
    "model_switch.mp3",
    "wakeup.mp3",
    "error.mp3"
]

# 音乐文件夹路径
music_dir = "music"

# 确保音乐文件夹存在
os.makedirs(music_dir, exist_ok=True)

# 创建音频文件占位符
for audio_file in audio_files:
    file_path = os.path.join(music_dir, audio_file)
    # 创建空文件
    open(file_path, 'a').close()
    print(f"创建音频文件占位符: {file_path}")

print("\n音频文件占位符创建完成！")
print("\n请将实际的音频文件替换这些占位符文件。")
print("文件列表:")
for audio_file in audio_files:
    print(f"- {audio_file}")
