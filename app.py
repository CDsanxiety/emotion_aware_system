"""
整合版 app.py: 修复音效播放、增强硬件联动、优化事件绑定。
"""
import time
import os
import threading
from pickle import FALSE

import cv2
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config import ROS_BRIDGE_URI, CAMERA_INDEX
from audio import transcribe_file, set_system_shutting_down
from llm_api import clear_memory, get_response_with_multi_agent as get_response
from ros_client import global_ros_manager
from utils import logger, setup_logger
from vision import process_image
from blackboard import Blackboard
from memory_rag import LongTermMemory
from pad_model import PADEmotionEngine
from agent_loop import start_agentic_main_loop, AgentLoopHandle, GlobalAgentState


# ================== 硬件联动修复：真实音效播放 ==================
def play_system_audio(event_type):
    """
    调用系统播放器播放音效。
    树莓派建议安装: sudo apt-get install mpg123
    """
    audio_map = {
        "startup": "music/startup.mp3",
        "shutdown": "music/shutdown.mp3",
        "wake": "music/wake.mp3"
    }
    file_path = audio_map.get(event_type)
    if file_path and os.path.exists(file_path):
        from config import AUDIO_OUTPUT_DEVICE
        logger.info(f"[Audio] 正在播放音效: {file_path} -> {AUDIO_OUTPUT_DEVICE}")
        # 去掉 &，强制阻塞播放，防止在播放期间后台开启录音和视觉导致 USB 硬件争抢锁死
        os.system(f"mpg123 -a {AUDIO_OUTPUT_DEVICE} {file_path} > /dev/null 2>&1")
    else:
        logger.warning(f"[Audio] 音效文件不存在: {file_path}")


# ================== 核心组件初始化 ==================
global_blackboard = Blackboard()
global_memory = LongTermMemory()
global_pad_engine = PADEmotionEngine()
global_agent_loop = None


# ================== 核心处理逻辑 ==================
def main_process(frame, audio_path):
    overall_start = time.time()

    # 播放唤醒音
    play_system_audio("wake")

    # [1. 感知阶段]
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_vision = executor.submit(process_image, frame)
        f_stt = executor.submit(transcribe_file, audio_path)

        vision_desc, is_vision_fallback = f_vision.result()
        voice_text = f_stt.result()

    # 更新状态
    global_blackboard.update_vision(vision_desc, frame is not None)
    if voice_text:
        global_blackboard.update_speech(voice_text)

    # [2. 认知决策]
    hardware_status = global_ros_manager.get_status()
    res, response_audio = get_response(
        face_emotion="neutral",
        voice_text=voice_text,
        enable_tts=True,
        vision_desc=vision_desc
    )

    # [3. 硬件下发]
    # 这里的 res 会通过 ROS 发送给 led_hardware_driver.py 和 audio_player_driver.py
    global_ros_manager.publish_action(res)

    total_latency = time.time() - overall_start
    latency_report = {
        "端到端总延时": f"{total_latency:.2f}s",
        "视觉感知": vision_desc[:30] + "...",
        "硬件反馈": hardware_status if hardware_status else "已连接"
    }

    return response_audio, res, vision_desc, latency_report


# ================== UI 界面 ==================
with gr.Blocks(theme=gr.themes.Soft(), title="微影听镜") as demo:
    gr.Markdown("# 🤖 微影听镜 - 情感感知交互系统")

    with gr.Row():
        with gr.Column():
            # 这里的 webcam=True 会调用浏览器摄像头，如果想用树莓派本地摄像头，需确保 frame 传入
            input_cam = gr.Image(sources=["webcam"], label="实时画面", type="numpy")
            input_mic = gr.Audio(sources=["microphone"], type="filepath", label="语音输入")
            btn_run = gr.Button("🚀 立即唤醒", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(label="机器人回复", autoplay=True)
            output_vlm = gr.Textbox(label="感知详情")
            output_json = gr.JSON(label="决策数据")

    btn_run.click(
        main_process,
        inputs=[input_cam, input_mic],
        outputs=[output_audio, output_json, output_vlm]
    )

if __name__ == "__main__":
    logger.info("================ 系统初始化开始 ================")
    
    # 1. 优先挂载 ROS 底层物理链路
    from ros_client import global_ros_manager
    global_ros_manager.connect()
    
    # 2. 播放开机音效（阻塞等待播放完毕，宣告声卡就绪）
    play_system_audio("startup")
    
    # 3. 启动后台自主感知与决策大循环
    logger.info("正在拉起后台 Agent 感知大循环...")
    global_agent_loop = start_agentic_main_loop(
        blackboard=global_blackboard,
        enable_tts=True,
        camera_index=CAMERA_INDEX
    )

    # 4. 最后启动前端控制台界面
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    finally:
        set_system_shutting_down(True)
        import vision
        vision.release_global_camera()
        play_system_audio("shutdown")
        if global_agent_loop: global_agent_loop.stop()
        global_ros_manager.shutdown()