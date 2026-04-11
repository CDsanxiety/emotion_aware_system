# app.py
import gradio as gr
import json
import os
import cv2
import numpy as np
from audio import recognize_speech
from llm_api import get_response
from ros_client import ros_manager
from debug_mode import EMOTION_OPTIONS, PRESET_TESTS

# ================== 初始化 ==================
# 尝试连接 ROSBridge（地址可从环境变量读取，默认为 localhost）
ROS_HOST = os.getenv("ROS_HOST", "localhost")
ros_manager.connect()

# ================== 核心胶水逻辑 ==================
def main_process(image, audio_input):
    """
    V2.0 深度融合流程：一次性调用
    """
    # 语音转文字
    voice_text = recognize_speech() if audio_input is not None else ""
    print(f"[Log] 语音内容: {voice_text}")

    # 调用两阶段推理接口 (VLM -> LLM)
    # 此时图像直接传给 get_response，内部会调用 analyze_scene
    result = get_response(image, voice_text, enable_tts=True)

    # 发布指令到 ROS
    ros_manager.publish_action(result)

    audio_path = "response.mp3"
    return result, audio_path

def debug_process(emotion, text):
    """
    Debug 模式：模拟一个空的或预设的图帧
    """
    # 创建一个纯色背景作为模拟图帧，并在上面写上表情文字（供 VLM 参考）
    dummy_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, f"Emotion: {emotion}", (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    result = get_response(dummy_frame, text, enable_tts=True)
    
    # 同样发布到 ROS
    ros_manager.publish_action(result)

    audio_path = "response.mp3"
    return result, audio_path

def check_ros_status():
    """返回 ROS 连接状态的 Markdown 字符串"""
    if ros_manager.is_connected:
        return "### 🟢 ROS 状态：已连接 (ws://{}:9090)".format(ros_manager.host)
    return "### 🔴 ROS 状态：离线 (单机仿真模式)"

def format_reply_text(result):
    """格式化显示回复文字"""
    if result and isinstance(result, dict):
        reply = result.get("reply", "")
        emotion = result.get("emotion", "")
        action = result.get("action", "")
        # V2.0 增加 VLM 描述的预览
        vlm_desc = result.get("vlm_description", "无视觉描述")
        
        return (f"### 💬 暖暖说：{reply}\n\n"
                f"> 🧠 **深度感知**：{emotion}\n\n"
                f"> 🎬 **执行动作**：`{action}`\n\n"
                f"--- \n"
                f"🔍 **VLM 原始语义**：{vlm_desc}")
    return "### 💬 等待交互..."

# ================== UI 布局设计 ==================
with gr.Blocks(title="微影听镜 V2.0 - 情感智能交互系统") as demo:
    gr.Markdown("""
    # 🤖 微影听镜 V2.0：基于端云协同与 VLM 语义感知
    **核心能力：** Qwen-VL-Max 跨模态感知识别 ｜ Qwen-Max 逻辑决策 ｜ ROSBridge 硬件联动
    """)

    ros_status_display = gr.Markdown(check_ros_status)
    # 定时更新状态（Gradio 3.x/4.x 技巧）
    demo.load(check_ros_status, None, ros_status_display)

    with gr.Row():
        with gr.Column(scale=1):
            camera_input = gr.Image(
                label="📸 视觉感知 (VLM 实时采集)",
                sources=["webcam"],
                type="numpy"
            )
            gr.Markdown("<center>点击拍照并同步语音触发</center>")

        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎤 语音输入 (ASR)",
                sources=["microphone"],
                type="numpy"
            )
            run_btn = gr.Button("🚀 开启多模态协同感知", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=1):
            json_output = gr.JSON(label="🧠 多模态融合决策中心 (JSON)")
        with gr.Column(scale=1):
            speech_output = gr.Audio(label="📢 暖暖的语音回复 (TTS)", autoplay=True)

    robot_text_display = gr.Markdown("### 💬 暖暖说：等待交互...")

    # 事件绑定
    run_btn.click(
        fn=main_process,
        inputs=[camera_input, audio_input],
        outputs=[json_output, speech_output]
    ).then(
        fn=format_reply_text,
        inputs=[json_output],
        outputs=[robot_text_display]
    )

    # ========== Debug Mode ==========
    with gr.Accordion("🔧 开发者工具 (Debug Mode)", open=False):
        with gr.Row():
            with gr.Column():
                debug_emotion = gr.Dropdown(choices=EMOTION_OPTIONS, label="模拟表情", value="neutral")
                debug_text = gr.Textbox(label="模拟语音内容", lines=2)
                debug_btn = gr.Button("模拟发送", variant="secondary")
            with gr.Column():
                debug_json = gr.JSON(label="Debug JSON")
                debug_audio = gr.Audio(label="Debug 语音")

        debug_btn.click(
            fn=debug_process,
            inputs=[debug_emotion, debug_text],
            outputs=[debug_json, debug_audio]
        ).then(
            fn=format_reply_text,
            inputs=[debug_json],
            outputs=[robot_text_display]
        )

    gr.Markdown("""
    ---
    **技术架构：** 端侧采集 (Raspberry Pi) → 云端 VLM (Qwen-VL-Max) → 云端决策 (Qwen-Max) → ROS 指令下发
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)