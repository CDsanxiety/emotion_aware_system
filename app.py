# app.py
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv

import ros_client  # [Local] ROS 硬件支持
from audio import transcribe_file
from llm_api import clear_memory, get_response
from utils import logger, setup_logger
from vision import process_image

# 初始化
load_dotenv()
setup_logger()

# [Local] 初始化 ROS 管理器
ros_manager = ros_client.ROSManager()

# ================== 核心逻辑 (并发感知 + 逻辑决策 + 硬件下发) ==================

def main_process(frame, audio_path):
    """
    1. 并发感知：Vision (VLM) + STT (Speech) 并行处理，减少等待。
    2. 逻辑决策：LLM 融合理解并生成嵌套 JSON。
    3. 硬件下发：若有 ROS 指令，实时下发到树莓派。
    """
    if frame is None and not audio_path:
        return None, "请提供摄像头画面或语音输入。", None

    start_time = time.time()
    logger.info("--- 开始新一轮集成感知决策流 ---")

    # [并发阶段 1] 视觉与听觉并行
    vision_desc = "（无画面）"
    voice_text = ""
    face_emotion = "neutral"

    with ThreadPoolExecutor(max_workers=2) as executor:
        # 并行处理图像与语音
        f_vision = executor.submit(process_image, frame)
        f_stt = executor.submit(transcribe_file, audio_path)

        # 获取结果
        v_res = f_vision.result()
        vision_desc = v_res.get("description", "（描述生成失败）")
        voice_text = f_stt.result()

    logger.info(f"感知耗时: {time.time() - start_time:.2f}s")
    
    # [逻辑阶段 2] LLM 聚合理解
    # get_response 内部处理了 Memory 与感知融合
    res, response_audio = get_response(
        face_emotion, 
        voice_text, 
        enable_tts=True, 
        vision_desc=vision_desc
    )

    # [集成阶段 3] ROS 硬件执行
    # 将决策结果发布到 ROS 话题，由树莓派订阅并执行（转头、亮灯等）
    if ros_manager:
        try:
            ros_manager.publish_action(res)
        except Exception as e:
            logger.warning(f"ROS 动作下发失败（可能未连接）: {e}")

    logger.info(f"全链路总耗时: {time.time() - start_time:.2f}s")

    # 返回给 Gradio 展示：音频, JSON 状态, VLM 描述
    return response_audio, res, vision_desc


def debug_process(face_input, text_input):
    """旁路调试：手动输入模拟感知结果。"""
    res, audio = get_response(face_input, text_input, enable_tts=True, vision_desc="（调试模式：手动输入）")
    
    if ros_manager:
        ros_manager.publish_action(res)
        
    return audio, res


# ================== Gradio UI 设计 (Premium 风格) ==================

with gr.Blocks(theme=gr.themes.Soft(), title="微影听镜 V2.0 - 实机集成版") as demo:
    gr.Markdown("""
    # 🤖 暖暖智能伴侣机器人 (Nuannuan Robot V2.0)
    **集成状态**：已开启端云协同感知 | **硬件连接**：ROSBridge 待命
    """)
    
    with gr.Tab("📱 实时交互模式"):
        with gr.Row():
            with gr.Column(scale=1):
                input_cam = gr.Image(sources=["webcam"], label="实时预览 (VLM 采集)", type="numpy")
                input_mic = gr.Audio(sources=["microphone"], type="filepath", label="语音对话 (STT)")
                btn_run = gr.Button("🚀 开启同步交互", variant="primary")
                btn_clear = gr.Button("🧹 清空机器人记忆")
            
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="暖暖的回复", autoplay=True)
                output_vlm = gr.Textbox(label="🔍 视觉感知详情 (VLM Description)")
                output_json = gr.JSON(label="🧠 核心决策链路 (Perception-Decision-Execution)")

        btn_run.click(
            main_process, 
            inputs=[input_cam, input_mic], 
            outputs=[output_audio, output_json, output_vlm]
        )
        btn_clear.click(clear_memory, outputs=[output_json])

    with gr.Tab("🔧 专业调试工具"):
        with gr.Row():
            db_face = gr.Dropdown(
                ["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"], 
                label="模拟表情标签", value="neutral"
            )
            db_text = gr.Textbox(label="模拟语音输入", placeholder="例如：我今天过得不太好...")
            db_btn = gr.Button("发送测试流")
        
        db_audio = gr.Audio(label="调试音频回复")
        db_json = gr.JSON(label="调试决策结果")
        
        db_btn.click(debug_process, inputs=[db_face, db_text], outputs=[db_audio, db_json])

    gr.Markdown("""
    ---
    *注：本系统正处于实机集成阶段。所有动作指令将通过 ROSBridge 同步发送至树莓派控制节点。*
    """)

if __name__ == "__main__":
    # 启动前确认 Memory 已清空
    clear_memory()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
