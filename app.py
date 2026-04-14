# app.py
"""
微影听镜 V2.0 - 实机集成重塑版
核心特性：并行感知、闭环反馈、延时监控、硬件状态同步。
"""
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

# 这里的 import 顺序按系统层级排列
from config import ROS_BRIDGE_URI
from audio import transcribe_file
from llm_api import clear_memory, get_response
from ros_client import global_ros_manager
from utils import logger, setup_logger
from vision import process_image

# 初始化日志与硬件连接
setup_logger()
global_ros_manager.connect()

# ================== 核心集成流水线 (Integrated Pipeline) ==================

def main_process(frame, audio_path):
    """
    竞赛级全链路：感知并行化 -> 决策结构化 -> 执行闭环化。
    """
    if frame is None and not audio_path:
        return None, "等待输入中...", "（无画面）", {}

    overall_start = time.time()
    logger.info("--- [Pipeline] 开启新一轮感知脉冲 ---")

    # [1. 并行感知阶段] - 减少阻塞，利用多核性能
    perception_start = time.time()
    vision_desc = "（未采集）"
    voice_text = ""
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_vision = executor.submit(process_image, frame)
        f_stt = executor.submit(transcribe_file, audio_path)

        res_v = f_vision.result()
        vision_desc = res_v.get("description", "视觉链路异常")
        is_vision_fallback = res_v.get("is_fallback", False)
        
        voice_text = f_stt.result()

    perception_latency = time.time() - perception_start

    # [2. 认知决策阶段] - 结合当前硬件状态进行决策
    # 获取当前的硬件闭环反馈（如有）
    hardware_status = global_ros_manager.get_status()
    
    # 将感知结果送入 LLM
    # 注意：此处 face_emotion 暂时占位，主要依赖 VLM 的 description
    res, response_audio = get_response(
        face_emotion="neutral", 
        voice_text=voice_text, 
        enable_tts=True, 
        vision_desc=vision_desc
    )

    # [3. 硬件下发阶段] - 异步执行，不阻塞 UI 响应
    global_ros_manager.publish_action(res)

    total_latency = time.time() - overall_start
    
    # 构造性能监控数据
    latency_report = {
        "感知层延时 (STT+VLM)": f"{perception_latency:.2f}s",
        "端到端总延时 (E2E)": f"{total_latency:.2f}s",
        "视觉模式": "端侧兜底" if is_vision_fallback else "云端 VLM",
        "硬件反馈": hardware_status if hardware_status else "等待同步..."
    }

    logger.info(f"脉冲完成: {total_latency:.2f}s | 状态: {res.get('emotion', '未知')}")

    return response_audio, res, vision_desc, latency_report


def debug_process(face_input, text_input):
    """调试通道：绕过感知层直达决策。"""
    res, audio = get_response(face_input, text_input, enable_tts=True, vision_desc="（调试模式：手动输入）")
    global_ros_manager.publish_action(res)
    return audio, res


# ================== 竞赛级交互界面 (National Prize UI) ==================

with gr.Blocks(theme=gr.themes.Soft(), title="Nuannuan V2.0 Pro") as demo:
    gr.Markdown(f"""
    # 🤖 暖暖 (Nuannuan) - 情感感知伴侣机器人
    > **[端云协同 V2.0]**：当前已连接至 `{ROS_BRIDGE_URI}` | 并行感知引擎已就绪
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tab("📱 实时交互"):
                with gr.Row():
                    input_cam = gr.Image(sources=["webcam"], label="视觉采集 (VLM)", type="numpy")
                    input_mic = gr.Audio(sources=["microphone"], type="filepath", label="语言意图 (STT)")
                
                with gr.Row():
                    btn_run = gr.Button("🚀 立即唤醒", variant="primary")
                    btn_clear = gr.Button("🧹 重置记忆", variant="secondary")
                
                output_audio = gr.Audio(label="暖暖的回复 (TTS)", autoplay=True)
                output_vlm = gr.Textbox(label="🔍 视觉感知详情", lines=3)

        with gr.Column(scale=3):
            gr.Markdown("### 🧠 核心决策链路")
            output_json = gr.JSON(label="Pipeline (JSON)")
            
            gr.Markdown("### 📊 工程性能与硬件反馈")
            output_latency = gr.JSON(label="Performance Monitor")

    with gr.Accordion("🔧 高级调试与后门 (Competition Recovery)", open=False):
        with gr.Row():
            db_face = gr.Dropdown(["happy", "sad", "neutral", "fear"], label="情绪占位", value="neutral")
            db_text = gr.Textbox(label="模拟文本输入")
            db_btn = gr.Button("发送模拟流")
        db_audio = gr.Audio(label="调试音频")
        db_json = gr.JSON(label="调试决策")
        db_btn.click(debug_process, inputs=[db_face, db_text], outputs=[db_audio, db_json])

    # 事件绑定
    btn_run.click(
        main_process, 
        inputs=[input_cam, input_mic], 
        outputs=[output_audio, output_json, output_vlm, output_latency]
    )
    btn_clear.click(clear_memory, outputs=[output_json])

    gr.Markdown("""
    ---
    *注：本系统遵循“端云协同”原则，优先在边缘侧进行风险预审，关键决策由云端大模型完成。*
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
