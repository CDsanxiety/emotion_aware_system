# app.py
import gradio as gr
import json
from vision import process_image
from audio import transcribe_file
from llm_api import get_response
from debug_mode import EMOTION_OPTIONS, PRESET_TESTS


# ================== 核心胶水逻辑 ==================
def main_process(image, audio_input):
    """
    正常模式：摄像头拍照 + 麦克风录音
    """
    # Step 1: 视觉分析
    detected_emotion = process_image(image) if image is not None else "neutral"
    print(f"[Log] 检测到表情: {detected_emotion}")

    # Step 2: 语音转文字
    voice_text = transcribe_file(audio_input) if audio_input is not None else ""
    print(f"[Log] 语音内容: {voice_text}")

    # Step 3 & 4: 调用 LLM 并在内部生成 TTS 语音，获取动态生成的音频文件路径
    result, audio_path = get_response(detected_emotion, voice_text, enable_tts=True)

    return result, audio_path


def debug_process(emotion, text):
    """
    Debug 模式：手动输入
    """
    result, audio_path = get_response(emotion, text, enable_tts=True)
    return result, audio_path


def fill_preset(preset_name):
    """填充预设"""
    if preset_name and preset_name in PRESET_TESTS:
        return PRESET_TESTS[preset_name]
    return "neutral", ""


def format_reply_text(result):
    """格式化显示回复文字"""
    if result and isinstance(result, dict):
        reply = result.get("reply", "")
        emotion = result.get("emotion", "")
        action = result.get("action", "")
        return f"### 💬 暖暖说：{reply}\n\n> 😊 情绪：{emotion} ｜ 🎬 动作：{action}"
    return "### 💬 等待交互..."


# ================== UI 布局设计 ==================
with gr.Blocks(title="暖暖情感机器人仿真系统") as demo:
    gr.Markdown("""
    # 🤖 暖暖：情感感知智能家居伴侣
    项目定位：国家级仿真比赛作品 | 硬件协同：多模态情感计算
    """)

    with gr.Row():
        # 左侧：视觉抓取
        with gr.Column(scale=1):
            camera_input = gr.Image(
                label="📸 实时面部捕捉",
                sources=["webcam"],
                type="numpy"
            )
            gr.Markdown("<center>点击上方按钮拍照</center>")

        # 右侧：语音交互
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎤 按住说话",
                sources=["microphone"],
                type="filepath"
            )
            run_btn = gr.Button("🚀 触发深度融合感知", variant="primary", size="lg")

    # 中间：输出展示
    with gr.Row():
        with gr.Column(scale=1):
            json_output = gr.JSON(label="🧠 多模态融合决策中心")
        with gr.Column(scale=1):
            speech_output = gr.Audio(label="📢 暖暖的语音回复", autoplay=True)

    # 机器人语录
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

    # ========== Debug Mode 折叠栏 ==========
    with gr.Accordion("🔧 Debug Mode ", open=False):
        gr.Markdown("### ⚠️ 仅限摄像头/麦克风/网络故障时使用")
        gr.Markdown("手动输入情绪和文字，绕过硬件直接测试大模型能力")

        with gr.Row():
            with gr.Column():
                debug_emotion = gr.Dropdown(
                    choices=EMOTION_OPTIONS,
                    label="😊 手动选择表情",
                    value="neutral"
                )
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESET_TESTS.keys()),
                    label="📋 快捷预设场景",
                    value=None
                )
                debug_text = gr.Textbox(
                    label="💬 手动输入文字",
                    placeholder="输入你想说的话...",
                    lines=3
                )
                debug_btn = gr.Button("🚀 Debug 发送", variant="secondary")

            with gr.Column():
                debug_json = gr.JSON(label="📤 Debug 输出")
                debug_audio = gr.Audio(label="📢 语音输出", autoplay=True)
                debug_reply = gr.Markdown("### 💬 等待 Debug 结果...")

        preset_dropdown.change(
            fn=fill_preset,
            inputs=[preset_dropdown],
            outputs=[debug_emotion, debug_text]
        )

        debug_btn.click(
            fn=debug_process,
            inputs=[debug_emotion, debug_text],
            outputs=[debug_json, debug_audio]
        ).then(
            fn=format_reply_text,
            inputs=[debug_json],
            outputs=[debug_reply]
        )

    gr.Markdown("""
    ---
    **技术支撑：** FER + Google STT + Qwen-Turbo + Edge-TTS
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True, theme=gr.themes.Soft())