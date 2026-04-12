# app.py
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import gradio as gr

from vision import process_image, no_input_vision
from audio import transcribe_file
from llm_api import get_response, clear_memory
from debug_mode import EMOTION_OPTIONS, PRESET_TESTS


def _format_vision_panel(vision_result: dict) -> str:
    """视觉感知描述：展示 VLM（或降级）对画面的文字理解，便于用户核对「到底观察到了什么」。"""
    if not vision_result:
        return "### 视觉感知描述\n*暂无：请先拍照并触发融合。*"
    desc = (vision_result.get("description") or "—").strip()
    if vision_result.get("success"):
        tag = "云端 VLM（qwen-vl-plus）"
    elif vision_result.get("is_fallback"):
        tag = "降级 · 本地参考（非云端整段描述）"
    else:
        tag = "视觉"
    meta = []
    if "success" in vision_result:
        meta.append(f"success={vision_result.get('success')}")
    if "is_fallback" in vision_result:
        meta.append(f"is_fallback={vision_result.get('is_fallback')}")
    meta_line = " · ".join(meta) if meta else ""
    return (
        f"### 视觉感知描述 · *{tag}*\n"
        f"{('> ' + meta_line) if meta_line else ''}\n\n"
        f"{desc}"
    )


def _format_pipeline_perception(result: dict) -> str:
    p = (result or {}).get("perception") if isinstance(result, dict) else None
    if not isinstance(p, dict):
        return "### 1. 感知（场景 · 状态 · 风险 · 情绪）\n*等待全链路结果…*"
    return (
        "### 1. 感知（场景 · 状态 · 风险 · 情绪）\n"
        f"- **场景内容**：{p.get('scene', '—')}\n"
        f"- **状态**：{p.get('state', '—')}\n"
        f"- **风险**：`{p.get('risk', '—')}`\n"
        f"- **情绪线索**：{p.get('emotion_signal', '—')}"
    )


def _format_pipeline_decision(result: dict) -> str:
    d = (result or {}).get("decision") if isinstance(result, dict) else None
    if not isinstance(d, dict):
        return "### 2. 决策\n*等待全链路结果…*"
    return (
        "### 2. 决策（判断 · 优先级）\n"
        f"- **判断**：{d.get('judgment', '—')}\n"
        f"- **优先级**：`{d.get('priority', '—')}`"
    )


def _format_pipeline_execution(result: dict) -> str:
    e = (result or {}).get("execution") if isinstance(result, dict) else None
    if not isinstance(e, dict):
        return "### 3. 执行（动作 · 语言 · 策略）\n*等待全链路结果…*"
    return (
        "### 3. 执行（动作 · 语言 · 策略）\n"
        f"- **情绪标签**：`{e.get('emotion', '—')}`\n"
        f"- **家居动作**：`{e.get('action', '—')}`\n"
        f"- **策略**：{e.get('strategy', '—')}\n"
        f"- **对用户说**：{e.get('reply', '—')}"
    )


def _latency_note(seconds: float) -> str:
    tip = (
        f"本轮 **感知→决策→执行** 墙钟约 **`{seconds:.2f}s`**。"
        " 感知阶段已对 **VLM 与 STT 并行** 以压缩等待；决策与执行在 LLM（及可选 TTS）。"
    )
    if seconds > 2.0:
        tip += (
            "\n\n> *提示：若常超过 2s，多为云端排队、VLM 耗时或 TTS；可在演示时关 TTS、缩短录音，"
            "或为本机设置更短 `LLM_REQUEST_TIMEOUT_SEC` / 检查网络。*"
        )
    else:
        tip += "\n\n> *已在 2s 目标内完成（实际随网络波动）。*"
    return tip


def _clear_memory_only():
    """记忆清除：重置 LLM 侧对话上下文（deque）。"""
    clear_memory()
    return "✅ **记忆已清除**：对话上下文已重置，可重新开始多轮共情对话。"


def _run_vision(image):
    return process_image(image) if image is not None else no_input_vision()


def _run_stt(audio_input):
    return transcribe_file(audio_input) if audio_input is not None else ""


# ================== 核心胶水逻辑 ==================
def main_process(image, audio_input):
    """
    生活场景模拟：感知（VLM∥STT）→ 决策+执行（LLM+TTS），全链路结果展示在 Web UI。
    """
    t0 = time.perf_counter()
    vision_result = no_input_vision()
    voice_text = ""

    with ThreadPoolExecutor(max_workers=2) as pool:
        fv = pool.submit(_run_vision, image)
        fs = pool.submit(_run_stt, audio_input)
        try:
            vision_result = fv.result(timeout=10.0)
        except (FuturesTimeout, Exception) as e:
            print(f"[Log] 视觉线程异常: {e}")
            vision_result = no_input_vision()
        try:
            voice_text = fs.result(timeout=10.0)
        except (FuturesTimeout, Exception) as e:
            print(f"[Log] 语音线程异常: {e}")
            voice_text = ""

    vision_desc = (vision_result.get("description") or "").strip()
    print(f"[Log] 视觉描述: {vision_desc[:80]}")
    print(f"[Log] 语音内容: {voice_text}")

    result, audio_path = get_response(
        "neutral",
        voice_text,
        enable_tts=True,
        vision_desc=vision_desc,
    )

    elapsed = time.perf_counter() - t0
    vision_panel = _format_vision_panel(vision_result)
    perc_md = _format_pipeline_perception(result)
    dec_md = _format_pipeline_decision(result)
    exec_md = _format_pipeline_execution(result)
    lat_md = _latency_note(elapsed)

    return (
        result,
        audio_path,
        vision_panel,
        perc_md,
        dec_md,
        exec_md,
        lat_md,
    )


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
    """格式化显示回复文字（兼容全链路 JSON）。"""
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

    gr.Markdown(
        "### 🏠 生活场景模拟（感知 → 决策 → 执行）\n"
        "每轮严格走：**感知**（画面内容、状态、风险、情绪线索）→ **决策**（判断与优先级）→ **执行**（家居动作、对用户话术、策略）。"
        "覆盖杂乱房间、昏暗、情绪倾诉及潜在风险描述等日常场景。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            camera_input = gr.Image(
                label="📸 实时面部捕捉",
                sources=["webcam"],
                type="numpy"
            )
            gr.Markdown("<center>点击上方按钮拍照</center>")
            gr.Markdown(
                "#### 视觉感知描述（VLM 所见）\n"
                "*以下为模型对当前画面的文字理解；成功时为云端 VLM，失败时为本地降级描述。*"
            )
            vision_desc_display = gr.Markdown(
                "### 视觉感知描述\n*拍照并点击「触发深度融合感知」后，将在此展示 VLM 观察到的内容。*"
            )
            with gr.Row():
                clear_memory_btn = gr.Button("🧹 记忆清除", variant="secondary", size="sm")
                memory_status = gr.Markdown("")

        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎤 按住说话",
                sources=["microphone"],
                type="filepath"
            )
            run_btn = gr.Button("🚀 触发深度融合感知", variant="primary", size="lg")

    with gr.Row():
        pipeline_perception = gr.Markdown("### 1. 感知（场景 · 状态 · 风险 · 情绪）\n*触发后展示…*")
        pipeline_decision = gr.Markdown("### 2. 决策\n*触发后展示…*")
        pipeline_execution = gr.Markdown("### 3. 执行\n*触发后展示…*")

    latency_display = gr.Markdown("*全链路耗时将在每轮结束后显示。*")

    with gr.Row():
        with gr.Column(scale=1):
            json_output = gr.JSON(label="🧠 多模态融合决策中心（含全链路 JSON）")
        with gr.Column(scale=1):
            speech_output = gr.Audio(label="📢 暖暖的语音回复", autoplay=True)

    robot_text_display = gr.Markdown("### 💬 暖暖说：等待交互...")

    run_btn.click(
        fn=main_process,
        inputs=[camera_input, audio_input],
        outputs=[
            json_output,
            speech_output,
            vision_desc_display,
            pipeline_perception,
            pipeline_decision,
            pipeline_execution,
            latency_display,
        ],
    ).then(
        fn=format_reply_text,
        inputs=[json_output],
        outputs=[robot_text_display],
    )

    clear_memory_btn.click(
        fn=_clear_memory_only,
        inputs=[],
        outputs=[memory_status],
    )

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
    **技术支撑：** Qwen-VL + Google STT + Qwen-Turbo + Edge-TTS · **全链路：** 感知-决策-执行 JSON
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True, theme=gr.themes.Soft())
