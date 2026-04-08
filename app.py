import gradio as gr
import json
import asyncio
import edge_tts
import os
import tempfile
import vision
import audio
import llm_api

# ================== 辅助函数：语音合成 ==================
async def text_to_speech(text):
    """使用 edge-tts 将文字转为语音文件"""
    if not text:
        return None
    
    # 挑一个温柔的中国女生声音：晓晓 (Xiaoxiao)
    VOICE = "zh-CN-XiaoxiaoNeural"
    communicate = edge_tts.Communicate(text, VOICE)
    
    # 创建临时文件保存音频
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    await communicate.save(temp_file.name)
    return temp_file.name

# ================== 核心胶水逻辑 ==================
def main_process(image, audio_input):
    """
    当录音触发时调用的主逻辑
    1. 分析面部表情
    2. 将语音转为文字
    3. 调用 LLM 获取分析结果（JSON 格式）
    4. 生成语音文件
    """
    # Step 1: 视觉分析
    detected_emotion = vision.process_image(image)
    print(f"[Log] 检测到表情: {detected_emotion}")

    # Step 2: 语音转文字
    voice_text = ""
    if audio_input:
        print(f"[Log] 正在处理语音文件: {audio_input}")
        voice_text = audio.transcribe_file(audio_input)
    
    print(f"[Log] 语音内容: {voice_text}")

    # Step 3: 智脑决策 (调用 LLM)
    llm_output_raw = llm_api.call_llm(detected_emotion, voice_text)
    
    # 尝试解析 JSON
    try:
        # 去掉 Markdown 的 ```json 标记（如果大模型不听话加了的话）
        clean_json = llm_output_raw.replace("```json", "").replace("```", "").strip()
        decision_data = json.loads(clean_json)
    except Exception as e:
        print(f"[Error] JSON 解析失败: {e}")
        decision_data = {
            "analyzed_emotion": "未知",
            "appliance_actions": ["由于解析错误，暂无指令"],
            "robot_speech": llm_output_raw # 即使解析失败也保留文字
        }

    # Step 4: 生成声音
    robot_speech_text = decision_data.get("robot_speech", "我没听清楚呢")
    audio_path = asyncio.run(text_to_speech(robot_speech_text))

    return decision_data, audio_path


# ================== UI 布局设计 ==================
with gr.Blocks(theme=gr.themes.Soft(), title="暖暖情感机器人仿真系统") as demo:
    gr.Markdown("""
    # 🤖 暖暖：情感感知智能家居伴侣
    项目定位：国家级仿真比赛作品 | 硬件协同：多模态情感计算
    """)

    with gr.Row():
        # 左侧：视觉抓取
        with gr.Column(scale=1):
            camera_input = gr.Image(label="实时面部捕捉", sources=["webcam"], mirror_webcam=True)
            gr.Markdown("<center>📸 系统会自动抓取快照进行情绪分析</center>")
        
        # 右侧：语音交互
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="按住说话 (Microphone)", sources=["microphone"], type="filepath")
            run_btn = gr.Button("🚀 触发深度融合感知", variant="primary")
            
    # 中间：输出展示
    with gr.Row():
        with gr.Column(scale=1):
            json_output = gr.JSON(label="🧠 多模态融合决策中心 (JSON Panel)")
        with gr.Column(scale=1):
            speech_output = gr.Audio(label="📢 暖暖的语音回复", autoplay=True)
            
    # 机器人语录（大号显示）
    robot_text_display = gr.Markdown("### 💬 机器人回复： 等待交互...")

    # 事件绑定
    def update_text(decision):
        return f"### 💬 机器人回复：\n> {decision.get('robot_speech', '')}"

    run_btn.click(
        fn=main_process,
        inputs=[camera_input, audio_input],
        outputs=[json_output, speech_output]
    ).then(
        fn=update_text,
        inputs=[json_output],
        outputs=[robot_text_display]
    )

    gr.Markdown("""
    ---
    **技术支撑：** FER (Facial Expression Recognition) + Whisper / Google STT + LLM (DeepSeek/Qwen) + Edge-TTS
    """)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)
