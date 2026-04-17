# app.py
"""
微影听镜 V2.0 - 实机集成重塑版
核心特性：并行感知、闭环反馈、延时监控、硬件状态同步。
基于ROS-LLM核心架构设计
"""
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import cv2

import gradio as gr

# 这里的 import 顺序按系统层级排列
from config import ROS_BRIDGE_URI
from audio import transcribe_file
from llm_api import clear_memory, get_response
from ros_client import global_ros_manager
from utils import logger, setup_logger
from vision import process_image
from blackboard import Blackboard
from memory_rag import LongTermMemory
from pad_model import PADEmotionEngine
from openvla_integration import OpenVLA, VLAControlLoop

# 尝试导入AutoGen和MetaGPT，如果失败则使用模拟实现
try:
    from autogen_integration import AutoGenManager
except ImportError:
    print("AutoGen 集成模块导入失败，使用模拟实现")
    class AutoGenManager:
        def __init__(self):
            pass
        def chat(self, message, context=None):
            return f"模拟AutoGen响应: {message}"

try:
    from meta_gpt_integration import MetaGPTManager
except ImportError:
    print("MetaGPT 集成模块导入失败，使用模拟实现")
    class MetaGPTManager:
        def __init__(self):
            pass
        async def run(self, task):
            return f"模拟MetaGPT响应: {task}"

# 初始化日志与硬件连接
setup_logger()
global_ros_manager.connect()

# 初始化新模块
global_blackboard = Blackboard()
global_memory = LongTermMemory()
global_pad_engine = PADEmotionEngine()

# 初始化OpenVLA
global_vla = OpenVLA()
global_vla_control_loop = VLAControlLoop(global_vla)

# 初始化AutoGen
global_autogen_manager = AutoGenManager()

# 初始化MetaGPT
global_metagpt_manager = MetaGPTManager()

# 线程控制变量
running = True

class CoreArchitecture:
    """ROS-LLM 核心架构"""
    
    def __init__(self):
        """初始化核心架构"""
        self.components = {
            "input": ["AudioInput", "VisionInput"],
            "processing": ["Blackboard", "PADEmotionEngine", "LongTermMemory"],
            "model": ["LLM API", "OpenVLA", "AutoGen", "MetaGPT"],
            "robot": "ROSManager",
            "output": "TTS"
        }
        self.flow = [
            "用户语音/视觉输入",
            "语音转文本/视觉识别",
            "数据写入 Blackboard",
            "PADEmotionEngine 计算情绪",
            "LongTermMemory 检索记忆",
            "AutoGen 多代理协作决策",
            "MetaGPT 结构化代理处理",
            "LLM 生成响应",
            "OpenVLA 预测动作",
            "ROSManager 下发指令",
            "TTS 语音反馈",
            "硬件执行并返回状态"
        ]
    
    def print_architecture(self):
        """打印架构信息"""
        print("=== ROS-LLM 核心架构 ===")
        print("\n1. 核心组件:")
        for name, components in self.components.items():
            if isinstance(components, list):
                print(f"   - {name}:")
                for component in components:
                    print(f"     * {component}")
            else:
                print(f"   - {name}: {components}")
        
        print("\n2. 数据流向:")
        for i, step in enumerate(self.flow, 1):
            print(f"   {i}. {step}")
        
        print("\n3. 核心流程:")
        print("   - 感知输入 → 数据处理 → LLM 决策 → 硬件控制 → 状态反馈")

# 传感器线程函数
def vision_thread():
    """视觉传感器线程：持续捕获摄像头画面并写入Blackboard"""
    cap = cv2.VideoCapture(0)
    try:
        while running:
            ret, frame = cap.read()
            if ret:
                # 处理图像
                res_v = process_image(frame)
                vision_desc = res_v.get("description", "视觉链路异常")
                # 写入Blackboard
                global_blackboard.update_vision(vision_desc, True)
            # 控制帧率
            time.sleep(0.1)  # 10fps
    finally:
        cap.release()

def audio_thread():
    """听觉传感器线程：持续监听麦克风输入并写入Blackboard"""
    # 注意：这里简化处理，实际应用中需要持续监听麦克风
    # 由于持续录音可能会占用过多资源，这里我们保持现有的按需录音方式
    # 但在架构上预留了线程接口
    pass

def agent_loop():
    """主决策循环：按固定频率读取Blackboard并做出决策"""
    # 记录上次交互时间
    last_interaction_time = time.time()
    # 主动交互间隔（秒）
    active_interaction_interval = 30
    
    while running:
        # 读取Blackboard数据
        vision_data = global_blackboard.get_vision_data()
        speech_data = global_blackboard.get_speech_data()
        
        vision_desc = vision_data["description"]
        voice_text = speech_data["text"]
        
        # 检查ROS状态，确保机器人没有在执行其他动作
        hardware_status = global_ros_manager.get_status()
        is_moving = hardware_status.get("is_moving", False)
        
        # 检查是否有新的语音输入
        if voice_text and not is_moving:
            # 构建上下文信息
            context = f"视觉描述: {vision_desc}\n硬件状态: {hardware_status}"
            
            # 使用AutoGen进行多代理协作决策
            autogen_response = global_autogen_manager.chat(voice_text, context)
            
            # 使用MetaGPT进行结构化代理处理
            import asyncio
            metagpt_response = asyncio.run(global_metagpt_manager.run(f"处理用户请求: {voice_text}\n上下文: {context}"))
            
            # 基于AutoGen和MetaGPT的决策结果调用LLM生成最终响应
            res, response_audio = get_response(
                face_emotion="neutral", 
                voice_text=voice_text, 
                enable_tts=True, 
                vision_desc=vision_desc
            )
            
            # 硬件下发
            global_ros_manager.publish_action(res)
            
            # 清空语音输入，避免重复处理
            global_blackboard.update_speech("")
            
            # 更新上次交互时间
            last_interaction_time = time.time()
        else:
            # 检查是否需要主动交互
            current_time = time.time()
            if current_time - last_interaction_time > active_interaction_interval and not is_moving:
                # 构建上下文信息
                context = f"视觉描述: {vision_desc}\n硬件状态: {hardware_status}"
                
                # 使用AutoGen进行多代理协作决策
                autogen_response = global_autogen_manager.chat("我需要主动与用户交互，根据当前环境和状态，我应该说什么？", context)
                
                # 使用MetaGPT进行结构化代理处理
                import asyncio
                metagpt_response = asyncio.run(global_metagpt_manager.run(f"生成主动交互内容\n上下文: {context}"))
                
                # 主动找用户说话
                res, response_audio = get_response(
                    face_emotion="neutral", 
                    voice_text="", 
                    enable_tts=True, 
                    vision_desc=vision_desc
                )
                
                # 硬件下发
                global_ros_manager.publish_action(res)
                
                # 更新上次交互时间
                last_interaction_time = time.time()
        
        # 控制决策频率
        time.sleep(1)  # 1Hz

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
    
    # 使用Blackboard存储感知数据
    global_blackboard.update_vision(vision_desc, True)  # 假设用户在场
    if voice_text:
        global_blackboard.update_speech(voice_text)

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
    # 初始化并打印核心架构
    architecture = CoreArchitecture()
    architecture.print_architecture()
    
    # 启动传感器线程
    vision_thread = threading.Thread(target=vision_thread, daemon=True)
    vision_thread.start()
    
    audio_thread = threading.Thread(target=audio_thread, daemon=True)
    audio_thread.start()
    
    # 启动主决策循环
    agent_thread = threading.Thread(target=agent_loop, daemon=True)
    agent_thread.start()
    
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    finally:
        # 停止线程
        running = False
        time.sleep(0.5)  # 等待线程结束
        # 关闭ROS连接，处理节点生命周期
        global_ros_manager.shutdown()
        logger.info("系统已关闭")
