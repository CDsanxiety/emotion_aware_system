🏠 微影听镜:基于多模态异步特征融合的深度情感感知与家居交互平台

国家级仿真比赛作品 | 多模态情感计算 + 大语言模型 + 智能家居协同

📖 项目简介

微影听镜是一个基于多模态情感计算的智能家居伴侣系统。通过摄像头实时分析用户面部表情，结合语音识别获取用户意图，调用大语言模型进行情感决策，最终通过语音合成与用户自然交互，并可联动智能家居设备（音乐、灯光）进行情绪安抚。

✨核心功能

- 😊 面部表情识别：基于 FER 库，支持 7 种情绪（开心、难过、愤怒、惊讶、平静、害怕、厌恶）
- 🎤 语音识别：Google Speech Recognition，支持中文
- 🧠 大模型决策：接入阿里云百炼 Qwen-Turbo，返回结构化 JSON（情绪分析 + 家居动作 + 回复话术）
- 🔊 语音合成：Edge-TTS，支持多音色（默认晓晓-活泼女声）
- 🖥️ Web 交互界面：Gradio 构建，支持摄像头拍照、麦克风录音
- 🔧 Debug 模式：手动输入情绪和文字，绕过硬件直接测试

🛠️ 技术栈

| 模块 | 技术 |
| :--- | :--- |
| 表情识别 | FER (Face Expression Recognition) + OpenCV |
| 语音识别 | SpeechRecognition + Google STT |
| 大语言模型 | 阿里云百炼 Qwen-Turbo / DeepSeek |
| 语音合成 | Edge-TTS |
| Web 框架 | Gradio |
| 日志系统 | Python logging |

📁 项目结构
emotion_aware_system/
├── app.py # Gradio 主界面入口
├── vision.py # 面部表情识别模块
├── audio.py # 语音识别模块
├── llm_api.py # 大模型 API 调用
├── tts.py # 语音合成模块
├── utils.py # 日志工具
├── debug_mode.py # Debug 模式预设
├── prompt.txt # 大模型人设与输出格式约束
├── requirements.txt # 依赖清单
└── logs/ # 日志文件目录（自动生成）

🚀 快速启动
1. 克隆仓库
```bash
git clone https://github.com/CDsanxiety/emotion_aware_system.git
cd emotion_aware_system
```
2. 创建虚拟环境（推荐）
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```
3. 安装依赖
```bash
pip install -r requirements.txt
```
> ⚠️ 如果 PyAudio 安装失败，Windows 用户可使用 `pip install pipwin && pipwin install pyaudio`

4. 配置 API 密钥
在项目根目录创建 `.env` 文件，写入：
```text
LLM_API_KEY=你的阿里云百炼API密钥
```
> 如需切换模型，修改 `llm_api.py` 中的 `MODEL` 和 `BASE_URL`

5. 运行项目
```bash
python app.py
```
6. 访问 Web 界面
启动成功后，浏览器打开：`http://127.0.0.1:7860`
