🏠 微影听镜: 基于云边协同多模态大模型的情感感知与具身伴侣系统

国家级仿真比赛作品 | 端到端多模态感知 | 视觉语言大模型 (VLM) | 边缘智能控制

📖 项目简介

微影听镜是一个专为资源受限边缘设备（如树莓派）打造的高性能智能陪伴系统。系统摒弃了传统的“级联小模型”架构，创新性地采用“Thin Edge, Cloud Brain”（瘦边缘，胖云端）架构。通过时分复用的单总线控制策略抓取环境视觉与音频，直接交由云端多模态大模型（VLM）进行跨模态情绪推断与安全研判。系统不仅能提供具备情感温度的语音陪伴，更能根据最高优先级的“安全守护规则”（如跌倒检测）或实时的情绪波动，触发底层的声光硬件与预留的 ROS 具身节点进行物理级响应。

✨ 核心技术与创新点

- 👁️ 端到端多模态感知 (VLM)：淘汰了传统的 FER 情绪分类库。系统直接捕获原始视觉帧，结合高低频降噪后的语音内容，送入阿里云通义千问多模态大模型（Qwen-VL）。由大模型直接进行从“面部微表情+肢体姿态+语音语义”到“高维情绪与安全状态”的推演。
- 🛡️ 硬件级安全守护 (Fall-Alert)：内置基于 Prompt Engineering 的严格状态机。在情感陪聊之前，优先对画面进行危险状态（如老人跌倒、异常倾斜）的零延迟判定，并触发最高优先级的红色警告灯光与紧急语音播报。
- 🎙️ 硬件级降噪与流式识别：针对边缘设备算力瓶颈，手搓 FFmpeg 底层管道进行音频的高通/低通滤波降噪，并无缝对接阿里云 Paraformer 接口，实现极低资源占用的高精度 STT。
- 💡 轻量化具身控制：抛弃了沉重的本地动作模型（如 OpenVLA）。当前在树莓派上使用底层的 `neopixel` 和 GPIO 实现零延迟的声光阵列情绪反馈。系统逻辑层已抽象 `PhysicalInterface`，为未来接入复杂的 ROS 机械臂生态预留了完整的扩展接口。
- 🔊 拟人情感合成：集成 Edge-TTS，通过定制的治愈系音色，将大模型生成的安抚话术转化为带有情感色彩的自然语音反馈。

🛠️ 技术架构栈

| 模块 | 核心技术 |
| :--- | :--- |
| **边缘控制 (Edge)** | RPi.GPIO, NeoPixel, FFmpeg (ALSA 采集与滤波), mpg123 |
| **视觉与视频流** | OpenCV (无头模式), 错峰抓拍时分复用算法 |
| **语音转写 (STT)** | 阿里云 Dashscope Paraformer (PCM 16bit 小端序流式上传) |
| **云端大脑 (Brain)** | 阿里云百炼 Qwen-VL (基于 OpenAI API 标准的多模态推理) |
| **语音合成 (TTS)** | Edge-TTS (治愈女声) |
| **具身联动接口** | 抽象 Hardware Interface，兼容未来 ROS Topic 发布 |

📁 项目结构
emotion_aware_system/
├── src/
│   ├── cloud/
│   │   ├── brain.py       # 多模态大模型核心推理中枢 (Qwen-VL)
│   │   ├── stt.py         # 基于 FFmpeg 降噪的语音识别
│   │   └── tts.py         # 边缘语音合成
│   ├── core/
│   │   ├── config.py      # 全局环境变量与引脚配置
│   │   └── orchestrator.py# 主循环：视觉/听觉时分复用与安全调度
│   └── hardware/
│       └── physical_interface.py # 底层声光联动与 GPIO 控制
├── main.py                # 边缘设备直连启动入口
├── requirements.txt       # 依赖清单
└── logs/                  # 系统运行日志目录

🚀 快速启动

1. 环境准备 (推荐树莓派 3B/4B 或同等 Linux 边缘设备)
```bash
git clone https://github.com/CDsanxiety/emotion_aware_system.git
cd emotion_aware_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 依赖工具安装
请确保系统已安装底层的音视频处理与播放工具：
```bash
sudo apt-get update
sudo apt-get install ffmpeg mpg123
```

3. 配置 API 密钥
在项目根目录创建 `.env` 文件，写入：
```text
QWEN_API_KEY=你的阿里云百炼API密钥
```

4. 启动情感系统
```bash
sudo .venv/bin/python main.py
```
> 注：由于需要调用 GPIO 控制硬件灯带，部分 Linux 系统下需要 `sudo` 权限运行。

## 📝 系统特性说明

### 硬件资源保护机制
为解决树莓派等边缘设备 USB 总线带宽和供电不足的问题，系统在 `orchestrator.py` 中实现了**串行错峰执行策略**。摄像头（Vision）抓拍完成后会立即释放硬件句柄，随后才拉起音频（Audio）采集管道。这种设计彻底避免了摄像头与麦克风同时工作导致的内核级 Deadlock，保障了系统 7x24 小时运行的极高稳定性。
