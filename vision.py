# vision.py
import cv2
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 初始化 OpenAI 客户端（兼容 DashScope）
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def encode_image(frame):
    """将 numpy array 图像编码为 base64"""
    if frame is None:
        return None
    # 调整大小以减少网络传输量 (最大 512)
    height, width = frame.shape[:2]
    if max(height, width) > 512:
        scale = 512 / max(height, width)
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode("utf-8")

def analyze_scene(frame, voice_text: str = "") -> str:
    """
    使用 Qwen-VL-Max 分析场景内容
    返回对场景、情绪、用户状态的自然语言描述
    """
    if frame is None:
        return "未捕捉到图像内容。"

    try:
        base64_image = encode_image(frame)
        if base64_image is None:
            return "图像编码失败。"
        
        prompt = (
            f"用户说：'{voice_text}'。请结合图中用户的面部神态、肢体语言、环境背景以及手中所持物品，"
            "综合判断用户的真实情绪与处境。特别注意语言与神态之间是否存在矛盾点（如强颜欢笑）。"
            "请给出一个详尽的情感语义描述。"
        )

        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        print(f"VLM 分析出错: {e}")
        return "无法看清当前画面，提示用户检查网络或摄像头。"

# 为了保持向下兼容，保留 process_image 接口，但内部重定向
def process_image(frame):
    """向下兼容接口：旧版系统可能调用此函数获取表情词"""
    # 在 2.0 中，如果强制需要一个词，我们可以让 VLM 简答，但此处保持通用性
    return analyze_scene(frame, "你好")