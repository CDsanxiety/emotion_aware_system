# llm_api.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)

load_dotenv()

# ================== 配置区 ==================
# 改成你申请到的模型（推荐 DeepSeek 或 智谱）
MODEL = "qwen-turbo"  # 或者 "glm-4-flash"（智谱） / "kimi"（Moonshot）
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 改成对应平台的 base_url
API_KEY = os.getenv("LLM_API_KEY")  # 放到 .env 文件里

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============================================

def call_llm(emotion: str, user_text: str, prompt_file: str = "prompt.txt") -> str:
    """
    核心函数：接收表情 + 用户语音文本 → 返回机器人回复
    """
    # 读取 Product 同学写的灵魂 Prompt
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # 拼接上下文
    full_prompt = f"""
    当前用户表情：{emotion}
    用户说的话：{user_text}

    请用温柔、体贴的语气回复（最多 80 字），直接输出回复内容，不要加任何解释。
    """

    # 方案三：显式类型定义，消除警告
    system_msg: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": system_prompt
    }
    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": full_prompt
    }

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[system_msg, user_msg],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return "嗯嗯，我在这里陪着你呢～"  # 兜底回复


# 测试用
if __name__ == "__main__":
    print(call_llm("happy", "今天天气真好"))