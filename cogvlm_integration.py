"""cogvlm_integration.py
CogVLM 视觉语言模型集成：同时处理图像和文字，直接输出情感和动作。
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import logger


class CogVLMIntegration:
    """CogVLM 集成类"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        try:
            # 加载 CogVLM 模型
            self.model = AutoModelForCausalLM.from_pretrained(
                "THUDM/cogvlm-chat-hf",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "THUDM/cogvlm-chat-hf",
                trust_remote_code=True
            )
            self.model = self.model.eval()
            logger.info("CogVLM 模型加载成功")
        except Exception as e:
            logger.error(f"CogVLM 模型加载失败: {e}")
            self.model = None
            self.tokenizer = None

    def process_multimodal(self, image: Image.Image, user_text: str) -> dict:
        """处理多模态输入"""
        if self.model is None or self.tokenizer is None:
            return {
                "emotion": "中性",
                "action": "无动作",
                "reply": "模型加载失败",
                "success": False
            }

        try:
            # 构建对话历史
            history = []

            # 生成响应
            input_ids = self.tokenizer.build_conversation_input_ids(
                self.tokenizer,
                query=user_text,
                history=history,
                images=[image]
            )

            inputs = {
                "input_ids": input_ids,
                "images": [image]
            }

            # 推理
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            # 解码输出
            response = self.tokenizer.decode(outputs[0])
            response = response.split("ASSISTANT:")[-1].strip()

            # 解析情感和动作
            emotion = self._infer_emotion(response)
            action = self._infer_action(response, emotion)

            return {
                "emotion": emotion,
                "action": action,
                "reply": response,
                "success": True
            }

        except Exception as e:
            logger.error(f"CogVLM 处理失败: {e}")
            return {
                "emotion": "中性",
                "action": "无动作",
                "reply": "处理失败",
                "success": False,
                "error": str(e)
            }

    def _infer_emotion(self, response: str) -> str:
        """从响应中推断情感"""
        emotion_keywords = {
            "开心": ["开心", "快乐", "高兴", "兴奋", "愉悦"],
            "难过": ["难过", "伤心", "疲惫", "累", "沮丧"],
            "愤怒": ["愤怒", "生气", "火大", "烦躁"],
            "害怕": ["害怕", "担心", "紧张", "恐惧"],
            "惊讶": ["惊讶", "震惊", "意外"],
            "平静": ["平静", "放松", "舒适"],
            "厌恶": ["厌恶", "讨厌", "反感"]
        }

        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in response:
                    return emotion

        return "中性"

    def _infer_action(self, response: str, emotion: str) -> str:
        """从响应和情感中推断动作"""
        # 基于情感的动作映射
        emotion_action_map = {
            "开心": "播放音乐",
            "难过": "调节灯光",
            "愤怒": "播放音乐",
            "害怕": "调节灯光",
            "惊讶": "无动作",
            "平静": "无动作",
            "厌恶": "无动作"
        }

        # 检查响应中是否有动作相关词汇
        if "音乐" in response:
            return "播放音乐"
        elif "灯光" in response:
            return "调节灯光"

        # 基于情感返回默认动作
        return emotion_action_map.get(emotion, "无动作")


def create_cogvlm_integration() -> CogVLMIntegration:
    """创建 CogVLM 集成实例"""
    return CogVLMIntegration()


def process_multimodal(image, user_text) -> dict:
    """处理多模态输入"""
    cogvlm = create_cogvlm_integration()
    return cogvlm.process_multimodal(image, user_text)


if __name__ == "__main__":
    # 测试 CogVLM
    from PIL import Image
    import cv2
    import numpy as np

    # 从摄像头获取图像
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # 转换为 PIL 图像
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 处理多模态输入
        result = process_multimodal(
            image,
            "你看到了什么？我现在感觉怎么样？"
        )
        
        print("CogVLM 处理结果:")
        print(f"情感: {result['emotion']}")
        print(f"动作: {result['action']}")
        print(f"回复: {result['reply']}")
    else:
        print("无法获取摄像头图像")
