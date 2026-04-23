"""vla_integration.py
OpenVLA 风格的动作预测集成：将视觉和语音输入转换为机器人动作。
基于 OpenVLA 的思想，实现感知-预测-执行的控制循环。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from vision import process_image
from audio import recognize_speech
from llm_api import get_response
from robot_functions import get_robot_functions
from blackboard import Blackboard


@dataclass
class VLAObservation:
    """VLA 观测数据"""
    image: Optional[np.ndarray] = None
    vision_desc: str = ""
    audio_text: str = ""
    timestamp: float = 0.0


@dataclass
class VLAPrediction:
    """VLA 预测结果"""
    action: str
    emotion: str
    reply: str
    confidence: float = 0.0


class VLAIntegration:
    """OpenVLA 风格的集成类"""

    def __init__(self, blackboard: Optional[Blackboard] = None):
        self.blackboard = blackboard or Blackboard()
        self.robot_functions = get_robot_functions()
        self.last_observation_time = 0.0
        self.observation_interval = 0.1  # 10Hz

    def get_observation(self, camera_index: int = 0) -> VLAObservation:
        """获取当前观测"""
        now = time.time()
        image = None
        vision_desc = ""

        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    image = frame
                    vision_result = process_image(frame)
                    vision_desc = vision_result.get("description", "").strip()
            cap.release()
        except Exception:
            pass

        audio_text = ""
        try:
            audio_text = recognize_speech(timeout=1, phrase_time_limit=3)
        except Exception:
            pass

        return VLAObservation(
            image=image,
            vision_desc=vision_desc,
            audio_text=audio_text,
            timestamp=now
        )

    def predict_action(self, observation: VLAObservation, instruction: str = "") -> VLAPrediction:
        """预测动作（单步预测）"""
        # 构建提示词
        prompt_builder = self._build_prompt(observation, instruction)
        
        # 调用 LLM 进行预测
        res, audio_path = get_response(
            "neutral",
            prompt_builder,
            enable_tts=False,
            vision_desc=observation.vision_desc
        )

        # 解析预测结果
        action = res.get("action", "无动作")
        emotion = res.get("emotion", "neutral")
        reply = res.get("reply", "")

        return VLAPrediction(
            action=action,
            emotion=emotion,
            reply=reply,
            confidence=0.9  # 假设高置信度
        )

    def _build_prompt(self, observation: VLAObservation, instruction: str) -> str:
        """构建 VLA 提示词"""
        vision_block = f"视觉观测：{observation.vision_desc}" if observation.vision_desc else "无视觉观测"
        audio_block = f"语音输入：{observation.audio_text}" if observation.audio_text else "无语音输入"
        instruction_block = f"任务指令：{instruction}" if instruction else "无特定任务"

        prompt = f"""作为智能家居伴侣机器人，你需要根据当前观测预测并执行适当的动作。

观测信息：
{vision_block}
{audio_block}
{instruction_block}

请分析当前场景，预测最合适的动作，并生成相应的回复。
"""

        return prompt

    def execute_action(self, prediction: VLAPrediction) -> Dict[str, Any]:
        """执行动作"""
        result = self.robot_functions.execute_action(prediction.action)
        return {
            "success": result.success,
            "message": result.message,
            "action_type": result.action_type.value,
            "emotion": prediction.emotion,
            "reply": prediction.reply
        }

    def control_loop(self, camera_index: int = 0, instruction: str = "", task_complete: callable = None) -> None:
        """控制循环"""
        control_frequency = 10.0  # 10Hz
        control_interval = 1.0 / control_frequency

        while True:
            start_time = time.time()

            # 1. 获取观测
            obs = self.get_observation(camera_index)

            # 2. 预测动作
            prediction = self.predict_action(obs, instruction)

            # 3. 执行动作
            execution_result = self.execute_action(prediction)

            # 4. 更新黑板
            self._update_blackboard(obs, prediction, execution_result)

            # 5. 检查任务是否完成
            if task_complete and task_complete():
                break

            # 6. 控制频率
            elapsed = time.time() - start_time
            if elapsed < control_interval:
                time.sleep(control_interval - elapsed)

    def _update_blackboard(self, obs: VLAObservation, pred: VLAPrediction, exec_result: Dict[str, Any]):
        """更新黑板状态"""
        if self.blackboard:
            updates = {
                "current_vision_desc": obs.vision_desc,
                "last_speech_text": obs.audio_text,
                "last_speech_time": obs.timestamp,
                "user_presence": bool(obs.vision_desc and "未检测到" not in obs.vision_desc),
                "robot_status": "acting" if exec_result.get("success") else "idle"
            }
            for key, value in updates.items():
                setattr(self.blackboard, key, value)


def create_vla_integration(blackboard: Optional[Blackboard] = None) -> VLAIntegration:
    """创建 VLA 集成实例"""
    return VLAIntegration(blackboard)


def run_vla_control_loop(camera_index: int = 0, instruction: str = "") -> None:
    """运行 VLA 控制循环"""
    vla = create_vla_integration()
    vla.control_loop(camera_index, instruction)


if __name__ == "__main__":
    print("启动 VLA 控制循环...")
    try:
        run_vla_control_loop(instruction="照顾用户情绪，提供适当的家居服务")
    except KeyboardInterrupt:
        print("控制循环已停止")
