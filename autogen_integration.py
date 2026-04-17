"""autogen_integration.py
AutoGen 多智能体协作集成：使用多个 AI 智能体协作完成情感分析和动作规划。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat import GroupChat, GroupChatManager

from utils import logger


class AutoGenIntegration:
    """AutoGen 集成类"""

    def __init__(self):
        self.emotion_agent = None
        self.action_agent = None
        self.group_chat = None
        self.manager = None
        self._initialize_agents()

    def _initialize_agents(self):
        """初始化智能体"""
        # 情感分析智能体
        self.emotion_agent = AssistantAgent(
            name="情感分析专家",
            system_message="你是一个专业的情感分析专家，擅长从用户的语言和视觉描述中分析情绪状态。\n"+
                           "请根据用户的输入和视觉描述，分析用户的情绪状态，并给出情绪标签。\n"+
                           "情绪标签包括：开心、难过、愤怒、害怕、惊讶、平静、厌恶。\n"+
                           "只需要输出情绪标签，不需要其他说明。",
            llm_config={
                "model": "gpt-4o",
                "temperature": 0.3,
            }
        )

        # 动作规划智能体
        self.action_agent = AssistantAgent(
            name="动作规划专家",
            system_message="你是一个智能家居动作规划专家，擅长根据用户的情绪状态和场景规划合适的家居动作。\n"+
                           "可用的动作包括：播放音乐、调节灯光、无动作。\n"+
                           "请根据情绪状态和场景，选择最合适的动作，并给出简短的理由。\n"+
                           "输出格式：动作：[动作名称]\n理由：[简短理由]",
            llm_config={
                "model": "gpt-4o",
                "temperature": 0.5,
            }
        )

        # 群组聊天
        self.group_chat = GroupChat(
            agents=[self.emotion_agent, self.action_agent],
            messages=[],
            max_round=5
        )

        # 群组管理器
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "model": "gpt-4o",
                "temperature": 0.3,
            }
        )

    def analyze_emotion_and_plan_action(self, user_text: str, vision_desc: str) -> Dict[str, Any]:
        """分析情绪并规划动作"""
        try:
            # 构建输入消息
            input_message = f"用户输入：{user_text}\n视觉描述：{vision_desc}"

            # 启动群组聊天
            result = self.emotion_agent.initiate_chat(
                self.manager,
                message=input_message
            )

            # 解析结果
            emotion = "中性"
            action = "无动作"

            # 从聊天历史中提取信息
            for msg in result.chat_history:
                if msg.get("name") == "情感分析专家":
                    content = msg.get("content", "").strip()
                    if content:
                        emotion = content
                elif msg.get("name") == "动作规划专家":
                    content = msg.get("content", "").strip()
                    lines = content.split("\n")
                    for line in lines:
                        if line.startswith("动作："):
                            action = line.replace("动作：", "").strip()

            return {
                "emotion": emotion,
                "action": action,
                "success": True
            }

        except Exception as e:
            logger.error(f"AutoGen 分析失败: {e}")
            return {
                "emotion": "中性",
                "action": "无动作",
                "success": False,
                "error": str(e)
            }

    def plan_task(self, task: str) -> List[Dict[str, Any]]:
        """规划任务"""
        try:
            # 创建任务规划智能体
            planner_agent = AssistantAgent(
                name="任务规划师",
                system_message="你是一个智能家居任务规划师，擅长将用户的请求分解为具体的步骤。\n"+
                               "请将用户的请求分解为清晰的步骤，每个步骤包括：\n"+
                               "1. 步骤描述\n"+
                               "2. 需要执行的动作\n"+
                               "3. 预期结果\n"+
                               "输出格式：\n"+
                               "步骤 1: [描述]\n动作: [动作]\n预期: [结果]\n\n"+
                               "步骤 2: [描述]\n动作: [动作]\n预期: [结果]\n",
                llm_config={
                    "model": "gpt-4o",
                    "temperature": 0.7,
                }
            )

            # 启动聊天
            result = planner_agent.initiate_chat(
                UserProxyAgent(
                    name="用户",
                    system_message="你是用户，只需要提出任务，不需要其他回应。",
                    llm_config=False
                ),
                message=task
            )

            # 解析规划结果
            plan = []
            content = result.chat_history[-1].get("content", "")
            lines = content.split("\n")

            current_step = {}
            for line in lines:
                line = line.strip()
                if line.startswith("步骤"):
                    if current_step:
                        plan.append(current_step)
                        current_step = {}
                    current_step["description"] = line
                elif line.startswith("动作:"):
                    current_step["action"] = line.replace("动作:", "").strip()
                elif line.startswith("预期:"):
                    current_step["expected"] = line.replace("预期:", "").strip()

            if current_step:
                plan.append(current_step)

            return plan

        except Exception as e:
            logger.error(f"任务规划失败: {e}")
            return [{
                "description": "默认步骤",
                "action": "无动作",
                "expected": "完成任务"
            }]


def create_autogen_integration() -> AutoGenIntegration:
    """创建 AutoGen 集成实例"""
    return AutoGenIntegration()


def analyze_emotion_and_plan_action(user_text: str, vision_desc: str) -> Dict[str, Any]:
    """分析情绪并规划动作"""
    autogen = create_autogen_integration()
    return autogen.analyze_emotion_and_plan_action(user_text, vision_desc)


def plan_task(task: str) -> List[Dict[str, Any]]:
    """规划任务"""
    autogen = create_autogen_integration()
    return autogen.plan_task(task)


if __name__ == "__main__":
    # 测试情感分析和动作规划
    result = analyze_emotion_and_plan_action(
        "今天工作好累啊",
        "人物靠在沙发上，神情疲惫"
    )
    print("情感分析结果:", result)

    # 测试任务规划
    plan = plan_task("我饿了，想吃点东西")
    print("任务规划结果:")
    for step in plan:
        print(f"- {step['description']}")
        print(f"  动作: {step['action']}")
        print(f"  预期: {step['expected']}")
        print()
