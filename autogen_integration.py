# autogen_integration.py
"""
AutoGen 集成模块
实现多代理系统，增强机器人决策能力
"""
import os
from config import API_KEY

# 尝试导入AutoGen，如果失败则使用模拟实现
try:
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTO_GEN_AVAILABLE = True
except ImportError:
    AUTO_GEN_AVAILABLE = False

class AutoGenManager:
    """
    AutoGen 管理器
    用于创建和管理多代理系统
    """
    def __init__(self):
        if AUTO_GEN_AVAILABLE:
            self.config_list = [
                {
                    "model": "gpt-4",
                    "api_key": API_KEY,
                }
            ]
            self.llm_config = {
                "config_list": self.config_list,
                "seed": 42,
            }
            self.agents = {}
            self.group_chat = None
            self.group_chat_manager = None
        else:
            print("AutoGen 未安装，使用模拟实现")
    
    def initialize_agents(self):
        """
        初始化代理系统
        """
        if not AUTO_GEN_AVAILABLE:
            return
        
        # 创建用户代理
        self.agents["user"] = UserProxyAgent(
            name="User",
            system_message="用户代理，负责与用户交互并执行最终决策",
            code_execution_config=False,
        )
        
        # 创建助手代理
        self.agents["assistant"] = AssistantAgent(
            name="Assistant",
            system_message="助手代理，负责提供专业建议和解决方案",
            llm_config=self.llm_config,
        )
        
        # 创建情感分析代理
        self.agents["emotion_analyzer"] = AssistantAgent(
            name="EmotionAnalyzer",
            system_message="情感分析代理，负责分析用户情绪并提供情感响应建议",
            llm_config=self.llm_config,
        )
        
        # 创建机器人控制代理
        self.agents["robot_controller"] = AssistantAgent(
            name="RobotController",
            system_message="机器人控制代理，负责将决策转换为具体的机器人动作",
            llm_config=self.llm_config,
        )
        
        # 创建群组聊天
        self.group_chat = GroupChat(
            agents=list(self.agents.values()),
            messages=[],
            max_round=5,
        )
        
        # 创建群组聊天管理器
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )
    
    def chat(self, message, context=None):
        """
        与代理系统聊天
        """
        if not AUTO_GEN_AVAILABLE:
            # 模拟AutoGen响应
            return f"模拟AutoGen响应: {message}"
        
        if not self.group_chat_manager:
            self.initialize_agents()
        
        # 构建完整的消息，包括上下文
        full_message = message
        if context:
            full_message = f"上下文: {context}\n\n消息: {message}"
        
        # 开始聊天
        response = self.agents["user"].initiate_chat(
            self.group_chat_manager,
            message=full_message,
        )
        
        return response
    
    def get_agents(self):
        """
        获取所有代理
        """
        if not AUTO_GEN_AVAILABLE:
            return {}
        return self.agents

# 全局AutoGen管理器
global_autogen_manager = AutoGenManager()
