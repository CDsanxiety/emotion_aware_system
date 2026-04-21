# identity_registration_agent.py
"""
身份注册代理
引导用户完成初次识别数据录入的自动化流程
"""
import os
import time
import uuid
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from utils import logger
from identity_manager import get_identity_manager, UserType, RecognitionMethod
from dialogue_manager import get_dialogue_manager
from vision import get_vision_manager
from audio import get_audio_manager


class RegistrationState(Enum):
    """注册状态枚举"""
    INIT = "init"  # 初始化状态
    ASK_NAME = "ask_name"  # 询问用户姓名
    COLLECT_FACE = "collect_face"  # 采集人脸
    COLLECT_VOICE = "collect_voice"  # 采集声纹
    CONFIRM = "confirm"  # 确认信息
    COMPLETE = "complete"  # 完成注册
    FAILED = "failed"  # 注册失败


@dataclass
class RegistrationData:
    """注册数据"""
    user_id: str
    name: str = ""
    face_images: List[np.ndarray] = None
    voice_audio: Optional[np.ndarray] = None
    sample_rate: int = 22050
    current_state: RegistrationState = RegistrationState.INIT
    error_message: str = ""
    
    def __post_init__(self):
        if self.face_images is None:
            self.face_images = []


class IdentityRegistrationAgent:
    """
    身份注册代理
    引导用户完成初次识别数据录入
    """
    def __init__(self):
        self.identity_manager = get_identity_manager()
        self.dialogue_manager = get_dialogue_manager()
        self.vision_manager = get_vision_manager()
        self.audio_manager = get_audio_manager()
        self.registration_data = None
        self._lock = None
    
    def start_registration(self) -> str:
        """
        开始注册流程
        
        返回:
            str: 初始提示信息
        """
        # 生成唯一用户ID
        user_id = str(uuid.uuid4())[:8]
        self.registration_data = RegistrationData(user_id=user_id)
        self.registration_data.current_state = RegistrationState.ASK_NAME
        
        # 向用户发出初始提示
        prompt = "您好！我需要为您创建一个个人档案，以便更好地识别和服务您。请问您的姓名是什么？"
        logger.info("[身份注册] 开始注册流程")
        return prompt
    
    def process_input(self, user_input: str) -> Tuple[str, bool]:
        """
        处理用户输入，推进注册流程
        
        参数:
            user_input: 用户输入
        
        返回:
            Tuple[str, bool]: (响应信息, 是否完成)
        """
        if not self.registration_data:
            return "请先开始注册流程。", False
        
        current_state = self.registration_data.current_state
        
        if current_state == RegistrationState.ASK_NAME:
            return self._handle_ask_name(user_input)
        elif current_state == RegistrationState.COLLECT_FACE:
            return self._handle_collect_face(user_input)
        elif current_state == RegistrationState.COLLECT_VOICE:
            return self._handle_collect_voice(user_input)
        elif current_state == RegistrationState.CONFIRM:
            return self._handle_confirm(user_input)
        elif current_state == RegistrationState.COMPLETE:
            return "注册已完成，感谢您的配合！", True
        elif current_state == RegistrationState.FAILED:
            return f"注册失败: {self.registration_data.error_message}", True
        else:
            return "注册流程出现错误，请重新开始。", True
    
    def _handle_ask_name(self, user_input: str) -> Tuple[str, bool]:
        """
        处理用户姓名输入
        """
        if not user_input or len(user_input.strip()) == 0:
            return "请告诉我您的姓名，以便我为您创建个人档案。", False
        
        self.registration_data.name = user_input.strip()
        self.registration_data.current_state = RegistrationState.COLLECT_FACE
        
        prompt = f"您好，{self.registration_data.name}！接下来我需要采集您的面部信息，请确保您在光线充足的地方，然后按照我的提示转动头部。"
        logger.info(f"[身份注册] 已获取用户姓名: {self.registration_data.name}")
        return prompt, False
    
    def _handle_collect_face(self, user_input: str) -> Tuple[str, bool]:
        """
        处理人脸采集
        """
        # 检查用户是否准备好
        if user_input.lower() not in ["是", "好", "准备好了", "开始"]:
            return "请在准备好后说'是'或'开始'，我将开始采集您的面部信息。", False
        
        # 开始采集人脸
        face_images = []
        positions = ["正面", "左侧", "右侧", "抬头", "低头"]
        
        for position in positions:
            prompt = f"请将脸转向{position}方向，保持微笑。"
            logger.info(f"[身份注册] 请用户转向{position}方向")
            
            # 等待用户准备
            time.sleep(2)
            
            # 采集图像
            success, image = self.vision_manager.capture_image()
            if success:
                face_images.append(image)
                logger.info(f"[身份注册] 已采集{position}方向的面部图像")
            else:
                self.registration_data.current_state = RegistrationState.FAILED
                self.registration_data.error_message = "无法采集面部图像，请检查摄像头是否正常。"
                return self.registration_data.error_message, True
        
        self.registration_data.face_images = face_images
        self.registration_data.current_state = RegistrationState.COLLECT_VOICE
        
        prompt = "面部信息采集完成！接下来我需要采集您的声纹信息，请按照我的提示朗读一段文字。"
        logger.info("[身份注册] 面部信息采集完成")
        return prompt, False
    
    def _handle_collect_voice(self, user_input: str) -> Tuple[str, bool]:
        """
        处理声纹采集
        """
        # 检查用户是否准备好
        if user_input.lower() not in ["是", "好", "准备好了", "开始"]:
            return "请在准备好后说'是'或'开始'，我将开始采集您的声纹信息。", False
        
        # 开始采集声纹
        prompt = "请朗读以下文字：'你好，我是这个家庭的成员，很高兴认识你。'"
        logger.info("[身份注册] 开始采集声纹")
        
        # 等待用户准备
        time.sleep(1)
        
        # 采集音频
        audio, sample_rate = self.audio_manager.record_audio(duration=5)
        if audio is not None:
            self.registration_data.voice_audio = audio
            self.registration_data.sample_rate = sample_rate
            self.registration_data.current_state = RegistrationState.CONFIRM
            
            confirmation_prompt = f"声纹信息采集完成！请确认以下信息是否正确：\n姓名：{self.registration_data.name}\nID：{self.registration_data.user_id}\n\n如果正确，请说'确认'或'是'；如果不正确，请说'重新开始'。"
            logger.info("[身份注册] 声纹信息采集完成")
            return confirmation_prompt, False
        else:
            self.registration_data.current_state = RegistrationState.FAILED
            self.registration_data.error_message = "无法采集声纹信息，请检查麦克风是否正常。"
            return self.registration_data.error_message, True
    
    def _handle_confirm(self, user_input: str) -> Tuple[str, bool]:
        """
        处理信息确认
        """
        if user_input.lower() in ["确认", "是", "正确"]:
            # 开始注册用户
            success = self._register_user()
            if success:
                self.registration_data.current_state = RegistrationState.COMPLETE
                prompt = f"注册成功！{self.registration_data.name}，欢迎加入我们的家庭。我会记住您的面部和声音特征，以便在未来更好地为您服务。"
                logger.info(f"[身份注册] 用户 {self.registration_data.name} 注册成功")
            else:
                self.registration_data.current_state = RegistrationState.FAILED
                self.registration_data.error_message = "注册失败，请稍后重试。"
                prompt = self.registration_data.error_message
            return prompt, True
        elif user_input.lower() in ["重新开始", "取消"]:
            # 重新开始注册
            return self.start_registration(), False
        else:
            return "请说'确认'或'是'来完成注册，或说'重新开始'来重新填写信息。", False
    
    def _register_user(self) -> bool:
        """
        注册用户
        """
        try:
            # 使用第一张正面图像进行注册
            if self.registration_data.face_images:
                face_image = self.registration_data.face_images[0]
            else:
                face_image = None
            
            # 注册用户
            success = self.identity_manager.register_user(
                user_id=self.registration_data.user_id,
                name=self.registration_data.name,
                user_type=UserType.FAMILY,
                face_image=face_image,
                voice_audio=self.registration_data.voice_audio,
                sample_rate=self.registration_data.sample_rate
            )
            
            # 保存其他角度的面部图像作为备份
            if success and self.registration_data.face_images:
                face_data_dir = os.path.join("./identity_data", "face")
                os.makedirs(face_data_dir, exist_ok=True)
                
                for i, image in enumerate(self.registration_data.face_images):
                    image_path = os.path.join(face_data_dir, f"{self.registration_data.user_id}_{i}.jpg")
                    cv2.imwrite(image_path, image)
                
                logger.info(f"[身份注册] 已保存用户 {self.registration_data.name} 的面部图像备份")
            
            return success
        except Exception as e:
            logger.error(f"[身份注册] 注册用户失败: {e}")
            return False
    
    def cancel_registration(self) -> str:
        """
        取消注册
        """
        self.registration_data = None
        logger.info("[身份注册] 注册流程已取消")
        return "注册已取消，如有需要可以随时重新开始。"
    
    def get_current_state(self) -> Optional[RegistrationState]:
        """
        获取当前注册状态
        """
        if self.registration_data:
            return self.registration_data.current_state
        return None


# 全局身份注册代理实例
_global_registration_agent = None


def get_registration_agent() -> IdentityRegistrationAgent:
    """
    获取身份注册代理实例
    """
    global _global_registration_agent
    if _global_registration_agent is None:
        _global_registration_agent = IdentityRegistrationAgent()
    return _global_registration_agent


def start_user_registration() -> str:
    """
    开始用户注册流程
    """
    agent = get_registration_agent()
    return agent.start_registration()


def process_registration_input(user_input: str) -> Tuple[str, bool]:
    """
    处理注册输入
    """
    agent = get_registration_agent()
    return agent.process_input(user_input)


def cancel_registration() -> str:
    """
    取消注册
    """
    agent = get_registration_agent()
    return agent.cancel_registration()
