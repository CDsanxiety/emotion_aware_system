"""agent_loop.py
后台"智能体循环"：不阻塞主程序，持续对接语音识别与视觉模块。
与 app.py、llm_api.py、blackboard.py、memory_rag.py、pad_model.py 无缝对接。
支持主动关心开关，不打断当前交互。
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import cv2

from audio import recognize_speech
from llm_api import get_response
from blackboard import Blackboard
from ros_bridge import get_ros_bridge
from vla_integration import create_vla_integration
from autogen_integration import global_autogen_manager
from memory_rag import LongTermMemory
from utils import logger
import vision

# 初始化CogVLM集成标志
COGVLM_AVAILABLE = False
create_cogvlm_integration = None

# 初始化长期记忆
global_memory = LongTermMemory()


@dataclass
class ProactiveCareResult:
    """主动关心结果：温柔插话不打断当前交互"""
    reply: str
    audio_path: Optional[str]
    emotion: str
    action: str


def _infer_presence_from_vision(vision_desc: str) -> bool:
    if not vision_desc:
        return False
    if "未检测到稳定人脸" in vision_desc:
        return False
    if "未能稳定检测到人脸" in vision_desc:
        return False
    return True


def _default_idle_prompt(vision_desc: str) -> str:
    vd = (vision_desc or "").strip()
    if vd:
        return f"我注意到画面与环境里有一些情绪线索：{vd}。你已经有一段时间没说话了，能不能用简短、温柔的话关心一下用户的状态？"
    return "你已经有一段时间没说话了，能不能用简短、温柔的话关心一下用户的状态？"


def _semantic_scene_match(vision_desc_lower: str, scene_type: str) -> bool:
    """基于语义覆盖的场景匹配"""
    semantic_keywords = {
        "回家": {
            "direct": ["回家", "回来了", "进门", "到家", "进入家门", "踏入家门", "推门进入", "开门进来", "刚回来"],
            "context": ["主人", "走进来", "进来", "进入", "回来"]
        },
        "疲惫": {
            "direct": ["很累", "疲惫", "累", "疲惫不堪", "无精打采", "有气无力", "精神不振", "疲劳"],
            "context": ["主人", "看起来", "好像", "似乎", "感觉"]
        },
        "光线变暗": {
            "direct": ["光线变暗", "变暗", "天黑", "灯光关闭", "变黑了", "暗下来", "黑暗", "光线不足"],
            "context": ["环境", "房间", "室内", "光线", "周围"]
        }
    }

    if scene_type not in semantic_keywords:
        return False

    keywords = semantic_keywords[scene_type]
    direct_match = any(kw in vision_desc_lower for kw in keywords.get("direct", []))

    if direct_match:
        return True

    context_match_count = sum(1 for kw in keywords.get("context", []) if kw in vision_desc_lower)
    return context_match_count >= 2


def detect_scene_triggers(vision_desc: str) -> Optional[str]:
    """检测场景触发事件（基于语义匹配）"""
    if not vision_desc:
        return None

    vision_desc_lower = vision_desc.lower()

    if _semantic_scene_match(vision_desc_lower, "回家"):
        return "主人回家了"

    if _semantic_scene_match(vision_desc_lower, "疲惫"):
        return "主人看起来很累"

    if _semantic_scene_match(vision_desc_lower, "光线变暗"):
        return "环境光线突然变暗"

    return None


def check_memory_triggers() -> Optional[str]:
    """检查长期记忆触发事件"""
    # 获取当前日期
    today = time.strftime("%m-%d")
    
    # 检索长期记忆中的生日信息
    birthday_query = f"今天是{today}，用户的生日是什么时候？"
    memory_result = global_memory.query(birthday_query)
    
    if "生日" in memory_result and today in memory_result:
        return "今天是你生日"
    
    return None


def _safe_update_blackboard(blackboard: Optional[Blackboard], updates: Dict[str, Any]) -> None:
    if blackboard is None:
        return
    if not hasattr(blackboard, "lock"):
        for k, v in updates.items():
            setattr(blackboard, k, v)
        return
    with blackboard.lock:
        for k, v in updates.items():
            setattr(blackboard, k, v)


class GlobalAgentState:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.proactive_enabled = True
        self.is_interacting = False
        self.pending_proactive: Optional[ProactiveCareResult] = None
        self.last_proactive_time = 0.0
        self.proactive_cooldown_sec = 15.0

    @classmethod
    def get_instance(cls) -> "GlobalAgentState":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_proactive_enabled(self, enabled: bool) -> None:
        with self._lock:
            self.proactive_enabled = enabled

    def is_proactive_enabled(self) -> bool:
        with self._lock:
            return self.proactive_enabled

    def set_interacting(self, interacting: bool) -> None:
        with self._lock:
            self.is_interacting = interacting

    def is_interacting(self) -> bool:
        with self._lock:
            return self.is_interacting

    def set_pending_proactive(self, result: Optional[ProactiveCareResult]) -> None:
        with self._lock:
            self.pending_proactive = result
            if result:
                self.last_proactive_time = time.time()

    def get_pending_proactive(self) -> Optional[ProactiveCareResult]:
        with self._lock:
            return self.pending_proactive

    def clear_pending_proactive(self) -> None:
        with self._lock:
            self.pending_proactive = None

    def can_trigger_proactive(self) -> bool:
        with self._lock:
            if not self.proactive_enabled:
                return False
            if self.is_interacting:
                return False
            if time.time() - self.last_proactive_time < self.proactive_cooldown_sec:
                return False
            return True


_global_state = GlobalAgentState.get_instance()


def _agent_worker(
    blackboard: Optional[Blackboard],
    enable_tts: bool,
    stt_timeout_sec: int,
    stt_phrase_time_limit_sec: int,
    vision_interval_sec: float,
    idle_trigger_sec: float,
    proactive_cooldown_sec: float,
    idle_prompt_func: Optional[Callable[[str], str]],
    on_proactive_output: Optional[Callable[[ProactiveCareResult], None]],
    camera_index: int,
    stop_event: threading.Event,
) -> None:
    cap = None
    vision_desc = ""
    presence = False
    last_vision_time = 0.0
    last_ros_publish_time = 0.0
    ros_publish_interval = 2.0  # 每 2 秒发布一次状态
    control_frequency = 10.0  # 控制频率 10Hz
    control_interval = 1.0 / control_frequency

    # 初始化 ROS 桥接
    ros_bridge = get_ros_bridge()
    # 初始化 VLA 集成
    vla_integration = create_vla_integration(blackboard)
    # 使用全局 AutoGen 管理器并初始化
    autogen_integration = global_autogen_manager
    if autogen_integration and hasattr(autogen_integration, 'initialize_agents'):
        autogen_integration.initialize_agents()
    # 初始化 CogVLM 集成
    global COGVLM_AVAILABLE, create_cogvlm_integration
    if not COGVLM_AVAILABLE:
        try:
            from cogvlm_integration import create_cogvlm_integration
            COGVLM_AVAILABLE = True
        except Exception as e:
            print(f"CogVLM 集成导入失败: {e}")
            create_cogvlm_integration = None
            COGVLM_AVAILABLE = False
    cogvlm_integration = create_cogvlm_integration() if COGVLM_AVAILABLE else None

    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap = None
            _safe_update_blackboard(blackboard, {"current_vision_desc": ""})
    except Exception:
        cap = None

    try:
        while not stop_event.is_set():
            start_time = time.time()
            now = start_time

            # 1. 获取当前观测
            if cap is not None and (now - last_vision_time) >= vision_interval_sec:
                last_vision_time = now
                try:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        # 2. 图像预处理与分析
                        vr = vision.process_image(frame)
                        vision_desc = (vr.get("description") or "").strip()
                        presence = _infer_presence_from_vision(vision_desc)
                        _safe_update_blackboard(
                            blackboard,
                            {
                                "current_vision_desc": vision_desc,
                                "user_presence": presence,
                            },
                        )
                        
                        # 3. 检测场景触发事件
                        scene_trigger = detect_scene_triggers(vision_desc)
                        if scene_trigger and _global_state.can_trigger_proactive():
                            _global_state.last_proactive_time = time.time()
                            logger.info(f"[场景触发] 检测到: {scene_trigger}")
                            
                            # 构建场景触发的提示
                            if scene_trigger == "主人回家了":
                                prompt_text = "主人刚回家，用温暖的语气问候并欢迎主人。"
                            elif scene_trigger == "主人看起来很累":
                                prompt_text = "主人看起来很累，用关心的语气询问是否需要帮助，并提供休息建议。"
                            elif scene_trigger == "环境光线突然变暗":
                                prompt_text = "环境光线突然变暗，询问主人是否需要开灯或其他帮助。"
                            else:
                                prompt_text = f"检测到场景: {scene_trigger}，用适当的语气回应。"
                            
                            # 执行主动关心动作
                            res, audio_path = get_response(
                                "neutral",
                                prompt_text,
                                enable_tts=enable_tts,
                                vision_desc=vision_desc,
                            )
                            
                            proactive_result = ProactiveCareResult(
                                reply=res.get("reply", "") if isinstance(res, dict) else "",
                                audio_path=audio_path,
                                emotion=res.get("emotion", "neutral") if isinstance(res, dict) else "neutral",
                                action=res.get("action", "无动作") if isinstance(res, dict) else "无动作",
                            )
                            
                            # 发布主动关心到 ROS
                            proactive_msg = {
                                "type": "proactive_care",
                                "reply": proactive_result.reply,
                                "emotion": proactive_result.emotion,
                                "action": proactive_result.action,
                            }
                            ros_bridge.publish_message("/robot/proactive_care", proactive_msg)
                            
                            _global_state.set_pending_proactive(proactive_result)
                            
                            if on_proactive_output is not None:
                                try:
                                    on_proactive_output(proactive_result)
                                except Exception as e:
                                    print(f"主动关心输出处理失败: {e}")
                        
                        # 4. 检查长期记忆触发事件
                        if presence:
                            memory_trigger = check_memory_triggers()
                            if memory_trigger and _global_state.can_trigger_proactive():
                                _global_state.last_proactive_time = time.time()
                                logger.info(f"[记忆触发] 检测到: {memory_trigger}")
                                
                                # 构建记忆触发的提示
                                if memory_trigger == "今天是你生日":
                                    prompt_text = "今天是主人的生日，播放生日快乐音乐并送上温馨的生日祝福。"
                                    # 这里可以添加播放音乐的逻辑
                                else:
                                    prompt_text = f"记忆触发: {memory_trigger}，用适当的语气回应。"
                                
                                # 执行主动关心动作
                                res, audio_path = get_response(
                                    "happy",
                                    prompt_text,
                                    enable_tts=enable_tts,
                                    vision_desc=vision_desc,
                                )
                                
                                proactive_result = ProactiveCareResult(
                                    reply=res.get("reply", "") if isinstance(res, dict) else "",
                                    audio_path=audio_path,
                                    emotion=res.get("emotion", "happy") if isinstance(res, dict) else "happy",
                                    action=res.get("action", "播放音乐") if isinstance(res, dict) else "播放音乐",
                                )
                                
                                # 发布主动关心到 ROS
                                proactive_msg = {
                                    "type": "proactive_care",
                                    "reply": proactive_result.reply,
                                    "emotion": proactive_result.emotion,
                                    "action": proactive_result.action,
                                }
                                ros_bridge.publish_message("/robot/proactive_care", proactive_msg)
                                
                                _global_state.set_pending_proactive(proactive_result)
                                
                                if on_proactive_output is not None:
                                    try:
                                        on_proactive_output(proactive_result)
                                    except Exception as e:
                                        print(f"主动关心输出处理失败: {e}")
                except Exception as e:
                    print(f"[agent_loop] 场景检测出错: {e}")
                    traceback.print_exc()

            # 3. 语音观测
            user_text = ""
            try:
                user_text = recognize_speech(
                    timeout=stt_timeout_sec,
                    phrase_time_limit=stt_phrase_time_limit_sec,
                )
            except Exception:
                user_text = ""

            if stop_event.is_set():
                break

            # 4. 基于观测预测动作（单步预测）
            if user_text:
                _global_state.set_interacting(True)
                ts = time.time()
                _safe_update_blackboard(
                    blackboard,
                    {
                        "last_speech_time": ts,
                        "last_speech_text": user_text,
                        "user_presence": True,
                    },
                )

                # 5. 执行动作（使用 AutoGen 和 CogVLM 集成）
                # 使用 AutoGen 分析情感和规划动作
                autogen_result = autogen_integration.analyze_emotion_and_plan_action(user_text, vision_desc)
                emotion = autogen_result.get("emotion", "neutral")
                action = autogen_result.get("action", "无动作")
                confidence = autogen_result.get("confidence", 1.0)
                decision_mode = autogen_result.get("decision_mode", "confident")
                pad_state = autogen_result.get("pad_state", {})

                # 如果是询问模式，生成询问用户的提示
                if autogen_result.get("needs_query", False):
                    logger.info(f"[不确定性] 置信度: {confidence:.2f}，进入询问模式")
                    inquiry_text = action
                    res, audio_path = get_response(
                        emotion,
                        inquiry_text,
                        enable_tts=enable_tts,
                        vision_desc=vision_desc,
                    )
                    _safe_update_blackboard(
                        blackboard,
                        {
                            "last_robot_result": res,
                            "last_robot_audio_path": audio_path,
                            "last_emotion_confidence": confidence,
                            "last_decision_mode": decision_mode,
                        },
                    )
                    _global_state.set_interacting(False)
                    continue

                # 如果有图像，使用 CogVLM 进行多模态分析
                if cap is not None and COGVLM_AVAILABLE and cogvlm_integration:
                    try:
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            from PIL import Image
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            cogvlm_result = cogvlm_integration.process_multimodal(image, user_text)
                            if cogvlm_result.get("success"):
                                emotion = cogvlm_result.get("emotion", emotion)
                                action = cogvlm_result.get("action", action)
                    except Exception as e:
                        print(f"CogVLM 处理失败: {e}")
                
                # 构建 VLA 观测
                from vla_integration import VLAObservation
                vla_obs = VLAObservation(
                    vision_desc=vision_desc,
                    audio_text=user_text,
                    timestamp=ts
                )
                # 预测动作
                prediction = vla_integration.predict_action(vla_obs, instruction="响应用户需求")
                # 执行动作
                execution_result = vla_integration.execute_action(prediction)
                
                res, audio_path = get_response(
                    emotion,
                    user_text,
                    enable_tts=enable_tts,
                    vision_desc=vision_desc,
                )
                _safe_update_blackboard(
                    blackboard,
                    {
                        "last_robot_result": res,
                        "last_robot_audio_path": audio_path,
                    },
                )

                _global_state.set_interacting(False)

            else:
                last_speech_time = 0.0
                if blackboard is not None:
                    last_speech_time = getattr(blackboard, "last_speech_time", 0.0) or 0.0
                presence_now = presence
                if blackboard is not None:
                    presence_now = getattr(blackboard, "user_presence", presence)
                idle_time = time.time() - last_speech_time

                # 6. 主动关心（基于视觉观测的预测）
                if (
                    presence_now
                    and idle_time >= idle_trigger_sec
                    and _global_state.can_trigger_proactive()
                ):
                    _global_state.last_proactive_time = time.time()
                    prompt_text = (
                        idle_prompt_func(vision_desc)
                        if callable(idle_prompt_func)
                        else _default_idle_prompt(vision_desc)
                    )

                    # 7. 执行主动关心动作
                    # 使用 AutoGen 分析场景和规划动作
                    autogen_result = autogen_integration.analyze_emotion_and_plan_action("", vision_desc)
                    emotion = autogen_result.get("emotion", "neutral")
                    action = autogen_result.get("action", "无动作")
                    
                    # 如果有图像，使用 CogVLM 进行多模态分析
                    if cap is not None and COGVLM_AVAILABLE and cogvlm_integration:
                        try:
                            ok, frame = cap.read()
                            if ok and frame is not None:
                                from PIL import Image
                                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                cogvlm_result = cogvlm_integration.process_multimodal(image, "分析当前场景，用户需要什么？")
                                if cogvlm_result.get("success"):
                                    emotion = cogvlm_result.get("emotion", emotion)
                                    action = cogvlm_result.get("action", action)
                        except Exception as e:
                            print(f"CogVLM 处理失败: {e}")
                    
                    res, audio_path = get_response(
                        emotion,
                        prompt_text,
                        enable_tts=enable_tts,
                        vision_desc=vision_desc,
                    )

                    proactive_result = ProactiveCareResult(
                        reply=res.get("reply", "") if isinstance(res, dict) else "",
                        audio_path=audio_path,
                        emotion=res.get("emotion", "neutral") if isinstance(res, dict) else "neutral",
                        action=res.get("action", "无动作") if isinstance(res, dict) else "无动作",
                    )

                    # 发布主动关心到 ROS
                    proactive_msg = {
                        "type": "proactive_care",
                        "reply": proactive_result.reply,
                        "emotion": proactive_result.emotion,
                        "action": proactive_result.action,
                    }
                    ros_bridge.publish_message("/robot/proactive_care", proactive_msg)

                    _global_state.set_pending_proactive(proactive_result)

                    if on_proactive_output is not None:
                        try:
                            on_proactive_output(proactive_result)
                        except Exception:
                            pass

            # 8. 发布状态到 ROS
            if (now - last_ros_publish_time) >= ros_publish_interval:
                last_ros_publish_time = now
                if blackboard:
                    state_msg = {
                        "last_speech_text": getattr(blackboard, "last_speech_text", ""),
                        "last_speech_time": getattr(blackboard, "last_speech_time", 0.0),
                        "user_presence": getattr(blackboard, "user_presence", False),
                        "robot_status": getattr(blackboard, "robot_status", "idle"),
                        "vision_desc": getattr(blackboard, "current_vision_desc", ""),
                        "vision_update_time": getattr(blackboard, "vision_update_time", 0.0),
                        "proactive_enabled": _global_state.is_proactive_enabled(),
                        "is_interacting": _global_state.is_interacting(),
                    }
                    ros_bridge.publish_message("/robot/status", state_msg)

            # 9. 控制频率
            elapsed = time.time() - start_time
            if elapsed < control_interval:
                time.sleep(control_interval - elapsed)

    except Exception as e:
        print(f"[agent_loop] worker crashed: {e}")
        traceback.print_exc()
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        _global_state.set_interacting(False)


def start_agentic_main_loop(
    blackboard: Optional[Blackboard] = None,
    *,
    enable_tts: bool = False,
    on_proactive_output: Optional[Callable[[ProactiveCareResult], None]] = None,
    camera_index: int = 0,
    stt_timeout_sec: int = 2,
    stt_phrase_time_limit_sec: int = 6,
    vision_interval_sec: float = 6.0,
    idle_trigger_sec: float = 30.0,
    proactive_cooldown_sec: float = 15.0,
    idle_prompt_func: Optional[Callable[[str], str]] = None,
) -> "AgentLoopHandle":
    if blackboard is None:
        blackboard = Blackboard()
    stop_event = threading.Event()

    t = threading.Thread(
        target=_agent_worker,
        args=(
            blackboard,
            enable_tts,
            stt_timeout_sec,
            stt_phrase_time_limit_sec,
            vision_interval_sec,
            idle_trigger_sec,
            proactive_cooldown_sec,
            idle_prompt_func,
            on_proactive_output,
            camera_index,
            stop_event,
        ),
        daemon=True,
    )
    t.start()
    return AgentLoopHandle(thread=t, stop_event=stop_event, blackboard=blackboard)


class AgentLoopHandle:
    def __init__(self, thread: threading.Thread, stop_event: threading.Event, blackboard: Optional[Blackboard] = None):
        self.thread = thread
        self.stop_event = stop_event
        self.blackboard = blackboard

    def stop(self) -> None:
        self.stop_event.set()

    def is_alive(self) -> bool:
        return self.thread.is_alive()


def agentic_main_loop(blackboard: Optional[Blackboard] = None, **kwargs) -> AgentLoopHandle:
    return start_agentic_main_loop(blackboard, **kwargs)


def set_proactive_enabled(enabled: bool) -> None:
    _global_state.set_proactive_enabled(enabled)


def is_proactive_enabled() -> bool:
    return _global_state.is_proactive_enabled()


def get_pending_proactive() -> Optional[ProactiveCareResult]:
    return _global_state.get_pending_proactive()


def clear_pending_proactive() -> None:
    _global_state.clear_pending_proactive()


if __name__ == "__main__":
    h = start_agentic_main_loop(enable_tts=False)
    print("agent_loop 已在后台运行，Ctrl+C 退出。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        h.stop()
        print("stopped")
