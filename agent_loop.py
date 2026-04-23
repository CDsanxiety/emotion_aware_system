<<<<<<< HEAD
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
from ros_client import global_ros_manager
#from openvla_integration import create_vla_integration
from memory_rag import LongTermMemory
from multi_agent import MultiAgentCoordinator
from utils import logger
import vision
from identity_manager import recognize_user
from decision_tracer import DecisionTracer, NodeType, ModelType
from audio_manager import play_system_audio



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
        self._is_interacting = False
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
            self._is_interacting = interacting

    def is_interacting(self) -> bool:
        with self._lock:
            return self._is_interacting

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
            if self.is_interacting():
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
    vision_desc = ""
    presence = False
    last_vision_time = 0.0
    last_ros_publish_time = 0.0
    ros_publish_interval = 2.0  # 每 2 秒发布一次状态
    control_frequency = 10.0  # 控制频率 10Hz
    control_interval = 1.0 / control_frequency
    last_stt_time = 0.0
    stt_interval = 1.0  # 语音检测间隔，避免频繁占用麦克风

    # 初始化 ROS 管理器
    ros_manager = global_ros_manager
    # 确保 ROS 连接
    if not ros_manager.is_connected:
        ros_manager.connect()
    # 初始化 VLA 集成
    #vla_integration = create_vla_integration(blackboard)
    # 初始化多代理协调器
    multi_agent_coordinator = MultiAgentCoordinator()

    # 初始化决策追踪器
    decision_tracer = DecisionTracer.get_instance()

    try:
        while not stop_event.is_set():
            start_time = time.time()
            now = start_time

            # 1. 获取当前观测
            if (now - last_vision_time) >= vision_interval_sec:
                last_vision_time = now
                cap = None
                try:
                    # 临时打开摄像头
                    cap = cv2.VideoCapture(camera_index)
                    if cap.isOpened():
                        # 播放摄像头打开识别音
                        play_system_audio("camara")
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            # 1. 用户身份识别
                            user_identity = recognize_user(face_image=frame)
                            user_id = user_identity.user_id if user_identity else "unknown"
                            user_name = user_identity.name if user_identity else "主人"
                            user_type = user_identity.user_type.value if user_identity else "unknown"
                            
                            # 2. 图像预处理与分析
                            decision_tracer.start_latency_tracking(NodeType.PERCEPTION, ModelType.LOCAL, user_id)
                            vr = vision.process_image(frame)
                            decision_tracer.end_latency_tracking(NodeType.PERCEPTION, ModelType.LOCAL, user_id=user_id)
                            vision_desc = (vr.get("description") or "").strip()
                            presence = _infer_presence_from_vision(vision_desc)
                            
                            # 检查是否正在用户交互，如果是，则不更新黑板数据，避免状态冲突
                            if not _global_state.is_interacting():
                                _safe_update_blackboard(
                                    blackboard,
                                    {
                                        "current_vision_desc": vision_desc,
                                        "user_presence": presence,
                                        "current_user_id": user_id,
                                        "current_user_name": user_name,
                                        "current_user_type": user_type,
                                    },
                                )
                            
                            # 3. 检测场景触发事件
                            scene_trigger = detect_scene_triggers(vision_desc)
                            if scene_trigger and _global_state.can_trigger_proactive():
                                _global_state.last_proactive_time = time.time()
                                logger.info(f"[场景触发] 检测到: {scene_trigger}")
                                
                                # 构建场景触发的提示
                                if scene_trigger == "主人回家了":
                                    prompt_text = f"{user_name}刚回家，用温暖的语气问候并欢迎{user_name}。"
                                elif scene_trigger == "主人看起来很累":
                                    prompt_text = f"{user_name}看起来很累，用关心的语气询问是否需要帮助，并提供休息建议。"
                                elif scene_trigger == "环境光线突然变暗":
                                    prompt_text = f"环境光线突然变暗，询问{user_name}是否需要开灯或其他帮助。"
                                else:
                                    prompt_text = f"检测到场景: {scene_trigger}，用适当的语气回应{user_name}。"
                                
                                # 执行主动关心动作
                                decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                                res, audio_path = get_response(
                                    "neutral",
                                    prompt_text,
                                    enable_tts=enable_tts,
                                    vision_desc=vision_desc,
                                )
                                decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
                                
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
                                ros_manager.publish_action(proactive_msg)
                                
                                _global_state.set_pending_proactive(proactive_result)
                                
                                if on_proactive_output is not None:
                                    try:
                                        on_proactive_output(proactive_result)
                                    except Exception as e:
                                        print(f"主动关心输出处理失败: {e}")
                            
                            # 4. 检查长期记忆触发事件
                            if presence:
                                decision_tracer.start_latency_tracking(NodeType.MEMORY_ASSOCIATION, ModelType.LOCAL, user_id)
                                memory_trigger = check_memory_triggers()
                                decision_tracer.end_latency_tracking(NodeType.MEMORY_ASSOCIATION, ModelType.LOCAL, user_id=user_id)
                                if memory_trigger and _global_state.can_trigger_proactive():
                                    _global_state.last_proactive_time = time.time()
                                    logger.info(f"[记忆触发] 检测到: {memory_trigger}")
                                    
                                    # 构建记忆触发的提示
                                    if memory_trigger == "今天是你生日":
                                        prompt_text = f"今天是{user_name}的生日，播放生日快乐音乐并送上温馨的生日祝福。"
                                        # 这里可以添加播放音乐的逻辑
                                    else:
                                        prompt_text = f"记忆触发: {memory_trigger}，用适当的语气回应{user_name}。"
                                    
                                    # 执行主动关心动作
                                    decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                                    res, audio_path = get_response(
                                        "happy",
                                        prompt_text,
                                        enable_tts=enable_tts,
                                        vision_desc=vision_desc,
                                    )
                                    decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
                                    
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
                                    ros_manager.publish_action(proactive_msg)
                                    
                                    _global_state.set_pending_proactive(proactive_result)
                                    
                                    if on_proactive_output is not None:
                                        try:
                                            on_proactive_output(proactive_result)
                                        except Exception as e:
                                            print(f"主动关心输出处理失败: {e}")
                except Exception as e:
                    print(f"[agent_loop] 场景检测出错: {e}")
                    traceback.print_exc()
                finally:
                    # 无论如何都释放摄像头
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass

            # 3. 语音观测
            user_text = ""
            # 只有在用户未交互且达到语音检测间隔时才进行语音识别
            if not _global_state.is_interacting() and (now - last_stt_time) >= stt_interval:
                try:
                    decision_tracer.start_latency_tracking(NodeType.RAW_SENSOR, ModelType.LOCAL)
                    logger.info("开始语音识别...")
                    user_text = recognize_speech(
                        timeout=stt_timeout_sec,
                        phrase_time_limit=stt_phrase_time_limit_sec,
                    )
                    decision_tracer.end_latency_tracking(NodeType.RAW_SENSOR, ModelType.LOCAL)
                    last_stt_time = now  # 更新语音检测时间
                    
                    if user_text:
                        logger.info(f"检测到用户语音: {user_text}")
                    else:
                        logger.info("未检测到用户语音")
                except Exception as e:
                    user_text = ""
                    decision_tracer.end_latency_tracking(NodeType.RAW_SENSOR, ModelType.LOCAL)
                    logger.info(f"语音识别错误: {e}")

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

                # 5. 执行动作（使用多代理协调器）
                # 使用多代理协调器分析情感和规划动作
                logger.info("模型调度中...")
                
                decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                execution_result = multi_agent_coordinator.think(
                    vision_desc=vision_desc,
                    audio_text=user_text,
                    current_emotion="neutral",
                    context=f"视觉: {vision_desc} 语音: {user_text} 情绪: neutral"
                )
                decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
                emotion = execution_result.emotion
                action = execution_result.action
                confidence = 1.0  # 多代理协调器暂时不返回置信度
                decision_mode = "confident"  # 多代理协调器暂时不返回决策模式

                # 如果需要抑制回复
                if execution_result.should_suppress:
                    logger.info("[多代理] 抑制回复: 无有效输入")
                    _global_state.set_interacting(False)
                    continue


                
                # VLA 功能已禁用（vla_integration 模块已删除）
                # 继续使用多代理协调器进行决策
                
                decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                res, audio_path = get_response(
                    emotion,
                    user_text,
                    enable_tts=enable_tts,
                    vision_desc=vision_desc,
                )
                decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
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
                    # 使用多代理协调器分析场景和规划动作
                    decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                    execution_result = multi_agent_coordinator.think(
                        vision_desc=vision_desc,
                        audio_text="",
                        current_emotion="neutral",
                        context=f"视觉: {vision_desc} 语音: 无 情绪: neutral"
                    )
                    decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
                    emotion = execution_result.emotion
                    action = execution_result.action
                    

                    
                    decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                    res, audio_path = get_response(
                        emotion,
                        prompt_text,
                        enable_tts=enable_tts,
                        vision_desc=vision_desc,
                    )
                    decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)

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
                    ros_manager.publish_action(proactive_msg)

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
                        "type": "status",
                        "last_speech_text": getattr(blackboard, "last_speech_text", ""),
                        "last_speech_time": getattr(blackboard, "last_speech_time", 0.0),
                        "user_presence": getattr(blackboard, "user_presence", False),
                        "robot_status": getattr(blackboard, "robot_status", "idle"),
                        "vision_desc": getattr(blackboard, "current_vision_desc", ""),
                        "vision_update_time": getattr(blackboard, "vision_update_time", 0.0),
                        "proactive_enabled": _global_state.is_proactive_enabled(),
                        "is_interacting": _global_state.is_interacting(),
                    }
                    ros_manager.publish_action(state_msg)

            # 9. 控制频率
            elapsed = time.time() - start_time
            if elapsed < control_interval:
                time.sleep(control_interval - elapsed)

    except Exception as e:
        print(f"[agent_loop] worker crashed: {e}")
        traceback.print_exc()
    finally:
        _global_state.set_interacting(False)


def start_agentic_main_loop(
    blackboard: Optional[Blackboard] = None,
    *,
    enable_tts: bool = False,
    on_proactive_output: Optional[Callable[[ProactiveCareResult], None]] = None,
    camera_index: int = 0,
    stt_timeout_sec: int = 5,
    stt_phrase_time_limit_sec: int = 8,
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
        # 等待线程结束，确保麦克风和其他资源被正确释放
        if self.thread.is_alive():
            logger.info("等待 agent_loop 线程结束...")
            self.thread.join(timeout=5.0)  # 最多等待 5 秒
            if self.thread.is_alive():
                logger.warning("agent_loop 线程未能在 5 秒内结束，可能存在资源泄露")
            else:
                logger.info("agent_loop 线程已成功结束")

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
=======
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
from ros_client import global_ros_manager
#from openvla_integration import create_vla_integration
from memory_rag import LongTermMemory
from multi_agent import MultiAgentCoordinator
from utils import logger
import vision
from identity_manager import recognize_user
from decision_tracer import DecisionTracer, NodeType, ModelType
from audio_manager import play_system_audio



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
        self._is_interacting = False
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
            self._is_interacting = interacting

    def is_interacting(self) -> bool:
        with self._lock:
            return self._is_interacting

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
            if self.is_interacting():
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
    vision_desc = ""
    presence = False
    last_vision_time = 0.0
    last_ros_publish_time = 0.0
    ros_publish_interval = 2.0  # 每 2 秒发布一次状态
    control_frequency = 10.0  # 控制频率 10Hz
    control_interval = 1.0 / control_frequency
    last_stt_time = 0.0
    stt_interval = 1.0  # 语音检测间隔，避免频繁占用麦克风

    # 初始化 ROS 管理器
    ros_manager = global_ros_manager
    # 确保 ROS 连接
    if not ros_manager.is_connected:
        ros_manager.connect()
    # 初始化 VLA 集成
    #vla_integration = create_vla_integration(blackboard)
    # 初始化多代理协调器
    multi_agent_coordinator = MultiAgentCoordinator()

    # 初始化决策追踪器
    decision_tracer = DecisionTracer.get_instance()

    try:
        while not stop_event.is_set():
            start_time = time.time()
            now = start_time

            # 1. 获取当前观测
            if (now - last_vision_time) >= vision_interval_sec:
                last_vision_time = now
                cap = None
                try:
                    # 临时打开摄像头
                    cap = cv2.VideoCapture(camera_index)
                    if cap.isOpened():
                        # 播放摄像头打开识别音
                        play_system_audio("camara")
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            # 1. 用户身份识别
                            user_identity = recognize_user(face_image=frame)
                            user_id = user_identity.user_id if user_identity else "unknown"
                            user_name = user_identity.name if user_identity else "主人"
                            user_type = user_identity.user_type.value if user_identity else "unknown"
                            
                            # 2. 图像预处理与分析
                            decision_tracer.start_latency_tracking(NodeType.PERCEPTION, ModelType.LOCAL, user_id)
                            vr = vision.process_image(frame)
                            decision_tracer.end_latency_tracking(NodeType.PERCEPTION, ModelType.LOCAL, user_id=user_id)
                            vision_desc = (vr.get("description") or "").strip()
                            presence = _infer_presence_from_vision(vision_desc)
                            
                            # 检查是否正在用户交互，如果是，则不更新黑板数据，避免状态冲突
                            if not _global_state.is_interacting():
                                _safe_update_blackboard(
                                    blackboard,
                                    {
                                        "current_vision_desc": vision_desc,
                                        "user_presence": presence,
                                        "current_user_id": user_id,
                                        "current_user_name": user_name,
                                        "current_user_type": user_type,
                                    },
                                )
                            
                            # 3. 检测场景触发事件
                            scene_trigger = detect_scene_triggers(vision_desc)
                            if scene_trigger and _global_state.can_trigger_proactive():
                                _global_state.last_proactive_time = time.time()
                                logger.info(f"[场景触发] 检测到: {scene_trigger}")
                                
                                # 构建场景触发的提示
                                if scene_trigger == "主人回家了":
                                    prompt_text = f"{user_name}刚回家，用温暖的语气问候并欢迎{user_name}。"
                                elif scene_trigger == "主人看起来很累":
                                    prompt_text = f"{user_name}看起来很累，用关心的语气询问是否需要帮助，并提供休息建议。"
                                elif scene_trigger == "环境光线突然变暗":
                                    prompt_text = f"环境光线突然变暗，询问{user_name}是否需要开灯或其他帮助。"
                                else:
                                    prompt_text = f"检测到场景: {scene_trigger}，用适当的语气回应{user_name}。"
                                
                                # 执行主动关心动作
                                decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                                res, audio_path = get_response(
                                    "neutral",
                                    prompt_text,
                                    enable_tts=enable_tts,
                                    vision_desc=vision_desc,
                                )
                                decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
                                
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
                                ros_manager.publish_action(proactive_msg)
                                
                                _global_state.set_pending_proactive(proactive_result)
                                
                                if on_proactive_output is not None:
                                    try:
                                        on_proactive_output(proactive_result)
                                    except Exception as e:
                                        print(f"主动关心输出处理失败: {e}")
                            
                            # 4. 检查长期记忆触发事件
                            if presence:
                                decision_tracer.start_latency_tracking(NodeType.MEMORY_ASSOCIATION, ModelType.LOCAL, user_id)
                                memory_trigger = check_memory_triggers()
                                decision_tracer.end_latency_tracking(NodeType.MEMORY_ASSOCIATION, ModelType.LOCAL, user_id=user_id)
                                if memory_trigger and _global_state.can_trigger_proactive():
                                    _global_state.last_proactive_time = time.time()
                                    logger.info(f"[记忆触发] 检测到: {memory_trigger}")
                                    
                                    # 构建记忆触发的提示
                                    if memory_trigger == "今天是你生日":
                                        prompt_text = f"今天是{user_name}的生日，播放生日快乐音乐并送上温馨的生日祝福。"
                                        # 这里可以添加播放音乐的逻辑
                                    else:
                                        prompt_text = f"记忆触发: {memory_trigger}，用适当的语气回应{user_name}。"
                                    
                                    # 执行主动关心动作
                                    decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                                    res, audio_path = get_response(
                                        "happy",
                                        prompt_text,
                                        enable_tts=enable_tts,
                                        vision_desc=vision_desc,
                                    )
                                    decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
                                    
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
                                    ros_manager.publish_action(proactive_msg)
                                    
                                    _global_state.set_pending_proactive(proactive_result)
                                    
                                    if on_proactive_output is not None:
                                        try:
                                            on_proactive_output(proactive_result)
                                        except Exception as e:
                                            print(f"主动关心输出处理失败: {e}")
                except Exception as e:
                    print(f"[agent_loop] 场景检测出错: {e}")
                    traceback.print_exc()
                finally:
                    # 无论如何都释放摄像头
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass

            # 3. 语音观测
            user_text = ""
            # 只有在用户未交互且达到语音检测间隔时才进行语音识别
            if not _global_state.is_interacting() and (now - last_stt_time) >= stt_interval:
                try:
                    decision_tracer.start_latency_tracking(NodeType.RAW_SENSOR, ModelType.LOCAL)
                    logger.info("开始语音识别...")
                    user_text = recognize_speech(
                        timeout=stt_timeout_sec,
                        phrase_time_limit=stt_phrase_time_limit_sec,
                    )
                    decision_tracer.end_latency_tracking(NodeType.RAW_SENSOR, ModelType.LOCAL)
                    last_stt_time = now  # 更新语音检测时间
                    
                    if user_text:
                        logger.info(f"检测到用户语音: {user_text}")
                    else:
                        logger.info("未检测到用户语音")
                except Exception as e:
                    user_text = ""
                    decision_tracer.end_latency_tracking(NodeType.RAW_SENSOR, ModelType.LOCAL)
                    logger.info(f"语音识别错误: {e}")

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

                # 5. 执行动作（使用多代理协调器）
                # 使用多代理协调器分析情感和规划动作
                logger.info("模型调度中...")
                
                decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                execution_result = multi_agent_coordinator.think(
                    vision_desc=vision_desc,
                    audio_text=user_text,
                    current_emotion="neutral",
                    context=f"视觉: {vision_desc} 语音: {user_text} 情绪: neutral"
                )
                decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
                emotion = execution_result.emotion
                action = execution_result.action
                confidence = 1.0  # 多代理协调器暂时不返回置信度
                decision_mode = "confident"  # 多代理协调器暂时不返回决策模式

                # 如果需要抑制回复
                if execution_result.should_suppress:
                    logger.info("[多代理] 抑制回复: 无有效输入")
                    _global_state.set_interacting(False)
                    continue


                
                # VLA 功能已禁用（vla_integration 模块已删除）
                # 继续使用多代理协调器进行决策
                
                decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                res, audio_path = get_response(
                    emotion,
                    user_text,
                    enable_tts=enable_tts,
                    vision_desc=vision_desc,
                )
                decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
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
                    # 使用多代理协调器分析场景和规划动作
                    decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                    execution_result = multi_agent_coordinator.think(
                        vision_desc=vision_desc,
                        audio_text="",
                        current_emotion="neutral",
                        context=f"视觉: {vision_desc} 语音: 无 情绪: neutral"
                    )
                    decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)
                    emotion = execution_result.emotion
                    action = execution_result.action
                    

                    
                    decision_tracer.start_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id)
                    res, audio_path = get_response(
                        emotion,
                        prompt_text,
                        enable_tts=enable_tts,
                        vision_desc=vision_desc,
                    )
                    decision_tracer.end_latency_tracking(NodeType.EMOTION_DETECTION, ModelType.CLOUD, user_id=user_id)

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
                    ros_manager.publish_action(proactive_msg)

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
                        "type": "status",
                        "last_speech_text": getattr(blackboard, "last_speech_text", ""),
                        "last_speech_time": getattr(blackboard, "last_speech_time", 0.0),
                        "user_presence": getattr(blackboard, "user_presence", False),
                        "robot_status": getattr(blackboard, "robot_status", "idle"),
                        "vision_desc": getattr(blackboard, "current_vision_desc", ""),
                        "vision_update_time": getattr(blackboard, "vision_update_time", 0.0),
                        "proactive_enabled": _global_state.is_proactive_enabled(),
                        "is_interacting": _global_state.is_interacting(),
                    }
                    ros_manager.publish_action(state_msg)

            # 9. 控制频率
            elapsed = time.time() - start_time
            if elapsed < control_interval:
                time.sleep(control_interval - elapsed)

    except Exception as e:
        print(f"[agent_loop] worker crashed: {e}")
        traceback.print_exc()
    finally:
        _global_state.set_interacting(False)


def start_agentic_main_loop(
    blackboard: Optional[Blackboard] = None,
    *,
    enable_tts: bool = False,
    on_proactive_output: Optional[Callable[[ProactiveCareResult], None]] = None,
    camera_index: int = 0,
    stt_timeout_sec: int = 5,
    stt_phrase_time_limit_sec: int = 8,
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
        # 等待线程结束，确保麦克风和其他资源被正确释放
        if self.thread.is_alive():
            logger.info("等待 agent_loop 线程结束...")
            self.thread.join(timeout=5.0)  # 最多等待 5 秒
            if self.thread.is_alive():
                logger.warning("agent_loop 线程未能在 5 秒内结束，可能存在资源泄露")
            else:
                logger.info("agent_loop 线程已成功结束")

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
>>>>>>> 483f2a96306b03f52efde3fc5895cf74d9121b3f
