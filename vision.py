# vision.py
"""
摄像头画面 → 阿里云百炼 qwen-vl-plus 场景理解（含杂乱室内、多物体、非人物）；
失败时用 OpenCV 启发式 + 可选 FER，保证 description 非空、便于 LLM 做社交共情。
"""
import base64
import os
from typing import Any, Dict, List, Optional

import cv2
from dotenv import load_dotenv
from fer.fer import FER
from openai import OpenAI

from utils import logger

load_dotenv()

# ================== 配置 ==================
VL_MODEL = "qwen-vl-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.getenv("LLM_API_KEY")
VL_TIMEOUT_SEC = 18.0

VL_SYSTEM_PROMPT = (
    "你是专业的室内与场景视觉理解助手。必须用中文输出，且**禁止输出空内容**。"
    "客观描述你看到的画面；不确定的用「可能」「似乎」表述，不要编造不存在的具体品牌或文字。"
)

VL_USER_PROMPT = """请用**一段连贯中文**描述这张照片（约 4～8 句），并自然覆盖下面维度（可用括号提示，但不要分条编号罗列成清单体）：

1）**场景类型**：室内/室外/走廊/厨房/卧室/客厅/办公室等，或混合。
2）**人物**：是否有人、大致人数、姿态与表情（若无人或背影、侧脸、被遮挡，请明确写「未出现清晰人脸」并继续描述环境）。
3）**主要物体与布局**：家具、电器、门窗、墙面、地面、杂物、堆叠物、桌面物品等，尽量点名可见大类。
4）**整洁与杂乱**：整体偏整洁、一般、偏杂乱或堆叠明显；若背景复杂、多物体同框，请说明「视觉信息较满」但仍要概括。
5）**光线与氛围**：明暗、冷暖、是否像生活/工作场景；给人偏放松、偏压抑或中性的感受（基于画面，勿过度推断心理病史）。

若画面完全无法辨认，请说明「画面过暗或过糊」并仍给出一句安全的环境猜测。不要输出 JSON，不要只回复「好的」等无信息短句。"""

VL_USER_PROMPT_RETRY = (
    "上一轮流式输出为空，请**务必**用中文写出完整一段（至少 3 句）："
    "室内还是室外；有无人物；主要家具与杂物；偏整洁还是偏乱；光线与整体氛围。"
    "禁止空回复。"
)

# 本地 FER（降级辅助）
_fer_detector: Optional[FER] = None
_client: Optional[OpenAI] = None

_EMOTION_NARRATIVE = {
    "happy": "整体偏开心、放松",
    "sad": "整体偏难过、低落",
    "angry": "整体偏愤怒、紧绷",
    "surprise": "整体偏惊讶",
    "neutral": "整体较平静、中性",
    "fear": "整体偏紧张、不安",
    "disgust": "整体偏反感、不适",
}


def _get_fer() -> FER:
    global _fer_detector
    if _fer_detector is None:
        _fer_detector = FER(mtcnn=True)
    return _fer_detector


def _get_client() -> Optional[OpenAI]:
    global _client
    if not API_KEY:
        return None
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def _make_result(success: bool, description: str, is_fallback: bool) -> Dict[str, Any]:
    text = (description or "").strip()
    if not text:
        text = "（兜底）已收到画面，但暂时无法生成有效文字描述；请用户简要说明场景或再拍一张。"
    return {
        "success": success,
        "description": text,
        "is_fallback": is_fallback,
    }


def _heuristic_scene_hint(frame) -> str:
    """
    纯本地像素级线索：杂乱度/明暗等，用于无云端或无人脸时的**非空**场景支撑。
    不替代 VLM 识物，只避免「完全不理解」。
    """
    try:
        h, w = frame.shape[:2]
        if h < 2 or w < 2:
            return "画面尺寸异常，难以分析。"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 35, 110)
        edge_density = cv2.countNonZero(edges) / float(h * w)
        brightness = float(cv2.mean(gray)[0])

        if edge_density > 0.14:
            clutter = "轮廓与细节边缘较多，整体**视觉信息很满**，常见于物品较多或背景复杂的房间。"
        elif edge_density > 0.08:
            clutter = "可见陈设与物体数量中等，环境介于**较整洁与略杂乱**之间。"
        else:
            clutter = "轮廓相对简洁，画面可能**较整洁、留白多或景深较浅**。"

        if brightness < 70:
            light = "整体**偏暗**，细节可能不易分辨。"
        elif brightness > 200:
            light = "整体**偏亮或略过曝**，部分区域细节可能被冲淡。"
        else:
            light = "亮度中等，主体与环境大致可辨。"

        return (
            f"（云端视觉暂不可用时的本地统计参考，非逐物体识别）"
            f"画面约 {w}×{h} 像素；{light}{clutter}"
            f"若用户分享的是生活空间，可从「整理压力、想要更舒服的环境」等角度温柔回应，避免指责。"
        )
    except Exception as e:
        logger.debug(f"heuristic scene hint failed: {e}")
        return "（本地兜底）已收到一帧图像，但像素统计失败；可请用户口头补充场景。"


def _fer_face_line(frame) -> Optional[str]:
    """若检测到人脸则返回一行表情叙述；否则 None。"""
    try:
        det = _get_fer()
        emotions = det.detect_emotions(frame)
        if not emotions:
            return None
        emo_dict = emotions[0]["emotions"]
        dominant = max(emo_dict, key=emo_dict.get)
        score = float(emo_dict.get(dominant, 0.0))
        narrative = _EMOTION_NARRATIVE.get(dominant, "情绪状态待进一步观察")
        return (
            f"本地人脸情绪参考：主导倾向为「{narrative}」（约 {score:.0%} 置信度），"
            f"可与环境描述一并交给对话模型做共情。"
        )
    except Exception as e:
        logger.warning(f"FER 辅助失败: {e}")
        return None


def _fallback_visual_description(frame) -> str:
    """启发式 + 可选人脸线，保证非空、可喂给 LLM。"""
    parts: List[str] = [_heuristic_scene_hint(frame)]
    fer_line = _fer_face_line(frame)
    if fer_line:
        parts.append(fer_line)
    else:
        parts.append(
            "未检测到稳定人脸：请对话侧按「纯室内/场景」理解，不要强依赖表情；"
            "可邀请用户调整角度或说明心情。"
        )
    return "\n".join(parts)


def _frame_to_data_url(frame) -> Optional[str]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok or buf is None:
        return None
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _call_qwen_vl(data_url: str, user_prompt: str) -> str:
    client = _get_client()
    if not client:
        return ""
    resp = client.chat.completions.create(
        model=VL_MODEL,
        messages=[
            {"role": "system", "content": VL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        temperature=0.25,
        max_tokens=512,
        timeout=VL_TIMEOUT_SEC,
    )
    return (resp.choices[0].message.content or "").strip()


def process_image(frame) -> Dict[str, Any]:
    """
    输入：Gradio / OpenCV 图像帧（numpy BGR）。
    输出：success / description / is_fallback；description 始终非空以利 LLM 联动。
    """
    if frame is None:
        return _make_result(False, "未获取到画面，无法描述场景。", True)

    data_url = _frame_to_data_url(frame)
    if not data_url:
        return _make_result(False, _fallback_visual_description(frame), True)

    client = _get_client()
    if client is None:
        logger.warning("未配置 LLM_API_KEY，视觉降级为本地启发式 + FER")
        return _make_result(False, _fallback_visual_description(frame), True)

    try:
        raw = _call_qwen_vl(data_url, VL_USER_PROMPT)
        if not raw:
            raw = _call_qwen_vl(data_url, VL_USER_PROMPT_RETRY)
        if raw:
            return _make_result(True, raw, False)
        logger.warning("qwen-vl-plus 返回空内容，降级本地 + 可选 FER")
        return _make_result(False, _fallback_visual_description(frame), True)
    except Exception as e:
        logger.warning(f"qwen-vl-plus 不可用，降级: {e}")
        return _make_result(False, _fallback_visual_description(frame), True)


def no_input_vision() -> Dict[str, Any]:
    return _make_result(False, "未获取到画面。", True)


def get_current_emotion() -> Dict[str, Any]:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return _make_result(False, "无法读取摄像头画面。", True)
    return process_image(frame)


if __name__ == "__main__":
    print("测试视觉：qwen-vl-plus（Ctrl+C 退出）")
    try:
        while True:
            out = get_current_emotion()
            print(
                f"[success={out['success']} fallback={out['is_fallback']}] "
                f"{out['description'][:120]}...",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\n结束")
