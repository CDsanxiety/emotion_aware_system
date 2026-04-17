# llm_api.py
import os
import json
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tts import speak_sync
from utils import logger
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from robot_functions import get_robot_functions
from autogen_integration import plan_task

load_dotenv()

# ================== 配置区 ==================
MODEL = "qwen-turbo"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 无密钥时不实例化 OpenAI、不发起网络请求，走本地模拟
API_KEY = (os.getenv("LLM_API_KEY") or "").strip()
USE_MOCK_LLM = not bool(API_KEY)
if USE_MOCK_LLM:
    logger.info("LLM_API_KEY 未设置：使用本地模拟回复，不连接云端大模型。")

# ---------- Memory 缓存：最近 5 轮对话（每轮 user + assistant，共 10 条）----------
_SESSION_MAX_TURNS = 5
_session_memory: Deque[Dict[str, Any]] = deque(maxlen=_SESSION_MAX_TURNS * 2)
# 兼容旧名称（模块内引用统一走 _session_memory）
_message_history = _session_memory

_client: Optional[OpenAI] = None

_EMPATHY_VISION_SUFFIX = """
## 多模态共情（与 prompt.txt 一致，必须遵守）
- 用户消息中已分块标注 **VLM 视觉感知** 与 **STT 语音意图**；reply 必须融合两路信息，可借鉴「看到画面…但你还…」这类深度共情，勿只复述一路。
- 画面与语音矛盾时（如疲惫画面 +「我没事」）：温柔留白、先陪伴，参考系统主提示中的示例风格。
- **纯场景 / 杂乱室内 / 多物体 / 无清晰人脸**：仍须认真阅读 VLM 段落，给出有内容的社交回应与情绪支持；**禁止**空回复、禁止假装没看见 VLM。
- 无画面描述时以 STT 与情绪参考标签为主。
## 会话记忆（与 prompt.txt「会话记忆」一节一致）
- system 之后到当前 user 之前，是**完整**历史消息；若主人追问「还记得我说过什么」「为什么不开心」等，必须据实从该历史中归纳或引用，**不可臆造**未出现过的内容。
## 生活场景全链路输出
- 必须输出 **perception → decision → execution** 嵌套 JSON（与 prompt.txt 一致），且 execution.reply 为对用户说的唯一主话术；历史记录仍只存 reply 正文。
"""


def clear_memory() -> None:
    """清空 Memory 缓存（最近 5 轮对话上下文）。"""
    _session_memory.clear()


def _get_client() -> Optional[OpenAI]:
    """仅在配置了 API_KEY 时懒加载客户端；无密钥时永不创建。"""
    global _client
    if USE_MOCK_LLM:
        return None
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def _build_system_prompt(base: str) -> str:
    return base.strip() + "\n" + _EMPATHY_VISION_SUFFIX.strip()


def _build_user_content(
    emotion: str, user_text: str, vision_desc: Optional[str]
) -> str:
    """多模态融合输入：VLM 视觉描述 + STT 文本意图，一并写入本轮 user 块。"""
    vd = (vision_desc or "").strip()
    stt = (user_text or "").strip()
    vision_block = vd if vd else "（本帧暂无有效 VLM 画面描述，请结合 STT 与情绪标签推理）"
    stt_block = stt if stt else "（暂无 STT 语音识别内容）"
    return f"""【多模态融合 · 本轮输入】
请将下列两路信息综合理解后，再输出系统要求的 JSON：

【1）VLM 视觉感知描述】（来自摄像头画面理解）
{vision_block}

【2）STT 语音 / 文本意图】（来自麦克风识别与用户表述）
{stt_block}

【辅助 · 情绪参考标签】（传统表情标签或占位，仅作补充）
{emotion}

请严格按照系统提示词输出**含 perception、decision、execution 的单一 JSON 对象**（不要 markdown）。"""


def _history_as_messages() -> List[Dict[str, Any]]:
    return list(_session_memory)


def _append_turn(user_content: str, assistant_reply: str) -> None:
    """写入 Memory 缓存（与云端共用同一条 deque，全程保留直至 clear_memory）。"""
    _session_memory.append({"role": "user", "content": user_content})
    _session_memory.append({"role": "assistant", "content": assistant_reply})


def _infer_risk_level(vision_desc: str, user_text: str) -> str:
    """粗粒度风险：火/烟/漏电/利器等 → 高；昏暗/湿滑/杂物 → 中；否则低。"""
    blob = f"{vision_desc}\n{user_text}"
    if any(
        k in blob
        for k in (
            "火", "明火", "浓烟", "漏电", "煤气", "燃气泄漏", "摔倒", "昏迷",
            "刀口", "流血", "溺水", "玻璃碎", "爆炸",
        )
    ):
        return "高"
    if any(
        k in blob
        for k in ("地滑", "湿滑", "很暗", "偏暗", "看不清脚", "杂物挡", "尖锐", "电线裸露")
    ):
        return "中"
    return "低"


def _ensure_full_chain_payload(
    result: Dict[str, Any],
    emotion: str,
    vision_desc: str,
    user_text: str,
) -> Dict[str, Any]:
    """
    感知-决策-执行：补齐 perception / decision / execution 三块，并与顶层 emotion/action/reply 对齐。
    若模型已输出嵌套结构，则只做校验与缺省填充，避免跳步矛盾。
    """
    vd = (vision_desc or "").strip()
    ut = (user_text or "").strip()

    ex = result.get("execution")
    if isinstance(ex, dict):
        result["execution"] = {
            "emotion": ex.get("emotion", result.get("emotion", emotion)),
            "action": ex.get("action", result.get("action", "无动作")),
            "reply": ex.get("reply", result.get("reply", "")),
            "strategy": ex.get("strategy", "先共情与安全确认，再执行家居动作"),
        }
    else:
        result["execution"] = {
            "emotion": result.get("emotion", emotion),
            "action": result.get("action", "无动作"),
            "reply": result.get("reply", ""),
            "strategy": result.get("strategy", "陪伴式应答与可选家居动作"),
        }

    perc = result.get("perception")
    if not isinstance(perc, dict):
        perc = {}
    scene = (perc.get("scene") or "").strip() or (vd[:600] if vd else "（本帧无有效画面描述）")
    state = (perc.get("state") or "").strip()
    if not state:
        state = (
            "用户有语音输入，可与画面交叉验证。"
            if ut
            else "暂无语音；可依据画面先做环境与安全判断，并鼓励用户开口。"
        )
    risk = perc.get("risk")
    if risk not in ("无", "低", "中", "高"):
        risk = _infer_risk_level(vd, ut)
    emo_sig = (perc.get("emotion_signal") or "").strip() or (ut[:400] if ut else "（无语音文本）")
    result["perception"] = {
        "scene": scene[:600],
        "state": state[:300],
        "risk": risk,
        "emotion_signal": emo_sig[:400],
    }

    dec = result.get("decision")
    if not isinstance(dec, dict):
        dec = {}
    judgment = (dec.get("judgment") or "").strip()
    if not judgment:
        ex0 = result["execution"]
        judgment = (
            f"在风险「{risk}」前提下，综合 VLM 与 STT；"
            f"输出情绪标签「{ex0['emotion']}」与家居动作「{ex0['action']}」，保证与 reply 一致、不跳步。"
        )
    priority = dec.get("priority") or ""
    if priority not in ("安全优先", "情绪优先", "陪伴与环境调节"):
        priority = (
            "安全优先"
            if risk == "高"
            else ("情绪优先" if any(x in ut for x in ("怕", "疼", "救", "帮帮我")) else "陪伴与环境调节")
        )
    result["decision"] = {"judgment": judgment[:500], "priority": priority}
    return result


def _sync_top_level_from_execution(result: Dict[str, Any]) -> None:
    ex = result.get("execution")
    if isinstance(ex, dict):
        if ex.get("emotion"):
            result["emotion"] = ex["emotion"]
        if ex.get("action"):
            result["action"] = ex["action"]
        if ex.get("reply"):
            result["reply"] = ex["reply"]


def finalize_pipeline_result(
    result: Dict[str, Any],
    emotion: str,
    vision_desc: str,
    user_text: str,
) -> Dict[str, Any]:
    """统一收口：全链路字段 + 顶层兼容字段。"""
    _ensure_full_chain_payload(result, emotion, vision_desc, user_text)
    _sync_top_level_from_execution(result)
    return result


# ---------- 本地模拟：从历史块中解析 STT / VLM（与 _build_user_content 结构一致）----------
_STT_TAG = "【2）STT 语音 / 文本意图】"
_VLM_TAG = "【1）VLM 视觉感知描述】"


def _parse_stt_from_session_user_block(content: str) -> str:
    if _STT_TAG not in content:
        return ""
    after = content.split(_STT_TAG, 1)[1].strip()
    if "【辅助" in after:
        after = after.split("【辅助", 1)[0].strip()
    return after.strip()


def _parse_vlm_from_session_user_block(content: str) -> str:
    if _VLM_TAG not in content:
        return ""
    after = content.split(_VLM_TAG, 1)[1].strip()
    if "【2）" in after:
        after = after.split("【2）", 1)[0].strip()
    return after.strip()


_UPSET_MARKERS = (
    "不开心", "难过", "失恋", "分手", "哭", "郁闷", "丧", "烦死了", "压力大", "崩溃",
    "考砸", "没考好", "挂科", "好累", "委屈", "加班", "吵架", "被骂", "失落", "焦虑",
    "害怕", "担心", "烦",
)
_VLM_UPSET_HINTS = ("疲惫", "低落", "难过", "愁", "沮丧", "憔悴", "倦", "无神", "暗淡")


def _iter_user_blocks_chronological() -> List[tuple[int, str]]:
    """按时间顺序列出历史中的 user 块（不含尚未入队的当前轮）。"""
    hist = list(_session_memory)
    out: List[tuple[int, str]] = []
    for i in range(0, len(hist), 2):
        if i < len(hist) and hist[i].get("role") == "user":
            out.append((i, hist[i].get("content") or ""))
    return out


def _all_stt_snippets_chronological() -> List[str]:
    """从完整会话中提取每轮 STT 非空片段（按轮次先后）。"""
    lines: List[str] = []
    for _, uc in _iter_user_blocks_chronological():
        stt = _parse_stt_from_session_user_block(uc)
        if not stt or stt.startswith("（暂无"):
            continue
        one = stt.replace("\n", " ").strip()
        if len(one) > 220:
            one = one[:220] + "…"
        lines.append(one)
    return lines


def _collect_all_distress_snippets() -> List[str]:
    """从完整会话中收集所有带负面/压力线索的 STT 或 VLM 摘要（按时间先后）。"""
    out: List[str] = []
    for _, uc in _iter_user_blocks_chronological():
        stt = _parse_stt_from_session_user_block(uc)
        vlm = _parse_vlm_from_session_user_block(uc)
        if stt and not stt.startswith("（暂无"):
            if any(m in stt for m in _UPSET_MARKERS) or any(
                m in stt for m in ("分手", "失恋", "被骂", "失业", "被裁")
            ):
                s = stt.replace("\n", " ").strip()
                out.append(s[:180] + ("…" if len(s) > 180 else ""))
                continue
        if vlm and not vlm.startswith("（本帧暂无"):
            if any(m in vlm for m in _VLM_UPSET_HINTS):
                v = vlm.replace("\n", " ").strip()
                out.append(f"画面线索：{v[:130]}" + ("…" if len(v) > 130 else ""))
    return out


def _wants_transcript_recall(t: str) -> bool:
    """是否在追问「之前说过什么」类，需要按轮次复述 / 归纳 STT。"""
    keys = (
        "之前说过什么", "说过什么", "说过啥", "说了啥", "讲过什么", "聊过什么",
        "我前面说", "我刚才说", "我前面说过", "重复一遍", "总结我说", "你都听见",
        "你听到我说", "还记得我说", "记得我说", "我跟你说过",
    )
    if any(k in t for k in keys):
        return True
    return ("记得" in t or "印象" in t) and ("说过" in t or "说了" in t)


def _is_distress_cause_recall(t: str) -> bool:
    """是否在追问「为什么不开心 / 原因」等（非全文复述）。"""
    t = (t or "").strip()
    if "为什么不开心" in t or "为啥不开心" in t or "为什么不高兴" in t:
        return True
    recall = any(
        x in t
        for x in ("还记得", "记得吗", "你还记得", "记得我", "有没有记得", "有没有印象", "你记得")
    )
    gist = any(
        x in t
        for x in ("上次", "之前", "什么原因", "因为什么", "我说过", "什么事", "不开心", "难过")
    )
    return recall and gist


def _is_any_recall_question(t: str) -> bool:
    """凡需从历史中取证回答的追问，走统一记忆分支（本地模拟）。"""
    t = (t or "").strip()
    if len(t) < 4:
        return False
    if _wants_transcript_recall(t):
        return True
    return _is_distress_cause_recall(t)


def _mock_reply_memory_recall(prefix: str, user_text: str) -> dict:
    """据完整 deque 历史作答：全文复述类 vs 情绪原因类。"""
    t = (user_text or "").strip()

    if _wants_transcript_recall(t):
        parts = _all_stt_snippets_chronological()
        if not parts:
            reply = (
                f"{prefix}我把咱俩从上次清空记忆以来的记录都翻了一遍，"
                f"暂时没有对得上号的原话片段。你愿意再用自己的话说一遍吗？我这次一定跟紧。"
            )
        else:
            joined = "；".join(parts)
            if len(joined) > 480:
                joined = joined[:480] + "…（更早的也还在上下文里，若要抠某一句可再点名问我）"
            reply = (
                f"{prefix}我都记着呢。按时间顺序，你前面陆续说过这些：{joined}。"
                f"若有哪句我复述得不够准，你直接纠正我就好。"
            )
        return {"emotion": "平静", "action": "无动作", "reply": reply}

    snippets = _collect_all_distress_snippets()
    if snippets:
        joined = "；".join(snippets)
        if len(joined) > 480:
            joined = joined[:480] + "…"
        reply = (
            f"{prefix}记得呀，这些我一直放在心上：{joined}。"
            f"现在想继续往下聊，还是想先安静待一会儿？我都陪着你。"
        )
    else:
        reply = (
            f"{prefix}我把完整对话又捋了一遍，没有对上特别清晰的「不开心原因」原句或画面线索。"
            f"要不你再跟我说一点点？我不催，只听你说。"
        )
    return {"emotion": "平静", "action": "无动作", "reply": reply}


def _mock_reply_distress(text: str, vd: str, prefix: str) -> dict:
    """难过、累、压力、考砸等：先安慰与共情，再轻量回应；可结合 VLM。"""
    emotion, action = "难过", "调节灯光"
    vlm_prefix = ""
    if vd and len(vd) > 6 and any(h in vd for h in _VLM_UPSET_HINTS):
        vlm_prefix = "看你这边的画面，我也能感觉到一点点沉。先不想别的，"

    if any(k in text for k in ("考砸", "没考好", "挂科", "考差了", "不及格", "考崩")):
        core = (
            "先抱抱你…分数真的不能定义你这个人。"
            "一次没发挥好，心里委屈太正常了。你愿意的话跟我说说哪一门最扎心，我们慢慢捋，不着急。"
        )
    elif "压力" in text or "绷" in text or "扛不住" in text:
        core = (
            "听到你说压力大，我先替你松一小口气——能撑到现在已经很不容易了。"
            "我不催你把事情都理清，就想让你知道：这种紧绷我懂，你不用说得很完整我也在这儿。"
        )
    elif any(k in text for k in ("好累", "累死了", "累坏了", "熬不住")):
        core = (
            "你说累的时候，我是真心疼。能撑到这里已经很了不起了，先把肩膀放下来半分钟好不好？"
            "灯光我帮你调柔一点，你想说话就说，想安静我就安静陪着。"
        )
    elif any(k in text for k in ("难过", "哭", "委屈", "郁闷", "丧")):
        core = (
            "难过的时候愿意说出来，其实已经很勇敢了。"
            "我不评价、不说教，就想陪着你把这一小段路慢慢走过去。想哭也没关系，我都在。"
        )
    else:
        core = (
            "听起来你心里不太轻松，我先把语气放软一点。"
            "不管具体是什么事，你都不是一个人在扛；想细说可以，想含糊一点也可以，我听着。"
        )

    return {
        "emotion": emotion,
        "action": action,
        "reply": f"{prefix}{vlm_prefix}{core}",
    }


def _mock_llm_result(emotion: str, user_text: str, vision_desc: str) -> dict:
    """
    无 API_KEY 时的本地模拟 JSON，不访问网络。
    仍与云端约定相同字段，便于 UI / TTS 复用。
    """
    text = (user_text or "").strip()
    vd = (vision_desc or "").strip()
    turn = len(_session_memory) // 2 + 1
    prefix = f"（本地模拟·第{turn}轮，未连接云端）"

    # 1）记忆追问：据完整 deque 历史归纳 / 复述，不丢早期轮次
    if _is_any_recall_question(text):
        return _mock_reply_memory_recall(prefix, text)

    # 2）负面情绪：先安慰、共情，再回应（不抢在记忆问句之前）
    _tired = "累" in text and "不累" not in text and "不太累" not in text
    if (
        any(k in text for k in _UPSET_MARKERS)
        or any(k in text for k in ("分手", "失恋", "被骂", "失业", "被裁"))
        or _tired
    ):
        return _mock_reply_distress(text, vd, prefix)

    # 2.5）VLM 指向杂乱房间 / 多物体 / 纯场景：给结构化环境共情，避免「只有画面没有回应」
    _clutter_vlm = len(vd) > 45 and any(
        k in vd
        for k in (
            "杂乱", "偏乱", "视觉信息很满", "物品较多", "轮廓与细节边缘较多",
            "陈设较满", "堆叠", "多物体", "本地统计参考", "未检测到稳定人脸",
        )
    )
    if _clutter_vlm:
        return {
            "emotion": "平静",
            "action": "调节灯光",
            "reply": (
                f"{prefix}我从画面里读出：东西挺多、信息也挺满的，这种空间真的容易让人脑壳嗡嗡的。"
                f"先不急着收拾啦，把灯光调柔一点，你跟我说说此刻更想吐槽还是想安静会儿，我都接得住。"
            ),
        }

    if any(k in text for k in ("开心", "棒", "太好了", "哈哈", "中奖", "喜欢")):
        return {
            "emotion": "开心",
            "action": "播放音乐",
            "reply": (
                f"{prefix}哈哈，光听你这么说我都跟着开心起来了。"
                f"要不要放首轻快的歌，把好心情再拉长一点点？"
            ),
        }
    if any(k in text for k in ("气死了", "气人", "好气", "愤怒", "火大", "气炸了", "讨厌死了")):
        return {
            "emotion": "愤怒",
            "action": "播放音乐",
            "reply": (
                f"{prefix}先深呼吸一下，生气的时候身体也会跟着绷。"
                f"我给你放点舒缓的音乐，你慢慢说，我都在听。"
            ),
        }
    if any(k in text for k in ("怕", "吓", "紧张", "慌")):
        return {
            "emotion": "害怕",
            "action": "调节灯光",
            "reply": (
                f"{prefix}别怕，我把灯调亮一点，家里亮堂些心里也会稳一点。"
                f"你慢慢讲，不着急，我哪儿也不去。"
            ),
        }
    if vd and ("疲惫" in vd or "累" in vd or "低落" in vd or "倦" in vd):
        if any(k in text for k in ("没事", "还行", "坚持", "撑", "熬过去", "交掉", "做完")):
            return {
                "emotion": "平静",
                "action": "调节灯光",
                "reply": (
                    f"{prefix}看到你虽然累但还在坚持，真为你骄傲。"
                    f"我把灯光调柔一点，你慢慢歇口气，不用逞强说很多也行。"
                ),
            }
        return {
            "emotion": "难过",
            "action": "调节灯光",
            "reply": (
                f"{prefix}从画面里能感觉到你有点倦，情绪也不太高。"
                f"先歇一小会儿吧，不想说话也没关系，我陪你坐着。"
            ),
        }
    snippet = text[:36] + ("…" if len(text) > 36 else "")
    return {
        "emotion": "平静",
        "action": "无动作",
        "reply": (
            f"{prefix}我在呢～{'「' + snippet + '」我也听到了。' if snippet else ''}"
            f"{'想往深聊还是随便扯两句都行，我跟着你的节奏来。' if snippet else '随时想聊都可以跟我说，我跟着你的节奏来。'}"
        ),
    }


def call_llm(
    emotion: str,
    user_text: str,
    vision_desc: str = "",
    prompt_file: str = "prompt.txt",
) -> dict:
    """
    核心函数：表情/情绪参考 + 用户语音文本 + 可选视觉描述 → JSON
    有 API_KEY 时走云端；否则本地模拟。成功或模拟路径均写入同一条 Memory（deque，完整会话直至 clear_memory）。
    """
    user_content = _build_user_content(emotion, user_text, vision_desc)

    if USE_MOCK_LLM:
        result = _mock_llm_result(emotion, user_text, vision_desc)
        if "emotion" not in result:
            result["emotion"] = emotion
        if "action" not in result:
            result["action"] = "无动作"
        if "reply" not in result:
            result["reply"] = "嗯嗯，我在这里陪着你呢～"
        finalize_pipeline_result(result, emotion, vision_desc, user_text)
        assistant_text = result.get("reply", "").strip() or json.dumps(
            result, ensure_ascii=False
        )
        _append_turn(user_content, assistant_text)
        logger.debug("LLM 本地模拟回复已生成并写入记忆")
        return result

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = _build_system_prompt(f.read())
    except FileNotFoundError:
        logger.error(f"Prompt 文件不存在: {prompt_file}")
        system_prompt = _build_system_prompt("你是一个温柔、善于共情的智能家居伴侣「暖暖」。")

    system_msg: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": system_prompt,
    }
    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_content,
    }

    messages: List[Any] = [system_msg] + _history_as_messages() + [user_msg]

    client = _get_client()
    if client is None:
        result = _mock_llm_result(emotion, user_text, vision_desc)
        finalize_pipeline_result(result, emotion, vision_desc, user_text)
        assistant_text = result.get("reply", "").strip() or json.dumps(
            result, ensure_ascii=False
        )
        _append_turn(user_content, assistant_text)
        return result

    try:
        _to = float(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "8"))
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.65,
            max_tokens=420,
            timeout=_to,
        )
        raw_reply = (response.choices[0].message.content or "").strip()
        logger.debug(f"LLM 原始回复: {raw_reply}")

        if raw_reply.startswith("```json"):
            raw_reply = raw_reply[7:]
        if raw_reply.startswith("```"):
            raw_reply = raw_reply[3:]
        if raw_reply.endswith("```"):
            raw_reply = raw_reply[:-3]
        raw_reply = raw_reply.strip()

        result = json.loads(raw_reply)

        if "emotion" not in result:
            result["emotion"] = emotion
        if "action" not in result:
            result["action"] = "无动作"
        if "reply" not in result:
            result["reply"] = "嗯嗯，我在这里陪着你呢～"

        finalize_pipeline_result(result, emotion, vision_desc, user_text)
        assistant_text = result.get("reply", "").strip() or json.dumps(
            result, ensure_ascii=False
        )
        _append_turn(user_content, assistant_text)

        return result

    except json.JSONDecodeError:
        logger.warning("JSON 解析失败")
        result = {
            "emotion": emotion,
            "action": "无动作",
            "reply": "嗯嗯，我在这里陪着你呢～",
        }
        finalize_pipeline_result(result, emotion, vision_desc, user_text)
        return result
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        result = {
            "emotion": emotion,
            "action": "无动作",
            "reply": "稍微等一下哦，我在整理思绪～",
        }
        finalize_pipeline_result(result, emotion, vision_desc, user_text)
        return result


def get_response(
    face_emotion: str,
    voice_text: str,
    enable_tts: bool = True,
    vision_desc: str = "",
) -> tuple:
    """
    统一接口：表情参考、语音文本、可选 vision_desc（摄像头自然语言描述）；
    内部带完整会话记忆（与云端共用 deque，直至 clear_memory）。
    返回格式: (result_dict, audio_path)
    """
    if not voice_text or voice_text.strip() == "":
        result = {
            "emotion": face_emotion,
            "action": "无动作",
            "reply": "我在听呢，你想说什么呀～",
        }
        finalize_pipeline_result(result, face_emotion, vision_desc, "")
    else:
        result = call_llm(face_emotion, voice_text, vision_desc=vision_desc)

    reply_text = result.get("reply", "")
    mode = "mock" if USE_MOCK_LLM else "remote"
    logger.info(
        f"[{mode}] 表情: {face_emotion} | 动作: {result.get('action')} | 回复: {reply_text[:30]}..."
    )

    # 执行机器人动作
    rf = get_robot_functions()
    action_results = rf.execute_from_llm_result(result)
    for ar in action_results:
        logger.info(f"执行动作: {ar.message}")

    audio_path = None
    if enable_tts and reply_text:
        audio_path = speak_sync(reply_text)

    return result, audio_path


if __name__ == "__main__":
    print("测试 LLM + 会话记忆 + vision_desc ...")
    clear_memory()
    r1, _ = get_response("neutral", "今天有点累", enable_tts=False, vision_desc="人物靠在沙发上，神情略显疲惫。")
    print(json.dumps(r1, ensure_ascii=False, indent=2))
    r2, _ = get_response("neutral", "嗯，但还是想和你聊两句", enable_tts=False, vision_desc="同上。")
    print(json.dumps(r2, ensure_ascii=False, indent=2))


def get_task_plan(task: str) -> List[Dict[str, Any]]:
    """获取任务规划
    
    Args:
        task: 任务描述
    
    Returns:
        任务规划步骤列表
    """
    try:
        plan = plan_task(task)
        logger.info(f"任务规划成功: {task}")
        return plan
    except Exception as e:
        logger.error(f"任务规划失败: {e}")
        return [{
            "description": "默认步骤",
            "action": "无动作",
            "expected": "完成任务"
        }]


def execute_task_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """执行任务规划
    
    Args:
        plan: 任务规划步骤列表
    
    Returns:
        执行结果列表
    """
    results = []
    rf = get_robot_functions()
    
    for step in plan:
        action = step.get("action", "无动作")
        result = rf.execute_action(action)
        results.append({
            "step": step,
            "execution_result": {
                "success": result.success,
                "message": result.message,
                "action_type": result.action_type.value
            }
        })
    
    return results
