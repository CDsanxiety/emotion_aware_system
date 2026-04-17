"""pad_model.py
PAD 情感模型实现：Pleasure-Arousal-Dominance 三维情感计算
用于将表情标签、语音情感等映射到统一的 PAD 空间，并与 llm_api 的 emotion 标签互转。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

EMOJI_PAD: Dict[str, Tuple[float, float, float]] = {
    "happy": (0.85, 0.45, 0.75),
    "sad": (-0.65, -0.25, -0.45),
    "angry": (-0.75, 0.75, 0.45),
    "fear": (-0.70, 0.30, -0.65),
    "surprise": (0.45, 0.80, 0.30),
    "disgust": (-0.60, 0.35, -0.25),
    "neutral": (0.0, 0.0, 0.0),
    "平静": (0.0, -0.2, 0.1),
    "开心": (0.85, 0.45, 0.75),
    "高兴": (0.80, 0.50, 0.70),
    "快乐": (0.90, 0.48, 0.78),
    "难过": (-0.65, -0.25, -0.45),
    "悲伤": (-0.70, -0.30, -0.50),
    "愤怒": (-0.75, 0.75, 0.45),
    "生气": (-0.80, 0.70, 0.40),
    "害怕": (-0.70, 0.30, -0.65),
    "恐惧": (-0.75, 0.35, -0.70),
    "惊讶": (0.45, 0.80, 0.30),
    "厌恶": (-0.60, 0.35, -0.25),
    "郁闷": (-0.50, -0.30, -0.30),
    "焦虑": (-0.40, 0.50, -0.40),
    "担心": (-0.35, 0.25, -0.50),
    "平静": (0.0, -0.2, 0.1),
    "放松": (0.40, -0.40, 0.30),
    "满足": (0.60, -0.20, 0.50),
    "欣慰": (0.55, -0.10, 0.55),
    "温馨": (0.50, -0.10, 0.45),
    "感动": (0.30, 0.20, 0.40),
    "委屈": (-0.45, -0.10, -0.55),
    "失落": (-0.50, -0.15, -0.40),
    "孤独": (-0.55, -0.10, -0.60),
    "疲惫": (-0.45, -0.50, -0.20),
    "累": (-0.40, -0.45, -0.15),
    "压力大": (-0.50, 0.40, -0.35),
    "害羞": (0.20, 0.30, -0.20),
    "尴尬": (-0.20, 0.10, -0.40),
    "同情": (0.10, -0.10, 0.10),
    "期待": (0.50, 0.60, 0.40),
    "好奇": (0.40, 0.55, 0.35),
    "无聊": (-0.20, -0.50, 0.10),
    "困惑": (-0.10, 0.20, -0.10),
}

PAD_LABELS: List[Tuple[str, float, float, float]] = [
    ("快乐", 0.85, 0.45, 0.75),
    ("兴奋", 0.75, 0.85, 0.60),
    ("放松", 0.40, -0.40, 0.30),
    ("满足", 0.60, -0.20, 0.50),
    ("平静", 0.0, -0.2, 0.1),
    ("郁闷", -0.50, -0.30, -0.30),
    ("悲伤", -0.70, -0.30, -0.50),
    ("害怕", -0.70, 0.30, -0.65),
    ("愤怒", -0.75, 0.75, 0.45),
    ("惊讶", 0.45, 0.80, 0.30),
    ("厌恶", -0.60, 0.35, -0.25),
    ("无聊", -0.20, -0.50, 0.10),
    ("焦虑", -0.40, 0.50, -0.40),
    ("疲惫", -0.45, -0.50, -0.20),
]


@dataclass
class PAD:
    pleasure: float
    arousal: float
    dominance: float

    def __post_init__(self):
        self.pleasure = max(-1.0, min(1.0, self.pleasure))
        self.arousal = max(-1.0, min(1.0, self.arousal))
        self.dominance = max(-1.0, min(1.0, self.dominance))

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.pleasure, self.arousal, self.dominance)

    def distance_to(self, other: "PAD") -> float:
        return math.sqrt(
            (self.pleasure - other.pleasure) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )

    def to_label(self) -> str:
        best_label = "平静"
        best_dist = float("inf")
        for label, p, a, d in PAD_LABELS:
            dist = math.sqrt(
                (self.pleasure - p) ** 2
                + (self.arousal - a) ** 2
                + (self.dominance - d) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label

    def to_emotion_category(self) -> str:
        p, a, d = self.pleasure, self.arousal, self.dominance
        if p > 0.5 and a > 0.3:
            return "开心"
        elif p > 0.3 and a < -0.2:
            return "平静"
        elif p < -0.4 and a > 0.3:
            return "愤怒"
        elif p < -0.4 and a < -0.2:
            return "悲伤"
        elif p < -0.3 and a > 0.0:
            return "害怕"
        elif abs(p) < 0.3 and a > 0.5:
            return "惊讶"
        elif p < -0.3 and abs(a) < 0.3:
            return "郁闷"
        return "neutral"

    def to_dict(self) -> Dict[str, float]:
        return {"pleasure": self.pleasure, "arousal": self.arousal, "dominance": self.dominance}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PAD":
        return cls(
            pleasure=float(data.get("pleasure", 0.0)),
            arousal=float(data.get("arousal", 0.0)),
            dominance=float(data.get("dominance", 0.0)),
        )

    @classmethod
    def from_label(cls, label: str) -> "PAD":
        label_lower = label.lower()
        if label_lower in EMOJI_PAD:
            p, a, d = EMOJI_PAD[label_lower]
            return cls(p, a, d)
        for key, (p, a, d) in EMOJI_PAD.items():
            if key in label_lower or label_lower in key:
                return cls(p, a, d)
        return cls(0.0, 0.0, 0.0)


def emotion_to_pad(emotion: str) -> PAD:
    return PAD.from_label(emotion)


def text_to_pad(text: str) -> PAD:
    text_lower = text.lower()
    scores = {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}
    count = 0
    positive_words = ["开心", "快乐", "高兴", "棒", "好", "喜欢", "爱", "幸福", "满足", "哈哈", "太好了", "完美"]
    negative_words = ["难过", "悲伤", "痛苦", "伤心", "哭", "累", "疲惫", "压力", "焦虑", "害怕", "恐惧", "生气", "愤怒", "气", "讨厌"]
    high_arousal_words = ["兴奋", "激动", "惊讶", "震惊", "紧张", "激动", "抓狂"]
    low_arousal_words = ["平静", "安静", "放松", "困", "累", "困倦", "无聊", "懈怠"]
    high_dominance_words = ["自信", "骄傲", "坚定", "决心", "掌控", "控制"]
    low_dominance_words = ["无助", "迷茫", "困惑", "犹豫", "不确定", "被动"]

    for w in positive_words:
        if w in text:
            scores["pleasure"] += 0.5
            count += 1
    for w in negative_words:
        if w in text:
            scores["pleasure"] -= 0.4
            count += 1
    for w in high_arousal_words:
        if w in text:
            scores["arousal"] += 0.5
            count += 1
    for w in low_arousal_words:
        if w in text:
            scores["arousal"] -= 0.4
            count += 1
    for w in high_dominance_words:
        if w in text:
            scores["dominance"] += 0.5
            count += 1
    for w in low_dominance_words:
        if w in text:
            scores["dominance"] -= 0.4
            count += 1

    if count > 0:
        p = max(-1.0, min(1.0, scores["pleasure"] / count))
        a = max(-1.0, min(1.0, scores["arousal"] / count))
        d = max(-1.0, min(1.0, scores["dominance"] / count))
        return PAD(p, a, d)
    return PAD(0.0, 0.0, 0.0)


def merge_pad_values(pad_list: List[PAD], weights: Optional[List[float]] = None) -> PAD:
    if not pad_list:
        return PAD(0.0, 0.0, 0.0)
    if weights is None:
        weights = [1.0] * len(pad_list)
    while len(weights) < len(pad_list):
        weights.append(1.0)
    weights = weights[: len(pad_list)]
    total_weight = sum(weights)
    if total_weight == 0:
        total_weight = 1.0
    p = sum(pad.pleasure * w for pad, w in zip(pad_list, weights)) / total_weight
    a = sum(pad.arousal * w for pad, w in zip(pad_list, weights)) / total_weight
    d = sum(pad.dominance * w for pad, w in zip(pad_list, weights)) / total_weight
    return PAD(p, a, d)


def pad_to_emotion(pad: PAD) -> str:
    return pad.to_emotion_category()


def get_pad_intensity(pad: PAD) -> float:
    return math.sqrt(pad.pleasure ** 2 + pad.arousal ** 2 + pad.dominance ** 2) / math.sqrt(3)


def is_emotion_intense(pad: PAD, threshold: float = 0.6) -> bool:
    return get_pad_intensity(pad) > threshold


class PADTracker:
    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = smoothing_window
        self.history: List[PAD] = []

    def add(self, pad: PAD) -> None:
        self.history.append(pad)
        if len(self.history) > self.smoothing_window:
            self.history.pop(0)

    def get_smoothed_pad(self) -> PAD:
        if not self.history:
            return PAD(0.0, 0.0, 0.0)
        return merge_pad_values(self.history)

    def get_trend(self) -> str:
        if len(self.history) < 2:
            return "stable"
        recent = self.history[-1]
        older = self.history[0]
        p_change = recent.pleasure - older.pleasure
        if p_change > 0.2:
            return "improving"
        elif p_change < -0.2:
            return "worsening"
        return "stable"

    def clear(self) -> None:
        self.history.clear()
