# uncertainty.py
"""
不确定性建模模块
实现证据深度学习(EDL)和贝叶斯状态更新
"""
import math
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from utils import logger


class DecisionMode(Enum):
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    QUERY = "query"


@dataclass
class EvidenceResult:
    """证据检测结果（含置信度）"""
    label: str
    confidence: float
    evidence_strength: float
    supporting_features: Dict[str, float]
    decision_mode: DecisionMode
    reasoning: str

    def is_confident(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold

    def should_query(self, threshold: float = 0.5) -> bool:
        return self.confidence < threshold


class BayesianPADState:
    """
    贝叶斯 PAD 状态估计器
    使用高斯分布表示 PAD 值的不确定性
    """
    def __init__(self, initial_P=0.0, initial_A=0.0, initial_D=0.0, initial_variance=0.5):
        self.mean = {"P": initial_P, "A": initial_A, "D": initial_D}
        self.variance = {"P": initial_variance, "A": initial_variance, "D": initial_variance}
        self._decay_rate = 0.05

    def get_std(self, dimension: str) -> float:
        return math.sqrt(max(self.variance.get(dimension, 0.01), 0.0001))

    def update(self, observation: str, observation_confidence: float, emotion_type: str = None) -> Dict[str, Tuple[float, float]]:
        """
        使用贝叶斯更新规则更新 PAD 状态
        observation: 观测到的情绪类型
        observation_confidence: 观测置信度 (0-1)
        """
        if observation_confidence <= 0:
            return self._get_state()

        emotion_to_pad_delta = {
            "happy": {"P": 0.6, "A": 0.3, "D": 0.1},
            "sad": {"P": -0.6, "A": -0.2, "D": -0.4},
            "angry": {"P": -0.4, "A": 0.7, "D": 0.5},
            "fear": {"P": -0.5, "A": 0.6, "D": -0.6},
            "neutral": {"P": 0.0, "A": 0.0, "D": 0.0},
            "surprise": {"P": 0.3, "A": 0.5, "D": 0.0},
            "disgust": {"P": -0.5, "A": 0.4, "D": 0.2},
            "caring": {"P": 0.3, "A": 0.1, "D": -0.1},
            "supportive": {"P": 0.2, "A": 0.0, "D": -0.2},
            "empathetic": {"P": 0.1, "A": 0.0, "D": -0.3},
            "helpful": {"P": 0.2, "A": 0.1, "D": 0.2},
        }

        delta = emotion_to_pad_delta.get(emotion_type, {"P": 0.0, "A": 0.0, "D": 0.0})

        likelihood_precision = observation_confidence * 10.0

        for dim in ["P", "A", "D"]:
            prior_mean = self.mean[dim]
            prior_var = self.variance[dim]

            obs_delta = delta.get(dim, 0.0)
            obs_mean = prior_mean + obs_delta
            obs_var = 1.0 / likelihood_precision

            posterior_var = 1.0 / (1.0 / prior_var + likelihood_precision)
            posterior_mean = posterior_var * (prior_mean / prior_var + likelihood_precision * obs_mean)

            self.mean[dim] = max(-1.0, min(1.0, posterior_mean))
            self.variance[dim] = max(0.01, posterior_var)

            self.mean[dim] = (1 - self._decay_rate) * self.mean[dim]
            self.variance[dim] = self.variance[dim] + 0.001

        return self._get_state()

    def _get_state(self) -> Dict[str, Tuple[float, float]]:
        return {
            dim: (self.mean[dim], self.get_std(dim))
            for dim in ["P", "A", "D"]
        }

    def get_distribution_summary(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "std": {dim: self.get_std(dim) for dim in ["P", "A", "D"]},
            "uncertainty": sum(self.variance.values()) / 3.0
        }


class UncertaintyAwareDetector:
    """
    不确定性感知检测器
    实现证据深度学习(EDL)风格的检测
    """
    CONFIDENCE_THRESHOLD_HIGH = 0.75
    CONFIDENCE_THRESHOLD_LOW = 0.45
    CONTEXT_WEIGHT = 0.4
    DIRECT_WEIGHT = 0.6

    def __init__(self):
        self.emotion_keywords = {
            "happy": {
                "direct": ["开心", "高兴", "快乐", "欢快", "棒", "好开心", "太好了", "笑", "笑哈哈"],
                "context": ["表情", "声音", "兴奋", "兴奋地", "愉快的"]
            },
            "sad": {
                "direct": ["难过", "伤心", "悲伤", "哭", "沮丧", "郁闷", "消沉", "低落"],
                "context": ["声音", "低沉", "唉", "叹气", "无精打采"]
            },
            "angry": {
                "direct": ["生气", "愤怒", "恼火", "烦躁", "火", "气"],
                "context": ["大声", "吼", "提高音量", "语气"]
            },
            "fear": {
                "direct": ["害怕", "恐惧", "担心", "紧张", "怕", "不安"],
                "context": ["声音", "颤抖", "犹豫", "结巴"]
            },
            "surprise": {
                "direct": ["惊讶", "吃惊", "意外", "惊奇", "哇", "天哪"],
                "context": ["声音", "突然", "变化"]
            },
            "neutral": {
                "direct": [],
                "context": []
            }
        }

    def detect_with_confidence(self, text: str, vision_desc: str = "") -> EvidenceResult:
        """
        使用 EDL 风格检测情绪，返回置信度
        """
        combined = (text + " " + vision_desc).lower()

        if not combined.strip():
            return EvidenceResult(
                label="neutral",
                confidence=0.3,
                evidence_strength=0.0,
                supporting_features={},
                decision_mode=DecisionMode.QUERY,
                reasoning="输入为空，无法判断情绪"
            )

        scores = {}
        feature_details = {}

        for emotion, keywords in self.emotion_keywords.items():
            if emotion == "neutral":
                continue

            direct_matches = sum(1 for kw in keywords.get("direct", []) if kw in combined)
            context_matches = sum(1 for kw in keywords.get("context", []) if kw in combined)

            direct_score = min(direct_matches * 0.4, 1.0)
            context_score = min(context_matches * 0.2, 1.0) * self.CONTEXT_WEIGHT

            feature_details[emotion] = {
                "direct_matches": direct_matches,
                "context_matches": context_matches,
                "direct_score": direct_score,
                "context_score": context_score
            }

            scores[emotion] = self.DIRECT_WEIGHT * direct_score + context_score

        if not scores or max(scores.values()) == 0:
            context_coverage = self._check_context_coverage(combined)
            if context_coverage > 0:
                return EvidenceResult(
                    label="neutral",
                    confidence=0.4,
                    evidence_strength=0.2,
                    supporting_features={"context_signals": context_coverage},
                    decision_mode=DecisionMode.QUERY,
                    reasoning=f"检测到上下文信号但无明确情绪: {context_coverage:.2f}"
                )

            return EvidenceResult(
                label="neutral",
                confidence=0.5,
                evidence_strength=0.1,
                supporting_features={},
                decision_mode=DecisionMode.UNCERTAIN,
                reasoning="未检测到明显情绪特征"
            )

        top_emotion = max(scores, key=scores.get)
        top_score = scores[top_emotion]

        evidence_count = sum(1 for e, s in scores.items() if s > 0)
        evidence_strength = top_score * (1.0 if evidence_count == 1 else 0.8)

        confidence = self._compute_confidence(evidence_strength, evidence_count, feature_details.get(top_emotion, {}))

        decision_mode = self._determine_decision_mode(confidence, evidence_count)

        reasoning = self._generate_reasoning(top_emotion, confidence, feature_details.get(top_emotion, {}))

        return EvidenceResult(
            label=top_emotion,
            confidence=confidence,
            evidence_strength=evidence_strength,
            supporting_features=feature_details,
            decision_mode=decision_mode,
            reasoning=reasoning
        )

    def _check_context_coverage(self, text: str) -> float:
        context_signals = ["主人", "看起来", "好像", "似乎", "感觉", "似乎"]
        matches = sum(1 for signal in context_signals if signal in text)
        return matches / len(context_signals)

    def _compute_confidence(self, evidence_strength: float, evidence_count: int, feature_detail: Dict) -> float:
        base_confidence = min(evidence_strength * 1.2, 0.95)

        if evidence_count == 1:
            base_confidence *= 1.1
        elif evidence_count >= 3:
            base_confidence *= 0.85

        direct_ambiguous = feature_detail.get("direct_matches", 0) == 0
        context_only = feature_detail.get("direct_score", 0) < 0.2 and feature_detail.get("context_score", 0) > 0

        if direct_ambiguous and context_only:
            base_confidence *= 0.7

        return max(0.1, min(0.99, base_confidence))

    def _determine_decision_mode(self, confidence: float, evidence_count: int) -> DecisionMode:
        if confidence >= self.CONFIDENCE_THRESHOLD_HIGH and evidence_count <= 2:
            return DecisionMode.CONFIDENT
        elif confidence < self.CONFIDENCE_THRESHOLD_LOW:
            return DecisionMode.QUERY
        else:
            return DecisionMode.UNCERTAIN

    def _generate_reasoning(self, emotion: str, confidence: float, feature_detail: Dict) -> str:
        parts = [f"检测到情绪: {emotion}"]
        parts.append(f"置信度: {confidence:.2f}")

        if feature_detail.get("direct_matches", 0) > 0:
            parts.append(f"直接匹配: {feature_detail['direct_matches']}个")
        if feature_detail.get("context_matches", 0) > 0:
            parts.append(f"上下文匹配: {feature_detail['context_matches']}个")

        if confidence < 0.5:
            parts.append("(建议进入询问模式)")
        elif confidence >= 0.75:
            parts.append("(可执行决策)")

        return " | ".join(parts)


class UncertaintyManager:
    """
    不确定性管理器
    协调检测器、贝叶斯更新和决策模式
    """
    def __init__(self):
        self.detector = UncertaintyAwareDetector()
        self.bayesian_pad = BayesianPADState()
        self.last_query_time = 0.0
        self.query_cooldown = 30.0
        self._pending_query: Optional[EvidenceResult] = None

    def analyze_and_update(
        self,
        user_text: str,
        vision_desc: str,
        emotion_type: str = None
    ) -> Tuple[EvidenceResult, Dict[str, Any], DecisionMode]:
        """
        综合分析并更新状态
        返回: (检测结果, PAD状态更新, 决策模式)
        """
        evidence = self.detector.detect_with_confidence(user_text, vision_desc)

        effective_emotion = emotion_type if emotion_type else evidence.label
        self.bayesian_pad.update(
            observation=evidence.label,
            observation_confidence=evidence.confidence,
            emotion_type=effective_emotion
        )

        pad_state = self.bayesian_pad.get_distribution_summary()

        if evidence.should_query(self.CONFIDENCE_THRESHOLD_LOW):
            if time.time() - self.last_query_time > self.query_cooldown:
                decision_mode = DecisionMode.QUERY
                self._pending_query = evidence
            else:
                decision_mode = DecisionMode.UNCERTAIN
        elif evidence.is_confident(self.CONFIDENCE_THRESHOLD_HIGH):
            decision_mode = DecisionMode.CONFIDENT
        else:
            decision_mode = DecisionMode.UNCERTAIN

        return evidence, pad_state, decision_mode

    def should_ask_question(self) -> bool:
        if self._pending_query is None:
            return False
        if time.time() - self.last_query_time < self.query_cooldown:
            return False
        return True

    def acknowledge_query(self) -> Optional[EvidenceResult]:
        self.last_query_time = time.time()
        query = self._pending_query
        self._pending_query = None
        return query

    def generate_inquiry_prompt(self, evidence: EvidenceResult) -> str:
        """生成询问用户的提示"""
        emotion_hints = {
            "happy": "你看起来好像很开心？",
            "sad": "你今天心情怎么样？有什么让我帮忙的吗？",
            "angry": "你好像有点不高兴，发生了什么？",
            "fear": "你看起来有些担心，有什么我能帮忙的吗？",
            "surprise": "发生什么事了吗？需要我帮忙吗？",
            "neutral": "你怎么样？今天一切都好吗？"
        }

        base_prompt = emotion_hints.get(evidence.label, "你怎么样？有什么需要我帮忙的吗？")

        if evidence.confidence < 0.3:
            return f"我不太确定你的状态。{base_prompt}"
        else:
            return base_prompt


_global_uncertainty_manager: Optional[UncertaintyManager] = None


def get_uncertainty_manager() -> UncertaintyManager:
    global _global_uncertainty_manager
    if _global_uncertainty_manager is None:
        _global_uncertainty_manager = UncertaintyManager()
    return _global_uncertainty_manager
