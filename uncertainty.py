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
from config import CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_LOW, BAYESIAN_DECAY_RATE


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
        self._decay_rate = BAYESIAN_DECAY_RATE

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
        if confidence >= CONFIDENCE_THRESHOLD_HIGH and evidence_count <= 2:
            return DecisionMode.CONFIDENT
        elif confidence < CONFIDENCE_THRESHOLD_LOW:
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
        self.conflict_resolver = get_conflict_resolver()
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

        if evidence.should_query(CONFIDENCE_THRESHOLD_LOW):
            if time.time() - self.last_query_time > self.query_cooldown:
                decision_mode = DecisionMode.QUERY
                self._pending_query = evidence
            else:
                decision_mode = DecisionMode.UNCERTAIN
        elif evidence.is_confident(CONFIDENCE_THRESHOLD_HIGH):
            decision_mode = DecisionMode.CONFIDENT
        else:
            decision_mode = DecisionMode.UNCERTAIN

        return evidence, pad_state, decision_mode

    def resolve_multimodal_conflict(
        self,
        vision_evidence: EvidenceResult,
        audio_evidence: EvidenceResult,
        text_evidence: EvidenceResult = None,
        vision_brightness: float = 100.0,
        audio_noise_level: float = 0.0
    ) -> Dict[str, Any]:
        """
        解决多模态冲突
        """
        return self.conflict_resolver.resolve_conflict(
            vision_evidence, audio_evidence, text_evidence, vision_brightness, audio_noise_level
        )

    def handle_specific_conflict(
        self,
        conflict_type: str,
        vision_evidence: EvidenceResult,
        audio_evidence: EvidenceResult,
        text_evidence: EvidenceResult = None,
        vision_brightness: float = 100.0,
        audio_noise_level: float = 0.0
    ) -> Dict[str, Any]:
        """
        处理特定类型的冲突
        """
        return self.conflict_resolver.handle_specific_conflict(
            conflict_type, vision_evidence, audio_evidence, text_evidence, vision_brightness, audio_noise_level
        )

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


class ModalConflictResolver:
    """
    多模态冲突解决器
    使用贝叶斯权重来自动调解视觉与听觉的冲突
    """

    def __init__(self):
        # 模态可靠性先验概率
        self.modality_reliability = {
            "vision": 0.7,  # 视觉识别的基础可靠性
            "audio": 0.6,   # 音频识别的基础可靠性
            "text": 0.8     # 文本识别的基础可靠性
        }
        # 基础可靠性（用于动态调整时的参考）
        self.base_reliability = {
            "vision": 0.7,
            "audio": 0.6,
            "text": 0.8
        }
        
        # 情绪-模态关联度
        self.emotion_modality_relevance = {
            "happy": {"vision": 0.9, "audio": 0.8, "text": 0.7},
            "sad": {"vision": 0.8, "audio": 0.9, "text": 0.7},
            "angry": {"vision": 0.7, "audio": 0.9, "text": 0.8},
            "fear": {"vision": 0.8, "audio": 0.8, "text": 0.6},
            "surprise": {"vision": 0.9, "audio": 0.7, "text": 0.6},
            "neutral": {"vision": 0.6, "audio": 0.6, "text": 0.6}
        }
        
        # 冲突类型权重
        self.conflict_type_weights = {
            "vision_vs_audio": 0.5,  # 视觉与音频冲突的权重
            "vision_vs_text": 0.3,   # 视觉与文本冲突的权重
            "audio_vs_text": 0.2     # 音频与文本冲突的权重
        }

    def adjust_vision_reliability(self, brightness: float) -> float:
        """
        根据图像亮度调整视觉可靠性权重

        参数:
            brightness: 图像平均亮度 (0-255)

        返回:
            float: 调整后的视觉可靠性权重
        """
        # 亮度阈值设置
        BRIGHT_THRESHOLD = 80
        DARK_THRESHOLD = 30
        
        if brightness > BRIGHT_THRESHOLD:
            # 光线充足，使用基础可靠性
            return self.base_reliability["vision"]
        elif brightness < DARK_THRESHOLD:
            # 光线昏暗，大幅降低视觉可靠性
            return 0.2
        else:
            # 光线适中，线性调整
            ratio = (brightness - DARK_THRESHOLD) / (BRIGHT_THRESHOLD - DARK_THRESHOLD)
            return 0.2 + (self.base_reliability["vision"] - 0.2) * ratio

    def adjust_audio_reliability(self, noise_level: float) -> float:
        """
        根据环境噪声水平调整音频可靠性权重

        参数:
            noise_level: 噪声水平 (0-1)

        返回:
            float: 调整后的音频可靠性权重
        """
        # 噪声水平阈值设置
        QUIET_THRESHOLD = 0.3
        NOISY_THRESHOLD = 0.7
        
        if noise_level < QUIET_THRESHOLD:
            # 环境安静，使用基础可靠性
            return self.base_reliability["audio"]
        elif noise_level > NOISY_THRESHOLD:
            # 环境嘈杂，大幅降低音频可靠性
            return 0.3
        else:
            # 噪声适中，线性调整
            ratio = (NOISY_THRESHOLD - noise_level) / (NOISY_THRESHOLD - QUIET_THRESHOLD)
            return 0.3 + (self.base_reliability["audio"] - 0.3) * ratio

    def resolve_conflict(self, vision_evidence: EvidenceResult, 
                       audio_evidence: EvidenceResult, 
                       text_evidence: EvidenceResult = None, 
                       vision_brightness: float = 100.0, 
                       audio_noise_level: float = 0.0) -> Dict[str, Any]:
        """
        解决多模态冲突

        参数:
            vision_evidence: 视觉识别的证据结果
            audio_evidence: 音频识别的证据结果
            text_evidence: 文本识别的证据结果（可选）
            vision_brightness: 图像平均亮度 (0-255)
            audio_noise_level: 环境噪声水平 (0-1)

        返回:
            Dict: 解决结果，包含最终情绪、置信度和冲突分析
        """
        # 动态调整模态可靠性
        adjusted_vision_reliability = self.adjust_vision_reliability(vision_brightness)
        adjusted_audio_reliability = self.adjust_audio_reliability(audio_noise_level)
        
        # 临时覆盖模态可靠性进行计算
        original_vision_reliability = self.modality_reliability["vision"]
        original_audio_reliability = self.modality_reliability["audio"]
        
        self.modality_reliability["vision"] = adjusted_vision_reliability
        self.modality_reliability["audio"] = adjusted_audio_reliability
        
        # 计算各模态的贝叶斯权重
        weights = self._calculate_bayesian_weights(vision_evidence, audio_evidence, text_evidence)
        
        # 恢复原始模态可靠性
        self.modality_reliability["vision"] = original_vision_reliability
        self.modality_reliability["audio"] = original_audio_reliability
        
        # 计算加权后的情绪得分
        emotion_scores = self._calculate_emotion_scores(
            vision_evidence, audio_evidence, text_evidence, weights
        )
        
        # 确定最终情绪
        final_emotion = max(emotion_scores, key=emotion_scores.get)
        final_confidence = emotion_scores[final_emotion]
        
        # 分析冲突类型
        conflict_analysis = self._analyze_conflicts(
            vision_evidence, audio_evidence, text_evidence, weights
        )
        
        # 添加环境因素分析到推理过程
        reasoning = self._generate_reasoning(weights, emotion_scores, conflict_analysis)
        reasoning += f"\n环境因素分析:"
        reasoning += f"\n  - 视觉亮度: {vision_brightness:.1f} (可靠性调整: {adjusted_vision_reliability:.2f})"
        reasoning += f"\n  - 环境噪声: {audio_noise_level:.2f} (可靠性调整: {adjusted_audio_reliability:.2f})"
        
        return {
            "final_emotion": final_emotion,
            "final_confidence": final_confidence,
            "weights": weights,
            "conflict_analysis": conflict_analysis,
            "reasoning": reasoning
        }

    def _calculate_bayesian_weights(self, vision_evidence: EvidenceResult, 
                                  audio_evidence: EvidenceResult, 
                                  text_evidence: EvidenceResult = None) -> Dict[str, float]:
        """
        计算各模态的贝叶斯权重
        """
        weights = {}
        
        # 计算视觉权重
        vision_relevance = self.emotion_modality_relevance.get(
            vision_evidence.label, self.emotion_modality_relevance["neutral"]
        ).get("vision", 0.5)
        weights["vision"] = (self.modality_reliability["vision"] * 
                           vision_evidence.confidence * vision_relevance)
        
        # 计算音频权重
        audio_relevance = self.emotion_modality_relevance.get(
            audio_evidence.label, self.emotion_modality_relevance["neutral"]
        ).get("audio", 0.5)
        weights["audio"] = (self.modality_reliability["audio"] * 
                          audio_evidence.confidence * audio_relevance)
        
        # 计算文本权重（如果有）
        if text_evidence:
            text_relevance = self.emotion_modality_relevance.get(
                text_evidence.label, self.emotion_modality_relevance["neutral"]
            ).get("text", 0.5)
            weights["text"] = (self.modality_reliability["text"] * 
                             text_evidence.confidence * text_relevance)
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            for modality in weights:
                weights[modality] /= total_weight
        
        return weights

    def _calculate_emotion_scores(self, vision_evidence: EvidenceResult, 
                                 audio_evidence: EvidenceResult, 
                                 text_evidence: EvidenceResult = None, 
                                 weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        计算加权后的情绪得分
        """
        if weights is None:
            weights = self._calculate_bayesian_weights(vision_evidence, audio_evidence, text_evidence)
        
        emotion_scores = {}
        
        # 收集所有可能的情绪
        emotions = set()
        emotions.add(vision_evidence.label)
        emotions.add(audio_evidence.label)
        if text_evidence:
            emotions.add(text_evidence.label)
        
        # 计算每种情绪的加权得分
        for emotion in emotions:
            score = 0.0
            
            # 视觉贡献
            if vision_evidence.label == emotion:
                score += weights.get("vision", 0) * vision_evidence.confidence
            
            # 音频贡献
            if audio_evidence.label == emotion:
                score += weights.get("audio", 0) * audio_evidence.confidence
            
            # 文本贡献
            if text_evidence and text_evidence.label == emotion:
                score += weights.get("text", 0) * text_evidence.confidence
            
            emotion_scores[emotion] = score
        
        return emotion_scores

    def _analyze_conflicts(self, vision_evidence: EvidenceResult, 
                         audio_evidence: EvidenceResult, 
                         text_evidence: EvidenceResult = None, 
                         weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        分析冲突类型和程度
        """
        if weights is None:
            weights = self._calculate_bayesian_weights(vision_evidence, audio_evidence, text_evidence)
        
        conflicts = {
            "has_conflict": False,
            "conflict_type": None,
            "conflict_strength": 0.0,
            "dominant_modality": None
        }
        
        # 检查视觉与音频冲突
        if vision_evidence.label != audio_evidence.label:
            conflicts["has_conflict"] = True
            conflicts["conflict_type"] = "vision_vs_audio"
            conflicts["conflict_strength"] = self.conflict_type_weights["vision_vs_audio"]
        
        # 检查视觉与文本冲突
        elif text_evidence and vision_evidence.label != text_evidence.label:
            conflicts["has_conflict"] = True
            conflicts["conflict_type"] = "vision_vs_text"
            conflicts["conflict_strength"] = self.conflict_type_weights["vision_vs_text"]
        
        # 检查音频与文本冲突
        elif text_evidence and audio_evidence.label != text_evidence.label:
            conflicts["has_conflict"] = True
            conflicts["conflict_type"] = "audio_vs_text"
            conflicts["conflict_strength"] = self.conflict_type_weights["audio_vs_text"]
        
        # 确定主导模态
        if weights:
            conflicts["dominant_modality"] = max(weights, key=weights.get)
        
        return conflicts

    def _generate_reasoning(self, weights: Dict[str, float], 
                          emotion_scores: Dict[str, float], 
                          conflict_analysis: Dict[str, Any]) -> str:
        """
        生成冲突解决的推理过程
        """
        parts = []
        
        # 模态权重
        parts.append("模态权重分析:")
        for modality, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            parts.append(f"  - {modality}: {weight:.2f}")
        
        # 情绪得分
        parts.append("\n情绪得分:")
        for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
            parts.append(f"  - {emotion}: {score:.2f}")
        
        # 冲突分析
        if conflict_analysis["has_conflict"]:
            parts.append(f"\n冲突分析: {conflict_analysis['conflict_type']} (强度: {conflict_analysis['conflict_strength']:.2f})")
            parts.append(f"主导模态: {conflict_analysis['dominant_modality']}")
        else:
            parts.append("\n无冲突: 各模态一致")
        
        return "\n".join(parts)

    def handle_specific_conflict(self, conflict_type: str, 
                               vision_evidence: EvidenceResult, 
                               audio_evidence: EvidenceResult, 
                               text_evidence: EvidenceResult = None, 
                               vision_brightness: float = 100.0, 
                               audio_noise_level: float = 0.0) -> Dict[str, Any]:
        """
        处理特定类型的冲突
        """
        # 对于视觉识别出'笑'但语义分析出'反讽或哭泣'的情况
        if conflict_type == "vision_vs_audio" and \
           "笑" in vision_evidence.reasoning and \
           ("反讽" in (text_evidence.reasoning if text_evidence else "") or \
            "哭泣" in (text_evidence.reasoning if text_evidence else "")):
            
            # 这种情况下，文本/语义分析更可靠
            weights = self._calculate_bayesian_weights(vision_evidence, audio_evidence, text_evidence)
            # 增加文本权重
            if "text" in weights:
                weights["text"] *= 1.5
                # 重新归一化
                total = sum(weights.values())
                for k in weights:
                    weights[k] /= total
            
            emotion_scores = self._calculate_emotion_scores(
                vision_evidence, audio_evidence, text_evidence, weights
            )
            
            final_emotion = max(emotion_scores, key=emotion_scores.get)
            
            return {
                "final_emotion": final_emotion,
                "final_confidence": emotion_scores[final_emotion],
                "weights": weights,
                "conflict_analysis": {
                    "has_conflict": True,
                    "conflict_type": "vision_vs_semantic",
                    "conflict_strength": 0.7,
                    "dominant_modality": "text"
                },
                "reasoning": "检测到视觉与语义冲突：'笑'可能是反讽或假笑，优先采用语义分析结果"
            }
        
        # 默认冲突处理
        return self.resolve_conflict(vision_evidence, audio_evidence, text_evidence, vision_brightness, audio_noise_level)


# 全局多模态冲突解决器
_global_conflict_resolver: Optional[ModalConflictResolver] = None


def get_conflict_resolver() -> ModalConflictResolver:
    """获取全局多模态冲突解决器单例"""
    global _global_conflict_resolver
    if _global_conflict_resolver is None:
        _global_conflict_resolver = ModalConflictResolver()
    return _global_conflict_resolver
