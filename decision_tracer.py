# decision_tracer.py
"""
决策链路追踪模块 (Decision Path Tracing)
实现可解释AI (Explainable AI) 决策追踪
从原始感知到最终决策的完整逻辑链路图
"""
import time
import threading
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import json


class NodeType(Enum):
    RAW_SENSOR = "raw_sensor"
    PERCEPTION = "perception"
    MEMORY_ASSOCIATION = "memory_association"
    EMOTION_DETECTION = "emotion_detection"
    SCENE_UNDERSTANDING = "scene_understanding"
    UNCERTAINTY_REASONING = "uncertainty_reasoning"
    SAFETY_CHECK = "safety_check"
    INHIBITION_RULE = "inhibition_rule"
    INTENT_DECISION = "intent_decision"
    ACTION_SELECTION = "action_selection"
    FINAL_ACTION = "final_action"


class EdgeType(Enum):
    CAUSES = "causes"
    INHIBITS = "inhibits"
    SUPPORTS = "supports"
    LEADS_TO = "leads_to"
    OVERRIDES = "overrides"


class ModelType(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass
class LatencyRecord:
    """延迟记录"""
    node_type: NodeType
    model_type: ModelType
    start_time: float
    end_time: float
    duration_ms: float
    network_condition: str = "GOOD"
    error: str = None

    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type.value,
            "model_type": self.model_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "network_condition": self.network_condition,
            "error": self.error
        }


class LatencyStats:
    """延迟统计聚合器"""

    def __init__(self):
        self.local_latencies: Dict[NodeType, List[float]] = defaultdict(list)
        self.cloud_latencies: Dict[NodeType, List[float]] = defaultdict(list)
        self.network_conditions: List[str] = []
        self._lock = threading.Lock()

    def add_latency(self, node_type: NodeType, model_type: ModelType,
                   duration_ms: float, network_condition: str = "GOOD") -> None:
        with self._lock:
            if model_type == ModelType.LOCAL:
                self.local_latencies[node_type].append(duration_ms)
            else:
                self.cloud_latencies[node_type].append(duration_ms)
            self.network_conditions.append(network_condition)

    def get_stats(self, node_type: NodeType) -> Dict[str, Any]:
        with self._lock:
            local = self.local_latencies.get(node_type, [])
            cloud = self.cloud_latencies.get(node_type, [])

            return {
                "node_type": node_type.value,
                "local": {
                    "count": len(local),
                    "avg_ms": sum(local) / len(local) if local else 0,
                    "min_ms": min(local) if local else 0,
                    "max_ms": max(local) if local else 0,
                    "p95_ms": sorted(local)[int(len(local) * 0.95)] if len(local) >= 20 else (sorted(local)[-1] if local else 0)
                },
                "cloud": {
                    "count": len(cloud),
                    "avg_ms": sum(cloud) / len(cloud) if cloud else 0,
                    "min_ms": min(cloud) if cloud else 0,
                    "max_ms": max(cloud) if cloud else 0,
                    "p95_ms": sorted(cloud)[int(len(cloud) * 0.95)] if len(cloud) >= 20 else (sorted(cloud)[-1] if cloud else 0)
                }
            }

    def get_end_to_end_stats(self) -> Dict[str, Any]:
        with self._lock:
            all_local = []
            all_cloud = []

            for latencies in self.local_latencies.values():
                all_local.extend(latencies)
            for latencies in self.cloud_latencies.values():
                all_cloud.extend(latencies)

            return {
                "local_e2e": {
                    "total_samples": len(all_local),
                    "avg_ms": sum(all_local) / len(all_local) if all_local else 0,
                    "min_ms": min(all_local) if all_local else 0,
                    "max_ms": max(all_local) if all_local else 0,
                    "p95_ms": sorted(all_local)[int(len(all_local) * 0.95)] if len(all_local) >= 20 else (sorted(all_local)[-1] if all_local else 0)
                },
                "cloud_e2e": {
                    "total_samples": len(all_cloud),
                    "avg_ms": sum(all_cloud) / len(all_cloud) if all_cloud else 0,
                    "min_ms": min(all_cloud) if all_cloud else 0,
                    "max_ms": max(all_cloud) if all_cloud else 0,
                    "p95_ms": sorted(all_cloud)[int(len(all_cloud) * 0.95)] if len(all_cloud) >= 20 else (sorted(all_cloud)[-1] if all_cloud else 0)
                },
                "network_conditions": dict(Counter(self.network_conditions))
            }

    def get_comparison_data(self) -> Dict[str, Any]:
        node_types = set(self.local_latencies.keys()) | set(self.cloud_latencies.keys())
        comparison = {}

        for nt in node_types:
            stats = self.get_stats(nt)
            comparison[nt.value] = {
                "local_avg": stats["local"]["avg_ms"],
                "cloud_avg": stats["cloud"]["avg_ms"],
                "overhead_ms": stats["cloud"]["avg_ms"] - stats["local"]["avg_ms"] if stats["cloud"]["avg_ms"] and stats["local"]["avg_ms"] else 0
            }

        return comparison

    def to_dict(self) -> Dict[str, Any]:
        node_types = set(self.local_latencies.keys()) | set(self.cloud_latencies.keys())
        return {
            "node_stats": {nt.value: self.get_stats(nt) for nt in node_types},
            "e2e_stats": self.get_end_to_end_stats(),
            "comparison": self.get_comparison_data()
        }


@dataclass
class ReasoningNode:
    """推理图中的节点"""
    node_id: str
    node_type: NodeType
    label: str
    data: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_type: ModelType = ModelType.LOCAL
    network_condition: str = "GOOD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
            "model_type": self.model_type.value,
            "network_condition": self.network_condition
        }


@dataclass
class ReasoningEdge:
    """推理图中的边"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "label": self.label,
            "metadata": self.metadata
        }


class ReasoningGraph:
    """
    推理图：记录从原始感知到最终决策的完整链路
    """
    def __init__(self, session_id: str = None, user_id: str = None):
        self.graph_id = session_id or str(uuid.uuid4())[:8]
        self.user_id = user_id  # 用户标识
        self.nodes: Dict[str, ReasoningNode] = {}
        self.edges: List[ReasoningEdge] = []
        self.created_at = time.time()
        self.updated_at = time.time()
        self.metadata: Dict[str, Any] = {}

    def add_node(self, node_type: NodeType, label: str, data: Dict[str, Any],
                 metadata: Dict[str, Any] = None) -> str:
        node_id = f"{node_type.value}_{len(self.nodes)}"
        node = ReasoningNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            data=data,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        self.updated_at = time.time()
        return node_id

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType,
                 weight: float = 1.0, label: str = "", metadata: Dict[str, Any] = None) -> None:
        if source_id in self.nodes and target_id in self.nodes:
            edge = ReasoningEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                label=label,
                metadata=metadata or {}
            )
            self.edges.append(edge)
            self.updated_at = time.time()

    def get_path(self, start_node_id: str, end_node_id: str) -> List[str]:
        """获取从起始节点到结束节点的路径"""
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return []

        visited = set()
        path = []

        def dfs(current: str, target: str) -> bool:
            if current == target:
                path.append(current)
                return True
            if current in visited:
                return False

            visited.add(current)
            path.append(current)

            for edge in self.edges:
                if edge.source_id == current:
                    if dfs(edge.target_id, target):
                        return True

            path.pop()
            return False

        dfs(start_node_id, end_node_id)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "user_id": self.user_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }

    def to_markdown(self) -> str:
        """导出为 Markdown 格式"""
        lines = [
            f"# Reasoning Graph: {self.graph_id}",
            f"",
            f"**创建时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at))}",
            f"**用户ID**: {self.user_id or 'Unknown'}",
            f"**节点数**: {len(self.nodes)}",
            f"**边数**: {len(self.edges)}",
            f"",
            f"## 节点",
            f"",
            f"| 节点ID | 类型 | 标签 | 数据 |",
            f"|--------|------|------|------|"
        ]

        for node_id, node in self.nodes.items():
            data_str = json.dumps(node.data, ensure_ascii=False)[:50]
            lines.append(f"| {node_id} | {node.node_type.value} | {node.label} | {data_str} |")

        lines.extend([
            f"",
            f"## 边 (逻辑链路)",
            f"",
            f"| 源节点 | 目标节点 | 类型 | 标签 |",
            f"|--------|---------|------|------|"
        ])

        for edge in self.edges:
            lines.append(f"| {edge.source_id} | {edge.target_id} | {edge.edge_type.value} | {edge.label} |")

        final_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.FINAL_ACTION]
        if final_nodes:
            lines.extend([
                f"",
                f"## 决策结论",
                f""
            ])
            for node in final_nodes:
                lines.append(f"- **{node.label}**: {json.dumps(node.data, ensure_ascii=False)}")

        return "\n".join(lines)


class DecisionTracer:
    """
    决策链路追踪器
    实时记录每一轮交互的完整推理链路
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._current_graphs: Dict[str, ReasoningGraph] = {}  # 按用户ID存储当前推理图
        self._graph_history: Dict[str, List[ReasoningGraph]] = {}  # 按用户ID存储历史推理图
        self._max_history = 50
        self._node_mappings: Dict[str, Dict[str, str]] = {}  # 按用户ID存储节点映射
        self._lock_graph = threading.Lock()
        self._enabled = True
        self._latency_stats: Dict[str, LatencyStats] = {}
        self._active_timers: Dict[str, Dict[str, float]] = {}

    @classmethod
    def get_instance(cls) -> "DecisionTracer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_tracing(self, session_id: str = None, user_id: str = None) -> ReasoningGraph:
        """开始一轮新的追踪
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            新创建的推理图
        """
        with self._lock_graph:
            user_key = user_id or "default"
            if user_key in self._current_graphs:
                self._end_tracing(user_id)

            self._current_graphs[user_key] = ReasoningGraph(session_id, user_id)
            self._node_mappings[user_key] = {}
            return self._current_graphs[user_key]

    def _end_tracing(self, user_id: str = None) -> Optional[ReasoningGraph]:
        """结束当前追踪
        
        Args:
            user_id: 用户ID
            
        Returns:
            结束的推理图
        """
        with self._lock_graph:
            user_key = user_id or "default"
            if user_key in self._current_graphs:
                graph = self._current_graphs[user_key]
                if user_key not in self._graph_history:
                    self._graph_history[user_key] = []
                self._graph_history[user_key].append(graph)
                if len(self._graph_history[user_key]) > self._max_history:
                    self._graph_history[user_key].pop(0)
                del self._current_graphs[user_key]
                if user_key in self._node_mappings:
                    del self._node_mappings[user_key]
                return graph
        return None

    def end_tracing(self, user_id: str = None) -> Optional[ReasoningGraph]:
        """结束当前追踪
        
        Args:
            user_id: 用户ID
            
        Returns:
            结束的推理图
        """
        return self._end_tracing(user_id)

    def is_tracing(self, user_id: str = None) -> bool:
        """检查是否正在追踪
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否正在追踪
        """
        user_key = user_id or "default"
        return user_key in self._current_graphs

    def record_raw_sensor(self, sensor_type: str, raw_data: Any,
                          presence: bool = None, user_id: str = None) -> Optional[str]:
        """记录原始传感器数据
        
        Args:
            sensor_type: 传感器类型
            raw_data: 原始数据
            presence: 是否存在
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        node_id = graph.add_node(
            NodeType.RAW_SENSOR,
            f"原始{sensor_type}",
            {"sensor_type": sensor_type, "raw_data": raw_data, "presence": presence}
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key][f"raw_{sensor_type}"] = node_id
        return node_id

    def record_perception(self, description: str, confidence: float = 1.0,
                          raw_sensor_ref: str = None, user_id: str = None) -> Optional[str]:
        """记录感知结果
        
        Args:
            description: 感知描述
            confidence: 置信度
            raw_sensor_ref: 原始传感器引用
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        node_id = graph.add_node(
            NodeType.PERCEPTION,
            "感知细节",
            {"description": description, "confidence": confidence}
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["perception"] = node_id

        if raw_sensor_ref and user_key in self._node_mappings and raw_sensor_ref in self._node_mappings[user_key]:
            graph.add_edge(
                self._node_mappings[user_key][raw_sensor_ref],
                node_id,
                EdgeType.CAUSES,
                label="感知来源"
            )

        return node_id

    def record_scene_understanding(self, scene_type: str, scene_confidence: float,
                                   keywords_matched: List[str] = None, user_id: str = None) -> Optional[str]:
        """记录场景理解
        
        Args:
            scene_type: 场景类型
            scene_confidence: 场景置信度
            keywords_matched: 匹配的关键词
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        perception_ref = self._node_mappings.get(user_key, {}).get("perception")

        node_id = graph.add_node(
            NodeType.SCENE_UNDERSTANDING,
            f"场景: {scene_type}",
            {
                "scene_type": scene_type,
                "confidence": scene_confidence,
                "keywords_matched": keywords_matched or []
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["scene"] = node_id

        if perception_ref:
            graph.add_edge(
                perception_ref,
                node_id,
                EdgeType.SUPPORTS,
                label="场景推断"
            )

        return node_id

    def record_memory_association(self, memory_type: str, retrieved_content: Any,
                                  relevance_score: float = 1.0, user_id: str = None) -> Optional[str]:
        """记录记忆关联
        
        Args:
            memory_type: 记忆类型
            retrieved_content: 检索到的内容
            relevance_score: 相关度分数
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        node_id = graph.add_node(
            NodeType.MEMORY_ASSOCIATION,
            f"记忆关联: {memory_type}",
            {
                "memory_type": memory_type,
                "content": retrieved_content,
                "relevance": relevance_score
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["memory"] = node_id
        return node_id

    def record_emotion_detection(self, emotion: str, confidence: float,
                                 pad_state: Dict[str, Any] = None, user_id: str = None) -> Optional[str]:
        """记录情绪检测
        
        Args:
            emotion: 情绪类型
            confidence: 置信度
            pad_state: PAD状态
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        perception_ref = self._node_mappings.get(user_key, {}).get("perception")

        node_id = graph.add_node(
            NodeType.EMOTION_DETECTION,
            f"情绪: {emotion}",
            {
                "emotion": emotion,
                "confidence": confidence,
                "pad_state": pad_state or {}
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["emotion"] = node_id

        if perception_ref:
            graph.add_edge(
                perception_ref,
                node_id,
                EdgeType.CAUSES,
                label="情绪推断"
            )

        return node_id

    def record_uncertainty_reasoning(self, decision_mode: str, confidence: float,
                                     reasoning: str = "", user_id: str = None) -> Optional[str]:
        """记录不确定性推理
        
        Args:
            decision_mode: 决策模式
            confidence: 置信度
            reasoning: 推理过程
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        emotion_ref = self._node_mappings.get(user_key, {}).get("emotion")

        node_id = graph.add_node(
            NodeType.UNCERTAINTY_REASONING,
            f"决策模式: {decision_mode}",
            {
                "decision_mode": decision_mode,
                "confidence": confidence,
                "reasoning": reasoning
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["uncertainty"] = node_id

        if emotion_ref:
            graph.add_edge(
                emotion_ref,
                node_id,
                EdgeType.CAUSES,
                label="不确定性评估"
            )

        return node_id

    def record_safety_check(self, safety_level: str, risk_factors: List[str],
                           passed: bool, blocked_reason: str = None, user_id: str = None) -> Optional[str]:
        """记录安全检查
        
        Args:
            safety_level: 安全级别
            risk_factors: 风险因素
            passed: 是否通过
            blocked_reason: 阻止原因
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        node_id = graph.add_node(
            NodeType.SAFETY_CHECK,
            f"安全检查: {safety_level}",
            {
                "safety_level": safety_level,
                "risk_factors": risk_factors,
                "passed": passed,
                "blocked_reason": blocked_reason
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["safety"] = node_id
        return node_id

    def record_inhibition_rule(self, rule_name: str, triggered: bool,
                              inhibition_effect: str = None, user_id: str = None) -> Optional[str]:
        """记录抑制规则触发
        
        Args:
            rule_name: 规则名称
            triggered: 是否触发
            inhibition_effect: 抑制效果
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        safety_ref = self._node_mappings.get(user_key, {}).get("safety")
        uncertainty_ref = self._node_mappings.get(user_key, {}).get("uncertainty")

        node_id = graph.add_node(
            NodeType.INHIBITION_RULE,
            f"抑制规则: {rule_name}",
            {
                "rule_name": rule_name,
                "triggered": triggered,
                "inhibition_effect": inhibition_effect
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key][f"inhibition_{rule_name}"] = node_id

        if triggered:
            if safety_ref:
                graph.add_edge(
                    safety_ref,
                    node_id,
                    EdgeType.INHIBITS,
                    weight=0.8,
                    label="安全抑制"
                )
            if uncertainty_ref:
                graph.add_edge(
                    uncertainty_ref,
                    node_id,
                    EdgeType.INHIBITS,
                    weight=0.5,
                    label="不确定性抑制"
                )

        return node_id

    def record_intent_decision(self, intent: str, confidence: float,
                              alternatives: List[str] = None, user_id: str = None) -> Optional[str]:
        """记录意图决策
        
        Args:
            intent: 意图
            confidence: 置信度
            alternatives: 替代方案
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        node_id = graph.add_node(
            NodeType.INTENT_DECISION,
            f"用户意图: {intent}",
            {
                "intent": intent,
                "confidence": confidence,
                "alternatives": alternatives or []
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["intent"] = node_id
        return node_id

    def record_action_selection(self, selected_action: str, llm_suggestion: str,
                               modified: bool = False, modification_reason: str = None, user_id: str = None) -> Optional[str]:
        """记录动作选择
        
        Args:
            selected_action: 选择的动作
            llm_suggestion: LLM建议
            modified: 是否修改
            modification_reason: 修改原因
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        intent_ref = self._node_mappings.get(user_key, {}).get("intent")
        safety_ref = self._node_mappings.get(user_key, {}).get("safety")

        node_id = graph.add_node(
            NodeType.ACTION_SELECTION,
            f"选择动作: {selected_action}",
            {
                "selected_action": selected_action,
                "llm_suggestion": llm_suggestion,
                "was_modified": modified,
                "modification_reason": modification_reason
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["action"] = node_id

        if intent_ref:
            graph.add_edge(
                intent_ref,
                node_id,
                EdgeType.LEADS_TO,
                label="意图驱动"
            )

        if safety_ref:
            graph.add_edge(
                safety_ref,
                node_id,
                EdgeType.OVERRIDES if modified else EdgeType.SUPPORTS,
                weight=1.0 if modified else 0.5,
                label="安全修正" if modified else "安全确认"
            )

        return node_id

    def record_final_action(self, action: str, success: bool,
                           execution_details: Dict[str, Any] = None, user_id: str = None) -> Optional[str]:
        """记录最终执行的动作
        
        Args:
            action: 动作
            success: 是否成功
            execution_details: 执行详情
            user_id: 用户ID
            
        Returns:
            节点ID
        """
        if not self._enabled:
            return None
        
        user_key = user_id or "default"
        if user_key not in self._current_graphs:
            return None

        graph = self._current_graphs[user_key]
        action_ref = self._node_mappings.get(user_key, {}).get("action")
        scene_ref = self._node_mappings.get(user_key, {}).get("scene")
        memory_ref = self._node_mappings.get(user_key, {}).get("memory")

        node_id = graph.add_node(
            NodeType.FINAL_ACTION,
            f"执行: {action}",
            {
                "action": action,
                "success": success,
                "execution_details": execution_details or {}
            }
        )
        if user_key not in self._node_mappings:
            self._node_mappings[user_key] = {}
        self._node_mappings[user_key]["final"] = node_id

        if action_ref:
            graph.add_edge(
                action_ref,
                node_id,
                EdgeType.LEADS_TO,
                label="执行动作"
            )

        if scene_ref:
            graph.add_edge(
                scene_ref,
                node_id,
                EdgeType.SUPPORTS,
                weight=0.3,
                label="场景相关"
            )

        if memory_ref:
            graph.add_edge(
                memory_ref,
                node_id,
                EdgeType.SUPPORTS,
                weight=0.3,
                label="记忆驱动"
            )

        return node_id

    def get_current_graph(self, user_id: str = None) -> Optional[ReasoningGraph]:
        """获取当前推理图
        
        Args:
            user_id: 用户ID
            
        Returns:
            当前推理图
        """
        user_key = user_id or "default"
        return self._current_graphs.get(user_key)

    def get_graph_history(self, user_id: str = None, limit: int = 10) -> List[ReasoningGraph]:
        """获取历史推理图
        
        Args:
            user_id: 用户ID
            limit: 限制数量
            
        Returns:
            历史推理图列表
        """
        user_key = user_id or "default"
        history = self._graph_history.get(user_key, [])
        return history[-limit:]

    def get_last_graph(self, user_id: str = None) -> Optional[ReasoningGraph]:
        """获取最近的推理图
        
        Args:
            user_id: 用户ID
            
        Returns:
            最近的推理图
        """
        user_key = user_id or "default"
        history = self._graph_history.get(user_key, [])
        if history:
            return history[-1]
        return None

    def export_json(self, user_id: str = None) -> str:
        """导出为 JSON 格式
        
        Args:
            user_id: 用户ID
            
        Returns:
            JSON字符串
        """
        user_key = user_id or "default"
        graphs = [g.to_dict() for g in self._graph_history.get(user_key, [])]
        return json.dumps({
            "user_id": user_id,
            "graphs": graphs,
            "count": len(graphs)
        }, ensure_ascii=False, indent=2)

    def export_last_markdown(self, user_id: str = None) -> str:
        """导出最近的推理图为 Markdown
        
        Args:
            user_id: 用户ID
            
        Returns:
            Markdown字符串
        """
        graph = self.get_last_graph(user_id)
        if graph:
            return graph.to_markdown()
        return "无可用的推理图"

    def export_all_users_markdown(self) -> str:
        """导出所有用户的推理图为 Markdown
        
        Returns:
            Markdown字符串
        """
        lines = ["# 多用户推理逻辑展示", ""]
        
        for user_key, history in self._graph_history.items():
            if history:
                latest_graph = history[-1]
                lines.extend([
                    f"## 用户: {latest_graph.user_id or 'Unknown'}",
                    f"",
                    latest_graph.to_markdown(),
                    f"",
                    "---",
                    f""
                ])
        
        return "\n".join(lines)

    def start_latency_tracking(self, node_type: NodeType, model_type: ModelType,
                               user_id: str = None) -> str:
        """开始追踪某个节点的延迟

        Args:
            node_type: 节点类型
            model_type: 模型类型
            user_id: 用户ID

        Returns:
            追踪ID
        """
        user_key = user_id or "default"
        if user_key not in self._active_timers:
            self._active_timers[user_key] = {}
        if user_key not in self._latency_stats:
            self._latency_stats[user_key] = LatencyStats()

        tracker_id = f"{node_type.value}_{time.time()}"
        self._active_timers[user_key][node_type.value] = time.time()
        return tracker_id

    def end_latency_tracking(self, node_type: NodeType, model_type: ModelType,
                            network_condition: str = "GOOD", user_id: str = None) -> Optional[LatencyRecord]:
        """结束追踪某个节点的延迟

        Args:
            node_type: 节点类型
            model_type: 模型类型
            network_condition: 网络状况
            user_id: 用户ID

        Returns:
            延迟记录
        """
        user_key = user_id or "default"
        if user_key not in self._active_timers or node_type.value not in self._active_timers[user_key]:
            return None
        if user_key not in self._latency_stats:
            self._latency_stats[user_key] = LatencyStats()

        start_time = self._active_timers[user_key].pop(node_type.value)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        record = LatencyRecord(
            node_type=node_type,
            model_type=model_type,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            network_condition=network_condition
        )

        self._latency_stats[user_key].add_latency(node_type, model_type, duration_ms, network_condition)

        if user_key in self._current_graphs:
            graph = self._current_graphs[user_key]
            for node in graph.nodes.values():
                if node.node_type == node_type:
                    node.processing_time_ms = duration_ms
                    node.model_type = model_type
                    node.network_condition = network_condition
                    break

        return record

    def record_node_latency(self, node_type: NodeType, model_type: ModelType,
                           duration_ms: float, network_condition: str = "GOOD",
                           user_id: str = None) -> None:
        """直接记录节点延迟

        Args:
            node_type: 节点类型
            model_type: 模型类型
            duration_ms: 延迟毫秒数
            network_condition: 网络状况
            user_id: 用户ID
        """
        user_key = user_id or "default"
        if user_key not in self._latency_stats:
            self._latency_stats[user_key] = LatencyStats()
        self._latency_stats[user_key].add_latency(node_type, model_type, duration_ms, network_condition)

    def get_latency_stats(self, user_id: str = None) -> Optional[Dict[str, Any]]:
        """获取延迟统计

        Args:
            user_id: 用户ID

        Returns:
            延迟统计数据
        """
        user_key = user_id or "default"
        if user_key not in self._latency_stats:
            return None
        return self._latency_stats[user_key].to_dict()

    def get_e2e_latency_report(self, user_id: str = None) -> str:
        """生成端到端延迟报告

        Args:
            user_id: 用户ID

        Returns:
            延迟报告字符串
        """
        user_key = user_id or "default"
        if user_key not in self._latency_stats:
            return "无可用延迟数据"

        stats = self._latency_stats[user_key]
        e2e = stats.get_end_to_end_stats()
        comparison = stats.get_comparison_data()

        lines = [
            "# E2E 延迟性能报告",
            "",
            f"## 网络状况统计",
            f"- GOOD: {e2e['network_conditions'].get('GOOD', 0)}",
            f"- POOR: {e2e['network_conditions'].get('POOR', 0)}",
            f"- 未知: {e2e['network_conditions'].get('UNKNOWN', 0)}",
            "",
            f"## 本地模型 E2E 延迟",
            f"- 平均: {e2e['local_e2e']['avg_ms']:.2f} ms",
            f"- 最小: {e2e['local_e2e']['min_ms']:.2f} ms",
            f"- 最大: {e2e['local_e2e']['max_ms']:.2f} ms",
            f"- P95: {e2e['local_e2e']['p95_ms']:.2f} ms",
            f"- 样本数: {e2e['local_e2e']['total_samples']}",
            "",
            f"## 云端模型 E2E 延迟",
            f"- 平均: {e2e['cloud_e2e']['avg_ms']:.2f} ms",
            f"- 最小: {e2e['cloud_e2e']['min_ms']:.2f} ms",
            f"- 最大: {e2e['cloud_e2e']['max_ms']:.2f} ms",
            f"- P95: {e2e['cloud_e2e']['p95_ms']:.2f} ms",
            f"- 样本数: {e2e['cloud_e2e']['total_samples']}",
            "",
            "## 各节点延迟对比 (本地 vs 云端)",
            "",
            "| 节点类型 | 本地平均(ms) | 云端平均(ms) | 额外延迟(ms) |",
            "|----------|-------------|-------------|-------------|"
        ]

        for node_type, data in comparison.items():
            lines.append(f"| {node_type} | {data['local_avg']:.2f} | {data['cloud_avg']:.2f} | {data['overhead_ms']:.2f} |")

        if e2e['cloud_e2e']['avg_ms'] > 0 and e2e['local_e2e']['avg_ms'] > 0:
            overhead_pct = ((e2e['cloud_e2e']['avg_ms'] - e2e['local_e2e']['avg_ms']) / e2e['local_e2e']['avg_ms']) * 100
            lines.extend([
                "",
                f"**云端相对本地平均额外延迟: {overhead_pct:.1f}%**"
            ])

        return "\n".join(lines)

    def generate_latency_comparison_chart(self, user_id: str = None) -> str:
        """生成ASCII延迟对比图

        Args:
            user_id: 用户ID

        Returns:
            ASCII图表字符串
        """
        user_key = user_id or "default"
        if user_key not in self._latency_stats:
            return "无可用延迟数据"

        comparison = self._latency_stats[user_key].get_comparison_data()
        if not comparison:
            return "无可用延迟数据"

        lines = ["\n# 延迟对比图 (本地 vs 云端)\n"]

        max_val = 0
        for data in comparison.values():
            max_val = max(max_val, data['local_avg'], data['cloud_avg'])

        if max_val == 0:
            return "无可用延迟数据"

        chart_width = 50
        scale = chart_width / max_val

        lines.append("延迟(ms) | " + "─" * chart_width + " |")
        lines.append("")

        for node_type, data in comparison.items():
            local_bar = "█" * int(data['local_avg'] * scale)
            cloud_bar = "▓" * int(data['cloud_avg'] * scale)

            lines.append(f"{node_type[:20]:20s}")
            lines.append(f"  本地: {local_bar} {data['local_avg']:.1f}ms")
            lines.append(f"  云端: {cloud_bar} {data['cloud_avg']:.1f}ms")
            if data['overhead_ms'] > 0:
                lines.append(f"  额外: +{data['overhead_ms']:.1f}ms ({(data['overhead_ms']/data['local_avg']*100):.0f}%)")
            lines.append("")

        return "\n".join(lines)

    def export_latency_json(self, user_id: str = None) -> str:
        """导出延迟数据为JSON

        Args:
            user_id: 用户ID

        Returns:
            JSON字符串
        """
        user_key = user_id or "default"
        if user_key not in self._latency_stats:
            return json.dumps({"error": "无可用延迟数据"})

        return json.dumps({
            "user_id": user_id,
            "stats": self._latency_stats[user_key].to_dict()
        }, ensure_ascii=False, indent=2)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False


_global_tracer: Optional[DecisionTracer] = None


def get_decision_tracer() -> DecisionTracer:
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DecisionTracer.get_instance()
    return _global_tracer
