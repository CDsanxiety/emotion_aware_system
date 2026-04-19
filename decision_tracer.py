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
from collections import defaultdict
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


@dataclass
class ReasoningNode:
    """推理图中的节点"""
    node_id: str
    node_type: NodeType
    label: str
    data: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata
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
    def __init__(self, session_id: str = None):
        self.graph_id = session_id or str(uuid.uuid4())[:8]
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
        self._current_graph: Optional[ReasoningGraph] = None
        self._graph_history: List[ReasoningGraph] = []
        self._max_history = 50
        self._node_mapping: Dict[str, str] = {}
        self._lock_graph = threading.Lock()
        self._enabled = True

    @classmethod
    def get_instance(cls) -> "DecisionTracer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_tracing(self, session_id: str = None) -> ReasoningGraph:
        """开始一轮新的追踪"""
        with self._lock_graph:
            if self._current_graph:
                self._end_tracing()

            self._current_graph = ReasoningGraph(session_id)
            self._node_mapping = {}
            return self._current_graph

    def _end_tracing(self) -> Optional[ReasoningGraph]:
        """结束当前追踪"""
        with self._lock_graph:
            if self._current_graph:
                self._graph_history.append(self._current_graph)
                if len(self._graph_history) > self._max_history:
                    self._graph_history.pop(0)
                graph = self._current_graph
                self._current_graph = None
                return graph
        return None

    def end_tracing(self) -> Optional[ReasoningGraph]:
        return self._end_tracing()

    def is_tracing(self) -> bool:
        return self._current_graph is not None

    def record_raw_sensor(self, sensor_type: str, raw_data: Any,
                          presence: bool = None) -> Optional[str]:
        """记录原始传感器数据"""
        if not self._enabled or not self._current_graph:
            return None

        node_id = self._current_graph.add_node(
            NodeType.RAW_SENSOR,
            f"原始{sensor_type}",
            {"sensor_type": sensor_type, "raw_data": raw_data, "presence": presence}
        )
        self._node_mapping[f"raw_{sensor_type}"] = node_id
        return node_id

    def record_perception(self, description: str, confidence: float = 1.0,
                          raw_sensor_ref: str = None) -> Optional[str]:
        """记录感知结果"""
        if not self._enabled or not self._current_graph:
            return None

        node_id = self._current_graph.add_node(
            NodeType.PERCEPTION,
            "感知细节",
            {"description": description, "confidence": confidence}
        )
        self._node_mapping["perception"] = node_id

        if raw_sensor_ref and raw_sensor_ref in self._node_mapping:
            self._current_graph.add_edge(
                self._node_mapping[raw_sensor_ref],
                node_id,
                EdgeType.CAUSES,
                label="感知来源"
            )

        return node_id

    def record_scene_understanding(self, scene_type: str, scene_confidence: float,
                                   keywords_matched: List[str] = None) -> Optional[str]:
        """记录场景理解"""
        if not self._enabled or not self._current_graph:
            return None

        perception_ref = self._node_mapping.get("perception")

        node_id = self._current_graph.add_node(
            NodeType.SCENE_UNDERSTANDING,
            f"场景: {scene_type}",
            {
                "scene_type": scene_type,
                "confidence": scene_confidence,
                "keywords_matched": keywords_matched or []
            }
        )
        self._node_mapping["scene"] = node_id

        if perception_ref:
            self._current_graph.add_edge(
                perception_ref,
                node_id,
                EdgeType.SUPPORTS,
                label="场景推断"
            )

        return node_id

    def record_memory_association(self, memory_type: str, retrieved_content: Any,
                                  relevance_score: float = 1.0) -> Optional[str]:
        """记录记忆关联"""
        if not self._enabled or not self._current_graph:
            return None

        node_id = self._current_graph.add_node(
            NodeType.MEMORY_ASSOCIATION,
            f"记忆关联: {memory_type}",
            {
                "memory_type": memory_type,
                "content": retrieved_content,
                "relevance": relevance_score
            }
        )
        self._node_mapping["memory"] = node_id
        return node_id

    def record_emotion_detection(self, emotion: str, confidence: float,
                                 pad_state: Dict[str, Any] = None) -> Optional[str]:
        """记录情绪检测"""
        if not self._enabled or not self._current_graph:
            return None

        perception_ref = self._node_mapping.get("perception")

        node_id = self._current_graph.add_node(
            NodeType.EMOTION_DETECTION,
            f"情绪: {emotion}",
            {
                "emotion": emotion,
                "confidence": confidence,
                "pad_state": pad_state or {}
            }
        )
        self._node_mapping["emotion"] = node_id

        if perception_ref:
            self._current_graph.add_edge(
                perception_ref,
                node_id,
                EdgeType.CAUSES,
                label="情绪推断"
            )

        return node_id

    def record_uncertainty_reasoning(self, decision_mode: str, confidence: float,
                                     reasoning: str = "") -> Optional[str]:
        """记录不确定性推理"""
        if not self._enabled or not self._current_graph:
            return None

        emotion_ref = self._node_mapping.get("emotion")

        node_id = self._current_graph.add_node(
            NodeType.UNCERTAINTY_REASONING,
            f"决策模式: {decision_mode}",
            {
                "decision_mode": decision_mode,
                "confidence": confidence,
                "reasoning": reasoning
            }
        )
        self._node_mapping["uncertainty"] = node_id

        if emotion_ref:
            self._current_graph.add_edge(
                emotion_ref,
                node_id,
                EdgeType.CAUSES,
                label="不确定性评估"
            )

        return node_id

    def record_safety_check(self, safety_level: str, risk_factors: List[str],
                           passed: bool, blocked_reason: str = None) -> Optional[str]:
        """记录安全检查"""
        if not self._enabled or not self._current_graph:
            return None

        node_id = self._current_graph.add_node(
            NodeType.SAFETY_CHECK,
            f"安全检查: {safety_level}",
            {
                "safety_level": safety_level,
                "risk_factors": risk_factors,
                "passed": passed,
                "blocked_reason": blocked_reason
            }
        )
        self._node_mapping["safety"] = node_id
        return node_id

    def record_inhibition_rule(self, rule_name: str, triggered: bool,
                              inhibition_effect: str = None) -> Optional[str]:
        """记录抑制规则触发"""
        if not self._enabled or not self._current_graph:
            return None

        safety_ref = self._node_mapping.get("safety")
        uncertainty_ref = self._node_mapping.get("uncertainty")

        node_id = self._current_graph.add_node(
            NodeType.INHIBITION_RULE,
            f"抑制规则: {rule_name}",
            {
                "rule_name": rule_name,
                "triggered": triggered,
                "inhibition_effect": inhibition_effect
            }
        )
        self._node_mapping[f"inhibition_{rule_name}"] = node_id

        if triggered:
            if safety_ref:
                self._current_graph.add_edge(
                    safety_ref,
                    node_id,
                    EdgeType.INHIBITS,
                    weight=0.8,
                    label="安全抑制"
                )
            if uncertainty_ref:
                self._current_graph.add_edge(
                    uncertainty_ref,
                    node_id,
                    EdgeType.INHIBITS,
                    weight=0.5,
                    label="不确定性抑制"
                )

        return node_id

    def record_intent_decision(self, intent: str, confidence: float,
                              alternatives: List[str] = None) -> Optional[str]:
        """记录意图决策"""
        if not self._enabled or not self._current_graph:
            return None

        node_id = self._current_graph.add_node(
            NodeType.INTENT_DECISION,
            f"用户意图: {intent}",
            {
                "intent": intent,
                "confidence": confidence,
                "alternatives": alternatives or []
            }
        )
        self._node_mapping["intent"] = node_id
        return node_id

    def record_action_selection(self, selected_action: str, llm_suggestion: str,
                               modified: bool = False, modification_reason: str = None) -> Optional[str]:
        """记录动作选择"""
        if not self._enabled or not self._current_graph:
            return None

        intent_ref = self._node_mapping.get("intent")
        safety_ref = self._node_mapping.get("safety")

        node_id = self._current_graph.add_node(
            NodeType.ACTION_SELECTION,
            f"选择动作: {selected_action}",
            {
                "selected_action": selected_action,
                "llm_suggestion": llm_suggestion,
                "was_modified": modified,
                "modification_reason": modification_reason
            }
        )
        self._node_mapping["action"] = node_id

        if intent_ref:
            self._current_graph.add_edge(
                intent_ref,
                node_id,
                EdgeType.LEADS_TO,
                label="意图驱动"
            )

        if safety_ref:
            self._current_graph.add_edge(
                safety_ref,
                node_id,
                EdgeType.OVERRIDES if modified else EdgeType.SUPPORTS,
                weight=1.0 if modified else 0.5,
                label="安全修正" if modified else "安全确认"
            )

        return node_id

    def record_final_action(self, action: str, success: bool,
                           execution_details: Dict[str, Any] = None) -> Optional[str]:
        """记录最终执行的动作"""
        if not self._enabled or not self._current_graph:
            return None

        action_ref = self._node_mapping.get("action")
        scene_ref = self._node_mapping.get("scene")
        memory_ref = self._node_mapping.get("memory")

        node_id = self._current_graph.add_node(
            NodeType.FINAL_ACTION,
            f"执行: {action}",
            {
                "action": action,
                "success": success,
                "execution_details": execution_details or {}
            }
        )
        self._node_mapping["final"] = node_id

        if action_ref:
            self._current_graph.add_edge(
                action_ref,
                node_id,
                EdgeType.LEADS_TO,
                label="执行动作"
            )

        if scene_ref:
            self._current_graph.add_edge(
                scene_ref,
                node_id,
                EdgeType.SUPPORTS,
                weight=0.3,
                label="场景相关"
            )

        if memory_ref:
            self._current_graph.add_edge(
                memory_ref,
                node_id,
                EdgeType.SUPPORTS,
                weight=0.3,
                label="记忆驱动"
            )

        return node_id

    def get_current_graph(self) -> Optional[ReasoningGraph]:
        """获取当前推理图"""
        return self._current_graph

    def get_graph_history(self, limit: int = 10) -> List[ReasoningGraph]:
        """获取历史推理图"""
        return self._graph_history[-limit:]

    def get_last_graph(self) -> Optional[ReasoningGraph]:
        """获取最近的推理图"""
        if self._graph_history:
            return self._graph_history[-1]
        return None

    def export_json(self) -> str:
        """导出为 JSON 格式"""
        graphs = [g.to_dict() for g in self._graph_history]
        return json.dumps({
            "graphs": graphs,
            "count": len(graphs)
        }, ensure_ascii=False, indent=2)

    def export_last_markdown(self) -> str:
        """导出最近的推理图为 Markdown"""
        graph = self.get_last_graph()
        if graph:
            return graph.to_markdown()
        return "无可用的推理图"

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
