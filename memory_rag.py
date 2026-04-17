"""memory_rag.py
长期记忆模块（RAG）：与 llm_api.py 的会话记忆互补，提供跨会话的长期记忆检索能力。
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils import logger


@dataclass
class MemoryEntry:
    """单条记忆条目"""
    content: str
    timestamp: float = field(default_factory=time.time)
    emotion: str = "neutral"
    vision_desc: str = ""
    importance: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "vision_desc": self.vision_desc,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            emotion=data.get("emotion", "neutral"),
            vision_desc=data.get("vision_desc", ""),
            importance=data.get("importance", 1.0),
        )


class MemoryRAG:
    """基于向量相似度的简易 RAG 记忆库（无外部依赖，使用关键词匹配）"""

    def __init__(self, max_entries: int = 100, persist_path: Optional[str] = None):
        self.max_entries = max_entries
        self.persist_path = persist_path
        self.entries: List[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = [MemoryEntry.from_dict(e) for e in data]
                logger.info(f"已从 {self.persist_path} 加载 {len(self.entries)} 条记忆")
            except Exception as e:
                logger.warning(f"加载记忆失败: {e}")
                self.entries = []

    def _save(self) -> None:
        if self.persist_path:
            try:
                os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
                with open(self.persist_path, "w", encoding="utf-8") as f:
                    json.dump([e.to_dict() for e in self.entries], f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"保存记忆失败: {e}")

    def add(
        self,
        content: str,
        emotion: str = "neutral",
        vision_desc: str = "",
        importance: float = 1.0,
    ) -> None:
        entry = MemoryEntry(
            content=content,
            emotion=emotion,
            vision_desc=vision_desc,
            importance=importance,
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = sorted(self.entries, key=lambda x: (x.importance, x.timestamp), reverse=True)[
                : self.max_entries
            ]
        self._save()

    def add_turn(self, user_text: str, robot_reply: str, emotion: str = "neutral") -> None:
        combined = f"用户: {user_text} | 暖暖: {robot_reply}"
        self.add(combined, emotion=emotion)

    def _score_entry(self, entry: MemoryEntry, query: str) -> float:
        query_lower = query.lower()
        content_lower = entry.content.lower()
        score = 0.0
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = query_words & content_words
        score += len(overlap) * 0.5
        for word in query_words:
            if word in content_lower:
                score += 0.3
        emotion_markers = ["开心", "快乐", "高兴", "难过", "悲伤", "生气", "愤怒", "害怕", "担心", "焦虑"]
        for marker in emotion_markers:
            if marker in query_lower and marker in content_lower:
                score += 0.5
        score += entry.importance * 0.2
        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[MemoryEntry]:
        if not query or not self.entries:
            return []
        scored = [(self._score_entry(e, query), e) for e in self.entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k] if _ > 0]

    def retrieve_as_context(self, query: str, top_k: int = 3) -> str:
        entries = self.retrieve(query, top_k)
        if not entries:
            return ""
        context_parts = []
        for i, e in enumerate(entries, 1):
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.timestamp))
            context_parts.append(f"[历史记忆{i}]({ts}): {e.content}")
        return "\n".join(context_parts)

    def search_by_emotion(self, emotion: str) -> List[MemoryEntry]:
        return [e for e in self.entries if e.emotion == emotion]

    def search_by_time_range(self, start_time: float, end_time: float) -> List[MemoryEntry]:
        return [e for e in self.entries if start_time <= e.timestamp <= end_time]

    def clear(self) -> None:
        self.entries.clear()
        self._save()

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        sorted_entries = sorted(self.entries, key=lambda x: x.timestamp, reverse=True)
        return sorted_entries[:n]


_global_rag: Optional[MemoryRAG] = None


def get_memory_rag(
    max_entries: int = 100,
    persist_path: Optional[str] = None,
) -> MemoryRAG:
    global _global_rag
    if _global_rag is None:
        _global_rag = MemoryRAG(max_entries=max_entries, persist_path=persist_path)
    return _global_rag


def add_long_term_memory(
    user_text: str,
    robot_reply: str,
    emotion: str = "neutral",
) -> None:
    rag = get_memory_rag()
    rag.add_turn(user_text, robot_reply, emotion)


def retrieve_relevant_memory(query: str, top_k: int = 3) -> str:
    rag = get_memory_rag()
    return rag.retrieve_as_context(query, top_k)


def clear_long_term_memory() -> None:
    rag = get_memory_rag()
    rag.clear()
