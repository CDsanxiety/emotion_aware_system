# memory_rag.py
import time
from typing import List, Dict, Any, Optional
from utils import logger

# 尝试导入 chromadb，如果失败则使用 Mock 类
try:
    import chromadb

    HAS_CHROMADB = True
except ImportError:
    logger.warning("未检测到 chromadb 模块，长期记忆功能将以轻量化模式运行（仅内存暂存）。")
    HAS_CHROMADB = False


class LongTermMemory:
    def __init__(self):
        if HAS_CHROMADB:
            try:
                self.client = chromadb.PersistentClient(path="./chroma_db")
                self.collection = self.client.get_or_create_collection(name="emotion_history")
                logger.info("ChromaDB 长期记忆库已就绪")
            except Exception as e:
                logger.error(f"ChromaDB 初始化失败: {e}，切换至轻量化模式")
                self._init_mock_db()
        else:
            self._init_mock_db()

    def _init_mock_db(self):
        self.collection = None
        self.memory_buffer = []  # 简单的内存缓存
        self.user_profiles = {}  # 简单的用户档案缓存
        logger.info("轻量化记忆模式已启动（数据不会持久化保存）")

    def add_memory(self, user_text, bot_reply, emotion_tag):
        """保存一段记忆"""
        if not user_text: return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        memory_content = f"[{timestamp}] 用户: {user_text} | 机器人: {bot_reply} | 情绪: {emotion_tag}"

        if HAS_CHROMADB and self.collection:
            try:
                self.collection.add(
                    documents=[memory_content],
                    metadatas=[{"type": "chat", "emotion": emotion_tag, "time": timestamp}],
                    ids=[f"chat_{int(time.time())}"]
                )
                return
            except Exception as e:
                logger.error(f"保存记忆到 ChromaDB 失败: {e}")

        # 轻量化模式：存入内存
        self.memory_buffer.append({
            "text": memory_content,
            "user_text": user_text,
            "emotion_tag": emotion_tag,
            "time": timestamp
        })
        if len(self.memory_buffer) > 20:  # 限制内存占用
            self.memory_buffer.pop(0)

    def recall(self, current_context):
        """在调用大模型前，先检索相关历史"""
        if HAS_CHROMADB and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[current_context],
                    n_results=2
                )
                if results['documents'] and results['documents'][0]:
                    return "长期记忆辅助：" + "；".join(results['documents'][0])
            except Exception as e:
                logger.error(f"从 ChromaDB 检索记忆失败: {e}")

        # 轻量化模式：返回最近的 2 条记忆
        if self.memory_buffer:
            recent = [m["text"] for m in self.memory_buffer[-2:]]
            return "近期记忆辅助：" + "；".join(recent)
        return ""

    def save_user_profile(self, user_id: str, birthday: str = None, name: str = None, preferences: dict = None,
                          status: dict = None) -> None:
        profile_id = user_id or "main"
        if profile_id not in self.user_profiles:
            self.user_profiles[profile_id] = {}

        if birthday: self.user_profiles[profile_id]["birthday"] = birthday
        if name: self.user_profiles[profile_id]["name"] = name
        if preferences: self.user_profiles[profile_id].setdefault("preferences", {}).update(preferences)
        if status: self.user_profiles[profile_id].setdefault("status", {}).update(status)

        # 如果有 ChromaDB，尝试持久化（此处简化处理，实际可根据需要完善）
        logger.debug(f"用户档案已更新: {profile_id}")

    def get_user_profile(self, user_id: str = None) -> dict:
        profile_id = user_id or "main"
        return self.user_profiles.get(profile_id, {})

    def get_user_birthday(self, user_id: str = None) -> str:
        return self.get_user_profile(user_id).get("birthday", "")

    def update_user_status(self, user_id: str, status: dict) -> None:
        self.save_user_profile(user_id, status=status)

    def get_user_preference(self, user_id: str, preference_key: str) -> Any:
        return self.get_user_profile(user_id).get("preferences", {}).get(preference_key)

    def list_users(self) -> List[str]:
        return list(self.user_profiles.keys())