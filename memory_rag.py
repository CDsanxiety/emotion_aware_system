import time
import chromadb
from chromadb.utils import embedding_functions

class LongTermMemory:
    def __init__(self):
        # 纯本地运行的向量数据库
        self.client = chromadb.PersistentClient(path="./robot_memory_db")
        # 使用轻量级中文嵌入模型 (或者通过 API 调用 Qwen 的 Embedding)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(name="user_profile")
        # 会话记忆缓冲区
        self.session_buffer = []
        # 缓冲区大小阈值
        self.buffer_threshold = 5
        # 上次存储摘要的日期
        self.last_summary_date = time.strftime("%Y-%m-%d")

    def save_memory(self, user_text, llm_summary, emotion_tag):
        """记录有价值的记忆片段"""
        # 先存入会话缓冲区
        self.session_buffer.append({
            "user_text": user_text,
            "llm_summary": llm_summary,
            "emotion_tag": emotion_tag,
            "timestamp": time.time()
        })
        
        # 检查是否需要提取摘要并存储
        self.check_and_store_summary()

    def check_and_store_summary(self):
        """检查是否需要提取摘要并存储"""
        current_date = time.strftime("%Y-%m-%d")
        
        # 检查是否是新的一天
        if current_date != self.last_summary_date:
            self.store_daily_summary()
            self.last_summary_date = current_date
        # 检查缓冲区是否达到阈值
        elif len(self.session_buffer) >= self.buffer_threshold:
            self.store_buffer_summary()

    def store_daily_summary(self):
        """存储每日摘要"""
        if not self.session_buffer:
            return
        
        # 生成每日摘要
        summary = self.generate_summary(self.session_buffer)
        doc_id = f"daily_{int(time.time())}"
        
        # 存储摘要
        self.collection.add(
            documents=[summary],
            metadatas=[{"type": "daily_summary", "date": self.last_summary_date}],
            ids=[doc_id]
        )
        
        # 清空缓冲区
        self.session_buffer = []
        print(f"已存储每日摘要: {summary}")

    def store_buffer_summary(self):
        """存储缓冲区摘要"""
        if not self.session_buffer:
            return
        
        # 生成缓冲区摘要
        summary = self.generate_summary(self.session_buffer)
        doc_id = f"buffer_{int(time.time())}"
        
        # 存储摘要
        self.collection.add(
            documents=[summary],
            metadatas=[{"type": "buffer_summary", "count": len(self.session_buffer)}],
            ids=[doc_id]
        )
        
        # 清空缓冲区
        self.session_buffer = []
        print(f"已存储缓冲区摘要: {summary}")

    def generate_summary(self, memory_list):
        """生成摘要"""
        # 简单的摘要生成逻辑，实际应用中可以使用LLM生成更高级的摘要
        if not memory_list:
            return ""
        
        # 提取关键信息
        user_texts = [item["user_text"] for item in memory_list if item["user_text"]]
        emotions = [item["emotion_tag"] for item in memory_list]
        
        # 生成摘要
        summary_parts = []
        if user_texts:
            summary_parts.append(f"用户提到: {' '.join(user_texts[:3])}")
        if emotions:
            # 统计最常见的情绪
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            most_common_emotion = max(emotion_counts, key=emotion_counts.get)
            summary_parts.append(f"主要情绪: {most_common_emotion}")
        
        summary = "; ".join(summary_parts)
        return summary if summary else "日常对话"

    def recall(self, current_context):
        """在调用大模型前，先检索相关历史"""
        results = self.collection.query(
            query_texts=[current_context],
            n_results=2 # 召回最相关的2条记忆
        )
        if results['documents'][0]:
            return "长期记忆辅助：" + "；".join(results['documents'][0])
        return ""

    def save_user_profile(self, birthday: str = None, name: str = None, preferences: dict = None) -> None:
        """保存用户档案信息"""
        profile_data = {}
        if birthday:
            profile_data["birthday"] = birthday
        if name:
            profile_data["name"] = name
        if preferences:
            profile_data["preferences"] = preferences

        if not profile_data:
            return

        profile_id = "user_profile_main"
        existing = self.collection.get(ids=[profile_id])

        if existing and existing.get("documents"):
            self.collection.update(
                ids=[profile_id],
                documents=[str(profile_data)],
                metadatas=[{"type": "user_profile", "updated_at": time.strftime("%Y-%m-%d")}]
            )
        else:
            self.collection.add(
                documents=[str(profile_data)],
                metadatas=[{"type": "user_profile", "updated_at": time.strftime("%Y-%m-%d")}],
                ids=[profile_id]
            )

    def get_user_profile(self) -> dict:
        """获取用户档案"""
        profile_id = "user_profile_main"
        try:
            result = self.collection.get(ids=[profile_id])
            if result and result.get("documents") and len(result["documents"]) > 0:
                profile_str = result["documents"][0]
                if profile_str.startswith("{") and profile_str.endswith("}"):
                    import ast
                    try:
                        return ast.literal_eval(profile_str)
                    except:
                        pass
        except:
            pass
        return {}

    def get_user_birthday(self) -> str:
        """从用户档案获取生日信息"""
        profile = self.get_user_profile()
        return profile.get("birthday", "")
