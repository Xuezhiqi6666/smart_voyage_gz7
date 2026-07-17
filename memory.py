"""
需求：实现SmartVoyage的记忆管理模块，包括短期对话记忆、用户偏好记忆和当前任务上下文
支持 MySQL 数据库持久化，服务重启后自动恢复记忆数据

架构图：

┌────────────────────────────────────────────────────────────────────────────┐
│                         ConversationMemory  类                             │
│                                                                            │
│  ┌─────────────────── 内存层 (Memory) ───────────────────┐                │
│  │                                                        │                │
│  │  short_term_messages[]  ── 短期对话 (最多10条)         │                │
│  │  user_profile{}         ── 用户偏好 (座位/舱位等)      │                │
│  │  current_task{}         ── 当前任务上下文 (类型/城市)   │                │
│  │  entity_history[]       ── 提取的实体历史 (最多50条)    │                │
│  └────────────────────┬──────────────────────────────────┘                │
│                       │ add / update / get                                 │
│                       ▼                                                    │
│  ┌─────────────────── 持久层 (MySQL) ────────────────────┐                │
│  │                                                        │                │
│  │  _db_conn ── mysql.connector 连接                      │                │
│  │  set_db_connection() / _ensure_db()  连接管理          │                │
│  │                                                        │                │
│  │  ┌──────────────────────────────────────────────┐     │                │
│  │  │  user_profiles 表                             │     │                │
│  │  │  (profile_key PK, profile_value)              │     │                │
│  │  │  save_profile_to_db()  ← INSERT ... UPSERT   │     │                │
│  │  │  load_profile_from_db()  ← SELECT            │     │                │
│  │  └──────────────────────────────────────────────┘     │                │
│  │  ┌──────────────────────────────────────────────┐     │                │
│  │  │  query_history 表                             │     │                │
│  │  │  (id PK, intent_type, query_content,          │     │                │
│  │  │   query_time)                                 │     │                │
│  │  │  save_entity_to_db()   ← INSERT               │     │                │
│  │  │  load_entities_from_db() ← SELECT ORDER BY    │     │                │
│  │  └──────────────────────────────────────────────┘     │                │
│  │  ┌──────────────────────────────────────────────┐     │                │
│  │  │  short_term_messages 表                       │     │                │
│  │  │  (id PK, role, content, message_time,         │     │                │
│  │  │   message_order)                              │     │                │
│  │  │  save_messages_to_db()   ← DELETE + INSERT    │     │                │
│  │  │  load_messages_from_db() ← SELECT ORDER BY    │     │                │
│  │  └──────────────────────────────────────────────┘     │                │
│  │                                                        │                │
│  │  clear_all_from_db()  ← 清空三张表                     │                │
│  └────────────────────────────────────────────────────────┘                │
│                                                                            │
│  ┌─────────────────── 序列化层 ──────────────────────────┐                │
│  │                                                        │                │
│  │  to_dict()   → 导出记忆为 dict (实体仅取最近5条)       │                │
│  │  from_dict() ← 从 dict 恢复记忆 (@classmethod)         │                │
│  └────────────────────────────────────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────────┘

数据流向：

    用户消息 → add_message() → short_term_messages[] → save_messages_to_db()
                                                      → MySQL: short_term_messages 表
                                                      → (DELETE + 批量 INSERT)

    偏好更新 → update_profile() → user_profile{}     → save_profile_to_db()
                                                      → MySQL: user_profiles 表
                                                      → (INSERT ... ON DUPLICATE KEY UPDATE)

    实体提取 → extract_entities() → entity_history[]  → save_entity_to_db()
                                                      → MySQL: query_history 表
                                                      → (单条 INSERT)

    服务重启 → load_profile_from_db()   → 恢复 user_profile{}
              load_messages_from_db()   → 恢复 short_term_messages[]
              load_entities_from_db()   → 恢复 entity_history[]

方法分类：

    写操作 (Write)          读操作 (Read)           序列化 (Serialize)
    ──────────────────     ──────────────────      ──────────────────
    add_message()          get_short_term_text()   to_dict()
    update_profile()       get_profile_text()      from_dict()
    update_task_context()
    extract_entities()
    clear()

    持久化 (DB Write)       加载 (DB Read)          连接管理
    ──────────────────     ──────────────────      ──────────────────
    save_messages_to_db()  load_messages_from_db() set_db_connection()
    save_profile_to_db()   load_profile_from_db()  _ensure_db()
    save_entity_to_db()    load_entities_from_db()
    clear_all_from_db()
"""
from datetime import datetime
import json
import mysql.connector
import pytz


class ConversationMemory:
    """管理对话记忆的类，包括短期对话、用户偏好和任务上下文"""

    def __init__(self, short_term_limit: int = 10):
        self.short_term_messages = []  # 短期对话消息列表，最多保留short_term_limit条
        self.user_profile = {}  # 用户偏好，如 {"座位喜好": "二等座", "cabin_type": "经济舱"}
        self.current_task = {}  # 当前任务上下文，如 {"type": "train", "departure_city": "北京", "arrival_city": "上海"}
        self.short_term_limit = short_term_limit  # 短期记忆最大长度
        self.entity_history = []  # 历史提取的关键实体
        self._db_conn = None  # 数据库连接

    def set_db_connection(self, db_conn):
        """设置数据库连接（由 ChatService 注入）"""
        self._db_conn = db_conn

    def _ensure_db(self):
        """确保数据库连接有效，断开则自动重连；重连失败时清理死连接"""
        if self._db_conn is None:
            raise RuntimeError("数据库连接未初始化")
        try:
            self._db_conn.ping(reconnect=True)
        except Exception:
            self._db_conn = None  # 清理死连接，避免后续操作反复使用失效对象
            raise RuntimeError("数据库连接已断开")

    def add_message(self, role: str, content: str):
        """添加消息到短期记忆，并持久化到数据库"""
        self.short_term_messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%H:%M:%S')
        })
        # 超过限制则移除最旧的消息
        if len(self.short_term_messages) > self.short_term_limit:
            self.short_term_messages = self.short_term_messages[-self.short_term_limit:]
        # 持久化到数据库
        self.save_messages_to_db()

    def get_short_term_text(self) -> str:
        """获取短期对话的文本格式，用于意图识别和代理调用"""
        lines = []
        for msg in self.short_term_messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        return '\n'.join(lines)

    def update_profile(self, profile_update: dict):
        """更新用户偏好，并持久化到数据库"""
        self.user_profile.update(profile_update)
        self.save_profile_to_db()

    def update_task_context(self, task_update: dict):
        """更新当前任务上下文"""
        self.current_task.update(task_update)

    def extract_entities(self, intent_type: str, query: str):
        """从查询中提取关键实体到历史，并持久化到数据库"""
        self.entity_history.append({
            "type": intent_type,
            "query": query,
            "timestamp": datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        })
        # 只保留最近50条实体记录
        if len(self.entity_history) > 50:
            self.entity_history = self.entity_history[-50:]
        # 持久化单条到数据库
        self.save_entity_to_db(intent_type, query)

    def get_profile_text(self) -> str:
        """获取用户偏好的文本描述，用于注入到prompt中"""
        if not self.user_profile:
            return "无已知的用户偏好"
        items = [f"{k}: {v}" for k, v in self.user_profile.items()]
        return "，".join(items)

    def clear(self):
        """清空所有记忆，同时清除数据库数据"""
        self.short_term_messages = []
        self.user_profile = {}
        self.current_task = {}
        self.entity_history = []
        self.clear_all_from_db()

    def to_dict(self) -> dict:
        """导出记忆为字典，用于序列化"""
        return {
            "short_term_messages": self.short_term_messages,
            "user_profile": self.user_profile,
            "current_task": self.current_task,
            "entity_history": self.entity_history[-5:]  # 只导出最近5条
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationMemory':
        """从字典导入记忆"""
        memory = cls()
        memory.short_term_messages = data.get("short_term_messages", [])
        memory.user_profile = data.get("user_profile", {})
        memory.current_task = data.get("current_task", {})
        memory.entity_history = data.get("entity_history", [])
        return memory

    # ==================== 数据库持久化 ====================

    def save_profile_to_db(self):
        """将用户偏好持久化到数据库（UPSERT）"""
        if self._db_conn is None:
            return
        try:
            self._ensure_db()
            cursor = self._db_conn.cursor()
            for key, value in self.user_profile.items():
                cursor.execute(
                    "INSERT INTO user_profiles (profile_key, profile_value) "
                    "VALUES (%s, %s) ON DUPLICATE KEY UPDATE profile_value = %s",
                    (key, str(value), str(value))
                )
            self._db_conn.commit()
            cursor.close()
        except Exception as e:
            try:
                self._db_conn.rollback()
            except Exception:
                pass
            print(f"保存用户偏好到数据库失败: {e}")

    def load_profile_from_db(self):
        """从数据库加载用户偏好"""
        if self._db_conn is None:
            return
        try:
            self._ensure_db()
            cursor = self._db_conn.cursor(dictionary=True)
            cursor.execute("SELECT profile_key, profile_value FROM user_profiles")
            rows = cursor.fetchall()
            cursor.close()
            self.user_profile = {row["profile_key"]: row["profile_value"] for row in rows}
        except Exception as e:
            print(f"从数据库加载用户偏好失败: {e}")

    def save_entity_to_db(self, intent_type: str, query: str):
        """将单条查询实体持久化到数据库"""
        if self._db_conn is None:
            return
        try:
            self._ensure_db()
            cursor = self._db_conn.cursor()
            now = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "INSERT INTO query_history (intent_type, query_content, query_time) "
                "VALUES (%s, %s, %s)",
                (intent_type, json.dumps({"query": query}, ensure_ascii=False), now)
            )
            self._db_conn.commit()
            cursor.close()
        except Exception as e:
            try:
                self._db_conn.rollback()
            except Exception:
                pass
            print(f"保存查询实体到数据库失败: {e}")

    def load_entities_from_db(self, limit: int = 50):
        """从数据库加载查询历史，按时间倒序取最近N条"""
        if self._db_conn is None:
            return
        try:
            self._ensure_db()
            cursor = self._db_conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT intent_type, query_content, query_time FROM query_history "
                "ORDER BY query_time DESC LIMIT %s",
                (limit,)
            )
            rows = cursor.fetchall()
            cursor.close()
            # 按时间正序排列（倒序取出来后翻转）
            rows.reverse()
            self.entity_history = []
            for row in rows:
                try:
                    query_data = json.loads(row["query_content"])
                    query_text = query_data.get("query", "")
                except Exception:
                    query_text = ""
                self.entity_history.append({
                    "type": row["intent_type"],
                    "query": query_text,
                    "timestamp": row["query_time"].strftime('%Y-%m-%d %H:%M:%S') if row["query_time"] else ""
                })
        except Exception as e:
            print(f"从数据库加载查询历史失败: {e}")

    def save_messages_to_db(self):
        """将短期对话覆盖写入数据库（先删除旧数据，再写入当前列表）"""
        if self._db_conn is None:
            return
        try:
            self._ensure_db()
            cursor = self._db_conn.cursor()
            # 可以不删除，直接更新插入update，因为删除数据是不建议的
            cursor.execute("DELETE FROM short_term_messages")
            for i, msg in enumerate(self.short_term_messages):
                cursor.execute(
                    "INSERT INTO short_term_messages (role, content, message_time, message_order) "
                    "VALUES (%s, %s, %s, %s)",
                    (msg["role"], msg["content"], msg["timestamp"], i)
                )
            self._db_conn.commit()
            cursor.close()
        except Exception as e:
            try:
                self._db_conn.rollback()
            except Exception:
                pass
            print(f"保存短期对话到数据库失败: {e}")

    def load_messages_from_db(self):
        """从数据库加载短期对话"""
        if self._db_conn is None:
            return
        try:
            self._ensure_db()
            cursor = self._db_conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT role, content, message_time FROM short_term_messages ORDER BY message_order ASC"
            )
            rows = cursor.fetchall()
            cursor.close()
            self.short_term_messages = [
                {"role": row["role"], "content": row["content"], "timestamp": row["message_time"]}
                for row in rows
            ]
        except Exception as e:
            print(f"从数据库加载短期对话失败: {e}")

    def clear_all_from_db(self):
        """清空数据库中所有记忆数据"""
        if self._db_conn is None:
            return
        try:
            self._ensure_db()
            cursor = self._db_conn.cursor()
            cursor.execute("DELETE FROM short_term_messages")
            cursor.execute("DELETE FROM query_history")
            cursor.execute("DELETE FROM user_profiles")
            self._db_conn.commit()
            cursor.close()
        except Exception as e:
            print(f"清空数据库记忆失败: {e}")
