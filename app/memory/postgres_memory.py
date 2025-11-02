from langchain.memory import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from app.utils.config import settings
from app.utils.logging_config import logger
import json


class PostgreSQLChatMessageHistory(ChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conn = self._get_connection()
        self._create_table_if_not_exists()
        super().__init__()
        self.messages = self._load_messages()

    def _get_connection(self):
        return psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db
        )

    def _create_table_if_not_exists(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """)
                self.conn.commit()
        except Exception as e:
            logger.error(f"创建表失败: {str(e)}")
            self.conn.rollback()

    def _load_messages(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                SELECT message FROM chat_history 
                WHERE session_id = %s 
                ORDER BY created_at ASC
                """, (self.session_id,))
                rows = cur.fetchall()
                messages = []
                for row in rows:
                    msg_data = row[0]
                    if msg_data["type"] == "human":
                        messages.append(HumanMessage(
                            content=msg_data["content"]))
                    elif msg_data["type"] == "ai":
                        messages.append(AIMessage(content=msg_data["content"]))
                return messages
        except Exception as e:
            logger.error(f"加载消息失败: {str(e)}")
            return []

    def add_message(self, message: BaseMessage):
        super().add_message(message)
        try:
            with self.conn.cursor() as cur:
                msg_type = "human" if isinstance(
                    message, HumanMessage) else "ai"
                cur.execute("""
                INSERT INTO chat_history (session_id, message)
                VALUES (%s, %s)
                """, (self.session_id, Json({
                    "type": msg_type,
                    "content": message.content,
                    "timestamp": datetime.now().isoformat()
                })))
                self.conn.commit()
        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}")
            self.conn.rollback()

    def clear(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                DELETE FROM chat_history WHERE session_id = %s
                """, (self.session_id,))
                self.conn.commit()
            super().clear()
        except Exception as e:
            logger.error(f"清空消息失败: {str(e)}")
            self.conn.rollback()
