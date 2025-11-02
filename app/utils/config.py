from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o"
    # OpenAI 嵌入模型（用于向量化）
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # Embedding 后端：'openai' 或 'sentence_transformers'
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "openai")
    # SentenceTransformers 模型名称（仅在 embedding_backend='sentence_transformers' 时使用）
    sentence_transformers_model: str = os.getenv(
        "SENTENCE_TRANSFORMERS_MODEL", "all-mpnet-base-v2")

    # 向量数据库配置
    chroma_persist_directory: str = "./chroma_db"

    # PostgreSQL配置
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_db: str = os.getenv("POSTGRES_DB", "agent_memory")

    # 文档处理配置
    chunk_size: int = 1000
    chunk_overlap: int = 200


settings = Settings()
