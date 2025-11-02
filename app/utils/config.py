from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_endpoint: str = os.getenv(
        "OPENAI_ENDPOINT", "https://api.openai.com/v1")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Anthropic配置
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_endpoint: str = os.getenv(
        "ANTHROPIC_ENDPOINT", "https://api.anthropic.com")

    # Mistral配置
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    mistral_endpoint: str = os.getenv(
        "MISTRAL_ENDPOINT", "https://api.mistral.ai/v1")

    # Google配置
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # Azure OpenAI配置
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    # DeepSeek配置
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_endpoint: str = os.getenv(
        "DEEPSEEK_ENDPOINT", "https://api.deepseek.com")

    # Alibaba配置
    alibaba_api_key: str = os.getenv("ALIBABA_API_KEY", "")
    alibaba_endpoint: str = os.getenv(
        "ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # MoonShot配置
    moonshot_api_key: str = os.getenv("MOONSHOT_API_KEY", "")
    moonshot_endpoint: str = os.getenv(
        "MOONSHOT_ENDPOINT", "https://api.moonshot.cn/v1")

    # Unbound配置
    unbound_api_key: str = os.getenv("UNBOUND_API_KEY", "")
    unbound_endpoint: str = os.getenv(
        "UNBOUND_ENDPOINT", "https://api.getunbound.ai")

    # IBM配置
    ibm_api_key: str = os.getenv("IBM_API_KEY", "")
    ibm_endpoint: str = os.getenv(
        "IBM_ENDPOINT", "https://us-south.ml.cloud.ibm.com")
    ibm_project_id: str = os.getenv("IBM_PROJECT_ID", "")

    # Grok配置
    grok_api_key: str = os.getenv("GROK_API_KEY", "")
    grok_endpoint: str = os.getenv("GROK_ENDPOINT", "https://api.x.ai/v1")

    # Ollama配置
    ollama_endpoint: str = os.getenv(
        "OLLAMA_ENDPOINT", "http://localhost:11434")

    # SiliconFlow配置
    siliconflow_api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    siliconflow_endpoint: str = os.getenv(
        "SILICONFLOW_ENDPOINT", "https://api.siliconflow.cn/v1")

    # ModelScope配置
    modelscope_api_key: str = os.getenv("MODELSCOPE_API_KEY", "")
    modelscope_endpoint: str = os.getenv(
        "MODELSCOPE_ENDPOINT", "https://api-inference.modelscope.cn/v1")

    # Embedding 后端：'openai' 或 'sentence_transformers'
    embedding_backend: str = os.getenv(
        "EMBEDDING_BACKEND", "sentence_transformers")
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
