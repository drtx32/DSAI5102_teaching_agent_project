from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from app.agent.tools import AgentTools
from app.agent import llm_provider
from app.memory.postgres_memory import PostgreSQLChatMessageHistory
from app.utils.config import settings
from app.utils.logging_config import logger


class TeachingAgent:
    def __init__(self, session_id: str, provider: str | None = None, model: str | None = None, api_key: str | None = None):
        """创建教学 Agent

        Args:
            session_id: 会话 ID，用于内存持久化
            provider: LLM provider（默认使用 'openai'）
            model: 模型名称（默认使用 settings.openai_model）
            api_key: API key（默认使用 settings.openai_api_key）
        """
        self.session_id = session_id
        chosen_provider = provider or "openai"
        chosen_model = model or settings.openai_model
        chosen_api_key = api_key or settings.openai_api_key

        # 使用 llm_provider 工厂创建 LLM
        try:
            self.llm = llm_provider.get_llm_model(
                provider=chosen_provider,
                model_name=chosen_model,
                api_key=chosen_api_key,
                temperature=0.3,
            )
        except Exception as e:
            logger.exception(
                "llm_provider 创建模型失败，回退到 ChatOpenAI: %s", e)
            # 回退到直接使用 ChatOpenAI
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                from langchain.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(
                api_key=chosen_api_key,
                model_name=chosen_model,
                temperature=0.3
            )

        self.tools = AgentTools().get_tools()
        self.memory = self._initialize_memory()
        self.agent = self._initialize_agent()

    def _initialize_memory(self):
        """初始化带PostgreSQL存储的记忆"""
        message_history = PostgreSQLChatMessageHistory(
            session_id=self.session_id)
        return ConversationBufferWindowMemory(
            chat_memory=message_history,
            memory_key="chat_history",
            return_messages=True,
            k=10  # 保留最近10轮对话
        )

    def _initialize_agent(self):
        """初始化Agent"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, query: str):
        """运行Agent处理查询"""
        try:
            logger.info(f"Agent处理查询: {query} (session_id: {self.session_id})")
            result = self.agent.run(query)
            logger.info(f"Agent返回结果 (session_id: {self.session_id})")
            return result
        except Exception as e:
            logger.error(
                f"Agent运行错误: {str(e)} (session_id: {self.session_id})")
            return f"处理查询时发生错误: {str(e)}"
