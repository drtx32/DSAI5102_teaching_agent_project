# worker.py
import threading
import queue
import asyncio
from contextlib import AsyncExitStack
from typing import List
from typing_extensions import TypedDict
from typing import Annotated

# LangGraph / LangChain 相关
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langchain.chat_models import BaseChatModel

from ..llm.llm_provider import get_llm_model


SYSTEM_PROMPT = """
You are a Data-Science Teaching Assistant Agent.

Your goals:
1. Provide clear, technically accurate explanations and code for all data-science-related topics.
2. Use external tools when appropriate and ALWAYS report which tool you used.
3. Strictly follow the tool routing rules below.

=====================
TOOL ROUTING RULES
=====================

TOOLS YOU HAVE:
- rag.search  : Searches the local FAISS-based knowledge base.
- websearch.search : Searches the internet.

ROUTING PRIORITY:
1. For ANY question related to:
   - data science, statistics, probability
   - machine learning, deep learning, reinforcement learning
   - forecasting, time-series, quantitative finance
   - neural networks, transformers, embeddings, optimization
   - mathematical derivations or computational modeling
   - Python scientific stack (numpy, pandas, sklearn, pytorch)
   → You MUST call `rag.search` FIRST.

2. You may call `websearch.search` ONLY IF:
   - rag returns zero relevant information, OR
   - the user’s question explicitly requires real-time or current events.

3. For general news, current events, or real-world updates:
   → Prefer `websearch.search`.

4. If unsure, ALWAYS default to RAG first.

=====================
OUTPUT REQUIREMENTS
=====================

Each time you answer, at the END of your final answer you MUST add:

[Tool Used: <rag | websearch | none>]

Where:
- Use "rag" if you called rag.search.
- Use "websearch" if you called websearch.search.
- Use "none" if you answered directly without external tools.

Never omit this tag.

=====================
STYLE GUIDELINES
=====================
- Provide step-by-step, intuitive explanations suitable for students.
- When teaching, use diagrams, pseudocode, or math formulas when helpful.
- Prefer clarity over brevity.
- Always mention assumptions or limitations.
- When code is provided, ensure it is correct and runnable.

=====================
BEHAVIORAL IDENTITY
=====================
You are not a general chatbot.
You are a Data-Science Teaching Assistant Agent.
Your primary purpose is to help users understand technical concepts deeply.
"""


class LangGraphWorker(threading.Thread):
    """
    后台线程工作器，用于在独立的线程中运行 LangGraph agent
    """

    def __init__(self, request_q: queue.Queue[str], reply_q: queue.Queue[str], config: dict) -> None:
        super().__init__(daemon=True)
        self.request_q = request_q
        self.reply_q = reply_q
        self.config = config
        self.thread_id = config.get("thread_id", "worker-thread")

        self.connections = {
            "websearch": {
                "url": "http://127.0.0.1:8881/mcp",
                "transport": "streamable_http",
            },
            "rag": {
                "url": "http://127.0.0.1:8882/mcp",
                "transport": "streamable_http",
            }
        }
        # MCP 客户端配置
        self.client = MultiServerMCPClient(self.connections)

        # 这些将在 initialize 中设置
        self.graph = None
        self._stop = False

    async def create_graph(self, *sessions) -> CompiledStateGraph:
        """
        创建 LangGraph 工作流
        """
        # 初始化 LLM
        llm: BaseChatModel = get_llm_model(
            provider=self.config["provider"],
            model_name=self.config["model"],
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
            temperature=0
        )

        tools = []
        for session in sessions:
            tools += await load_mcp_tools(session)
        llm_with_tool = llm.bind_tools(tools, parallel_tool_calls=True)

        # 加载系统提示词, 必须是第一个 session
        # system_prompt = await load_mcp_prompt(sessions[0], "system_prompt")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),  # system_prompt[0].content
            MessagesPlaceholder("messages")
        ])
        chat_llm = prompt_template | llm_with_tool

        # 定义状态
        class State(TypedDict):
            messages: Annotated[List[AnyMessage], add_messages]

        # 定义节点
        def chat_node(state: State) -> State:
            state["messages"] = chat_llm.invoke(
                {"messages": state["messages"]})
            return state

        # 构建图
        graph_builder = StateGraph(State)
        graph_builder.add_node("chat_node", chat_node)
        graph_builder.add_node("tool_node", ToolNode(tools=tools))
        graph_builder.add_edge(START, "chat_node")
        graph_builder.add_conditional_edges("chat_node", tools_condition, {
            "tools": "tool_node",
            "__end__": END
        })
        graph_builder.add_edge("tool_node", "chat_node")

        return graph_builder.compile(checkpointer=MemorySaver())

    async def async_main(self) -> None:
        loop = asyncio.get_running_loop()

        sessions = list(self.connections.keys())
        async with AsyncExitStack() as stack:
            session_objs = []
            for name in sessions:
                session = await stack.enter_async_context(self.client.session(name))
                session_objs.append(session)
            agent = await self.create_graph(*session_objs)
            while not self._stop:
                message = await loop.run_in_executor(None, self.request_q.get)
                if message is None:
                    print("收到停止信号，退出线程")
                    self._stop = True
                    break
                response = await agent.ainvoke({
                    "messages": [HumanMessage(content=message)]},
                    config=RunnableConfig(
                        configurable={"thread_id": self.thread_id})
                )
                reply = response["messages"][-1].content
                await loop.run_in_executor(None, self.reply_q.put, reply)

    def run(self) -> None:
        """
        线程入口点
        """
        # 创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 运行主协程
            loop.run_until_complete(self.async_main())
        finally:
            # 清理事件循环
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
