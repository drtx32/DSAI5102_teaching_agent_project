from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.tools import AgentTools
from app.agent import llm_provider
from app.memory.postgres_memory import PostgreSQLChatMessageHistory
from app.utils.config import settings
from app.utils.logging_config import logger
from typing import Union, List, Dict, Any
import re


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
        self.tool_dict = {tool.name: tool for tool in self.tools}
        self.message_history = self._initialize_memory()
        self.agent = self._initialize_agent()

    def _initialize_memory(self):
        """初始化带PostgreSQL存储的记忆"""
        return PostgreSQLChatMessageHistory(session_id=self.session_id)

    def _get_chat_history(self, limit: int = 10) -> List[Any]:
        """获取最近的聊天历史"""
        try:
            messages = self.message_history.messages
            # 只保留最近 limit 条消息
            return messages[-limit*2:] if len(messages) > limit*2 else messages
        except Exception as e:
            logger.error(f"获取聊天历史失败: {e}")
            return []

    def _parse_agent_output(self, output: str) -> Union[AgentAction, AgentFinish]:
        """解析 agent 输出"""
        # 查找 Final Answer
        if "Final Answer:" in output:
            return AgentFinish(
                return_values={"output": output.split(
                    "Final Answer:")[-1].strip()},
                log=output,
            )

        # 查找 Action 和 Action Input
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", output)
        action_input_match = re.search(
            r"Action Input:\s*(.+?)(?:\n|$)", output, re.DOTALL)

        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            return AgentAction(tool=action, tool_input=action_input, log=output)

        # 如果无法解析，返回输出作为最终答案
        return AgentFinish(
            return_values={"output": output},
            log=output,
        )

    def _initialize_agent(self):
        """初始化Agent（使用 LangChain 1.0 新API）"""
        # 创建工具描述
        tools_desc = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in self.tools])
        tool_names = ", ".join([tool.name for tool in self.tools])

        # 创建提示词模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是一个教学助手，可以帮助学生学习课程内容。
你有以下工具可以使用：

{tools_desc}

使用以下格式回答问题：

Question: 用户的问题
Thought: 思考应该做什么
Action: 使用的工具名称（{tool_names}中的一个）
Action Input: 工具的输入
Observation: 工具返回的结果
... (可以重复 Thought/Action/Action Input/Observation 多次)
Thought: 我现在知道最终答案了
Final Answer: 给用户的最终回答

开始！记住在给出最终答案时要详细和有帮助。"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
        ])

        # 创建简单的 chain
        chain = prompt | self.llm | StrOutputParser()

        # 创建 agent executor（简化版本，手动处理工具调用）
        return chain

    def run(self, query: str):
        """运行Agent处理查询"""
        try:
            logger.info(f"Agent处理查询: {query} (session_id: {self.session_id})")

            # 获取历史消息
            chat_history = self._get_chat_history()

            # 最多迭代5次
            max_iterations = 5
            current_input = query

            for i in range(max_iterations):
                # 调用 LLM
                response = self.agent.invoke({
                    "input": current_input,
                    "chat_history": chat_history
                })

                logger.info(f"Agent 响应 (迭代 {i+1}): {response[:200]}...")

                # 解析输出
                parsed = self._parse_agent_output(response)

                # 如果是最终答案，返回
                if isinstance(parsed, AgentFinish):
                    answer = parsed.return_values["output"]
                    # 保存到记忆
                    self.message_history.add_user_message(query)
                    self.message_history.add_ai_message(answer)
                    logger.info(f"Agent返回结果 (session_id: {self.session_id})")
                    return answer

                # 如果是工具调用
                if isinstance(parsed, AgentAction):
                    tool_name = parsed.tool
                    tool_input = parsed.tool_input

                    # 执行工具
                    if tool_name in self.tool_dict:
                        tool = self.tool_dict[tool_name]
                        try:
                            observation = tool.run(tool_input)
                            logger.info(f"工具 {tool_name} 执行成功")
                        except Exception as e:
                            observation = f"工具执行错误: {str(e)}"
                            logger.error(f"工具 {tool_name} 执行失败: {e}")
                    else:
                        observation = f"错误: 未找到工具 {tool_name}"

                    # 更新输入，包含观察结果
                    current_input = f"{response}\nObservation: {observation}\nThought:"
                else:
                    # 无法解析，直接返回
                    self.message_history.add_user_message(query)
                    self.message_history.add_ai_message(response)
                    return response

            # 达到最大迭代次数
            final_answer = "抱歉，我在处理您的问题时遇到了困难。请尝试重新表述您的问题。"
            self.message_history.add_user_message(query)
            self.message_history.add_ai_message(final_answer)
            return final_answer

        except Exception as e:
            logger.error(
                f"Agent运行错误: {str(e)} (session_id: {self.session_id})")
            return f"处理查询时发生错误: {str(e)}"
