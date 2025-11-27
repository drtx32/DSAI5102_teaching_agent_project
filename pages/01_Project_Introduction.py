import streamlit as st
from teaching_agent.module.streamlit_bottom_bar import bottom_bar
from teaching_agent.module.streamlit_ask_ai_dialog import ask_ai_button
from teaching_agent.module.streamlit_settings_dialog import settings_button

st.set_page_config(page_title="项目介绍", page_icon="📘")
settings_button()
ask_ai_button()

st.title("📘 项目介绍 — RAG & WebSearch Chatbot")
st.markdown("""
该项目是一个教学演示级别的文档问答 (RAG) 和联网搜索系统原型，侧重于可运行的最小实现，当前已实现并测试的功能包括：

- 基本的 Streamlit 前端与多页面框架（主页面为 [首页.py](../首页.py) / `首页.py`），用于展示与交互。
- 后台的 MCP 服务：
  - 本地 RAG 服务（`teaching_agent/mcp/rag/server.py`）管理 FAISS 向量库并提供检索接口（`RAGVectorStore`）。
  - 网络检索服务（`teaching_agent/mcp/websearch/server.py`）用于在线搜索与抓取网页内容。
- 向量库构建工具：[tools/build_vectordb.py](/tools/build_vectordb.py)，当前以 PDF 为输入样本，生成 embeddings 并保存 FAISS 索引。
- Embedding 与向量存储采用 Ollama embeddings + FAISS 的组合（在 RAG 服务与构建脚本中使用）。
- 异步与多线程运行：使用后台线程 / 协程来承载 agent（`teaching_agent/agent/worker.py` 中的 `teaching_agent.agent.worker.LangGraphWorker`）与 MCP 客户端，实现非阻塞的聊天交互。
- 配置与日志：通过 [`teaching_agent.utils.config.settings`](teaching_agent/utils/config.py) 读取环境配置，通过 [`teaching_agent.utils.logging_config.logger`](teaching_agent/utils/logging_config.py) 记录运行信息。
- 简单的设置 UI（见 `teaching_agent/module/streamlit_settings_dialog.py` 的 `teaching_agent.module.streamlit_settings_dialog.settings_button`），用于输入 provider / API Key 等参数。

注意（已去除尚未实现的功能说明）：
- 当前实现以 PDF 文档处理为主（见构建脚本），`.docx` 的高级兼容性或更多厂商整合尚未完整实现。
- 有些演示页面占位内容可能尚未连接到后端工具，请以实际能运行的模块为准。

快速运行提示：
- 启动所有服务的快捷方式：运行根目录的 [main.py](main.py)（会并行启动 RAG、websearch 和 Streamlit 主页面）。
- 单独启动 Streamlit：`streamlit run 首页.py` 或打开 [app.py](app.py) 的主界面。
""")

bottom_bar(previous_alias="首页", previous_page="首页.py",
           next_alias="Usage Guide", next_page="pages/02_Usage_Guide.py")
