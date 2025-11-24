import streamlit as st
from app.module.streamlit_bottom_bar import bottom_bar
from app.module.streamlit_ask_ai_dialog import ask_ai_button
from app.module.streamlit_settings_dialog import settings_button

st.set_page_config(page_title="使用方法", page_icon="🛠️")
settings_button()
ask_ai_button()

st.title("🛠️ 使用方法（快速起步）")

st.markdown("""
本页只描述已实现并经过测试的功能与最小运行步骤 — 不包含尚未实现的页面或功能。
""")

st.header("1) 安装依赖")
st.markdown("""
推荐使用虚拟环境（例如 Conda 或 venv）。项目依赖在 [pyproject.toml](pyproject.toml) 中声明，可以用如下方式安装：

- 可用的快速方式（在项目根目录）：
  - pip（如果你有 requirements.txt）：`pip install -r requirements.txt`
  - 或直接从项目安装：`pip install -e .`

请确保安装了关键依赖：Streamlit、FAISS、PyMuPDF、langchain 及相关 provider 包（见 `pyproject.toml`）。
""")

st.header("2) 必要配置")
st.markdown("""
将 API Key / 端点放入环境变量或 Streamlit secrets（见 `.env.example` 与 `.streamlit/secrets.toml.example`）。

常用配置项示例（使用环境变量或在设置中填写）：
- `OPENAI_API_KEY`, `OPENAI_ENDPOINT`
- `OLLAMA_ENDPOINT`（若使用本地 Ollama）
- 也可通过页面右下角的设置弹窗进行输入（使用：[`app.module.streamlit_settings_dialog.settings_button`](app/module/streamlit_settings_dialog.py)）。
查看运行时配置对象：[`app.utils.config.settings`](app/utils/config.py)。
""")

st.header("3) 构建向量索引（当前仅支持 PDF 输入）")
st.markdown("""
项目目前的向量构建脚本以 PDF 为输入（不是 .docx）。构建流程如下：

1. 将要索引的 PDF 放到 `assets/pdfs/`（或在运行脚本时指定路径）。
2. 运行向量构建脚本：
   - `python tools/build_vectordb.py`
   该脚本会读取 PDF、生成 embeddings（使用 `nomic-embed-text` via Ollama）并保存 FAISS 索引到 `vectordb/faiss`。
   参考实现：[`tools.build_vectordb.build`](tools/build_vectordb.py)

构建完成后，向量库位于 `vectordb/faiss`，RAG 服务会从该路径加载索引。
""")

st.header("4) 启动服务（可选：整组或单独启动）")
st.markdown("""
- 一键并行启动（同时启动 RAG、websearch 和 Streamlit 主页面）：
  - `python main.py`（内部会分别在后台启动 `app/mcp/rag/server.py`、`app/mcp/websearch/server.py` 以及 Streamlit）
  - 参见启动脚本：[main.py](main.py)

- 单独启动：
  - 启动 RAG MCP：`python app/mcp/rag/server.py`（服务监听端口 8002，参考：[`app.mcp.rag.server.RAGVectorStore`](app/mcp/rag/server.py)）
  - 启动 WebSearch MCP：`python app/mcp/websearch/server.py`（端口 8001）
  - 仅启动前端：`streamlit run 首页.py` 或 `streamlit run app.py`

注意：RAG 服务从 `vectordb/faiss` 加载索引（见：[`app/mcp/rag/server.py`](app/mcp/rag/server.py)），如更新索引可调用 RAG 的 reload 工具或重启服务。
""")

st.header("5) 使用聊天与 Agent（已实现）")
st.markdown("""
- 在 Streamlit 页面启动后，可通过右下角的 Ask AI 按钮（实现：[`app.module.streamlit_ask_ai_dialog.ask_ai_button`](app/module/streamlit_ask_ai_dialog.py)）打开聊天窗口。
- 后台的 LangGraph Worker 在需要时由前端启动（实现：[`app.agent.worker.LangGraphWorker`](app/agent/worker.py)），它会调用 MCP 的 RAG 与 websearch 工具进行检索式问答。
- 日志与运行信息记录由：[`app.utils.logging_config.logger`](app/utils/logging_config.py) 管理。
""")

st.header("6) 常见问题与调试要点")
st.markdown("""
- 无法找到向量库：确认 `vectordb/faiss` 是否存在，或在 RAG 服务启动后使用 `reload_vectorstore` 工具重新加载（参见 `app/mcp/rag/server.py`）。
- 构建失败或 embedding 错误：检查 Ollama / embedding 后端是否可用，以及 CPU/并发限制（构建脚本会使用多进程并行）。
- 若要测试本地 Ollama，请先启动本地 Ollama 服务并在设置里把 `OLLAMA_ENDPOINT` 指向 `http://localhost:11434`。
""")

bottom_bar(previous_page="pages/01_Project_Introduction.py",
           previous_alias="Project Introduction",
           next_alias="Models 说明", next_page="pages/03_Models_说明.py")
