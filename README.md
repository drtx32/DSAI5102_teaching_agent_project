# RAG & Web Search Chatbot（教学演示级别）

> 一个以 Streamlit 为前端、FAISS 向量库与 Ollama embeddings 为后端的文档问答（RAG）和联网搜索教学原型。

## 快速概览
- **前端**：Streamlit 多页面（入口：`首页.py` / `app.py`）。
- **向量构建**：`tools/build_vectordb.py`（将 PDF 生成 embeddings 并保存 FAISS 索引）。
- **RAG 服务**：`teaching_agent/mcp/rag/server.py`（本地 MCP 服务，管理 FAISS 向量库并提供检索接口）。
- **模型适配**：`teaching_agent/llm/llm_provider.py`（封装多厂商 LLM 调用）。
- **UI 组件**：右下角设置与 Ask AI（`teaching_agent/module/streamlit_settings_dialog.py`、`teaching_agent/module/streamlit_ask_ai_dialog.py`）。

## 快速开始
1. 在项目根目录创建并激活虚拟环境（建议使用 `venv` 或 `conda`）。
2. 安装依赖（可编辑安装）：

```powershell
pip install -e .
```

或按照 `pyproject.toml` 中声明的依赖安装。

3. 配置 API Key / 端点
- 使用环境变量或 Streamlit secrets（参考项目中的示例文件，如 `.env.example`、`.streamlit/secrets.toml.example`）。
- 常见配置项示例：`OPENAI_API_KEY`、`OLLAMA_ENDPOINT`（本地 Ollama 默认为 `http://localhost:11434`）等。

4. 构建向量索引（PDF -> FAISS）
- 将欲索引的 PDF 放到 `assets/pdfs/`。
- 运行：

```powershell
python tools/build_vectordb.py
```

该脚本会读取 PDF、生成 embeddings（项目中使用 Ollama / 指定的 embed 方案），并把 FAISS 索引保存到 `vectordb/faiss`。

5. 重命名 `.streamlit` 文件夹下的 `secrets.toml.example` 为 `secrets.toml`，并修改其中的内容

6. 启动服务
- 一键并行启动（同时启动 RAG、websearch 与前端 Streamlit）：

```powershell
python main.py
```

- 单独启动：
	- RAG MCP：`python teaching_agent/mcp/rag/server.py`（默认监听端口 8002）。
	- WebSearch MCP：`python teaching_agent/mcp/websearch/server.py`（默认监听端口 8001）。
	- 前端：

```powershell
streamlit run 首页.py
```

- Docker启动：
	- 前置要求 - Ollama (Docker):
		```bash
		docker pull ollama/ollama
		docker run -d -v ollama_data:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
		docker exec -it ollama /bin/bash
		# 在ollama docker容器中下载模型
		>>> ollama pull nomic-embeded-text
		```
	- 后台运行
		```bash
		chmod +x ./start.sh
		./start.sh
		```
	- 查看日志
		```bash
		docker logs -f teaching-agent
		```

7. 在 Streamlit 页面中使用
- 点击右下角的设置按钮填写/保存配置（见 `teaching_agent/module/streamlit_settings_dialog.py`），随后使用 Ask AI 按钮（`teaching_agent/module/streamlit_ask_ai_dialog.py`）开启对话与检索式问答。

## 关键文件与目录
- `main.py`：项目快捷启动脚本。
- `首页.py` / `app.py`：Streamlit 前端入口。
- `pages/`：Streamlit 多页面（项目介绍、使用指南、模型说明、FAQ 等）。
- `tools/build_vectordb.py`：向量构建工具（PDF -> embeddings -> FAISS）。
- `vectordb/faiss/`：保存的 FAISS 索引位置。
- `teaching_agent/mcp/rag/server.py`：RAG 向量检索服务（`RAGVectorStore` 类）。
- `teaching_agent/llm/llm_provider.py`：LLM provider 适配层。
- `teaching_agent/module/`：Streamlit 模块化 UI（settings、ask_ai、bottom_bar 等）。

## 常见问题与调试要点
- 设置填写后未生效：在设置弹窗填写后须点击保存，保存后配置才会被前端/后端读取生效。
- 无法找到向量库：确认 `vectordb/faiss` 是否存在；若更新索引，请重启 RAG 服务或使用其 reload 接口。
- 本地 Ollama 无响应：确认 Ollama 服务已启动且 `OLLAMA_ENDPOINT` 指向正确地址（如 `http://localhost:11434`）。

## 可扩展点与贡献指南
- 添加或支持更多模型/Provider：修改 `teaching_agent/llm/llm_provider.py` 并在前端设置中暴露所需参数。
- 添加新的文档格式支持（如 DOCX）：扩展 `tools/build_vectordb.py` 的文档解析逻辑。
- 扩展 Agent / Tools：在 `teaching_agent/agent/` 添加新工具，并在 `teaching_agent/mcp/` 中提供对应的 MCP 接口。

欢迎通过 Pull Request 提交改进，或在课堂演示中按需定制本项目作为教学示例。

---

项目页面摘要：
- `pages/01_Project_Introduction.py`：项目介绍
- `pages/02_Usage_Guide.py`：快速使用指南
- `pages/03_Models_说明.py`：模型与 Provider 说明
- `pages/04_FAQ.py`：常见问题与排错

