import streamlit as st
from app.module.streamlit_bottom_bar import bottom_bar
from app.module.streamlit_ask_ai_dialog import ask_ai_button
from app.module.streamlit_settings_dialog import settings_button

st.set_page_config(page_title="项目介绍", page_icon="📘")
settings_button()
ask_ai_button()

st.title("📘 项目介绍 — DOCX RAG Chatbot")
st.markdown("""
这个项目是一个基于 LangChain + FAISS + Streamlit 的文档问答（RAG）示例，特色如下：

- 支持多文档 `.docx` 上传并做合法性校验（避免 BadZipFile 问题）。
- 中文友好的分块器（`RecursiveCharacterTextSplitter`），支持自定义分块大小/重叠。
- 支持多厂商 LLM 与 Embeddings（通过工厂函数动态实例化）。
- 建索引时显示进度条与状态信息，支持增量批量入库。
- 检索支持 Top-K、MMR 与基于距离的阈值过滤。

本页面概述项目目标、架构与设计决策，便于教学和讲解。
""")

st.header("目标")
st.write(
    "帮助学生或研发人员快速理解如何把本地文档（.docx）接入向量索引并用多厂商大模型做文档问答（RAG）。")

st.header("高层架构")
with st.expander("点击查看架构概览", expanded=True):
    st.markdown("""
    - 文档加载：`Docx2txtLoader` 将 `.docx` 转为文本块。
    - 文本分割：`RecursiveCharacterTextSplitter` 进行分块。
    - 嵌入生成：支持 OpenAI / Anthropic/Voyage / Google / 本地 Ollama / Sentence-Transformers 等。
    - 向量索引：使用 FAISS 承载向量并提供检索接口。
    - 检索策略：Top-K / MMR / 距离阈值过滤相结合。
    - 生成回答：把检索到的上下文拼接到提示词，调用所选 LLM 生成回答。
    """)

st.header("项目文件说明")
st.markdown("""
- `app.py`：主应用入口（上传、构建索引、聊天界面、设置对话窗）。
- `pages/`：教学文档页面（当前自动注册为 Streamlit 多页应用）。
- `doc/architecture.md`：架构文档（项目内可直接查看）。
""")

st.header("课堂/演示建议")
st.markdown("""
- 先运行主应用并展示上传 -> 构建索引 -> 查询的完整流程。
- 再展示 `pages/02_Usage_Guide.py` 中的逐步安装与环境变量说明。
- 讲解 `make_llm` 与 `make_embeddings` 的工厂模式（为何要做参数过滤与动态导入）。
- 最后演示如何扩展到新的提供商（添加 `pages/` 新页面记录步骤）。
""")

st.info("提示：要返回主应用页面（可以在左上角 Streamlit 菜单或侧边栏导航里找到主页面）。")

bottom_bar(previous_alias="首页", previous_page="首页.py",
           next_alias="Usage Guide", next_page="pages/02_Usage_Guide.py")