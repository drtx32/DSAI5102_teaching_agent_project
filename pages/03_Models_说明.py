import streamlit as st
from app.module.streamlit_bottom_bar import bottom_bar
from app.module.streamlit_ask_ai_dialog import ask_ai_button
from app.module.streamlit_settings_dialog import settings_button

st.set_page_config(page_title="模型说明", page_icon="🤖")
settings_button()
ask_ai_button()

st.title("🤖 模型与 Embeddings 说明")

st.markdown("""
本页列出项目中常用的 LLM 与 Embeddings 提示、选型建议与注意事项。
""")

st.header("LLM 提示与建议")
st.markdown("""
- 小模型 vs 大模型：大模型（如 `gpt-4o`/`claude-3`）通常在复杂推理上更好，但延迟与费用更高。
- 本地模型（Ollama、Mistral 本地部署）快速且可离线测试，但可能需要更多本地资源。
- 使用温度（`temperature`）控制回答多样性：教学演示建议 `0.0 - 0.3` 保持确定性。
""")

st.header("Embeddings 选型")
st.markdown("""
- 语义搜索一般使用文本嵌入模型（如 OpenAI 的 `text-embedding-3-small` / `text-embedding-3-large`）。
- 中文任务可选择 `BAAI/bge-large-zh-v1.5`、或本地 `sentence-transformers` 的中文模型。
- 对于没有原生 embeddings 的 LLM（如部分 Claude 旧型号），可以配对 Voyage / Nomic / 本地 HF 模型。
""")

st.header("推荐快速参考表")
st.markdown("""
- `openai` LLM: `gpt-4o-mini`； Embedding: `text-embedding-3-small`。
- `anthropic` LLM: `claude-3-...`； Embedding: `voyage-3`（via Voyage 插件）。
- `google` LLM: `gemini-2.0-flash`； Embedding: `models/text-embedding-004`。
- `ollama`（本地）: 按本地镜像名选择；Embedding: `nomic-embed-text`。
""")

st.header("如何选择")
st.markdown("""
- 优先考虑成本/延迟/隐私：如果数据敏感，优先本地 Embeddings + 本地 LLM。 
- 如果希望高准确率与较好可解释性，优先使用大模型 + 高质量 embedding。
- 在教学中建议先用 `sentence-transformers` 做本地 Embedding，逐步演示云服务接入。
""")

st.info("提示：如果需要，我可以把这张推荐表转换成可下载的 CSV 或漂亮的表格页面。")

bottom_bar(previous_alias="Usage Guide", previous_page="pages/02_Usage_Guide.py",
           next_alias="FAQ", next_page="pages/04_FAQ.py")
