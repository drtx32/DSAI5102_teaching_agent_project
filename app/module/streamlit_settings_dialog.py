from typing import Optional

import os
import streamlit as st
from ..llm.llm_provider import model_names
from ..agent.worker import LangGraphWorker


# -----------------------------
# Utility functions
# -----------------------------
def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Safely read configs: prefer env vars, fallback to st.secrets, or use default."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return default


# -----------------------------
# 设置弹窗（st.dialog）
# -----------------------------
@st.dialog("⚙️ 设置与参数", width="medium")
def show_settings():
    st.markdown("**模型参数**")
    default_provider = read_secret("LLM_PROVIDER", "OpenAI")
    default_base_url = read_secret("BASE_URL", "https://api.openai.com/v1")
    default_api_key = read_secret("API_KEY", "")
    # default_embed = read_secret("EMBEDDING_MODEL", "text-embedding-3-small")
    default_llm = read_secret("LLM_MODEL", "gpt-4o-mini")

    # 用于全局参数传递
    if "llm_provider" not in st.session_state:
        st.session_state["llm_provider"] = default_provider
    if "base_url" not in st.session_state:
        st.session_state["base_url"] = default_base_url
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = default_api_key
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = default_llm
    if "embed_provider" not in st.session_state:
        st.session_state["embed_provider"] = "openai"
    if "cfg" not in st.session_state:
        st.session_state["cfg"] = {}
    # if "embed_model" not in st.session_state:
    #     st.session_state["embed_model"] = default_embed
    # if "top_k" not in st.session_state:
    #     st.session_state["top_k"] = 4
    # if "use_mmr" not in st.session_state:
    #     st.session_state["use_mmr"] = True
    # if "use_threshold" not in st.session_state:
    #     st.session_state["use_threshold"] = True
    # if "dist_threshold" not in st.session_state:
    #     st.session_state["dist_threshold"] = 0.45
    # if "chunk_size" not in st.session_state:
    #     st.session_state["chunk_size"] = 800
    # if "chunk_overlap" not in st.session_state:
    #     st.session_state["chunk_overlap"] = 100

    # llm_provider_list = [
    #     "openai", "azure_openai", "anthropic", "deepseek", "google", "alibaba",
    #     "moonshot", "unbound", "ibm", "grok", "ollama", "mistral", "siliconflow",
    #     "modelscope", "custom"
    # ]
    with st.form("settings_form"):
        llm_provider = st.selectbox(
            "LLM Provider", model_names.keys(),
            index=list(model_names.keys()).index(st.session_state.llm_provider))
        base_url = st.text_input(
            "Base URL (OpenAI兼容)", value=st.session_state.base_url, help="如 https://api.openai.com/v1")
        api_key = st.text_input(
            "API Key (不会存储)", type="password", value=st.session_state.api_key)
        llm_model = st.selectbox(
            "Chat Model (模型名或自定义类路径)", options=model_names[st.session_state.llm_provider],
            accept_new_options=True)
        # embed_provider_list = [
        #     "openai", "azure_openai", "sentence-transformers", "anthropic",
        #     "google", "alibaba", "deepseek", "ollama", "mistral", "ibm", "siliconflow",
        #     "modelscope", "custom"
        # ]
        # embed_provider = st.selectbox("Embeddings Provider", embed_provider_list,
        #                               index=embed_provider_list.index(st.session_state.embed_provider))
        # embed_model = st.text_input(
        #     "Embedding Model (模型名或自定义类路径)", value=st.session_state.embed_model)

        # st.divider()
        # st.markdown("**检索与分块参数**")
        # top_k = st.slider("Top-K (返回块数)", 1, 10, st.session_state.top_k, 1)
        # use_mmr = st.checkbox("启用MMR (最大边际相关性)", value=st.session_state.use_mmr)
        # use_threshold = st.checkbox(
        #     "启用相似度阈值 (FAISS距离)", value=st.session_state.use_threshold)
        # dist_threshold = st.slider(
        #     "距离阈值 (越小越严格)", 0.10, 1.00, st.session_state.dist_threshold, 0.05, disabled=not use_threshold)
        # chunk_size = st.number_input(
        #     "分块大小", 200, 2000, st.session_state.chunk_size, 50)
        # chunk_overlap = st.number_input(
        #     "分块重叠", 0, 800, st.session_state.chunk_overlap, 10)

        st.divider()
        # if st.button("🧹 清除会话与索引", use_container_width=True):
        #     for k in list(st.session_state.keys()):
        #         del st.session_state[k]
        #     st.rerun()
        if st.form_submit_button("保存设置"):
            st.session_state.cfg = {
                "provider": llm_provider,
                "model": llm_model,
                "api_key": api_key,
                "base_url": base_url,
            }
            st.toast("设置已保存", icon="✅")

    # 用于全局参数传递
    st.session_state.llm_provider = llm_provider
    st.session_state.base_url = base_url
    st.session_state.api_key = api_key
    st.session_state.llm_model = llm_model
    # st.session_state.embed_provider = embed_provider
    # st.session_state.embed_model = embed_model
    # st.session_state.top_k = top_k
    # st.session_state.use_mmr = use_mmr
    # st.session_state.use_threshold = use_threshold
    # st.session_state.dist_threshold = dist_threshold
    # st.session_state.chunk_size = chunk_size
    # st.session_state.chunk_overlap = chunk_overlap


def settings_button() -> None:
    float_settings = st.container(
        width=121, vertical_alignment="bottom", horizontal_alignment="center")
    with float_settings:
        st.button("Settings", key="settings_button", icon="⚙️",
                  on_click=show_settings, type="primary")
    float_settings.float(css="bottom: 80px; right: 20px; z-index: 120;")
