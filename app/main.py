import streamlit as st
from streamlit_ace import st_ace
import uuid
from app.agent.agent_builder import TeachingAgent
from app.agent.llm_provider import PROVIDER_DISPLAY_NAMES
from app.rag.vector_store import RAGVectorStore
from app.utils.logging_config import logger
from app.utils.config import settings


@st.dialog("创建教学 Agent")
def create_agent_dialog():
    # 模型提供商选择
    provider = st.selectbox(
        "选择模型提供商",
        options=list(PROVIDER_DISPLAY_NAMES.keys()),
        format_func=lambda x: PROVIDER_DISPLAY_NAMES[x],
        index=0
    )

    # 模型名称输入
    model = st.text_input(
        "模型名称",
        value=settings.openai_model  # 默认值从环境变量获取
    )

    # API Key 输入
    api_key = st.text_input(
        "API Key",
        value=settings.openai_api_key,  # 默认值从环境变量获取
        type="password"
    )

    # 提交按钮
    if st.button("确认创建"):
        try:
            st.session_state.agent = TeachingAgent(
                session_id=st.session_state.session_id,
                provider=provider,
                model=model,
                api_key=api_key
            )
            st.success("Agent 已成功创建！")
        except Exception as e:
            st.error(f"创建 Agent 失败: {e}")


# 初始化页面
st.set_page_config(page_title="DSAI5102教学辅助Agent", layout="wide")
st.title("📚 DSAI5102教学辅助Agent")

# 初始化session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"新会话创建: {st.session_state.session_id}")

# 初始化Agent（延迟初始化，避免启动时强制要求 API key）
if "agent" not in st.session_state:
    st.session_state.agent = None

# 侧边栏 - 文档上传与 embedding 后端选择
with st.sidebar:
    st.header("课件管理")

    # Embedding 后端选择
    backend = st.selectbox("Embedding 后端", options=["openai", "sentence_transformers"], index=(
        0 if settings.embedding_backend == "openai" else 1))

    # 如果选择发生变化，重新初始化 RAG
    if "embedding_backend" not in st.session_state or st.session_state.embedding_backend != backend:
        st.session_state.embedding_backend = backend
        # 初始化或替换 RAG 实例
        try:
            st.session_state.rag = RAGVectorStore(embedding_backend=backend)
            st.success(f"已选择 embedding 后端: {backend}")
        except Exception as e:
            st.error(f"初始化向量后端失败: {e}")
            st.session_state.rag = None

    uploaded_files = st.file_uploader(
        "上传课程课件(PDF)", accept_multiple_files=True, type=["pdf"])
    use_ocr = st.checkbox("使用OCR提取（适用于扫描版PDF）", value=False)

    if st.button("添加到知识库"):
        if uploaded_files and st.session_state.get("rag"):
            for file in uploaded_files:
                with open(f"temp_{file.name}", "wb") as f:
                    f.write(file.getbuffer())
                success = st.session_state.rag.add_document(
                    f"temp_{file.name}", use_ocr)
                if success:
                    st.success(f"成功添加: {file.name}")
                else:
                    st.error(f"添加失败: {file.name}")
        else:
            st.warning("请先上传文件并初始化向量后端")

    st.divider()
    st.subheader("Agent 设置")

    # Agent 创建/重建控件
    if st.button("创建/重建 Agent"):
        create_agent_dialog()


# 代码编辑器
st.subheader("💻 代码执行区")
code = st_ace(
    language="python",
    theme="monokai",
    keybinding="vscode",
    font_size=14,
    tab_size=4,
    show_gutter=True,
    show_print_margin=False,
    wrap=True,
    height=300,
    value="# 在这里编写Python代码\n# 例如: 计算1+1\nprint(1+1)"
)

if st.button("运行代码"):
    if st.session_state.agent and hasattr(st.session_state.agent, 'tools'):
        try:
            # 调用Python REPL工具
            result = st.session_state.agent.tools[2].func(code)
            st.code(result)
        except Exception as e:
            st.error(f"代码执行错误: {str(e)}")
    else:
        st.warning("请先创建 Agent")

# 对话区
st.subheader("💬 对话区")
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理新消息
if prompt := st.chat_input("有什么可以帮助你的吗？"):
    if not st.session_state.agent:
        st.warning("请先在侧边栏创建 Agent")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = st.session_state.agent.run(prompt)
                st.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
