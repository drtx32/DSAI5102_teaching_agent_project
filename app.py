# app.py
import streamlit as st
import queue
import uuid
from app.agent.worker import LangGraphWorker

st.set_page_config(page_title="MCP + LangGraph Chat", page_icon="🧠")

st.title("🧠 MCP + LangGraph Chat with Worker Thread")

# ========== 初始化 session 状态 ==========
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "worker" not in st.session_state:
    st.session_state.request_q = queue.Queue()
    st.session_state.reply_q = queue.Queue()

    cfg = {
        "thread_id": st.session_state.thread_id,
        "model": st.secrets["OPENAI_MODEL_NAME"],
        "api_key": st.secrets["OPENAI_API_KEY"],
        "base_url": st.secrets["OPENAI_BASE_URL"],
    }

    worker = LangGraphWorker(
        st.session_state.request_q,
        st.session_state.reply_q,
        cfg
    )
    worker.start()
    st.session_state.worker = worker

if "messages" not in st.session_state:
    st.session_state.messages = []


# 显示历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 输入
if prompt := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 发送到后台线程
    st.session_state.request_q.put(prompt)

    # 等待 worker 回答
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reply = st.session_state.reply_q.get()  # 阻塞等待
        placeholder.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


# 清空按钮
if st.button("🗑 清除聊天"):
    st.session_state.messages = []
    st.rerun()
