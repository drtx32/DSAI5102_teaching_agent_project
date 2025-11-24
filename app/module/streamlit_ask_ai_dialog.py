import queue
import streamlit as st
from ..utils.streamlit_float import *
from ..agent.worker import SYSTEM_PROMPT, LangGraphWorker
from langchain_core.messages import AIMessage
from ..llm.llm_provider import PROVIDER_DISPLAY_NAMES


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是您的数据科学助教AI，能够使用RAG和网页搜索的功能。请随时向我提问。"}]


# -----------------------------
# Chat area
# -----------------------------
@st.dialog("💬 AI Chat", width="large")
def ask_ai() -> None:
    if len(st.session_state.get("cfg", {})) == 4:
        if "worker" not in st.session_state:
            st.session_state.request_q = queue.Queue()
            st.session_state.reply_q = queue.Queue()

            cfg = st.session_state.cfg
            cfg["thread_id"] = st.session_state.thread_id

            worker = LangGraphWorker(
                st.session_state.request_q,
                st.session_state.reply_q,
                cfg
            )
            worker.start()
            st.session_state.worker = worker
        st.toast("AI 聊天线程已启动")
    else:
        st.toast("请先在设置中配置模型参数，然后再使用 AI 聊天功能")
        return

    # Display previous messages
    placeholder = st.container(height=800, border=False)
    for msg in st.session_state.messages:
        with placeholder.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("请输入你的问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with placeholder.chat_message("user"):
            st.markdown(prompt)

        # 发送到后台线程
        st.session_state.request_q.put(prompt)

        # 等待 worker 回答
        with placeholder.chat_message("assistant"):
            ans_placeholder = st.empty()
            reply = st.session_state.reply_q.get()  # 阻塞等待
            ans_placeholder.markdown(reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply})


def ask_ai_button() -> None:
    float_init()
    float_ai_asking = st.container(width=100)
    with float_ai_asking:
        st.button("Ask AI", key="AI_asking", on_click=ask_ai,
                  type="primary", icon=":material/question_answer:")
    float_ai_asking.float(
        css="position: fixed; bottom: 20px; right: 20px; z-index: 120;")
