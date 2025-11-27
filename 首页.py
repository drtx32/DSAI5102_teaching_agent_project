# -*- coding: utf-8 -*-
import base64

import streamlit as st
from teaching_agent.module.streamlit_bottom_bar import bottom_bar
from teaching_agent.module.streamlit_ask_ai_dialog import ask_ai_button
from teaching_agent.module.streamlit_settings_dialog import settings_button
from streamlit.components.v1 import html

# -----------------------------
# Page configuration & styles
# -----------------------------
st.set_page_config(
    page_title="AMA5102 Project Homepage",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings_button()
ask_ai_button()

# Light UI styling
st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.stChatMessage p {font-size:1.02rem; line-height:1.6;}
section[data-testid="stSidebar"] .st-emotion-cache-1vt4y43 {margin-bottom: 0.5rem;}
code, pre {font-size: 0.92rem;}
</style>
""", unsafe_allow_html=True)

st.title(":computer: Principles of Data Science")
st.caption(
    "LangGraph + LangChain + MCP(websearch & rag) + Streamlit")
st.caption("Github Repository: "
            "[DSAI5102_teaching_agent_project](https://github.com/drtx32/DSAI5102_teaching_agent_project)")

# -----------------------------
# Main documentation content
# -----------------------------
documentation = st.container(width=1200)

with documentation:
    st.markdown("## 📖 首页")
    with open("assets/index.svg", "rb") as f:
        svg_base64 = base64.b64encode(f.read()).decode()
    html(f'''
        <img src="data:image/svg+xml;base64,{svg_base64}" style="width: 100%;">
    ''', height=520)


bottom_bar(next_page="pages/01_Project_Introduction.py",
           next_alias="Project Introduction")
