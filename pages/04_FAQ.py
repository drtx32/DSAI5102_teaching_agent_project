import streamlit as st
from app.module.streamlit_bottom_bar import bottom_bar
from app.module.streamlit_ask_ai_dialog import ask_ai_button
from app.module.streamlit_settings_dialog import settings_button

st.set_page_config(page_title="FAQ", page_icon="❓")
settings_button()
ask_ai_button()

st.title("❓ 常见问题（FAQ）与排错")

st.markdown("""
列出项目常见问题与排错步骤，便于快速解决学生遇到的问题。
""")

st.header("API 与模型调用")
st.markdown("""
- 问：‘创建嵌入模型失败’/‘Model call error’？
  - 答：先确认 API Key 是否设置（环境变量或在设置弹窗中填写），并确认对应 provider 的包已安装。
""")


st.header("会话启动 ".strip())
st.markdown("""
- 问：我配置了 `secrets.toml` 后怎么不能直接对话？
  - 答：需要点击右下角设置按钮，在弹窗中点击保存设置后才能生效。

- 问：AI 为什么搜索不了网页？
  - 答：本项目使用的是Duckduckgo_search (ddgs)，它不需要 API Key，但由于内地访问限制，可能无法正常使用。
""")

st.header("Streamlit UI 问题")
st.markdown("""
- 问：首次加载页面或切换页面时，Settings和Ask AI按钮会有闪烁？
  - 答：Streamlit 的结构对自定义浮动元素支持有限。我们通过 Streamlit_float 第三方component来实现悬浮效果（重写了部分代码以兼容最新的Streamlit）。

- 问：我想后续丰富和改进内容，能不能告诉我主要修改的地方？
  - 答：如果添加更多的 Streamlit 页面，推荐使用 `pages/` 多页面结构（每个页面为独立 `.py`），比把所有内容塞在 `首页.py` 更清晰，也便于课堂分段展示。
  - 另外，`app/module/` 目录下的模块化组件可以复用到各个页面，便于维护和扩展。
  - 想要添加工具（Tool）或自定义 Agent，可以修改 `app/agent/`（添加工具还需要修改 `app/mcp/`）目录下的代码。
  - 如果想添加更多 AI 模型支持，可以在 `app/llm/` 目录下添加对应的模型调用代码。
""")

bottom_bar(previous_page="pages/02_Usage_Guide.py",
           previous_alias="Usage Guide")
