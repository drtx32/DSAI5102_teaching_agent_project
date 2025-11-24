import streamlit as st
from app.module.streamlit_bottom_bar import bottom_bar
from app.module.streamlit_ask_ai_dialog import ask_ai_button
from app.module.streamlit_settings_dialog import settings_button

st.set_page_config(page_title="FAQ", page_icon="❓")
settings_button()
ask_ai_button()

st.title("❓ 常见问题（FAQ）与排错")

st.markdown("""
列出项目常见问题与排错步骤，便于课堂上快速解决学生遇到的问题。
""")

st.header("上传与解析 ".strip())
st.markdown("""
- 问：上传 .docx 后没有文本或报错 `BadZipFile`？
  - 答：说明文件可能不是正确的 `.docx`（例如改后缀的 .doc）。请打开 Word 另存为 `.docx` 后重试。

- 问：分块后信息丢失？
  - 答：尝试增加 `chunk_size` 或减少 `chunk_overlap`，并确认文档中是否存在大量非文本对象（表格/图片）。
""")

st.header("API 与模型调用")
st.markdown("""
- 问：‘创建嵌入模型失败’/‘Model call error’？
  - 答：先确认 API Key 是否设置（环境变量或在设置弹窗中填写），并确认对应 provider 的包已安装。

- 问：某些 provider 返回结构不同，文本无法抽取？
  - 答：我们在 `call_llm()` 中做了兼容处理（`invoke`、`generate`、可调用对象），如果遇见新类型，请把返回样例贴给我帮适配。
""")

st.header("Streamlit UI 问题")
st.markdown("""
- 问：设置按钮尝试浮动但不在预期位置？
  - 答：Streamlit 的 iframe 结构对自定义浮动元素支持有限。项目最终把按钮放在侧边栏顶部以保证兼容性。

- 问：要把教学内容做得更丰富，是否必须在 `pages/` 下写文件？
  - 答：推荐使用 `pages/` 多页面结构（每个页面为独立 `.py`），比把所有内容塞在 `app.py` 更清晰，也便于课堂分段展示。
""")

st.header("如果需要我可以帮你：")
st.markdown("""
- 自动把侧边栏导航改成链接到 `pages/` 页（或在主页面嵌入页面内容）。
- 为每个页面添加示例数据、截图或交互式演示（例如：在线嵌入示例、显式向量可视化）。
""")

st.success("如需我现在把侧边栏导航替换为指向这些页面的链接或在页面中加入示例数据，请告诉我我会继续修改。")

bottom_bar(previous_page="pages/02_Usage_Guide.py",
           previous_alias="Usage Guide")
