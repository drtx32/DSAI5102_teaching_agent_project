import streamlit as st
from app.module.streamlit_bottom_bar import bottom_bar
from app.module.streamlit_ask_ai_dialog import ask_ai_button
from app.module.streamlit_settings_dialog import settings_button

st.set_page_config(page_title="使用方法", page_icon="🛠️")
settings_button()
ask_ai_button()

st.title("🛠️ 使用方法（快速起步）")

st.markdown("""
本页包含：运行环境、依赖安装、运行命令、常见配置（在 Windows PowerShell 下的示例）。
""")

st.header("1) 环境与依赖")
st.markdown("""
建议使用 Conda 创建一个专用环境，例如 `sml`（项目测试时使用此环境）。

示例安装（PowerShell）：
```powershell
conda create -n sml python=3.11 -y; conda activate sml
pip install -r requirements.txt
```

`requirements.txt` 应包含 `streamlit`, `langchain`, `langchain-community`, `faiss-cpu` 等依赖。
""")

st.header("2) 必要环境变量")
st.markdown("""
将以下关键变量设置到环境或 `st.secrets`：

- `OPENAI_API_KEY` (或替代代理的 `OPENAI_BASE_URL`)
- `ANTHROPIC_API_KEY`（如果使用 Anthropic）
- `GOOGLE_API_KEY`（如果使用 Google Generative API）
- `IBM_PROJECT_ID`（如果使用 IBM Watsonx）

PowerShell 设置举例：
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:OPENAI_BASE_URL = "https://your-proxy.example/v1"
```
""")

st.header("3) 启动应用")
st.code("streamlit run app.py", language="bash")
st.write("在浏览器打开地址后，主页面提供上传 .docx、构建索引与聊天界面。")

st.header("4) 构建索引（操作步骤）")
st.markdown("""
- 点击左侧 `⚙️ 打开设置` 输入 API Key、选择 LLM/Embeddings 提供商与模型。
- 在主页面上传一个或多个 `.docx` 文件。
- 点击 `🚀 Build / Update Index`，观察进度条与提示信息。
- 构建完成后可在聊天框中提问，系统会基于检索到的上下文生成回答。
""")

st.header("5) 测试不同厂商")
st.write("如果你想测试本地 Ollama：")
with st.expander("Ollama 本地测试要点"):
    st.markdown("""
    - 在本地安装并启动 Ollama 服务（默认 `http://localhost:11434`）。
    - 在设置里选择 `ollama`，并把模型写成 `your-model-name:tag` 或接受默认。
    - 因为 Ollama 在本地运行，API Key 通常留空。
    """)

st.success("运行提示：如遇到权限或网络问题，先检查环境变量与防火墙，然后重启 Streamlit 服务。")

bottom_bar(previous_page="pages/01_Project_Introduction.py",
           previous_alias="Project Introduction",
           next_alias="Models 说明", next_page="pages/03_Models_说明.py")
