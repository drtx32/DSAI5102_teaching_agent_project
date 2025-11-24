import streamlit as st
from app.module.streamlit_bottom_bar import bottom_bar
from app.module.streamlit_ask_ai_dialog import ask_ai_button
from app.module.streamlit_settings_dialog import settings_button

st.set_page_config(page_title="模型说明", page_icon="🤖")
settings_button()
ask_ai_button()

st.title("🤖 模型说明")
st.markdown("""
本页基于 `app/llm/llm_provider.py` 的实现，列出当前项目已支持的 LLM 提供商、示例 model 名称、配置要点与选型建议。只包含已实现的功能与包装器说明。
""")

st.header("可用 Provider（已实现/常用）")
st.markdown("""
- `openai`（OpenAI）
- `azure_openai`（Azure OpenAI）
- `anthropic`（Anthropic）
- `google`（Google Generative AI）
- `ollama`（本地 Ollama）
- `deepseek`（DeepSeek / DeepSeek R1 包装器）
- `mistral`（Mistral）
- `ibm`（WatsonX）
- 以及少量厂商适配：`moonshot`、`unbound`、`grok`、`alibaba`、`siliconflow`、`modelscope`
""")

st.header("示例模型名称（快速参考）")
st.markdown("""
- `openai`: `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`
- `anthropic`: `claude-3-5-sonnet-20241022`（示例）
- `google`: `gemini-2.0-flash`
- `ollama`（本地镜像示例）: `qwen2.5:7b`, `qwen2.5:14b`, `llama2:7b`
- `deepseek`: `deepseek-chat`（普通），`deepseek-reasoner`（专用的 reasoning wrapper）
- `mistral`: `mistral-large-latest`
""")

st.header("本地与自定义包装器说明")
st.markdown("""
- `DeepSeekR1ChatOpenAI` / `DeepSeekR1ChatOllama`：项目对 DeepSeek R1 做了自定义封装，返回结果中可能包含 `reasoning_content`（可用于展示链式推理或中间思路）。
- `Ollama`：默认 `OLLAMA_ENDPOINT` 为 `http://localhost:11434`，本地模型通过 `ChatOllama` 访问，不强制要求 API Key，但需在设置中填写 `OLLAMA_ENDPOINT`（若未使用默认地址）。
""")

st.header("配置要点")
st.markdown("""
- API Key 环境变量示例（按 provider）：`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `AZURE_OPENAI` 相关变量等。
- 本地 Ollama：设置 `OLLAMA_ENDPOINT`（示例：`http://localhost:11434`）。
- IBM WatsonX 需要 `IBM_PROJECT_ID`（见 `app/llm/llm_provider.py` 中的说明，Streamlit的前端界面中没做适配，使用会报错）。
- 项目会优先从 kwargs 中读取参数，若未提供则从环境变量加载。也可通过页面右下角的设置弹窗在运行时填写（`app.module.streamlit_settings_dialog.settings_button`）。
""")

st.header("选型与使用建议")
st.markdown("""
- 成本/延迟/隐私三角权衡：敏感数据优先本地 Embedding + 本地 LLM；对高准确率与复杂推理需求时优先大模型。
- 温度设置：教学演示推荐 `0.0 - 0.3` 保持回答确定性；探索性任务可提高温度。
- 若使用 DeepSeek 的 `reasoner` 模型，注意解析返回的 `reasoning_content`（项目包装器会把中间推理与最终回答分离）。
""")

st.header("调试小贴士")
st.markdown("""
- 若出现认证错误，先确认对应的环境变量已设置或在设置弹窗中填写正确的 key/endpoint。
- 本地 Ollama 无响应时，确认 Ollama 服务已启动并能在 `OLLAMA_ENDPOINT` 访问。
- 若模型返回包含思考标记（例如 `<think>`/`</think>`），项目包装器会在部分 provider 中做提取与清洗。
""")

bottom_bar(previous_alias="Usage Guide", previous_page="pages/02_Usage_Guide.py",
           next_alias="FAQ", next_page="pages/04_FAQ.py")
