"""
LLM Provider Factory - 支持多个AI服务商
"""
import os
from typing import Any, Optional, Sequence, Callable, Literal
from openai import OpenAI

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.tools.base import BaseTool
from langchain.chat_models import BaseChatModel
from langchain_openai.chat_models.base import _DictOrPydanticClass
try:
    from langchain_openai import ChatOpenAI
    from langchain_ollama import ChatOllama
    from langchain_openai import AzureChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.chat_models import ChatOllama
    from langchain.chat_models import AzureChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI


# Provider Display Names
PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "alibaba": "Alibaba",
    "moonshot": "MoonShot",
    "unbound": "Unbound AI",
    "ibm": "IBM",
    "grok": "Grok",
    "ollama": "Ollama",
    "mistral": "Mistral",
    "siliconflow": "SiliconFlow",
    "modelscope": "ModelScope",
}

# Predefined model names for common providers
model_names = {
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "google": ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp", "gemini-1.5-flash-latest",
               "gemini-1.5-flash-8b-latest", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-pro-exp-02-05",
               "gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"],
    "ollama": ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5-coder:14b", "qwen2.5-coder:32b", "llama2:7b",
               "deepseek-r1:14b", "deepseek-r1:32b"],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "mistral": ["pixtral-large-latest", "mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"],
    "alibaba": ["qwen-plus", "qwen-max", "qwen-vl-max", "qwen-vl-plus", "qwen-turbo", "qwen-long"],
    "moonshot": ["moonshot-v1-32k-vision-preview", "moonshot-v1-8k-vision-preview"],
    "unbound": ["gemini-2.0-flash", "gpt-4o-mini", "gpt-4o", "gpt-4.5-preview"],
    "grok": [
        "grok-3",
        "grok-3-fast",
        "grok-3-mini",
        "grok-3-mini-fast",
        "grok-2-vision",
        "grok-2-image",
        "grok-2",
    ],
    "siliconflow": [
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-V2.5",
        "deepseek-ai/deepseek-vl2",
        "Qwen/Qwen2.5-72B-Instruct-128K",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/QwQ-32B-Preview",
        "Qwen/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "TeleAI/TeleChat2",
        "THUDM/glm-4-9b-chat",
        "Vendor-A/Qwen/Qwen2.5-72B-Instruct",
        "internlm/internlm2_5-7b-chat",
        "internlm/internlm2_5-20b-chat",
        "Pro/Qwen/Qwen2.5-7B-Instruct",
        "Pro/Qwen/Qwen2-7B-Instruct",
        "Pro/Qwen/Qwen2-1.5B-Instruct",
        "Pro/THUDM/chatglm3-6b",
        "Pro/THUDM/glm-4-9b-chat",
    ],
    "ibm": ["ibm/granite-vision-3.1-2b-preview", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            "meta-llama/llama-3-2-90b-vision-instruct"],
    "modelscope": [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/QwQ-32B-Preview",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-235B-A22B",
    ],
}


class DeepSeekR1ChatOpenAI:
    """DeepSeek R1 模型的自定义包装器，支持推理内容"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # 延迟导入 ChatOpenAI
        self.base_model = ChatOpenAI(*args, **kwargs)
        self.model_name = kwargs.get("model_name", "deepseek-reasoner")
        self.client = OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key")
        )

    async def ainvoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append(
                    {"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append(
                    {"role": "assistant", "content": input_.content})
            else:
                message_history.append(
                    {"role": "user", "content": input_.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history,
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append(
                    {"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append(
                    {"role": "assistant", "content": input_.content})
            else:
                message_history.append(
                    {"role": "user", "content": input_.content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        response_format: _DictOrPydanticClass | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """绑定工具到基础模型"""
        return self.base_model.bind_tools(
            tools,
            tool_choice=tool_choice,
            strict=strict,
            parallel_tool_calls=parallel_tool_calls,
            response_format=response_format,
            **kwargs,
        )


class DeepSeekR1ChatOllama:
    """Ollama DeepSeek R1 模型的自定义包装器"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.base_model = ChatOllama(*args, **kwargs)

    async def ainvoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = await self.base_model.ainvoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split(
            "</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def invoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = self.base_model.invoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split(
            "</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | Literal["auto", "any"] | bool | None = None,  # noqa: PYI051, ARG002
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """绑定工具到基础模型"""
        return self.base_model.bind_tools(
            tools,
            tool_choice=tool_choice,
            **kwargs,
        )


def get_llm_model(provider: str, **kwargs) -> ChatAnthropic | ChatMistralAI | ChatOpenAI | DeepSeekR1ChatOllama | DeepSeekR1ChatOpenAI | ChatGoogleGenerativeAI | ChatOllama | AzureChatOpenAI | ChatWatsonx:
    """
    获取LLM模型实例

    Args:
        provider: LLM provider名称
        **kwargs: 其他参数，包括 model_name, temperature, api_key, base_url 等

    Returns:
        LLM模型实例

    Raises:
        ValueError: 如果provider不支持或缺少必需的API key
    """
    # 对于非本地provider，检查API key
    if provider not in ["ollama", "bedrock"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key:
            provider_display = PROVIDER_DISPLAY_NAMES.get(
                provider, provider.upper())
            error_msg = f"💥 {provider_display} API key not found! 🔑 Please set the `{env_var}` environment variable or provide it in the UI."
            raise ValueError(error_msg)
        kwargs["api_key"] = api_key

    # 根据provider创建对应的LLM实例
    if provider == "anthropic":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("ANTHROPIC_ENDPOINT",
                                 "https://api.anthropic.com")
        else:
            base_url = kwargs.get("base_url")

        return ChatAnthropic(
            model=kwargs.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs["api_key"],
        )

    elif provider == 'mistral':
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("MISTRAL_ENDPOINT",
                                 "https://api.mistral.ai/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatMistralAI(
            model=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            endpoint=base_url,
            api_key=kwargs["api_key"],
        )

    elif provider == "openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv(
                "OPENAI_ENDPOINT", "https://api.openai.com/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs["api_key"],
        )

    elif provider == "grok":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("GROK_ENDPOINT", "https://api.x.ai/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "grok-3"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs["api_key"],
        )

    elif provider == "deepseek":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("DEEPSEEK_ENDPOINT",
                                 "https://api.deepseek.com")
        else:
            base_url = kwargs.get("base_url")

        if kwargs.get("model_name", "deepseek-chat") == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model_name=kwargs.get("model_name", "deepseek-reasoner"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=kwargs["api_key"],
            )
        else:
            return ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-chat"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=kwargs["api_key"],
            )

    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash"),
            temperature=kwargs.get("temperature", 0.0),
            google_api_key=kwargs["api_key"],
        )

    elif provider == "ollama":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        else:
            base_url = kwargs.get("base_url")

        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )

    elif provider == "azure_openai":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        else:
            base_url = kwargs.get("base_url")

        api_version = kwargs.get("api_version", "") or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version=api_version,
            azure_endpoint=base_url,
            api_key=kwargs["api_key"],
        )

    elif provider == "alibaba":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv(
                "ALIBABA_ENDPOINT", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "qwen-plus"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs["api_key"],
        )

    elif provider == "ibm":
        parameters = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("num_ctx", 32000)
        }

        if not kwargs.get("base_url", ""):
            base_url = os.getenv(
                "IBM_ENDPOINT", "https://us-south.ml.cloud.ibm.com")
        else:
            base_url = kwargs.get("base_url")

        return ChatWatsonx(
            model_id=kwargs.get(
                "model_name", "ibm/granite-vision-3.1-2b-preview"),
            url=base_url,
            project_id=os.getenv("IBM_PROJECT_ID"),
            apikey=kwargs["api_key"],
            params=parameters
        )

    elif provider == "moonshot":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("MOONSHOT_ENDPOINT",
                                 "https://api.moonshot.cn/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "moonshot-v1-32k-vision-preview"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs["api_key"],
        )

    elif provider == "unbound":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("UNBOUND_ENDPOINT",
                                 "https://api.getunbound.ai")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o-mini"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs["api_key"],
        )

    elif provider == "siliconflow":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("SILICONFLOW_ENDPOINT",
                                 "https://api.siliconflow.cn/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            api_key=kwargs["api_key"],
            base_url=base_url,
            model_name=kwargs.get("model_name", "Qwen/QwQ-32B-Preview"),
            temperature=kwargs.get("temperature", 0.0),
        )

    elif provider == "modelscope":
        if not kwargs.get("base_url", ""):
            base_url = os.getenv("MODELSCOPE_ENDPOINT",
                                 "https://api-inference.modelscope.cn/v1")
        else:
            base_url = kwargs.get("base_url")

        return ChatOpenAI(
            api_key=kwargs["api_key"],
            base_url=base_url,
            model_name=kwargs.get("model_name", "Qwen/QwQ-32B-Preview"),
            temperature=kwargs.get("temperature", 0.0),
            model_kwargs={"enable_thinking": False}
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
