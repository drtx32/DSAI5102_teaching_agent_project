import os
import importlib
from typing import Any, Dict


# --- Generic instantiation helpers ---------------------------------
def safe_instantiate(path: str, params: Dict[str, Any]):
    """Instantiate a class given a full path like 'module.ClassName' with filtered params."""
    module_name, class_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    # kw = _filter_init_kwargs(cls, params)
    return cls(**params)


def make_llm(provider: str, model: str, base_url: str, api_key: str):
    """Create an LLM instance for the selected provider.

    - provider: one of 'openai', 'anthropic', 'deepseek', etc.
    - model/base_url/api_key: passed as potential kwargs; filtered per class signature
    """
    provider = (provider or "openai").lower()

    # For non-local providers, validate API key
    if provider not in ["ollama", "custom"]:
        if not api_key:
            raise st.toast(
                f"API key is required for provider '{provider}'", icon="⚠️", duration="long")

    if provider == "openai":
        params = {"model": model or "gpt-4o-mini", "base_url": base_url or "https://api.openai.com/v1",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_openai.ChatOpenAI", params)

    elif provider == "azure_openai":
        params = {"model": model or "gpt-4o", "azure_endpoint": base_url,
                  "api_key": api_key, "temperature": 0.2,
                  "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")}
        return safe_instantiate("langchain_openai.AzureChatOpenAI", params)

    elif provider == "anthropic":
        params = {"model": model or "claude-3-5-sonnet-20241022",
                  "base_url": base_url or "https://api.anthropic.com",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_anthropic.ChatAnthropic", params)

    elif provider == "deepseek":
        params = {"model": model or "deepseek-chat",
                  "base_url": base_url or "https://api.deepseek.com",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_openai.ChatOpenAI", params)

    elif provider == "google":
        params = {"model": model or "gemini-2.0-flash",
                  "google_api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_google_genai.ChatGoogleGenerativeAI", params)

    elif provider == "alibaba":
        params = {"model": model or "qwen-plus",
                  "base_url": base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_openai.ChatOpenAI", params)

    elif provider == "moonshot":
        params = {"model": model or "moonshot-v1-8k",
                  "base_url": base_url or "https://api.moonshot.cn/v1",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_openai.ChatOpenAI", params)

    elif provider == "unbound":
        params = {"model": model or "gpt-4o-mini",
                  "base_url": base_url or "https://api.getunbound.ai",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_openai.ChatOpenAI", params)

    elif provider == "grok":
        params = {"model": model or "grok-3",
                  "base_url": base_url or "https://api.x.ai/v1",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_openai.ChatOpenAI", params)

    elif provider == "ollama":
        params = {"model": model or "qwen2.5:7b",
                  "base_url": base_url or "http://localhost:11434",
                  "temperature": 0.2, "num_ctx": 32000}
        return safe_instantiate("langchain_ollama.ChatOllama", params)

    elif provider == "mistral":
        params = {"model": model or "mistral-large-latest",
                  "endpoint": base_url or "https://api.mistral.ai/v1",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_mistralai.ChatMistralAI", params)

    elif provider == "ibm":
        params = {"model_id": model or "ibm/granite-vision-3.1-2b-preview",
                  "url": base_url or "https://us-south.ml.cloud.ibm.com",
                  "apikey": api_key,
                  "project_id": os.getenv("IBM_PROJECT_ID"),
                  "params": {"temperature": 0.2, "max_tokens": 32000}}
        return safe_instantiate("langchain_ibm.ChatWatsonx", params)

    elif provider == "siliconflow":
        params = {"model_name": model or "Qwen/QwQ-32B-Preview",
                  "base_url": base_url or "https://api.siliconflow.cn/v1",
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate("langchain_openai.ChatOpenAI", params)

    elif provider == "modelscope":
        params = {"model_name": model or "Qwen/QwQ-32B-Preview",
                  "base_url": base_url or "https://api-inference.modelscope.cn/v1",
                  "api_key": api_key, "temperature": 0.2,
                  "model_kwargs": {"enable_thinking": False}}
        return safe_instantiate("langchain_openai.ChatOpenAI", params)

    elif provider == "custom":
        # for custom, expect `model` to contain the full class path
        params = {"model": model, "base_url": base_url,
                  "api_key": api_key, "temperature": 0.2}
        return safe_instantiate(model, params)

    else:
        # fallback: try to interpret provider as a module.Class path
        try:
            params = {"model": model, "base_url": base_url,
                      "api_key": api_key, "temperature": 0.2}
            return safe_instantiate(provider, params)
        except Exception as e:
            raise ValueError(
                f"Cannot create LLM for provider '{provider}': {e}")


def call_llm(llm, messages):
    """Call the LLM in a compatible way: prefer .invoke, then .generate, then callable."""
    if hasattr(llm, "invoke"):
        return llm.invoke(messages)
    if hasattr(llm, "generate"):
        # some LLMs expect a `messages=` kwarg
        try:
            return llm.generate(messages=messages)
        except TypeError:
            return llm.generate(messages)
    if callable(llm):
        return llm(messages)
    raise RuntimeError(
        "Provided LLM instance is not callable and has no known call method.")
