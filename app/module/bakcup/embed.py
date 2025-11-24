def is_valid_docx_bytes(b: bytes) -> bool:
    """A valid .docx file starts with 'PK' since it's a zip archive."""
    return b[:2] == b"PK"


def safe_instantiate(path: str, params: Dict[str, Any]):
    """Instantiate a class given a full path like 'module.ClassName' with filtered params."""
    module_name, class_name = path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    # kw = _filter_init_kwargs(cls, params)
    return cls(**params)


def load_docs_from_uploads(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    docs = []
    tmp_dir = "uploaded_docx"
    os.makedirs(tmp_dir, exist_ok=True)
    for uf in uploaded_files:
        raw = uf.read()
        if not is_valid_docx_bytes(raw):
            st.warning(
                f"⚠️ `{uf.name}` is not a valid .docx (or corrupted). Skipped.")
            continue
        path = os.path.join(tmp_dir, uf.name)
        with open(path, "wb") as f:
            f.write(raw)
        docs.extend(Docx2txtLoader(path).load())
    return docs


def split_documents(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
    )
    return splitter.split_documents(docs)


def make_embeddings(provider: str, model: str, base_url: str, api_key: str):
    """Create an embeddings instance for the selected provider.

    - provider: 'openai', 'sentence-transformers', 'anthropic', etc.
    - model may be a model name or a full class path for custom
    """
    provider = (provider or "openai").lower()

    # For non-local providers, validate API key
    if provider not in ["sentence-transformers", "sentence_transformers", "ollama", "custom"]:
        if not api_key:
            raise ValueError(f"API key is required for provider '{provider}'")

    if provider == "openai":
        params = {"model": model or "text-embedding-3-small",
                  "base_url": base_url or "https://api.openai.com/v1",
                  "api_key": api_key}
        return safe_instantiate("langchain_openai.OpenAIEmbeddings", params)

    elif provider == "azure_openai":
        params = {"model": model or "text-embedding-3-small",
                  "azure_endpoint": base_url,
                  "api_key": api_key,
                  "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")}
        return safe_instantiate("langchain_openai.AzureOpenAIEmbeddings", params)

    elif provider == "anthropic":
        # Anthropic doesn't have a native embeddings API, use Voyage AI (common pairing)
        params = {"model": model or "voyage-3",
                  "api_key": api_key}
        return safe_instantiate("langchain_community.embeddings.VoyageEmbeddings", params)

    elif provider == "google":
        params = {"model": model or "models/text-embedding-004",
                  "google_api_key": api_key}
        return safe_instantiate("langchain_google_genai.GoogleGenerativeAIEmbeddings", params)

    elif provider == "deepseek":
        # DeepSeek uses OpenAI-compatible embeddings endpoint
        params = {"model": model or "text-embedding-v3",
                  "base_url": base_url or "https://api.deepseek.com",
                  "api_key": api_key}
        return safe_instantiate("langchain_openai.OpenAIEmbeddings", params)

    elif provider == "alibaba":
        # Alibaba DashScope embeddings (OpenAI-compatible)
        params = {"model": model or "text-embedding-v3",
                  "base_url": base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                  "api_key": api_key}
        return safe_instantiate("langchain_openai.OpenAIEmbeddings", params)

    elif provider == "ollama":
        params = {"model": model or "nomic-embed-text",
                  "base_url": base_url or "http://localhost:11434"}
        return safe_instantiate("langchain_ollama.OllamaEmbeddings", params)

    elif provider == "mistral":
        params = {"model": model or "mistral-embed",
                  "api_key": api_key}
        return safe_instantiate("langchain_mistralai.MistralAIEmbeddings", params)

    elif provider == "ibm":
        # IBM watsonx.ai embeddings
        params = {"model_id": model or "ibm/slate-125m-english-rtrvr",
                  "url": base_url or "https://us-south.ml.cloud.ibm.com",
                  "apikey": api_key,
                  "project_id": os.getenv("IBM_PROJECT_ID")}
        return safe_instantiate("langchain_ibm.WatsonxEmbeddings", params)

    elif provider == "siliconflow":
        # SiliconFlow uses OpenAI-compatible embeddings
        params = {"model": model or "BAAI/bge-large-zh-v1.5",
                  "base_url": base_url or "https://api.siliconflow.cn/v1",
                  "api_key": api_key}
        return safe_instantiate("langchain_openai.OpenAIEmbeddings", params)

    elif provider == "modelscope":
        # ModelScope uses OpenAI-compatible embeddings
        params = {"model": model or "AI-ModelScope/bge-large-zh-v1.5",
                  "base_url": base_url or "https://api-inference.modelscope.cn/v1",
                  "api_key": api_key}
        return safe_instantiate("langchain_openai.OpenAIEmbeddings", params)

    elif provider in ("sentence-transformers", "sentence_transformers"):
        # uses sentence-transformers locally via LangChain wrapper
        params = {"model_name": model or "all-MiniLM-L6-v2"}
        return safe_instantiate("langchain_community.embeddings.HuggingFaceEmbeddings", params)

    elif provider == "custom":
        params = {"model": model, "model_name": model,
                  "base_url": base_url, "api_key": api_key}
        return safe_instantiate(model, params)

    else:
        try:
            params = {"model": model, "model_name": model,
                      "base_url": base_url, "api_key": api_key}
            return safe_instantiate(provider, params)
        except Exception as e:
            raise ValueError(
                f"Cannot create Embeddings for provider '{provider}': {e}")


def build_vectordb_with_progress(splits, embeddings) -> FAISS:
    if not splits:
        raise ValueError("No text chunks available for indexing.")

    progress = st.progress(
        0.0, text="Embedding and building FAISS index… (initializing)")
    status = st.empty()

    # `embeddings` is an embeddings instance created by `make_embeddings` below

    total = len(splits)
    batch_size = max(16, min(128, total // 10 or 16))

    first = splits[:batch_size]
    status.info(f"Initializing index… (1/{(total-1)//batch_size + 1} batches)")
    vectordb = FAISS.from_documents(first, embeddings)
    built = len(first)
    progress.progress(min(0.08 + 0.80 * (built/total), 0.95),
                      text=f"Indexed {built}/{total} chunks…")

    while built < total:
        end = min(built + batch_size, total)
        batch = splits[built:end]
        status.info(
            f"Incremental indexing… ({end//batch_size + (1 if end%batch_size else 0)}/{(total-1)//batch_size + 1} batches)")
        vectordb.add_documents(batch)
        built = end
        progress.progress(min(0.08 + 0.80 * (built/total), 0.98),
                          text=f"Indexed {built}/{total} chunks…")

    progress.progress(1.0, text="Indexing completed ✅")
    progress.empty()
    status.empty()
    return vectordb


# -----------------------------
# File upload & vector index building
# -----------------------------
st.subheader("📤 Upload DOCX files (multiple allowed)")
uploaded_files = st.file_uploader(
    "Only .docx supported", type=["docx"], accept_multiple_files=True)

c1, c2 = st.columns([1.2, 2.8])
with c1:
    build_btn = st.button("🚀 Build / Update Index",
                          type="primary", use_container_width=True)
with c2:
    if st.session_state.ready and st.session_state.vectordb is not None:
        st.success(
            f"Index ready. {len(st.session_state.files)} files loaded in this session.")

if build_btn:
    api_key = st.session_state.get("api_key", "")
    base_url = st.session_state.get("base_url", "")
    embed_provider = st.session_state.get("embed_provider", "openai")
    embed_model = st.session_state.get("embed_model", "text-embedding-3-small")
    chunk_size = st.session_state.get("chunk_size", 800)
    chunk_overlap = st.session_state.get("chunk_overlap", 100)
    if not api_key:
        st.error("请先在设置弹窗中输入API Key。")
    elif not uploaded_files:
        st.error("请至少上传一个.docx文件。")
    else:
        with st.spinner("Loading and splitting documents…"):
            docs = load_docs_from_uploads(uploaded_files)
            st.session_state.files = [
                f.name for f in uploaded_files if f is not None]
            if not docs:
                st.error("未加载到有效的.docx文件。")
            else:
                splits = split_documents(
                    docs, int(chunk_size), int(chunk_overlap))
                st.info(f"已分割为 {len(splits)} 个块。")
        try:
            try:
                embeddings = make_embeddings(
                    embed_provider, embed_model, base_url, api_key)
            except Exception as e:
                st.error(f"创建嵌入模型失败: {e}")
                raise
            st.session_state.vectordb = build_vectordb_with_progress(
                splits, embeddings)
            st.session_state.ready = True
            st.success("✅ 向量索引已就绪！可以开始提问。")
        except Exception as e:
            st.exception(e)
