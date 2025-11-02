# 基于LangChain的智能教学Agent完整方案（含部署配置）

## 项目架构总览
```plaintext
langchain-teaching-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # Streamlit前端入口
│   ├── agent/               #Agent核心逻辑
│   │   ├── __init__.py
│   │   ├── agent_builder.py #Agent构建逻辑
│   │   └── tools.py         #工具定义
│   ├── rag/                 #RAG相关组件
│   │   ├── __init__.py
│   │   ├── document_loader.py #文档加载与处理
│   │   └── vector_store.py  #Chroma向量存储
│   ├── memory/              #记忆管理
│   │   ├── __init__.py
│   │   └── postgres_memory.py #PostgreSQL记忆存储
│   └── utils/               #工具函数
│       ├── __init__.py
│       ├── logging_config.py #日志配置
│       └── config.py        #配置管理
├── Dockerfile               #容器化配置
├── pyproject.toml           #项目依赖
├── .env.example             #环境变量示例
└── README.md                #项目说明
```

## 一、核心功能实现

### 1. 环境配置与依赖（pyproject.toml）
```toml
[project]
name = "langchain-teaching-agent"
version = "0.1.0"
description = "DSAI5102教学辅助Agent"
requires-python = ">=3.10"

[project.dependencies]
langchain = "==0.1.20"
langchain-openai = "==0.1.7"
langchain-community = "==0.0.38"
chromadb = "==0.4.24"
nomic-embed = "==1.0.5"
duckduckgo-search = "==5.3.0"
streamlit = "==1.34.0"
streamlit-ace = "==0.1.1"
psycopg2-binary = "==2.9.9"
python-dotenv = "==1.0.1"
uvicorn = "==0.29.0"
logging = "==0.4.9.6"
pdf2image = "==1.17.0"  #PDF处理
pytesseract = "==0.3.10" #OCR支持（视觉提取备用）

[tool.poetry.scripts]
start = "uvicorn app.main:app --reload"
run-streamlit = "streamlit run app/main.py"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### 2. 日志配置（app/utils/logging_config.py）
```python
import logging
import os
from datetime import datetime

def configure_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("teaching_agent")

logger = configure_logging()
```

### 3. 配置管理（app/utils/config.py）
```python
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4o"
    
    # 向量数据库配置
    chroma_persist_directory: str = "./chroma_db"
    
    # PostgreSQL配置
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD")
    postgres_db: str = os.getenv("POSTGRES_DB", "agent_memory")
    
    # 文档处理配置
    chunk_size: int = 1000
    chunk_overlap: int = 200

settings = Settings()
```

### 4. RAG模块实现

#### 文档加载与处理（app/rag/document_loader.py）
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from app.utils.logging_config import logger
from app.utils.config import settings
import tempfile
import os

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_document(self, file_path: str, use_ocr: bool = False):
        """加载文档，支持PDF（可选OCR）和普通文本"""
        try:
            if file_path.endswith(".pdf"):
                if use_ocr:
                    # 视觉大模型OCR提取（示例用基础OCR替代）
                    from pdf2image import convert_from_path
                    pages = convert_from_path(file_path)
                    texts = []
                    for i, page in enumerate(pages):
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                            page.save(f, "PNG")
                            loader = UnstructuredImageLoader(f.name)
                            text = loader.load()[0].page_content
                            texts.append(text)
                        os.unlink(f.name)
                    return "\n".join(texts)
                else:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    return self.text_splitter.split_documents(documents)
            else:
                loader = UnstructuredFileLoader(file_path)
                documents = loader.load()
                return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"文档加载错误: {str(e)}")
            raise
```

#### 向量存储（app/rag/vector_store.py）
```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import NomicEmbeddings
from app.rag.document_loader import DocumentProcessor
from app.utils.config import settings
from app.utils.logging_config import logger

class RAGVectorStore:
    def __init__(self):
        self.embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
        self.document_processor = DocumentProcessor()
        self.vector_store = Chroma(
            persist_directory=settings.chroma_persist_directory,
            embedding_function=self.embeddings,
            collection_name="teaching_materials"
        )
    
    def add_document(self, file_path: str, use_ocr: bool = False):
        """添加文档到向量库"""
        try:
            documents = self.document_processor.load_document(file_path, use_ocr)
            if documents:
                self.vector_store.add_documents(documents)
                self.vector_store.persist()
                logger.info(f"成功添加文档: {file_path}, 分片数: {len(documents)}")
                return True
            return False
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 3):
        """检索相似文档"""
        return self.vector_store.similarity_search(query, k=k)
```

### 5. 记忆模块（PostgreSQL）（app/memory/postgres_memory.py）
```python
from langchain.memory import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from app.utils.config import settings
from app.utils.logging_config import logger
import json

class PostgreSQLChatMessageHistory(ChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conn = self._get_connection()
        self._create_table_if_not_exists()
        super().__init__()
        self.messages = self._load_messages()
    
    def _get_connection(self):
        return psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db
        )
    
    def _create_table_if_not_exists(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """)
                self.conn.commit()
        except Exception as e:
            logger.error(f"创建表失败: {str(e)}")
            self.conn.rollback()
    
    def _load_messages(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                SELECT message FROM chat_history 
                WHERE session_id = %s 
                ORDER BY created_at ASC
                """, (self.session_id,))
                rows = cur.fetchall()
                messages = []
                for row in rows:
                    msg_data = row[0]
                    if msg_data["type"] == "human":
                        messages.append(HumanMessage(content=msg_data["content"]))
                    elif msg_data["type"] == "ai":
                        messages.append(AIMessage(content=msg_data["content"]))
                return messages
        except Exception as e:
            logger.error(f"加载消息失败: {str(e)}")
            return []
    
    def add_message(self, message: BaseMessage):
        super().add_message(message)
        try:
            with self.conn.cursor() as cur:
                msg_type = "human" if isinstance(message, HumanMessage) else "ai"
                cur.execute("""
                INSERT INTO chat_history (session_id, message)
                VALUES (%s, %s)
                """, (self.session_id, Json({
                    "type": msg_type,
                    "content": message.content,
                    "timestamp": datetime.now().isoformat()
                })))
                self.conn.commit()
        except Exception as e:
            logger.error(f"添加消息失败: {str(e)}")
            self.conn.rollback()
    
    def clear(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                DELETE FROM chat_history WHERE session_id = %s
                """, (self.session_id,))
                self.conn.commit()
            super().clear()
        except Exception as e:
            logger.error(f"清空消息失败: {str(e)}")
            self.conn.rollback()
```

### 6. 工具定义（app/agent/tools.py）
```python
from langchain.tools import Tool, StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.python.tool import PythonREPLTool
from app.rag.vector_store import RAGVectorStore
from app.utils.logging_config import logger

class AgentTools:
    def __init__(self):
        self.rag = RAGVectorStore()
        self.search = DuckDuckGoSearchRun()
        self.python_repl = PythonREPLTool()
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self):
        """初始化所有工具"""
        return [
            Tool(
                name="课件检索",
                func=self.rag.similarity_search,
                description="用于检索课程课件中的知识点，当问题涉及教材内容、公式原理、课程案例时使用"
            ),
            Tool(
                name="网页搜索",
                func=self.search.run,
                description="用于获取最新信息、外部案例或实时数据，当问题涉及当前时间、最新趋势、外部资源时使用"
            ),
            Tool(
                name="Python代码执行",
                func=self.python_repl.run,
                description="用于执行Python代码、数据分析、模型计算，当需要运行代码或处理数据时使用"
            )
        ]
    
    def get_tools(self):
        return self.tools
```

### 7. Agent构建（app/agent/agent_builder.py）
```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from app.agent.tools import AgentTools
from app.memory.postgres_memory import PostgreSQLChatMessageHistory
from app.utils.config import settings
from app.utils.logging_config import logger

class TeachingAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            temperature=0.3
        )
        self.tools = AgentTools().get_tools()
        self.memory = self._initialize_memory()
        self.agent = self._initialize_agent()
    
    def _initialize_memory(self):
        """初始化带PostgreSQL存储的记忆"""
        message_history = PostgreSQLChatMessageHistory(session_id=self.session_id)
        return ConversationBufferWindowMemory(
            chat_memory=message_history,
            memory_key="chat_history",
            return_messages=True,
            k=10  # 保留最近10轮对话
        )
    
    def _initialize_agent(self):
        """初始化Agent"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, query: str):
        """运行Agent处理查询"""
        try:
            logger.info(f"Agent处理查询: {query} (session_id: {self.session_id})")
            result = self.agent.run(query)
            logger.info(f"Agent返回结果 (session_id: {self.session_id})")
            return result
        except Exception as e:
            logger.error(f"Agent运行错误: {str(e)} (session_id: {self.session_id})")
            return f"处理查询时发生错误: {str(e)}"
```

### 8. 前端实现（app/main.py）
```python
import streamlit as st
from streamlit_ace import st_ace
import uuid
from app.agent.agent_builder import TeachingAgent
from app.rag.vector_store import RAGVectorStore
from app.utils.logging_config import logger

# 初始化页面
st.set_page_config(page_title="DSAI5102教学辅助Agent", layout="wide")
st.title("📚 DSAI5102教学辅助Agent")

# 初始化session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"新会话创建: {st.session_state.session_id}")

# 初始化Agent
if "agent" not in st.session_state:
    st.session_state.agent = TeachingAgent(session_id=st.session_state.session_id)

# 初始化RAG
if "rag" not in st.session_state:
    st.session_state.rag = RAGVectorStore()

# 侧边栏 - 文档上传
with st.sidebar:
    st.header("课件管理")
    uploaded_files = st.file_uploader("上传课程课件(PDF)", accept_multiple_files=True, type=["pdf"])
    use_ocr = st.checkbox("使用OCR提取（适用于扫描版PDF）", value=False)
    
    if st.button("添加到知识库"):
        if uploaded_files:
            for file in uploaded_files:
                with open(f"temp_{file.name}", "wb") as f:
                    f.write(file.getbuffer())
                success = st.session_state.rag.add_document(f"temp_{file.name}", use_ocr)
                if success:
                    st.success(f"成功添加: {file.name}")
                else:
                    st.error(f"添加失败: {file.name}")
        else:
            st.warning("请先上传文件")

# 代码编辑器
st.subheader("💻 代码执行区")
code = st_ace(
    language="python",
    theme="monokai",
    keybinding="vscode",
    font_size=14,
    tab_size=4,
    show_gutter=True,
    show_print_margin=False,
    wrap=True,
    height=300,
    value="# 在这里编写Python代码\n# 例如: 计算1+1\nprint(1+1)"
)

if st.button("运行代码"):
    try:
        result = st.session_state.agent.tools[2].func(code)  # 调用Python REPL工具
        st.code(result)
    except Exception as e:
        st.error(f"代码执行错误: {str(e)}")

# 对话区
st.subheader("💬 对话区")
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理新消息
if prompt := st.chat_input("有什么可以帮助你的吗？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = st.session_state.agent.run(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
```

## 二、Docker部署配置

### Dockerfile
```dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    poppler-utils \  #PDF处理
    tesseract-ocr \  #OCR支持
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN pip install --upgrade pip && pip install uv

# 复制项目文件
COPY pyproject.toml .
COPY .env.example .env
COPY . .

# 用uv安装依赖
RUN uv pip install .

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml（可选，含PostgreSQL）
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - POSTGRES_HOST=db
      - POSTGRES_PORT=5432
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=agent_password
      - POSTGRES_DB=agent_memory
    depends_on:
      - db
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./logs:/app/logs

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=agent_password
      - POSTGRES_DB=agent_memory
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 三、使用说明

1. **环境变量配置**：复制`.env.example`为`.env`，填入OpenAI API密钥和PostgreSQL信息
2. **本地运行**：
   ```bash
   # 安装依赖
   uv pip install .
   
   # 启动streamlit
   streamlit run app/main.py
   ```
3. **Docker部署**：
   ```bash
   # 构建镜像
   docker build -t teaching-agent .
   
   # 启动容器（需先配置.env）
   docker-compose up -d
   ```
4. **功能使用**：
   - 侧边栏上传课程PDF课件（支持普通和扫描版）
   - 对话区提问，Agent会自动调用课件检索/网页搜索/代码执行工具
   - 代码执行区可编写并运行Python代码，支持数据分析等操作

## 四、关键技术亮点

1. **混合文档处理**：支持普通PDF文本提取和扫描版PDF的OCR提取，适配不同类型课件
2. **持久化记忆**：通过PostgreSQL存储对话历史，支持跨会话记忆
3. **可扩展工具链**：基于LangChain的Tool机制，可轻松添加新工具（如公式计算、图表生成）
4. **开发友好**：使用uv管理依赖，pyproject.toml清晰展示项目结构，便于团队协作
5. **快速部署**：Docker容器化配置，一键部署完整环境（含数据库）

该方案完全满足您提出的技术栈要求，同时针对教学场景做了专门优化，可直接作为DSAI5102课程的教学案例展示Agent在教育领域的应用。