from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from app.rag.document_loader import DocumentProcessor
from app.utils.config import settings
from app.utils.logging_config import logger


class RAGVectorStore:
    def __init__(self, embedding_backend: str | None = None):
        """初始化向量存储。

        embedding_backend: 'openai' 或 'sentence_transformers'（默认为 settings.embedding_backend）
        """
        self.embedding_backend = embedding_backend or settings.embedding_backend
        self.document_processor = DocumentProcessor()

        if self.embedding_backend == "sentence_transformers":
            # 延迟导入，避免在未安装依赖时立即失败
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                logger.error("sentence-transformers 未安装或无法导入: %s", e)
                raise

            self.st_model = SentenceTransformer(
                settings.sentence_transformers_model)

            # 创建自定义 embedding 函数（兼容 Chroma 的接口）
            class _SentenceTransformerEmbeddings:
                def __init__(self, model):
                    self.model = model

                def embed_documents(self, texts):
                    # Chroma expects a list of lists
                    return [vec.tolist() for vec in self.model.encode(list(texts))]

                def embed_query(self, text):
                    # Chroma expects a single list
                    return self.model.encode([text])[0].tolist()

            embedding_function = _SentenceTransformerEmbeddings(self.st_model)
        else:
            # 默认使用 OpenAI Embeddings
            embedding_function = OpenAIEmbeddings(
                api_key=settings.openai_api_key, model=settings.openai_embedding_model)

        # 创建 Chroma 向量库实例
        self.vector_store = Chroma(
            persist_directory=settings.chroma_persist_directory,
            embedding_function=embedding_function,
            collection_name="teaching_materials"
        )

    def add_document(self, file_path: str, use_ocr: bool = False):
        """添加文档到向量库"""
        try:
            documents = self.document_processor.load_document(
                file_path, use_ocr)
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
        docs = self.vector_store.similarity_search(query, k=k)
        # 返回格式化的结果字符串，便于Agent使用
        if docs:
            result = "\n\n".join(
                [f"[文档片段 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])
            return result
        return "未找到相关内容"
