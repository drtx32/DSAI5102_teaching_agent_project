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
                    # OCR结果也要分块
                    from langchain.schema import Document
                    full_doc = Document(page_content="\n".join(
                        texts), metadata={"source": file_path})
                    return self.text_splitter.split_documents([full_doc])
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
