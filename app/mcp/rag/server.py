import os
from typing import List, Optional
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


os.chdir(Path(__file__).parent.parent.parent.parent)
print(f"Current working directory: {os.getcwd()}")

# =====================================================
# 配置
# =====================================================
VECTORDB_DIR = "vectordb/faiss"
DEFAULT_TOP_K = 5

# 使用与 build_vectordb.py 相同的 embedding 模型
embedder = OllamaEmbeddings(model="nomic-embed-text")

# =====================================================
# MCP Server
# =====================================================
mcp = FastMCP("rag-server", port=8002)


# =====================================================
# FAISS 向量数据库管理
# =====================================================
class RAGVectorStore:
    """管理 FAISS 向量数据库的加载和查询"""

    def __init__(self, vectordb_path: str):
        self.vectordb_path = vectordb_path
        self.vectorstore: Optional[FAISS] = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        """加载 FAISS 向量数据库"""
        if not os.path.exists(self.vectordb_path):
            print(f"⚠️ 警告: 向量数据库路径不存在: {self.vectordb_path}")
            self.vectorstore = None
            return

        try:
            self.vectorstore = FAISS.load_local(
                self.vectordb_path,
                embedder,
                allow_dangerous_deserialization=True
            )
            print(f"✅ 成功加载向量数据库: {self.vectordb_path}")
        except Exception as e:
            print(f"❌ 加载向量数据库失败: {e}")
            self.vectorstore = None

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[tuple]:
        """
        搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回最相关的 top_k 个结果

        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未加载，请先构建向量数据库")

        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return results

    def reload(self):
        """重新加载向量数据库"""
        self._load_vectorstore()


# 初始化 RAG 向量存储
rag_store = RAGVectorStore(VECTORDB_DIR)


# =====================================================
# MCP Tools
# =====================================================
@mcp.tool()
def search_documents(query: str, top_k: int = DEFAULT_TOP_K) -> TextContent:
    """
    在向量数据库中搜索与查询最相关的文档片段。

    Args:
        query: 搜索查询文本
        top_k: 返回最相关的文档数量（默认5个）

    Returns:
        包含相关文档内容和相似度分数的文本
    """
    try:
        results = rag_store.search(query, top_k=top_k)

        if not results:
            return TextContent(
                type="text",
                text="未找到相关文档。请确保已构建向量数据库。"
            )

        # 格式化结果
        output_lines = [f"📚 找到 {len(results)} 个相关文档片段:\n"]

        for idx, (doc, score) in enumerate(results, 1):
            similarity = 1 - score  # FAISS 返回的是距离，转换为相似度
            output_lines.append(f"--- 文档 {idx} (相似度: {similarity:.4f}) ---")
            output_lines.append(doc.page_content)
            output_lines.append("")

        return TextContent(
            type="text",
            text="\n".join(output_lines)
        )

    except Exception as e:
        return TextContent(
            type="text",
            text=f"❌ 搜索出错: {str(e)}"
        )


@mcp.tool()
def get_context_for_query(query: str, top_k: int = 3) -> TextContent:
    """
    获取查询的上下文，用于 RAG 问答。返回合并后的相关文档内容。

    Args:
        query: 查询文本
        top_k: 返回最相关的文档数量（默认3个）

    Returns:
        合并后的相关文档内容，可直接作为 LLM 的上下文
    """
    try:
        results = rag_store.search(query, top_k=top_k)

        if not results:
            return TextContent(
                type="text",
                text="未找到相关上下文。"
            )

        # 合并所有相关文档
        context_parts = []
        for idx, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[上下文片段 {idx}]")
            context_parts.append(doc.page_content)
            context_parts.append("")

        return TextContent(
            type="text",
            text="\n".join(context_parts)
        )

    except Exception as e:
        return TextContent(
            type="text",
            text=f"❌ 获取上下文出错: {str(e)}"
        )


@mcp.tool()
def reload_vectorstore() -> TextContent:
    """
    重新加载向量数据库。当向量数据库更新后，使用此工具重新加载。

    Returns:
        重新加载的状态信息
    """
    try:
        rag_store.reload()
        if rag_store.vectorstore is not None:
            return TextContent(
                type="text",
                text=f"✅ 向量数据库已重新加载: {VECTORDB_DIR}"
            )
        else:
            return TextContent(
                type="text",
                text=f"⚠️ 向量数据库加载失败，请检查路径: {VECTORDB_DIR}"
            )
    except Exception as e:
        return TextContent(
            type="text",
            text=f"❌ 重新加载出错: {str(e)}"
        )


@mcp.tool()
def check_vectorstore_status() -> TextContent:
    """
    检查向量数据库的状态和信息。

    Returns:
        向量数据库的状态信息
    """
    status_lines = [
        "📊 向量数据库状态:",
        f"路径: {VECTORDB_DIR}",
        f"是否存在: {os.path.exists(VECTORDB_DIR)}",
    ]

    if rag_store.vectorstore is not None:
        try:
            # 获取向量数据库中的文档数量
            index_size = rag_store.vectorstore.index.ntotal
            status_lines.append(f"状态: ✅ 已加载")
            status_lines.append(f"文档数量: {index_size}")
            status_lines.append(f"Embedding 模型: nomic-embed-text")
        except Exception as e:
            status_lines.append(f"状态: ⚠️ 已加载但读取信息出错: {e}")
    else:
        status_lines.append("状态: ❌ 未加载")

    return TextContent(
        type="text",
        text="\n".join(status_lines)
    )


# =====================================================
# 启动服务器
# =====================================================
if __name__ == "__main__":
    print("🚀 启动 RAG MCP 服务器...")
    print(f"📁 向量数据库路径: {VECTORDB_DIR}")
    print(f"🤖 Embedding 模型: nomic-embed-text")
    mcp.run(transport="streamable-http")
