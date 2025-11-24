import os
import asyncio
import fitz  # PyMuPDF
from pathlib import Path
from tqdm.asyncio import tqdm
import multiprocessing
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

os.chdir(Path(__file__).parent.parent)
print(f"Current working directory: {os.getcwd()}")


# -----------------------------
# 🔹 1. 异步读取 PDF
# -----------------------------
async def read_pdf_async(pdf_path: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, read_pdf_sync, pdf_path)


def read_pdf_sync(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# -----------------------------
# 🔹 2. 异步生成 Embedding
# -----------------------------
async def embed_batch(
    texts: List[str],
    model: OllamaEmbeddings,
    sem: asyncio.Semaphore,
    pbar: tqdm
):
    """利用并发生成 embedding"""
    async with sem:   # 限制并行度
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model.embed_documents, texts)
        pbar.update(len(texts))
        return result


# -----------------------------
# 🔹 3. 主流程：多个 PDF → 文本 → Embedding → FAISS
# -----------------------------
async def build_faiss_from_pdfs(pdf_paths: List[str], faiss_path: str):
    print(f"📄 共 {len(pdf_paths)} 个 PDF 文件")

    if len(pdf_paths) == 0:
        print("❌ 错误：没有找到 PDF 文件!")
        return

    # ------------------------
    # Step 1: 异步读取全部 PDF
    # ------------------------
    print("📘 正在读取 PDF ...")
    pdf_tasks = [read_pdf_async(p) for p in pdf_paths]
    pdf_texts = await tqdm.gather(*pdf_tasks)
    print("📘 PDF 读取完成\n")

    # ------------------------
    # Step 2: 拆分文本（按段落）
    # ------------------------
    docs = []
    for pdf_text in pdf_texts:
        parts = [p.strip() for p in pdf_text.split("\n") if p.strip()]
        docs.extend(parts)

    print(f"✂️ 总文本段落数量： {len(docs)}")

    # ------------------------
    # Step 3: 并发 embedding
    # ------------------------
    embed_model = OllamaEmbeddings(model="nomic-embed-text")

    # 自动并行度 = CPU 核数
    max_workers = multiprocessing.cpu_count()
    sem = asyncio.Semaphore(max_workers)

    print(f"⚡ 开始生成 embeddings（并行度 = {max_workers}）")

    # 你可以调节 chunk_size，这里默认 32 行一个 batch
    chunk_size = 32
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

    pbar = tqdm(total=len(docs), desc="Embedding")

    embed_tasks = [
        embed_batch(chunk, embed_model, sem, pbar)
        for chunk in chunks
    ]

    all_vectors_nested = await tqdm.gather(*embed_tasks)
    all_vectors = [v for batch in all_vectors_nested for v in batch]  # flatten
    pbar.close()

    print("⚡ Embeddings 生成完毕\n")

    # ------------------------
    # Step 4: 建立 FAISS
    # ------------------------
    print("🧱 正在构建 FAISS index ...")
    # FAISS.from_embeddings 需要 (text, embedding) 的元组列表
    text_embedding_pairs = list(zip(docs, all_vectors))
    faiss_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embed_model
    )

    faiss_store.save_local(faiss_path)
    print(f"📦 FAISS 已保存到：{faiss_path}")


# -----------------------------
# 🔹 主入口（外部调用）
# -----------------------------
def build(pdf_paths: List[str], faiss_path="faiss_index"):
    asyncio.run(build_faiss_from_pdfs(pdf_paths, faiss_path))


# -----------------------------
# 🔹 脚本运行
# -----------------------------
if __name__ == "__main__":
    pdfs = list(Path("assets/pdfs").glob("*.pdf"))
    build(pdfs, "vectordb/faiss")
