import os
import sys
import platform
from threading import Thread


def main() -> None:
    rag_server = Thread(target=lambda: os.system(
        "python app/mcp/rag/server.py"))
    websearch_server = Thread(target=lambda: os.system(
        "python app/mcp/websearch/server.py"))
    if platform.system() == "Windows":
        addr = "127.0.0.1"
    elif platform.system() == "Linux":
        addr = "0.0.0.0"
    streamlit_server = Thread(target=lambda: os.system(
        f"streamlit run 首页.py --server.port=8501 --server.address={addr}"))

    rag_server.start()
    websearch_server.start()
    streamlit_server.start()


if __name__ == "__main__":
    main()
