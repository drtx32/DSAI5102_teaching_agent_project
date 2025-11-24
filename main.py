import os
import sys
from threading import Thread


def main() -> None:
    rag_server = Thread(target=lambda: os.system(
        "python app/mcp/rag/server.py"))
    websearch_server = Thread(target=lambda: os.system(
        "python app/mcp/websearch/server.py"))
    streamlit_server = Thread(target=lambda: os.system(
        "streamlit run 首页.py"))

    rag_server.start()
    websearch_server.start()
    streamlit_server.start()


if __name__ == "__main__":
    main()
