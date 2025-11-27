from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from ddgs import DDGS
from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup


mcp = FastMCP("websearch-server", port=8881)


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> TextContent:
    """搜索网络信息并返回相关链接和标题。使用DuckDuckGo搜索引擎。"""
    ddg = DDGS()
    results = ddg.text(query, max_results=max_results)
    text = "\n".join([f"{r['title']} - {r['href']}" for r in results])
    return TextContent(type="text", text=text)


@mcp.tool()
def fetch_webpage_content(url: str) -> TextContent:
    """获取网页内容（纯文本8000字符）。从URL中提取纯文本，自动清理HTML标签、脚本和样式。"""
    try:
        ua = UserAgent()
        headers = {"User-Agent": ua.random}

        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        cleaned = "\n".join([line.strip()
                            for line in text.split("\n") if line.strip()])

        return TextContent(type="text", text=cleaned[:8000])

    except Exception as e:
        return TextContent(type="text", text=f"[Error fetching webpage] {str(e)}")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
