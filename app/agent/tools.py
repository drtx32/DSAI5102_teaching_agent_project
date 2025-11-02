from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from app.rag.vector_store import RAGVectorStore
from app.utils.logging_config import logger

# PythonREPLTool 在不同 langchain 版本中的路径不同，使用回退导入
try:
    from langchain_community.tools.python.tool import PythonREPLTool
except ImportError:
    try:
        from langchain_experimental.tools import PythonREPLTool
    except ImportError:
        # 如果都失败，提供一个最小的本地实现
        from langchain.tools import Tool as BaseTool
        import sys
        from io import StringIO

        class PythonREPLTool:
            name = "Python REPL"
            description = "Execute Python code"

            def run(self, code: str) -> str:
                """执行 Python 代码并返回结果"""
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    exec(code, {})
                    output = sys.stdout.getvalue()
                    return output if output else "代码执行成功（无输出）"
                except Exception as e:
                    return f"执行错误: {str(e)}"
                finally:
                    sys.stdout = old_stdout


class AgentTools:
    def __init__(self):
        self.rag = RAGVectorStore()
        self.search = DuckDuckGoSearchRun()
        self.python_repl = PythonREPLTool()

    def get_tools(self):
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
