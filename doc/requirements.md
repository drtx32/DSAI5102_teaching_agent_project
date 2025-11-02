我的基本要求如下：
1. 用langchain作为主题架构，大模型的话用的是OpenAI库支持的api接口，rag组件就用chroma
2. embedding模型用类似nomic-embed-text可以吗（pdf类型的文件用视觉大模型提取可能会好些，到时候可以选择提取方式）
3. 网页搜索用duckduckgo-search绑定为langchain的装饰器tool吧，然后辅助工具的Python REPL要的
4. streamlit前端可以加一个streamlit-ace作为前端交互性代码编辑器
5. 最好把提问的内容记忆到postgresql，这样可以后期记忆
6. 利用logging把所有操作记录
7. uv工具管理，pyproject.toml方便查看项目整体架构
8. dockerfile方便快速部署