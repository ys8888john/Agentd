"""测试用的最小 MCP server（stdio）：暴露一个 echo 工具。

被 test_mcp_agent.py 以子进程方式拉起，验证 McpHub 真的能连、能列工具、能调用。
不放进 agentd 包，只属于测试夹具。
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool()
def echo(text: str) -> str:
    """原样回显输入文本（前面加个前缀便于断言）。"""
    return f"echo: {text}"


if __name__ == "__main__":
    server.run("stdio")
