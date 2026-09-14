"""MCP 客户端管理：把 ACP 传来的 mcpServers 连起来，暴露成 LLM 工具并负责执行。

ACP 的设计是「客户端声明 MCP server、agent 负责连」——所以这里由内核侧持有连接，
把每个 server 的工具列出来，拼成 LLM 的 tools schema（名字加 server 前缀防冲突），
模型要调时再路由回去执行。

生命周期说明（重要）：MCP 的 stdio 客户端内部是 anyio task group，**必须与 enter/exit
在同一个 task**。所以这里刻意做成「每次 run 现开现关」（`async with McpHub(...)`），
不做跨 run 长驻 —— 代价是每轮重连（stdio 会重开一次子进程），MVP 可接受。
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

try:  # mcp 是可选依赖：没装时 agentd 仍要能起来，只是工具功能不可用
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于环境是否装了 mcp
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False

_MCP_HINT = "需要 MCP 支持：pip install mcp"


def _log(msg: str) -> None:
    """日志一律走 stderr —— agentd 的 stdout 是纯 JSON-RPC 通道。"""
    print(msg, file=sys.stderr, flush=True)


@dataclass
class ToolBinding:
    """一个可用工具到「哪个 server 的哪个原始工具」的映射。"""

    server: str
    tool: str          # server 侧的原始工具名
    full_name: str     # 暴露给 LLM 的名字：{server}__{tool}
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})


def _as_str_map(raw: Any) -> dict[str, str]:
    """ACP 的 env/headers 是 [{name, value}] 列表；也兼容直接给 dict。"""
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    out: dict[str, str] = {}
    for item in raw or []:
        if isinstance(item, dict) and item.get("name"):
            out[str(item["name"])] = str(item.get("value", ""))
    return out


def _result_text(result: Any) -> str:
    """把 CallToolResult 折成纯文本（非文本块用占位标记）。"""
    prefix = "[错误] " if getattr(result, "is_error", False) else ""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(f"<{getattr(block, 'type', 'unknown')}>")
    body = "\n".join(parts)
    if not body and getattr(result, "structured_content", None) is not None:
        body = json.dumps(result.structured_content, ensure_ascii=False)
    return prefix + body


class McpHub:
    """一次 run 内连上若干 MCP server；列工具、执行工具。"""

    def __init__(self, servers: list[Any] | None, *, cwd: str | None = None) -> None:
        self._servers = list(servers or [])
        self._cwd = cwd
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, ToolBinding] = {}
        self._errors: list[str] = []

    async def __aenter__(self) -> "McpHub":
        for raw in self._servers:
            await self._connect_one(raw)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()
        self._sessions.clear()

    # ---- 连接 ----

    async def _connect_one(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            self._errors.append(f"非法 server 配置：{raw!r}")
            return
        name = str(raw.get("name") or f"server{len(self._sessions)}")
        try:
            read, write = await self._open_stream(raw)
        except Exception as exc:  # noqa: BLE001 - 单个 server 挂了不该拖垮整轮
            msg = f"MCP server {name} 连接失败：{type(exc).__name__}: {exc}"
            _log(f"[agentd] {msg}")
            self._errors.append(msg)
            return

        try:
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
        except Exception as exc:  # noqa: BLE001
            msg = f"MCP server {name} 初始化/列工具失败：{type(exc).__name__}: {exc}"
            _log(f"[agentd] {msg}")
            self._errors.append(msg)
            return

        self._sessions[name] = session
        for tool in listed.tools:
            full = f"{name}__{tool.name}"
            self._tools[full] = ToolBinding(
                server=name,
                tool=tool.name,
                full_name=full,
                description=tool.description or "",
                parameters=tool.input_schema or {"type": "object", "properties": {}},
            )
        _log(f"[agentd] MCP server {name} 接入，工具 {len(listed.tools)} 个")

    async def _open_stream(self, raw: dict) -> tuple[Any, Any]:
        if not _MCP_AVAILABLE:
            raise RuntimeError(_MCP_HINT)
        url = raw.get("url")
        if url:
            from mcp.client.streamable_http import streamable_http_client

            streams = await self._stack.enter_async_context(
                streamable_http_client(str(url))
            )
            # TransportStreams 是 (read, write, get_session_id)，取前两个
            return streams[0], streams[1]
        params = StdioServerParameters(
            command=str(raw.get("command") or ""),
            args=[str(a) for a in raw.get("args") or []],
            env=_as_str_map(raw.get("env")) or None,
            cwd=str(raw.get("cwd") or self._cwd) if (raw.get("cwd") or self._cwd) else None,
        )
        return await self._stack.enter_async_context(stdio_client(params))

    # ---- 对外 ----

    @property
    def has_tools(self) -> bool:
        return bool(self._tools)

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    def tool_schema(self) -> list[dict]:
        """转成 OpenAI/Ollama 通用的 tools 数组。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": b.full_name,
                    "description": b.description,
                    "parameters": b.parameters,
                },
            }
            for b in self._tools.values()
        ]

    def binding(self, full_name: str) -> ToolBinding | None:
        return self._tools.get(full_name)

    async def call(self, full_name: str, arguments: str) -> str:
        binding = self._tools.get(full_name)
        if binding is None:
            return f"[错误] 未知工具：{full_name}"
        session = self._sessions.get(binding.server)
        if session is None:
            return f"[错误] MCP server {binding.server} 不可用"
        try:
            args = json.loads(arguments) if arguments else {}
        except (ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        try:
            result = await session.call_tool(binding.tool, args)
        except Exception as exc:  # noqa: BLE001
            return f"[错误] 调用 {full_name} 失败：{type(exc).__name__}: {exc}"
        return _result_text(result)
