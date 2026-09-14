"""MCP 工具循环测试。

三块：
1. AgentMode 的循环逻辑（用假 McpHub + 脚本化 LLM，确定性强、不起进程）；
2. McpHub 连真 stdio MCP server（echo）的集成；
3. 端到端：kernel.handle 带 mcp_servers 跑通一轮（真 server + 脚本化 LLM）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentd.contracts import MessageDelta, MessageDone, ToolCallDone, ToolCallStart
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.mcp import McpHub
from agentd.kernel.modes.agent import AgentMode
from agentd.kernel.modes.base import ModeContext
from agentd.kernel.models import Message

_ECHO_SERVER = str(Path(__file__).parent / "_echo_mcp_server.py")


class ScriptedLLM(LLM):
    """按脚本逐轮产出事件；记录每次收到的 messages/tools 便于断言。"""

    def __init__(self, scripts: list[list]) -> None:
        self._scripts = list(scripts)
        self.calls: list[dict] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        script = self._scripts.pop(0) if self._scripts else [LLMText("（脚本耗尽）")]
        for event in script:
            yield event


def _echo_mcp_config() -> dict:
    return {"name": "echo", "command": sys.executable, "args": [_ECHO_SERVER]}


# ---------------------------------------------------------------------------
# 1) 循环逻辑（假 Hub）
# ---------------------------------------------------------------------------

class _FakeHub:
    def __init__(self, servers=None, *, cwd=None) -> None:
        self.servers = servers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def tool_schema(self):
        if not self.servers:
            return []
        return [{"type": "function", "function": {"name": "echo__echo", "description": "", "parameters": {}}}]

    def binding(self, name):
        return SimpleNamespace(tool="echo") if name == "echo__echo" else None

    async def call(self, name, arguments):
        return f"echo: {arguments}"


async def test_agent_mode_runs_tool_then_answers(monkeypatch):
    import agentd.kernel.modes.agent as agent_mod

    monkeypatch.setattr(agent_mod, "McpHub", _FakeHub)

    llm = ScriptedLLM(
        [
            # 第一轮：模型要求调 echo
            [LLMToolCall(id="c1", name="echo__echo", arguments='{"text":"hi"}')],
            # 第二轮：模型给出最终文本
            [LLMText("完成"), LLMText("了")],
        ]
    )
    ctx = ModeContext(session_id="s", run_id="r", llm=llm, history=[Message.user("hi")])
    events = [e async for e in AgentMode().run(ctx, "hi")]

    kinds = [type(e).__name__ for e in events]
    assert "ToolCallStart" in kinds
    assert "ToolCallDone" in kinds
    assert isinstance(events[-1], MessageDone)
    assert events[-1].text == "完成了"

    # 工具结果被回灌进第二轮消息（role=tool）
    second_messages = llm.calls[1]["messages"]
    assert any(m.role == "tool" for m in second_messages)
    # assistant 那轮带了 tool_calls
    assert any(m.role == "assistant" and m.tool_calls for m in second_messages)


async def test_agent_mode_no_tools_behaves_like_single(monkeypatch):
    import agentd.kernel.modes.agent as agent_mod

    monkeypatch.setattr(agent_mod, "McpHub", _FakeHub)

    llm = ScriptedLLM([[LLMText("你好")]])
    ctx = ModeContext(session_id="s", run_id="r", llm=llm, history=[Message.user("hi")])
    events = [e async for e in AgentMode().run(ctx, "hi")]

    assert [type(e).__name__ for e in events] == ["MessageDelta", "MessageDone"]
    assert events[-1].text == "你好"
    # 没工具的 hub，tools 传 None
    assert llm.calls[0]["tools"] is None


# ---------------------------------------------------------------------------
# 2) McpHub 连真 server
# ---------------------------------------------------------------------------

async def test_mcp_hub_connects_lists_and_calls():
    async with McpHub([_echo_mcp_config()], cwd=str(Path(__file__).parent)) as hub:
        assert hub.has_tools, f"没列到工具；errors={hub.errors}"
        schema = hub.tool_schema()
        names = [t["function"]["name"] for t in schema]
        assert "echo__echo" in names
        out = await hub.call("echo__echo", '{"text":"世界"}')
        assert out == "echo: 世界"


async def test_mcp_hub_unknown_tool_returns_error():
    async with McpHub([_echo_mcp_config()], cwd=str(Path(__file__).parent)) as hub:
        out = await hub.call("nope__nope", "{}")
        assert out.startswith("[错误]")


async def test_mcp_hub_bad_server_is_isolated():
    async with McpHub([{"name": "bad", "command": "definitely-not-a-real-cmd-xyz"}]) as hub:
        assert not hub.has_tools
        assert hub.errors  # 失败被记录，但没抛，整轮不崩


# ---------------------------------------------------------------------------
# 3) 端到端：kernel.handle 带 mcp_servers
# ---------------------------------------------------------------------------

async def test_kernel_handle_tool_loop_end_to_end():
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="echo__echo", arguments='{"text":"端到端"}')],
            [LLMText("工具说："), LLMText("echo: 端到端")],
        ]
    )
    kernel = AgentKernel(llm=llm)
    session_id = await kernel.create_session(
        cwd=str(Path(__file__).parent), mcp_servers=[_echo_mcp_config()]
    )

    events = [e async for e in kernel.handle(session_id, "调用 echo", mode="agent")]

    assert any(isinstance(e, ToolCallStart) for e in events)
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done and done[0].status == "completed"
    assert done[0].output == "echo: 端到端"
    # 内核的流末尾是 Done(整体收尾)，MessageDone 在它之前
    message_done = [e for e in events if isinstance(e, MessageDone)]
    assert message_done and message_done[-1].text == "工具说：echo: 端到端"

    # 最终文本作为 assistant 消息落库（内核统一负责持久化）
    history = await kernel.history(session_id)
    assert history[-1].role == "assistant"
    assert history[-1].content == "工具说：echo: 端到端"
