"""工具大结果外置化（blobstore）测试。

背景：2026-10-08 事故 —— web_fetch 一篇几百 KB 的文章被打成一行 JSON-RPC
大帧，打死 GUI 侧 asyncio 默认 64KB 的 stdout 读入上限，误报「agent 进程已
退出」。GUI 侧放大 limit 是止血，这里的**外置化**才是治本：管道里的帧本来
就不该有几百 KB（设计参考 WorkBuddy 的 ToolResultBlobService）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentd.kernel import blobstore
from agentd.kernel.blobstore import externalize
from agentd.kernel.models import Message
from agentd.kernel.modes.agent import AgentMode
from agentd.kernel.modes.base import ModeContext


# ---- 纯函数 ----


def test_small_output_passes_through(tmp_path):
    assert externalize("正常输出", "sess_1", "call_1", home=tmp_path) == "正常输出"


def test_big_output_is_persisted_with_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(blobstore, "THRESHOLD_BYTES", 1000)
    big = "甲" * 5000  # 15000 字节（中文 3 字节/字符），按字节判定必须触发
    wrapped = externalize(big, "sess_测试/1", "call:X.1", home=tmp_path)

    assert "<persisted-output>" in wrapped and "</persisted-output>" in wrapped
    assert "完整内容已保存到" in wrapped
    assert "预览（前 2048 字符）" in wrapped
    # 全文落盘，路径真实存在且内容完整
    files = list((tmp_path / "tool_results").rglob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == big
    # 非法文件名字符被清洗（: 被替换成下划线，字母数字与点保留）
    assert files[0].stem == "call_X.1"


def test_chinese_content_is_measured_in_bytes(tmp_path, monkeypatch):
    """中文内容按字符数会漏判（1 字符 = 3 字节），必须按字节判定。"""
    monkeypatch.setattr(blobstore, "THRESHOLD_BYTES", 1000)
    # 600 字符 = 1800 字节 > 1000 字节阈值；但 600 < 2048，字符数视角会漏判
    text = "中" * 600
    assert externalize(text, "s", "c", home=tmp_path) != text  # 被外置了


def test_exactly_at_threshold_not_persisted(tmp_path):
    text = "x" * blobstore.THRESHOLD_BYTES  # 恰好等于阈值（1 字节/字符）
    assert externalize(text, "s", "c", home=tmp_path) == text


# ---- 接进 AgentMode ----


class _BigToolHub:
    """提供一个返回 5000 字符结果的假 MCP 工具。"""

    def __init__(self, servers=None, *, cwd=None) -> None:
        self.servers = servers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def tool_schema(self):
        return [{"type": "function", "function": {"name": "big__boom", "description": "", "parameters": {}}}]

    def binding(self, name):
        return SimpleNamespace(tool="boom") if name == "big__boom" else None

    async def call(self, name, arguments):
        return "乙" * 5000


class _ScriptedLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.calls.append({"messages": list(messages)})
        if len(self.calls) == 1:
            from agentd.kernel.llm import LLMToolCall

            yield LLMToolCall(id="call_big_1", name="big__boom", arguments="{}")
        else:
            from agentd.kernel.llm import LLMText

            yield LLMText("拿到结果了")


async def test_agent_mode_externalizes_big_tool_result(tmp_path, monkeypatch):
    import agentd.kernel.modes.agent as agent_mod
    from agentd.contracts import ToolCallDone

    monkeypatch.setattr(agent_mod, "McpHub", _BigToolHub)
    monkeypatch.setattr(blobstore, "THRESHOLD_BYTES", 1000)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    llm = _ScriptedLLM()
    ctx = ModeContext(session_id="sess_ext", run_id="r", llm=llm, history=[Message.user("hi")])
    events = [e async for e in AgentMode().run(ctx, "hi")]

    done = next(e for e in events if isinstance(e, ToolCallDone))
    assert done.status == "completed"
    assert "<persisted-output>" in done.output
    # 模型拿到的 role=tool 消息也是包装后的（上下文不会灌进 5000 字符原文）
    second = llm.calls[1]["messages"]
    tool_msg = next(m for m in second if m.role == "tool")
    assert "<persisted-output>" in tool_msg.content
    # 全文在盘上（Path.home() 被补丁成 tmp_path，因此落在 tmp_path/.agentd/ 下）
    spilled = list(tmp_path.rglob("call_big_1.txt"))
    assert len(spilled) == 1
    assert len(spilled[0].read_text(encoding="utf-8")) == 5000
