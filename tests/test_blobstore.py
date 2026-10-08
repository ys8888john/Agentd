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
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM
from agentd.kernel.models import Message
from agentd.kernel.modes.agent import AgentMode
from agentd.kernel.modes.base import ModeContext
from agentd.kernel.store import InMemorySessionStore


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


# ---- 外置目录必须在工具的路径白名单里 ----
#
# 2026-10-08 实测事故（sess_65a0584cc8f94dd6 seq=479）：web_fetch 大输出被外置到
# ~/.agentd/tool_results/…，包装里写着"要全文就用 read_file 读上面的路径"，
# 但那个路径在工作目录之外 → read_file 直接回「路径越界」，模型照着提示读必然失败。
# 提示与守卫必须对齐：blobstore 落哪，工具箱就得允许读哪。


class _NullLLM(LLM):
    """占位 LLM：本段用例只验证工具箱的根目录，不真的跑一轮对话。"""

    async def stream_events(self, messages, *, system=None, tools=None):
        return
        yield  # pragma: no cover  —— 让这仍是生成器函数


def _kernel() -> AgentKernel:
    return AgentKernel(llm=_NullLLM(), store=InMemorySessionStore())


def test_tool_results_root_matches_externalize_location(tmp_path, monkeypatch):
    """`tool_results_root` 与实际落盘位置必须同源，不能各写一份。"""
    from agentd.kernel.blobstore import tool_results_root

    monkeypatch.setattr(blobstore, "THRESHOLD_BYTES", 10)
    externalize("x" * 1000, "sess_a", "call_b", home=tmp_path)
    spilled = list(tmp_path.rglob("call_b.txt"))
    assert len(spilled) == 1
    # 落盘文件就在 tool_results_root(home)/<session>/ 下
    assert spilled[0].parent.parent == tool_results_root(home=tmp_path)


def test_kernel_registers_tool_results_dir_as_a_readable_root(tmp_path, monkeypatch):
    """内核必须把 tool_results 目录加进工具箱的根，否则"要全文就 read_file"是死胡同。"""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    kernel = _kernel()
    roots = kernel._tool_roots(None)
    from agentd.kernel.blobstore import tool_results_root

    assert tool_results_root() in roots


def test_kernel_keeps_client_supplied_roots_alongside_blob_dir(tmp_path, monkeypatch):
    """客户端显式给的 additionalDirectories 不能被顶掉。"""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    kernel = _kernel()
    extra = tmp_path / "proj"
    extra.mkdir()
    roots = kernel._tool_roots([extra])
    from agentd.kernel.blobstore import tool_results_root

    assert extra in roots and tool_results_root() in roots


def test_blob_dir_is_actually_readable_through_the_toolbox(tmp_path, monkeypatch):
    """端到端：外置后按包装里给的路径 read_file，必须真的读得到全文。

    这条是事故的直接回归 —— 以前这一步会返回「路径越界」。
    """
    import asyncio
    import json as _json

    from agentd.kernel.tools import NativeToolbox

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(blobstore, "THRESHOLD_BYTES", 10)

    body = "正文" * 500
    # 不带 home=：走真实链路（Path.home()/.agentd/…），与内核注册的根同源
    wrapped = externalize(body, "sess_r", "call_r")
    # 包装里给的路径（externalize 用的是绝对路径）
    path = next(p for p in wrapped.splitlines() if str(tmp_path) in p)
    path = path.split("：", 1)[1].strip()

    (tmp_path / "work").mkdir(exist_ok=True)
    kernel = _kernel()
    tb = NativeToolbox(cwd=tmp_path / "work", additional_roots=kernel._tool_roots(None))
    out = asyncio.run(tb.call("read_file", _json.dumps({"path": path})))
    assert not out.startswith("[错误]"), out
    assert "正文" in out
