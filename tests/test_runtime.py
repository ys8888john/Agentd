"""三入口共享会话语义（SessionRuntime）的单测。

runtime 是 ACP / Gateway / CLI 三个入口共用的那层"会话运行时"：模式覆盖、
取消信号、审批记忆。这里把它的语义钉死 —— 三个入口的行为一致性靠它保证。
不依赖任何 LLM / 网络 / 文件系统。
"""

from __future__ import annotations

import asyncio

import pytest

from agentd.contracts import Done, MessageDelta
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.store import InMemorySessionStore
from agentd.kernel.tools import ApprovalRequest
from agentd.transports.runtime import (
    ALLOW_ONCE,
    ALLOW_SESSION,
    REJECT,
    SessionRuntime,
)


class EchoLLM(LLM):
    async def stream_events(self, messages, *, system=None, tools=None):
        yield LLMText("答")


async def _kernel() -> AgentKernel:
    return AgentKernel(llm=EchoLLM(), store=InMemorySessionStore())


def _req(tool: str = "write_file") -> ApprovalRequest:
    return ApprovalRequest(call_id="c1", tool=tool, title="写文件", kind="edit", detail="a.txt")


# ---- 模式 ----

async def test_default_mode_is_agent() -> None:
    runtime = SessionRuntime(await _kernel())
    sid = await runtime.kernel.create_session()
    assert runtime.mode_of(sid) == "agent"
    assert "single" in runtime.modes() and "agent" in runtime.modes()


async def test_set_mode_roundtrips() -> None:
    runtime = SessionRuntime(await _kernel())
    sid = await runtime.kernel.create_session()
    runtime.set_mode(sid, "single")
    assert runtime.mode_of(sid) == "single"


# ---- 取消 ----

async def test_cancel_without_running_round_is_noop() -> None:
    runtime = SessionRuntime(await _kernel())
    assert runtime.cancel("sess_nope") is False


async def test_run_is_cancellable_and_cleans_signal() -> None:
    class SlowLLM(LLM):
        async def stream_events(self, messages, *, system=None, tools=None):
            for t in ["一", "二", "三", "四"]:
                await asyncio.sleep(0.005)
                yield LLMText(t)

    kernel = AgentKernel(llm=SlowLLM(), store=InMemorySessionStore())
    runtime = SessionRuntime(kernel)
    sid = await kernel.create_session()

    events = []
    async for e in runtime.run(sid, "你好", mode="single"):
        events.append(e)
        if isinstance(e, MessageDelta):
            runtime.cancel(sid)  # 流出一点就停

    fins = [e for e in events if isinstance(e, Done)]
    assert [e.stop_reason for e in fins] == ["cancelled"]
    # 轮次结束必须把取消信号摘干净，否则会串到下一轮
    assert runtime._cancels == {}  # noqa: SLF001 - 锁内部不变量


# ---- 审批 ----

async def test_approver_allows_once() -> None:
    runtime = SessionRuntime(await _kernel())

    async def ask(req):
        assert req.tool == "write_file"
        return ALLOW_ONCE

    approve = runtime.approver("s1", ask)
    assert await approve(_req()) is True
    # allow_once 不该被记住：下次还要问
    assert runtime.granted("s1", "write_file") is False


async def test_approver_allow_session_is_remembered() -> None:
    runtime = SessionRuntime(await _kernel())
    asked = []

    async def ask(req):
        asked.append(req.tool)
        return ALLOW_SESSION

    approve = runtime.approver("s1", ask)
    assert await approve(_req()) is True
    # 本会话总是允许：第二次不再打扰
    assert await approve(_req()) is True
    assert asked == ["write_file"]  # 只问了一次
    assert runtime.granted("s1", "write_file") is True


async def test_approver_reject_is_false() -> None:
    runtime = SessionRuntime(await _kernel())

    async def ask(req):
        return REJECT

    approve = runtime.approver("s1", ask)
    assert await approve(_req()) is False
    assert runtime.granted("s1", "write_file") is False


async def test_approver_error_is_treated_as_reject() -> None:
    """问不到人（连接断了等）必须按拒绝 —— 反过来就是安全漏洞。"""
    runtime = SessionRuntime(await _kernel())

    async def ask(req):
        raise RuntimeError("客户端断了")

    approve = runtime.approver("s1", ask)
    assert await approve(_req()) is False


async def test_approver_none_means_no_gate() -> None:
    """ask=None（没人可问）时内核按放行处理，不返回回调。"""
    runtime = SessionRuntime(await _kernel())
    assert runtime.approver("s1", None) is None


async def test_run_propagates_ask_to_approver() -> None:
    """runtime.run 把 ask 注成审批回调，工具审批时真的能拦住。"""

    class ToolLLM(LLM):
        def __init__(self) -> None:
            self.calls = 0

        async def stream_events(self, messages, *, system=None, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield LLMToolCall(id="c1", name="write_file",
                                  arguments='{"path":"a.txt","content":"x"}')
            else:
                yield LLMText("收尾")

    kernel = AgentKernel(llm=ToolLLM(), store=InMemorySessionStore())
    runtime = SessionRuntime(kernel)
    sid = await kernel.create_session(cwd=".")

    asked = []

    async def ask(req):
        asked.append(req.tool)
        return REJECT

    events = [e async for e in runtime.run(sid, "写个文件", mode="agent", ask=ask)]
    assert asked == ["write_file"]  # 审批被触发过
    assert any(isinstance(e, Done) for e in events)


if __name__ == "__main__":
    pytest.main([__file__])