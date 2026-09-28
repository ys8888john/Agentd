"""取消（中断）链路测试。

取消是跨层能力：传输层把 session/cancel 变成 asyncio.Event，内核和模式在
chunk / 工具边界上查询并收尾。这里按层拆开测：

1. single 模式：流到一半叫停 —— 只出已流出的那部分，Done 是 cancelled，落库；
2. agent 模式：文本中途叫停 —— 举手过的工具**绝不执行**（没有 ToolCallStart）；
3. agent 模式：工具批次之间叫停 —— 剩下的调用不再派发；
4. 取消信号即使没人在跑也是无害 no-op；
5. 传输层：cancel() 与 prompt() 并发（SDK 每帧一个 task），轮次以
   PromptResponse(stop_reason="cancelled") 收尾，信号表清理干净。

节奏控制用带 on_chunk 钩子的假 LLM（同 test_mcp_agent.ScriptedLLM 的思路）：
停止信号"在第 N 个 chunk 之后到达"是确定性的，不靠 sleep 碰运气。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agentd.contracts import Done, MessageDelta, MessageDone, ToolCallStart
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.store import InMemorySessionStore
from agentd.transports.acp_stdio import AgentdAcpAgent

_CHUNK_DELAY = 0.001


class PaceLLM(LLM):
    """按剧本产事件；每产完一个事件调一次 on_chunk(i)，给测试挂钩子。

    注意 on_chunk 是在"产出第 i 个事件、消费者来取下一个"的时刻被调的 ——
    也就是第 i+1 个事件之前。想让信号在第 2 个 chunk 后到达，就传 i == 1。
    """

    def __init__(self, script: list, on_chunk=None, delay: float = _CHUNK_DELAY) -> None:
        self._script = script
        self._on_chunk = on_chunk
        self._delay = delay
        self.calls = 0

    async def stream_events(self, messages, *, system=None, tools=None):
        self.calls += 1
        for i, event in enumerate(self._script):
            await asyncio.sleep(self._delay)
            yield event
            if self._on_chunk is not None:
                self._on_chunk(i)


async def test_single_mode_cancel_stops_mid_stream() -> None:
    ev = asyncio.Event()

    def hook_factory():
        def hook(i: int) -> None:
            if i == 0:
                ev.set()

        return hook

    llm = PaceLLM([LLMText(t) for t in ["一", "二", "三", "四", "五"]], on_chunk=hook_factory())
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "你好", mode="single", cancel=ev)]

    deltas = [e for e in events if isinstance(e, MessageDelta)]
    dones = [e for e in events if isinstance(e, MessageDone)]
    finis = [e for e in events if isinstance(e, Done)]

    # 没流完五个 chunk：信号在第 1 个 chunk 后置位，第 3 个来不及出
    assert len(deltas) < 5
    assert "".join(d.text for d in deltas).endswith("二")
    assert [e.stop_reason for e in finis] == ["cancelled"]
    assert len(dones) == 1
    # MessageDone 承载落库文本，必须是已流出的部分
    assert dones[0].text

    history = await kernel.history(sid)
    assert history[-1].role == "assistant"
    assert history[-1].content == dones[0].text


async def test_single_mode_no_cancel_keeps_end_turn() -> None:
    """不设取消：行为与从前完全一致（回归保护）。"""
    llm = PaceLLM([LLMText("完整"), LLMText("回答")])
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "你好", mode="single")]

    finis = [e for e in events if isinstance(e, Done)]
    assert [e.stop_reason for e in finis] == ["end_turn"]
    dones = [e for e in events if isinstance(e, MessageDone)]
    assert dones[0].text == "完整回答"


async def test_agent_mode_cancel_skips_tool_dispatch() -> None:
    """文本中途叫停：模型已举手要调工具，但工具绝不能真的执行。"""
    ev = asyncio.Event()

    def hook(i: int) -> None:
        if i == 0:
            ev.set()  # "让我" 流出后就叫停

    llm = PaceLLM(
        [
            LLMText("让我"),
            LLMToolCall(id="c1", name="read_file", arguments='{"path":"x.txt"}'),
            LLMText("看看"),
        ],
        on_chunk=hook,
    )
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "你好", mode="agent", cancel=ev)]

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert starts == []  # 工具一次都没派
    fins = [e for e in events if isinstance(e, Done)]
    assert [e.stop_reason for e in fins] == ["cancelled"]
    dones = [e for e in events if isinstance(e, MessageDone)]
    # "看看"排在工具调用事件之后，信号一收到就收尾 —— 它永远不会流出
    assert dones[0].text == "让我"


async def test_agent_mode_cancel_between_tool_batches() -> None:
    """第一批工具派发后叫停：剩下的调用不派发、第二轮 LLM 不发起。

    这里没有配 MCP server，工具是"未知工具"——它会走到
    "[错误] 未知工具"的完成分支，正好当第一批。测试主循环看到
    ToolCallStart 就把信号 set 上，断言只剩这一张卡、没有第二轮。
    """

    ev = asyncio.Event()

    class TwoRoundLLM(LLM):
        def __init__(self) -> None:
            self.rounds = 0

        async def stream_events(self, messages, *, system=None, tools=None):
            self.rounds += 1
            if self.rounds == 1:
                yield LLMToolCall(id="c1", name="no_such_tool", arguments="{}")
                return
            # 第二轮：此时 cancel 已经被测试主循环在收到工具完成事件后 set
            yield LLMText("不该到这里")

    two = TwoRoundLLM()
    kernel = AgentKernel(llm=two, store=InMemorySessionStore())
    sid = await kernel.create_session()

    events = []
    agen = kernel.handle(sid, "你好", mode="agent", cancel=ev)
    async for e in agen:
        events.append(e)
        if isinstance(e, ToolCallStart):
            ev.set()  # 工具一起跑就叫停，后续轮次不该发生

    fins = [e for e in events if isinstance(e, Done)]
    assert [e.stop_reason for e in fins] == ["cancelled"]
    assert two.rounds == 1  # 第二次 LLM 调用从未发生
    dones = [e for e in events if isinstance(e, MessageDone)]
    assert dones[0].text == "（已手动停止）"


async def test_cancel_without_running_prompt_is_noop() -> None:
    """没人跑的时候收到取消：只记日志，不炸、不留状态。"""
    agent = AgentdAcpAgent(AgentKernel(llm=PaceLLM([LLMText("x")]), store=InMemorySessionStore()))
    agent.on_connect(_FakeConn())

    await agent.cancel("sess_nope")  # 不能抛

    assert agent._cancels == {}


class _FakeConn:
    """conn 的最小替身：只记录 session_update 调用。"""

    def __init__(self) -> None:
        self.updates: list = []

    async def session_update(self, session_id, update) -> None:
        self.updates.append((session_id, update))


async def test_transport_cancel_finishes_prompt_as_cancelled() -> None:
    """传输层并发：cancel() 在 prompt() 跑到一半时到达，轮次以 cancelled 收尾。"""

    class WatchedLLM(LLM):
        def __init__(self) -> None:
            self.first_out = asyncio.Event()

        async def stream_events(self, messages, *, system=None, tools=None):
            for i, t in enumerate(["A", "B", "C", "D", "E", "F"]):
                await asyncio.sleep(_CHUNK_DELAY)
                yield LLMText(t)
                if i == 0:
                    self.first_out.set()

    llm = WatchedLLM()
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    agent = AgentdAcpAgent(kernel)
    conn = _FakeConn()
    agent.on_connect(conn)
    sid = await kernel.create_session()

    async def stopper() -> None:
        await llm.first_out.wait()
        await agent.cancel(sid)

    stop_task = asyncio.ensure_future(stopper())
    resp = await agent.prompt(sid, [SimpleNamespace(text="你好")])
    await stop_task

    assert resp.stop_reason == "cancelled"
    texts = [
        # SDK 的 AgentMessageChunk 增量字段是 content，工具事件没有正文
        str(getattr(u, "content", None) or getattr(u, "text", None) or "")
        for _, u in conn.updates
    ]
    assert any(t for t in texts)  # 至少流出过一点增量
    assert not agent._cancels  # 轮次结束必须摘干净信号


async def test_cancel_after_finished_prompt_is_noop_for_next_round() -> None:
    """prompt 正常结束后再 cancel：no-op，且不影响下一轮（信号没串台）。"""

    llm = PaceLLM([LLMText("正常")])
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    agent = AgentdAcpAgent(kernel)
    agent.on_connect(_FakeConn())
    sid = await kernel.create_session()

    resp = await agent.prompt(sid, [SimpleNamespace(text="你好")])
    assert resp.stop_reason == "end_turn"

    await agent.cancel(sid)
    assert agent._cancels == {}
    resp2 = await agent.prompt(sid, [SimpleNamespace(text="再来")])
    assert resp2.stop_reason == "end_turn"
