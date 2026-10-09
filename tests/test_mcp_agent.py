"""MCP 工具循环测试。

三块：
1. AgentMode 的循环逻辑（用假 McpHub + 脚本化 LLM，确定性强、不起进程）；
2. McpHub 连真 stdio MCP server（echo）的集成；
3. 端到端：kernel.handle 带 mcp_servers 跑通一轮（真 server + 脚本化 LLM）。
"""

from __future__ import annotations

import json
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
# 1b) 工具循环耗尽（MAX_STEPS）时的收尾正文
#
# 2026-10-08 实测事故（sess_65a0584cc8f94dd6）：模型 12 步全烧在反复
# web_search / web_fetch 上，最后停在"让我尝试从另一个来源获取更详细的航班信息"，
# 用户看到的是「没给答案就突然中断」。
# ---------------------------------------------------------------------------


class _AlwaysCallsLLM:
    """每一步都只要调工具 —— 用来把 MAX_STEPS 耗光。记录每次调用供断言。"""

    def __init__(self, preamble: str) -> None:
        self.preamble = preamble
        self.calls: list[dict] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        yield LLMText(self.preamble)
        yield LLMToolCall(id=f"c{len(self.calls)}", name="echo__echo", arguments="{}")


async def test_agent_mode_hitting_step_limit_does_not_answer_with_preamble(monkeypatch):
    """耗尽循环时不能把"让我再试一个来源"当过场话交给用户。

    2026-10-09 起收尾逻辑改为两级护栏：软警告（剩 2 步时注入收敛指令）+
    硬护栏（步数耗尽后追加一轮 tools=None 的收尾调用）。收尾调用若还在说
    过场话，仍退回 _LOOP_LIMIT_NOTE。
    """
    import agentd.kernel.modes.agent as agent_mod
    from agentd.kernel.modes.agent import MAX_STEPS, _WRAPUP_ASK

    monkeypatch.setattr(agent_mod, "McpHub", _FakeHub)

    llm = _AlwaysCallsLLM("让我尝试从另一个来源获取更详细的信息：")
    ctx = ModeContext(session_id="s", run_id="r", llm=llm, history=[Message.user("hi")])
    events = [e async for e in AgentMode().run(ctx, "hi")]

    assert len(llm.calls) == MAX_STEPS + 1, "循环烧满上限 + 1 次收尾调用"
    # 收尾调用不带工具 —— 模型到了这步只能给答复，不能再拖
    assert llm.calls[-1]["tools"] is None
    # 收尾调用收到了硬护栏指令
    assert any(
        m.role == "user" and m.content == _WRAPUP_ASK for m in llm.calls[-1]["messages"]
    )
    done = events[-1]
    assert isinstance(done, MessageDone)
    # 关键断言：不能是那句过场话
    assert done.text != "让我尝试从另一个来源获取更详细的信息："
    assert "让我尝试" not in done.text
    # 要说清发生了什么、并给出下一步
    assert str(MAX_STEPS) in done.text
    assert "继续" in done.text


async def test_agent_mode_budget_warning_injected_two_steps_before_limit(monkeypatch):
    """软护栏：剩 2 步时上下文里出现收敛指令，且只进上下文（不 yield 落库）。"""
    import agentd.kernel.modes.agent as agent_mod
    from agentd.kernel.modes.agent import MAX_STEPS, _WARN_REMAINING, _BUDGET_WARN

    monkeypatch.setattr(agent_mod, "McpHub", _FakeHub)

    llm = _AlwaysCallsLLM("让我换个来源：")
    ctx = ModeContext(session_id="s", run_id="r", llm=llm, history=[Message.user("hi")])
    events = [e async for e in AgentMode().run(ctx, "hi")]

    # 警告只注入一次：此后每次调用（含收尾调用）的上下文里都有它。
    # 首次出现 = 耗尽前第 _WARN_REMAINING 次调用（0 基下标 MAX_STEPS-_WARN_REMAINING）。
    with_warn = [i for i, c in enumerate(llm.calls)
                 if any(m.role == "user" and m.content == _BUDGET_WARN
                        for m in c["messages"])]
    assert with_warn == list(range(MAX_STEPS - _WARN_REMAINING, MAX_STEPS + 1)), (
        f"警告应从剩余 {_WARN_REMAINING} 步的那次调用起持续到收尾，实际：{with_warn}"
    )
    # 事件流里没有警告文本 —— 它不该被渲染给用户
    assert not any(
        getattr(e, "text", "") == _BUDGET_WARN for e in events
    )


async def test_agent_mode_wrapup_call_delivers_final_answer(monkeypatch):
    """硬护栏的价值：步数烧完后，收尾调用的答复就是本轮最终结论。"""
    import agentd.kernel.modes.agent as agent_mod
    from agentd.kernel.modes.agent import MAX_STEPS

    monkeypatch.setattr(agent_mod, "McpHub", _FakeHub)

    class _CallsThenWraps:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def stream_events(self, messages, *, system=None, tools=None):
            self.calls.append({"messages": list(messages), "tools": tools})
            # 收尾调用的标志是上下文里有硬护栏指令（本测试没有工具 schema，
            # 循环调用也是 tools=None，不能拿 tools 区分）
            from agentd.kernel.modes.agent import _WRAPUP_ASK

            if any(m.role == "user" and m.content == _WRAPUP_ASK for m in messages):
                # 收尾调用：模型基于已有信息给出真正结论
                yield LLMText("基于已抓取的数据，成都飞北京最低 2100 元，详见下表。")
                return
            yield LLMText("让我再抓一个来源：")
            yield LLMToolCall(id=f"c{len(self.calls)}", name="echo__echo", arguments="{}")

    llm = _CallsThenWraps()
    ctx = ModeContext(session_id="s", run_id="r", llm=llm, history=[Message.user("hi")])
    events = [e async for e in AgentMode().run(ctx, "hi")]

    assert len(llm.calls) == MAX_STEPS + 1
    assert llm.calls[-1]["tools"] is None
    done = events[-1]
    assert isinstance(done, MessageDone)
    assert done.text == "基于已抓取的数据，成都飞北京最低 2100 元，详见下表。"
    assert "上限" not in done.text and "继续」" not in done.text


async def test_agent_mode_keeps_real_conclusion_at_step_limit(monkeypatch):
    """最后一步已经写出结论（不像过场话）时必须原样保留，不能被兜底文案盖掉。"""
    import agentd.kernel.modes.agent as agent_mod
    from agentd.kernel.modes.agent import MAX_STEPS

    monkeypatch.setattr(agent_mod, "McpHub", _FakeHub)

    llm = _AlwaysCallsLLM("已为你整理好 3 个航班，详见上表。")
    ctx = ModeContext(session_id="s", run_id="r", llm=llm, history=[Message.user("hi")])
    events = [e async for e in AgentMode().run(ctx, "hi")]

    assert len(llm.calls) == MAX_STEPS + 1
    assert events[-1].text == "已为你整理好 3 个航班，详见上表。"


async def test_agent_mode_cancel_keeps_streamed_text(monkeypatch):
    """中途被叫停：保留已流出的正文，不能套上"到达步数上限"的话术。

    这两件事原因完全不同 —— 用户自己按的停，不该被告知"这一轮没跑完"。
    """
    import asyncio

    import agentd.kernel.modes.agent as agent_mod
    from agentd.kernel.modes.agent import MAX_STEPS

    monkeypatch.setattr(agent_mod, "McpHub", _FakeHub)

    cancel = asyncio.Event()

    class _StopMidStream:
        def __init__(self) -> None:
            self.calls = 0

        async def stream_events(self, messages, *, system=None, tools=None):
            self.calls += 1
            yield LLMText("让我")
            cancel.set()  # 用户在这一刻点了停止
            yield LLMToolCall(id=f"c{self.calls}", name="echo__echo", arguments="{}")

    llm = _StopMidStream()
    ctx = ModeContext(
        session_id="s", run_id="r", llm=llm, history=[Message.user("hi")], cancel=cancel
    )
    events = [e async for e in AgentMode().run(ctx, "hi")]

    assert llm.calls < MAX_STEPS, "叫停了就不该再烧到上限"
    done = events[-1]
    assert isinstance(done, MessageDone)
    assert done.text == "让我"
    assert "上限" not in done.text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("让我尝试从另一个来源获取更详细的航班信息：", True),
        ("我再试一个来源", True),
        ("接下来我会先读取文件", True),
        ("我准备换一个搜索词", True),
        ("下面我分三步来做：", True),
        ("已为你整理好 28 个航班，见下表。", False),
        ("", False),
        (None, False),
        # 写得长 → 判断为已在给结论，不能误判成过场话丢掉用户已有的结果
        ("让我说明一下：" + "详" * 200, False),
    ],
)
def test_looks_like_preamble_matrix(text, expected):
    from agentd.kernel.modes.agent import _looks_like_preamble

    assert _looks_like_preamble(text) is expected


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


async def test_mcp_hub_accepts_acp_pydantic_model_config():
    """回归：ACP SDK 在路由层已把 mcpServers 校验成 pydantic 模型（不是 dict）。

    早先 McpHub 只认 dict，`isinstance(raw, dict)` 为假就把整条静默跳过 ——
    现象是 agentd 日志说"本会话接入 1 个 MCP server"，却一个工具都列不出来，
    工具调用回"[错误] 未知工具"。这个用例把这个形态钉死。
    """
    from acp.schema import McpServerStdio

    model = McpServerStdio(
        name="echo", command=sys.executable, args=[_ECHO_SERVER], env=[]
    )
    assert not isinstance(model, dict)  # 确保测的确实是"非 dict"这条路径

    async with McpHub([model], cwd=str(Path(__file__).parent)) as hub:
        assert hub.has_tools, f"pydantic 形态没被识别；errors={hub.errors}"
        out = await hub.call("echo__echo", '{"text":"模型形态"}')
        assert out == "echo: 模型形态"


async def test_mcp_hub_skips_illegal_config_with_error():
    """既不是 dict 也不是模型的配置要被记下来（不能静默吞掉）。"""
    async with McpHub(["oops"]) as hub:
        assert not hub.has_tools
        assert hub.errors


def test_as_str_map_accepts_model_items():
    """env/headers 在线上是 EnvVariable 之类的模型，不是 dict。"""
    from acp.schema import EnvVariable

    from agentd.kernel.mcp import _as_str_map

    assert _as_str_map([EnvVariable(name="A", value="1")]) == {"A": "1"}
    assert _as_str_map([{"name": "B", "value": "2"}]) == {"B": "2"}
    assert _as_str_map({"C": "3"}) == {"C": "3"}


# ---------------------------------------------------------------------------
# 4) ScriptLLM：端到端验证用的脚本回放后端
# ---------------------------------------------------------------------------

async def test_script_llm_replays_tool_calls_then_text():
    from agentd.kernel.llm import ScriptLLM

    llm = ScriptLLM(
        [
            {"tool_calls": [{"name": "echo__echo", "arguments": {"text": "x"}}]},
            {"text": "完成"},
        ]
    )
    first = [e async for e in llm.stream_events([])]
    calls = [e for e in first if isinstance(e, LLMToolCall)]
    assert len(calls) == 1
    assert calls[0].name == "echo__echo"
    assert json.loads(calls[0].arguments) == {"text": "x"}  # 对象被转成 JSON 字符串

    second = [e async for e in llm.stream_events([])]
    assert "".join(e.text for e in second if isinstance(e, LLMText)) == "完成"


async def test_script_llm_repeats_last_step_when_exhausted():
    """脚本用完不抛异常 —— 免得掩盖真正的问题（比如模型不收敛）。"""
    from agentd.kernel.llm import ScriptLLM

    llm = ScriptLLM([{"text": "只有一步"}])
    for _ in range(3):
        events = [e async for e in llm.stream_events([])]
        assert "".join(e.text for e in events if isinstance(e, LLMText)) == "只有一步"


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
