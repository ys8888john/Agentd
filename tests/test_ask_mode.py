"""Ask（只读）模式测试。

锁的行为：

1. 内核注册 ask，``validate`` 认它，界面三档清单里有它；
2. Ask 档的原生工具箱 = 只读四件套 + 联网两件套，写 / 执行类一个都没有；
3. MCP 工具按 kind 过滤：没声明 ``read_only_hint`` 的一律不给模型看；
4. 执行层兜底：模型硬调一个写工具（幻觉出工具名）→ 回明确错误、
   **不走审批通道**、磁盘上什么都没发生；
5. Ask 档仍然能读完文件正常作答 —— 它不是"没有工具的 single"。

驱动方式同 test_plan_mode.py：剧本化假 LLM，不起进程、不靠 sleep 碰运气。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentd.contracts import MessageDone, ToolCallDone, ToolCallStart
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.models import Message
from agentd.kernel.modes.ask import AskMode
from agentd.kernel.modes.base import ModeContext
from agentd.kernel.store import InMemorySessionStore
from agentd.kernel.tools import TOOL_PROFILES

# Ask 档允许的原生工具：读 + 搜 + 抓。写类（write_file / edit / make_xlsx
# / generate_image）与 run_command 一律不在其中。
_ASK_TOOLS = frozenset(
    {"read_file", "list_dir", "glob", "grep", "web_search", "web_fetch"}
)
_WRITE_TOOLS = ("write_file", "edit", "run_command", "make_xlsx", "generate_image")


class ScriptedLLM(LLM):
    """按调用次序发剧本，并记录每次收到的 tools（过滤断言要用）。"""

    def __init__(self, script: list[list]) -> None:
        self._script = script
        self.calls: list[dict] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        idx = len(self.calls) - 1
        for event in self._script[idx] if idx < len(self._script) else []:
            yield event


def _kernel(script: list[list], **kwargs) -> AgentKernel:
    return AgentKernel(llm=ScriptedLLM(script), store=InMemorySessionStore(), **kwargs)


def _ctx(llm: LLM, **kwargs) -> ModeContext:
    return ModeContext(
        session_id="s",
        run_id="r",
        llm=llm,
        history=[Message.user("q")],
        **kwargs,
    )


class _MixedHub:
    """两个 MCP 工具：docs__search 声明了只读，db__write 没声明。

    MCP 工具没有 profile 概念，内核只能从 annotations 反推 kind
    （见 agent._mcp_kind），所以这两个正好代表"该留"和"该剔"两种情况。
    """

    def __init__(self, servers=None, *, cwd=None) -> None:
        self.servers = servers
        self.called: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def tool_schema(self):
        if not self.servers:
            return []
        return [
            {
                "type": "function",
                "function": {"name": "docs__search", "description": "", "parameters": {}},
            },
            {
                "type": "function",
                "function": {"name": "db__write", "description": "", "parameters": {}},
            },
        ]

    def binding(self, name):
        if name == "docs__search":
            return SimpleNamespace(tool="search", read_only=True)
        if name == "db__write":
            return SimpleNamespace(tool="write", read_only=False)
        return None

    async def call(self, name, arguments):
        self.called.append(name)
        return f"called {name}"


# ---------------------------------------------------------------------------
# 1) 注册
# ---------------------------------------------------------------------------


async def test_kernel_registers_ask_mode():
    kernel = _kernel([[LLMText("x")]])
    assert "ask" in kernel.modes()
    sid = await kernel.create_session()
    await kernel.validate(sid, "ask")  # 不抛即通过


def test_ask_is_one_of_the_three_ui_modes():
    from agentd.kernel.routing import UI_MODES

    assert "ask" in UI_MODES


# ---------------------------------------------------------------------------
# 2) 原生工具档位
# ---------------------------------------------------------------------------


def test_ask_profile_is_read_only_plus_web():
    assert set(TOOL_PROFILES["ask"]) == _ASK_TOOLS
    for bad in _WRITE_TOOLS:
        assert bad not in TOOL_PROFILES["ask"]


def test_tool_profile_for_only_overrides_ask():
    """只有 Ask 档脱离会话配置；其余沿用 AGENTD_TOOLS。"""
    kernel = _kernel([[LLMText("x")]], native_tools="native")
    assert kernel.tool_profile_for("ask") == "ask"
    assert kernel.tool_profile_for(" ASK ") == "ask"   # 归一化后再比
    assert kernel.tool_profile_for("agent") == "native"
    assert kernel.tool_profile_for("plan") == "native"
    assert kernel.tool_profile_for("single") == "native"


async def test_ask_mode_toolbox_has_no_write_tools(tmp_path):
    kernel = _kernel([[LLMText("x")]], native_tools="native")
    sid = await kernel.create_session(cwd=str(tmp_path))
    box = kernel._make_toolbox(str(tmp_path), profile=kernel.tool_profile_for("ask"))
    assert box is not None
    assert set(box.names) == _ASK_TOOLS


# ---------------------------------------------------------------------------
# 3) MCP 工具过滤（schema 层）
# ---------------------------------------------------------------------------


async def test_ask_mode_hides_non_read_only_mcp_tools(monkeypatch):
    import agentd.kernel.modes.agent as agent_mod

    monkeypatch.setattr(agent_mod, "McpHub", _MixedHub)

    llm = ScriptedLLM([[LLMText("只读作答")]])
    ctx = _ctx(llm, mcp_servers=[{"name": "docs"}])
    events = [e async for e in AskMode().run(ctx, "q")]

    names = [f["function"]["name"] for f in (llm.calls[0]["tools"] or [])]
    assert "docs__search" in names, "声明了只读的 MCP 工具应该还在"
    assert "db__write" not in names, "没声明只读的 MCP 工具不能出现在 Ask 档"
    assert isinstance(events[-1], MessageDone)


# ---------------------------------------------------------------------------
# 4) 执行层兜底（模型硬调）
# ---------------------------------------------------------------------------


async def test_ask_mode_blocks_a_write_call_the_model_invented(monkeypatch):
    """schema 里已经没有 db__write，这里防的是模型幻觉出工具名硬调。

    拦下后回一条明确错误，**不走审批通道** —— 弹窗再拒等于让模型白等一次
    往返，还会诱导它换个工具重试。所以测试里故意挂一个"批准一切"的回调，
    断言它一次都没被叫到。
    """
    import agentd.kernel.modes.agent as agent_mod

    monkeypatch.setattr(agent_mod, "McpHub", _MixedHub)

    asked: list[str] = []

    async def approve_all(req):
        asked.append(req.tool)
        return True

    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="db__write", arguments='{"sql":"drop table x"}')],
            [LLMText("我只能读，不能写，所以……")],
        ]
    )
    ctx = _ctx(llm, mcp_servers=[{"name": "docs"}], approve=approve_all)
    events = [e async for e in AskMode().run(ctx, "q")]

    assert asked == [], "只读档的拦截不该走审批通道"
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert len(done) == 1
    assert done[0].status == "cancelled"
    assert "只读" in done[0].output
    # 事件流仍然配对了 Start/Done，客户端的卡片能正常收掉
    assert any(isinstance(e, ToolCallStart) for e in events)
    # 回灌给模型的是"不许执行"，不是执行结果 —— 它才知道要换路子
    fed = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert fed and "只读" in (fed[0].content or "")
    assert "drop table" not in (fed[0].content or "")


async def test_ask_mode_refuses_native_write_and_touches_nothing(tmp_path):
    """端到端：Ask 档下模型硬调原生 write_file → 拒掉，磁盘上一个字节都没动。"""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    target = tmp_path / "b.txt"

    kernel = _kernel(
        [
            [
                LLMToolCall(
                    id="c1",
                    name="write_file",
                    arguments=json.dumps({"path": str(target), "content": "x"}),
                )
            ],
            [LLMText("a.txt 里是 hello；Ask 档下我不能写文件。")],
        ],
        native_tools="native",
    )
    sid = await kernel.create_session(cwd=str(tmp_path))
    events = [e async for e in kernel.handle(sid, "写个 b.txt", mode="ask")]

    assert not target.exists(), "Ask 档被硬调的写工具竟然落盘了"
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done and done[0].status == "cancelled" and "只读" in done[0].output


# ---------------------------------------------------------------------------
# 5) 只读档仍然能干活
# ---------------------------------------------------------------------------


async def test_ask_mode_still_reads_files_and_answers(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

    kernel = _kernel(
        [
            [LLMToolCall(id="c1", name="read_file", arguments=json.dumps({"path": "a.txt"}))],
            [LLMText("a.txt 的内容是 hello")],
        ],
        native_tools="native",
    )
    sid = await kernel.create_session(cwd=str(tmp_path))
    events = [e async for e in kernel.handle(sid, "读 a.txt", mode="ask")]

    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert len(done) == 1
    assert done[0].status == "completed"
    assert "hello" in done[0].output
    # kernel.handle() 末尾还会补一个 Done（stop_reason），所以 MessageDone
    # 不是最后一条 —— 用 dones[0] 断言而不是 events[-1]。
    dones = [e for e in events if isinstance(e, MessageDone)]
    assert len(dones) == 1 and "hello" in dones[0].text
