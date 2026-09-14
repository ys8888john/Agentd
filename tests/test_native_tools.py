"""原生工具层（kernel/tools.py）测试。

分四块：
1. 六个工具各自的行为（真文件系统，全在 tmp_path 里）；
2. 路径边界与 profile（能不能越出 cwd、read_only 能不能挡住写）；
3. needs_approval 的判定矩阵 + AgentMode 的审批/kind 端到端；
4. ACP 映射：_ACP_KIND 必须覆盖 kernel 产出的每一个 ToolKind（漏一个 = 客户端静默卡死）。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from agentd.contracts import ToolCallDone, ToolCallStart, ToolCallStart as _TCS
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.modes.agent import AgentMode
from agentd.kernel.modes.base import ModeContext
from agentd.kernel.models import Message
from agentd.kernel.tools import (
    ALL_TOOL_NAMES,
    ApprovalRequest,
    NativeToolbox,
    TOOL_PROFILES,
    needs_approval,
)
from agentd.transports.acp_stdio import _ACP_KIND

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def box(tmp_path: Path, **kw) -> NativeToolbox:
    """默认把 root 钉在 tmp_path 上 —— 测试绝不允许碰到仓库里的真文件。"""
    kw.setdefault("cwd", tmp_path)
    return NativeToolbox(**kw)


# ---------------------------------------------------------------------------
# 1) 六个工具
# ---------------------------------------------------------------------------


async def test_read_file_numbers_lines_and_respects_offset(tmp_path):
    (tmp_path / "a.txt").write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
    tb = box(tmp_path)

    out = await tb.call("read_file", json.dumps({"path": "a.txt"}))
    assert "共 4 行" in out
    assert "     1\tl1" in out
    assert "     4\tl4" in out

    out = await tb.call("read_file", json.dumps({"path": "a.txt", "offset": 3, "limit": 1}))
    assert "第 3-3 行" in out
    assert "l3" in out
    assert "l4" not in out


async def test_read_file_errors_are_prefixed(tmp_path):
    tb = box(tmp_path)
    assert (await tb.call("read_file", json.dumps({"path": "nope.txt"}))).startswith("[错误]")
    (tmp_path / "d").mkdir()
    assert (await tb.call("read_file", json.dumps({"path": "d"}))).startswith("[错误]")
    assert (await tb.call("read_file", "{}")).startswith("[错误]")


async def test_glob_lists_relative_paths_newest_first(tmp_path):
    (tmp_path / "src").mkdir()
    old = tmp_path / "src" / "old.py"
    old.write_text("", encoding="utf-8")
    new = tmp_path / "src" / "new.py"
    new.write_text("", encoding="utf-8")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    out = await box(tmp_path).call("glob", json.dumps({"pattern": "**/*.py"}))
    assert "匹配 2 个文件" in out
    lines = [ln for ln in out.splitlines() if ln.endswith(".py")]
    assert lines == ["src/new.py", "src/old.py"]  # 新的在前，且是相对路径


async def test_glob_rejects_absolute_pattern(tmp_path):
    """Path.glob 不吃绝对 pattern，报的是 NotImplementedError —— 要接住并说人话。"""
    out = await box(tmp_path).call("glob", json.dumps({"pattern": str(tmp_path / "*")}))
    assert out.startswith("[错误]")
    assert "绝对路径" in out


async def test_glob_no_match_is_not_an_error(tmp_path):
    out = await box(tmp_path).call("glob", json.dumps({"pattern": "*.rs"}))
    assert not out.startswith("[错误]")
    assert "没有匹配" in out


async def test_grep_returns_file_line_and_content(tmp_path):
    (tmp_path / "m.py").write_text("import os\n\ndef handle_x():\n    pass\n", encoding="utf-8")
    out = await box(tmp_path).call("grep", json.dumps({"pattern": r"def handle_\w+"}))
    assert "m.py:3: def handle_x():" in out


async def test_grep_skips_noise_dirs_and_honors_glob(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hit.py").write_text("target\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "hit.py").write_text("target\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("target\n", encoding="utf-8")
    (tmp_path / "note.md").write_text("target\n", encoding="utf-8")

    tb = box(tmp_path)
    out = await tb.call("grep", json.dumps({"pattern": "target"}))
    assert "keep.py" in out
    assert "note.md" in out
    assert ".git" not in out and "node_modules" not in out

    only_py = await tb.call("grep", json.dumps({"pattern": "target", "glob": "*.py"}))
    assert "keep.py" in only_py
    assert "note.md" not in only_py


async def test_grep_bad_regex_and_missing_pattern(tmp_path):
    tb = box(tmp_path)
    assert "正则表达式非法" in await tb.call("grep", json.dumps({"pattern": "([unclosed"}))
    assert (await tb.call("grep", "{}")).startswith("[错误]")


async def test_grep_ignore_case(tmp_path):
    (tmp_path / "f.txt").write_text("Hello\n", encoding="utf-8")
    tb = box(tmp_path)
    assert "没有匹配" in await tb.call("grep", json.dumps({"pattern": "hello"}))
    assert "f.txt" in await tb.call("grep", json.dumps({"pattern": "hello", "ignore_case": True}))


async def test_write_file_creates_parents_and_reports_action(tmp_path):
    tb = box(tmp_path)
    out = await tb.call("write_file", json.dumps({"path": "deep/nest/x.txt", "content": "a\nb"}))
    assert "新建" in out and "2 行" in out
    assert (tmp_path / "deep" / "nest" / "x.txt").read_text(encoding="utf-8") == "a\nb"

    out = await tb.call("write_file", json.dumps({"path": "deep/nest/x.txt", "content": "z"}))
    assert "覆盖" in out


async def test_write_file_requires_content_explicitly(tmp_path):
    """空字符串是合法内容，缺字段不是 —— 两者要区分开，否则会静默清空文件。"""
    tb = box(tmp_path)
    assert (await tb.call("write_file", json.dumps({"path": "x.txt"}))).startswith("[错误]")
    out = await tb.call("write_file", json.dumps({"path": "x.txt", "content": ""}))
    assert "新建" in out
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == ""


async def test_edit_replaces_unique_match(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("def f():\n    return 1\n", encoding="utf-8")
    out = await box(tmp_path).call(
        "edit", json.dumps({"path": "c.py", "old_string": "return 1", "new_string": "return 2"})
    )
    assert "替换 1 处" in out
    assert f.read_text(encoding="utf-8") == "def f():\n    return 2\n"


async def test_edit_refuses_ambiguous_match_then_replace_all(tmp_path):
    f = tmp_path / "c.py"
    f.write_text("x = 1\ny = 1\n", encoding="utf-8")
    tb = box(tmp_path)

    out = await tb.call("edit", json.dumps({"path": "c.py", "old_string": "= 1", "new_string": "= 2"}))
    assert out.startswith("[错误]") and "2 处" in out
    assert f.read_text(encoding="utf-8") == "x = 1\ny = 1\n"  # 被拒时绝不能动文件

    out = await tb.call(
        "edit",
        json.dumps({"path": "c.py", "old_string": "= 1", "new_string": "= 2", "replace_all": True}),
    )
    assert "替换 2 处" in out
    assert f.read_text(encoding="utf-8") == "x = 2\ny = 2\n"


async def test_edit_reports_not_found_and_missing_file(tmp_path):
    (tmp_path / "c.py").write_text("abc\n", encoding="utf-8")
    tb = box(tmp_path)
    assert "找不到 old_string" in await tb.call(
        "edit", json.dumps({"path": "c.py", "old_string": "zzz", "new_string": "1"})
    )
    assert (await tb.call(
        "edit", json.dumps({"path": "no.py", "old_string": "a", "new_string": "b"})
    )).startswith("[错误]")
    # 空 old_string 会把每个字符之间都插一遍，必须拒
    assert (await tb.call(
        "edit", json.dumps({"path": "c.py", "old_string": "", "new_string": "b"})
    )).startswith("[错误]")


async def test_run_command_captures_stdout_and_stderr(tmp_path):
    script = tmp_path / "emit.py"
    script.write_text(
        "import sys\nprint('OUT')\nprint('ERR', file=sys.stderr)\n", encoding="utf-8"
    )
    tb = box(tmp_path)
    out = await tb.call(
        "run_command",
        json.dumps({"command": f'"{sys.executable}" "{script}"'}),
    )
    assert out.startswith("exit=0")
    assert "OUT" in out and "[stderr]" in out and "ERR" in out


async def test_run_command_nonzero_exit_is_reported_as_failure(tmp_path):
    tb = box(tmp_path)
    out = await tb.call(
        "run_command", json.dumps({"command": f'"{sys.executable}" -c "raise SystemExit(3)"'})
    )
    assert out.startswith("[错误]")
    assert "退出码 3" in out


async def test_run_command_timeout_kills_and_reports(tmp_path):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    tb = box(tmp_path, timeout=0.6)
    out = await tb.call("run_command", json.dumps({"command": f'"{sys.executable}" "{script}"'}))
    assert out.startswith("[错误]") and "超时" in out


async def test_run_command_uses_cwd_relative_to_session_root(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "marker.txt").write_text("m", encoding="utf-8")
    script = tmp_path / "ls.py"
    script.write_text("import os\nprint(sorted(os.listdir('.')))\n", encoding="utf-8")
    tb = box(tmp_path)
    out = await tb.call(
        "run_command", json.dumps({"command": f'"{sys.executable}" "{script}"', "cwd": "sub"})
    )
    assert "marker.txt" in out


# ---------------------------------------------------------------------------
# 2) 边界与 profile
# ---------------------------------------------------------------------------


async def test_path_escape_outside_root_is_rejected(tmp_path):
    """`..` 必须被 resolve 展开后再判，否则 ../ 就成了走后门的通道。"""
    root = tmp_path / "work"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")

    tb = NativeToolbox(cwd=root)
    out = await tb.call("read_file", json.dumps({"path": "../secret.txt"}))
    assert out.startswith("[错误]") and "越界" in out
    assert "top secret" not in out

    # 写同样要拦
    out = await tb.call("write_file", json.dumps({"path": "../evil.txt", "content": "x"}))
    assert out.startswith("[错误]")
    assert not (tmp_path / "evil.txt").exists()


async def test_allow_outside_opens_the_gate(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (tmp_path / "outer.txt").write_text("ok", encoding="utf-8")
    tb = NativeToolbox(cwd=root, allow_outside=True)
    out = await tb.call("read_file", json.dumps({"path": "../outer.txt"}))
    assert "ok" in out


async def test_profiles_control_which_tools_exist(tmp_path):
    assert NativeToolbox(cwd=tmp_path, profile="native").names == list(ALL_TOOL_NAMES)
    assert NativeToolbox(cwd=tmp_path, profile="read_only").names == [
        "read_file",
        "glob",
        "grep",
    ]
    assert NativeToolbox(cwd=tmp_path, profile="off").names == []
    assert not NativeToolbox(cwd=tmp_path, profile="off").has_tools
    assert TOOL_PROFILES["off"] == () and TOOL_PROFILES["none"] == ()


async def test_disabled_tool_is_unknown(tmp_path):
    tb = NativeToolbox(cwd=tmp_path, profile="read_only")
    out = await tb.call("run_command", json.dumps({"command": "echo hi"}))
    assert out.startswith("[错误]") and "未知工具" in out


async def test_bad_arguments_are_reported_not_raised(tmp_path):
    tb = box(tmp_path)
    assert "不是合法 JSON" in await tb.call("read_file", "{oops")
    assert "必须是 JSON 对象" in await tb.call("read_file", "[1,2]")
    assert (await tb.call("nope", "{}")).startswith("[错误]")


async def test_output_is_truncated(tmp_path):
    (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")
    tb = box(tmp_path, max_bytes=500)
    out = await tb.call("read_file", json.dumps({"path": "big.txt"}))
    assert "已截断" in out
    assert len(out) < 900


async def test_tool_schema_shape_matches_openai(tmp_path):
    schema = box(tmp_path).tool_schema()
    assert {t["function"]["name"] for t in schema} == set(ALL_TOOL_NAMES)
    for item in schema:
        assert item["type"] == "function"
        fn = item["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


# ---------------------------------------------------------------------------
# 3) 审批判定
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy,kind,requires,destructive,expected",
    [
        # 只读动作在任何策略下都不弹 —— 每个 ls 都弹窗，用户三分钟就学会无脑点允许
        ("native", "read", False, False, False),
        ("native", "search", False, False, False),
        ("all", "read", False, False, False),
        # 原生写/执行类
        ("native", "edit", True, False, True),
        ("native", "execute", True, False, True),
        # MCP 工具没声明 requires，只有 destructiveHint 能拦
        ("native", "execute", False, False, False),
        ("native", "execute", False, True, True),
        ("all", "execute", False, False, True),
        ("all", "edit", False, False, True),
        # none 是无人值守档，全放
        ("none", "execute", True, True, False),
        ("none", "edit", True, False, False),
        # 拼错的策略退回默认（native），不能因为拼错就全放行
        ("nonsense", "edit", True, False, True),
    ],
)
def test_needs_approval_matrix(policy, kind, requires, destructive, expected):
    assert (
        needs_approval(requires=requires, kind=kind, destructive=destructive, policy=policy)
        is expected
    )


# ---------------------------------------------------------------------------
# 4) AgentMode：kind 映射 + 审批 + 与 MCP 共存
# ---------------------------------------------------------------------------


class ScriptedLLM(LLM):
    """按轮产出事件；记录每次收到的 tools 便于断言。"""

    def __init__(self, rounds: list[list]) -> None:
        self._rounds = list(rounds)
        self.seen_tools: list[list[dict] | None] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.seen_tools.append(tools)
        round_events = self._rounds.pop(0) if self._rounds else [LLMText("（脚本耗尽）")]
        for event in round_events:
            yield event


def _ctx(tmp_path, llm, **kw) -> ModeContext:
    kw.setdefault("toolbox", NativeToolbox(cwd=tmp_path))
    return ModeContext(
        session_id="s", run_id="r", llm=llm, history=[Message.user("hi")], cwd=str(tmp_path), **kw
    )


async def test_agent_mode_maps_kind_per_tool(tmp_path):
    """kind 不能再写死 execute：读/搜/写/执行各是各的图标。"""
    (tmp_path / "a.txt").write_text("hi\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            [
                LLMToolCall(id="c1", name="read_file", arguments='{"path":"a.txt"}'),
                LLMToolCall(id="c2", name="glob", arguments='{"pattern":"*.txt"}'),
                LLMToolCall(
                    id="c3",
                    name="write_file",
                    arguments='{"path":"b.txt","content":"x"}',
                ),
            ],
            [LLMText("好了")],
        ]
    )
    events = [e async for e in AgentMode().run(_ctx(tmp_path, llm), "hi")]

    kinds = {e.call_id: e.kind for e in events if isinstance(e, ToolCallStart)}
    assert kinds == {"c1": "read", "c2": "search", "c3": "edit"}
    done = {e.call_id: e.status for e in events if isinstance(e, ToolCallDone)}
    assert done == {"c1": "completed", "c2": "completed", "c3": "completed"}


async def test_agent_mode_merges_native_and_mcp_schema(tmp_path):
    """两条工具来源都要进 tools 数组，且原生排在前面。"""
    llm = ScriptedLLM([[LLMText("ok")]])

    class _Hub:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def tool_schema(self):
            return [
                {"type": "function", "function": {"name": "srv__t", "description": "", "parameters": {}}}
            ]

        def binding(self, name):
            return None

        async def call(self, name, args):
            return ""

    import agentd.kernel.modes.agent as agent_mod

    original = agent_mod.McpHub
    agent_mod.McpHub = _Hub
    try:
        [e async for e in AgentMode().run(_ctx(tmp_path, llm), "hi")]
    finally:
        agent_mod.McpHub = original

    names = [t["function"]["name"] for t in llm.seen_tools[0]]
    assert names[: len(ALL_TOOL_NAMES)] == list(ALL_TOOL_NAMES)
    assert names[-1] == "srv__t"


async def test_agent_mode_asks_approval_and_refuses_when_denied(tmp_path):
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="write_file", arguments='{"path":"x.txt","content":"boom"}')],
            [LLMText("好")],
        ]
    )
    seen: list[ApprovalRequest] = []

    async def deny(req: ApprovalRequest) -> bool:
        seen.append(req)
        return False

    events = [
        e
        async for e in AgentMode().run(_ctx(tmp_path, llm, approve=deny), "hi")
    ]

    assert [r.tool for r in seen] == ["write_file"]
    assert seen[0].kind == "edit"
    assert "x.txt" in seen[0].detail  # 弹窗摘要来自参数
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done[0].status == "cancelled"
    assert not (tmp_path / "x.txt").exists()  # 被拒就绝不能落盘


async def test_agent_mode_executes_when_approved(tmp_path):
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="write_file", arguments='{"path":"x.txt","content":"ok"}')],
            [LLMText("好")],
        ]
    )
    asked: list[str] = []

    async def allow(req: ApprovalRequest) -> bool:
        asked.append(req.tool)
        return True

    [e async for e in AgentMode().run(_ctx(tmp_path, llm, approve=allow), "hi")]
    assert asked == ["write_file"]
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "ok"


async def test_agent_mode_does_not_ask_for_read_only_tools(tmp_path):
    (tmp_path / "a.txt").write_text("hi\n", encoding="utf-8")
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="read_file", arguments='{"path":"a.txt"}')],
            [LLMText("好")],
        ]
    )

    async def boom(req: ApprovalRequest) -> bool:
        raise AssertionError("只读工具不该触发审批")

    [e async for e in AgentMode().run(_ctx(tmp_path, llm, approve=boom), "hi")]


async def test_agent_mode_tool_failure_marks_status_failed(tmp_path):
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="read_file", arguments='{"path":"missing.txt"}')],
            [LLMText("文件不存在")],
        ]
    )
    events = [e async for e in AgentMode().run(_ctx(tmp_path, llm), "hi")]
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done[0].status == "failed"
    assert done[0].output.startswith("[错误]")


async def test_agent_mode_unknown_tool_still_closes_the_card(tmp_path):
    """模型幻觉出一个工具名时，start/done 必须配对发出去，否则客户端卡片永远转圈。"""
    llm = ScriptedLLM(
        [[LLMToolCall(id="c1", name="totally_made_up", arguments="{}")], [LLMText("好")]]
    )
    events = [e async for e in AgentMode().run(_ctx(tmp_path, llm), "hi")]
    start = [e for e in events if isinstance(e, ToolCallStart)]
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert len(start) == len(done) == 1
    assert start[0].kind == "other"
    assert done[0].status == "failed" and "未知工具" in done[0].output


async def test_approval_exception_is_treated_as_refusal(tmp_path):
    """审批通道坏了必须往"拒绝"倒 —— 反过来就是安全漏洞。"""
    llm = ScriptedLLM(
        [[LLMToolCall(id="c1", name="write_file", arguments='{"path":"x.txt","content":"a"}')],
         [LLMText("好")]]
    )

    async def broken(req: ApprovalRequest) -> bool:
        raise RuntimeError("客户端炸了")

    events = [e async for e in AgentMode().run(_ctx(tmp_path, llm, approve=broken), "hi")]
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert done[0].status == "cancelled"
    assert not (tmp_path / "x.txt").exists()


# ---------------------------------------------------------------------------
# 5) 内核接线
# ---------------------------------------------------------------------------


async def test_kernel_enables_native_tools_by_default(tmp_path):
    llm = ScriptedLLM([[LLMText("好")]])
    kernel = AgentKernel(llm=llm)
    session_id = await kernel.create_session(cwd=str(tmp_path))

    [e async for e in kernel.handle(session_id, "hi", mode="agent")]

    names = [t["function"]["name"] for t in llm.seen_tools[0]]
    assert set(ALL_TOOL_NAMES) <= set(names)


async def test_kernel_can_turn_native_tools_off(tmp_path):
    llm = ScriptedLLM([[LLMText("好")]])
    kernel = AgentKernel(llm=llm, native_tools="off")
    session_id = await kernel.create_session(cwd=str(tmp_path))

    [e async for e in kernel.handle(session_id, "hi", mode="agent")]
    assert llm.seen_tools[0] is None  # 无工具 ⇒ 与 single 等价


async def test_kernel_read_only_profile(tmp_path):
    llm = ScriptedLLM([[LLMText("好")]])
    kernel = AgentKernel(llm=llm, native_tools="read_only")
    session_id = await kernel.create_session(cwd=str(tmp_path))

    [e async for e in kernel.handle(session_id, "hi", mode="agent")]
    names = {t["function"]["name"] for t in llm.seen_tools[0]}
    assert names == {"read_file", "glob", "grep"}


async def test_kernel_tool_runs_against_session_cwd(tmp_path):
    (tmp_path / "only-here.txt").write_text("x", encoding="utf-8")
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="glob", arguments='{"pattern":"*.txt"}')],
            [LLMText("好")],
        ]
    )
    kernel = AgentKernel(llm=llm)
    session_id = await kernel.create_session(cwd=str(tmp_path))

    events = [e async for e in kernel.handle(session_id, "hi", mode="agent")]
    done = [e for e in events if isinstance(e, ToolCallDone)]
    assert "only-here.txt" in done[0].output

    # 工具往返是"本轮内的临时消息"，不落库：库里只有 user + 最终 assistant
    history = await kernel.history(session_id)
    assert [m.role for m in history] == ["user", "assistant"]


async def test_kernel_approve_callback_is_used(tmp_path):
    llm = ScriptedLLM(
        [
            [LLMToolCall(id="c1", name="run_command", arguments='{"command":"echo hi"}')],
            [LLMText("好")],
        ]
    )
    kernel = AgentKernel(llm=llm)
    session_id = await kernel.create_session(cwd=str(tmp_path))
    asked: list[str] = []

    async def approve(req: ApprovalRequest) -> bool:
        asked.append(req.tool)
        return True

    [e async for e in kernel.handle(session_id, "hi", mode="agent", approve=approve)]
    assert asked == ["run_command"]


async def test_kernel_unknown_tools_profile_falls_back_to_native(tmp_path, capsys):
    llm = ScriptedLLM([[LLMText("好")]])
    kernel = AgentKernel(llm=llm, native_tools="typo")
    session_id = await kernel.create_session(cwd=str(tmp_path))

    [e async for e in kernel.handle(session_id, "hi", mode="agent")]
    names = {t["function"]["name"] for t in llm.seen_tools[0]}
    assert names == set(ALL_TOOL_NAMES)  # 拼错就退回 native，不能变成"没有工具"


# ---------------------------------------------------------------------------
# 6) ACP 映射：kind 差集与审批翻译
# ---------------------------------------------------------------------------


def test_acp_kind_covers_every_kernel_toolkind():
    """回归：kernel 产出的 kind 必须全部有 ACP 译法。

    漏映射的后果不是报错，而是 **pydantic 校验失败后被客户端静默吞掉** ——
    现象是工具卡片永远停在"运行中"，而且两边日志都干干净净。
    所以这里直接从 ToolCallStart 的 Literal 里取全部取值来比，而不是手写一份清单：
    以后往 contracts 加了新 kind，这个测试会立刻红。
    """
    import typing

    allowed = set(typing.get_args(_TCS.model_fields["kind"].annotation))
    assert allowed, "取不到 ToolKind 取值，说明 contracts 里改了声明方式"
    assert allowed <= set(_ACP_KIND), f"这些 kind 没有 ACP 译法：{allowed - set(_ACP_KIND)}"
    assert set(_ACP_KIND.values()) <= {
        "read",
        "edit",
        "delete",
        "move",
        "search",
        "execute",
        "think",
        "fetch",
        "switch_mode",
        "other",
    }


class _FakeConn:
    """记录 session_update 与 request_permission 的假连接。"""

    def __init__(self, outcome: str = "selected", option_id: str = "allow_once") -> None:
        from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

        self.updates: list[tuple[str, object]] = []
        self.permission_calls: list[dict] = []
        # 注意 outcome 是**必填**的 Literal（没有默认值），构造时必须显式给 ——
        # 这和客户端回传的 JSON 形状一致（{"outcome":"selected","optionId":"..."}）。
        self._response = RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=option_id)
            if outcome == "selected"
            else DeniedOutcome(outcome="cancelled")
        )

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))

    async def request_permission(self, session_id, tool_call, options):
        self.permission_calls.append(
            {"session_id": session_id, "tool_call": tool_call, "options": options}
        )
        return self._response


def _agent_with(conn, tmp_path, script: list[dict]):
    from agentd.boot import Settings, build_kernel, build_store  # noqa: F401
    from agentd.kernel.llm import ScriptLLM
    from agentd.transports.acp_stdio import AgentdAcpAgent

    kernel = AgentKernel(llm=ScriptLLM(script))
    agent = AgentdAcpAgent(kernel)
    agent._conn = conn  # noqa: SLF001 - 这是传输层自己的字段，测试直接注入
    return agent, kernel


async def test_acp_request_permission_allow_once(tmp_path):
    conn = _FakeConn(outcome="selected", option_id="allow_once")
    agent, _ = _agent_with(conn, tmp_path, [{"text": "x"}])
    approve = agent._make_approver("sess_1")  # noqa: SLF001

    ok = await approve(
        ApprovalRequest(call_id="c1", tool="run_command", title="run_command", kind="execute", detail="command: ls")
    )
    assert ok is True
    call = conn.permission_calls[0]
    assert call["session_id"] == "sess_1"
    assert call["tool_call"].kind == "execute"
    assert call["tool_call"].raw_input == {"detail": "command: ls"}
    # option_id 是我们自己定义的字符串，客户端只原样回传
    assert {o.option_id for o in call["options"]} == {"allow_once", "allow_session", "reject"}


async def test_acp_request_permission_cancelled_means_denied(tmp_path):
    conn = _FakeConn(outcome="cancelled")
    agent, _ = _agent_with(conn, tmp_path, [{"text": "x"}])
    approve = agent._make_approver("sess_1")  # noqa: SLF001

    assert await approve(ApprovalRequest(call_id="c1", tool="edit", title="edit", kind="edit")) is False


async def test_acp_allow_session_remembers_and_stops_asking(tmp_path):
    """选了"本会话总是允许"之后不能每次都弹 —— 审批疲劳比没有审批更危险。"""
    conn = _FakeConn(outcome="selected", option_id="allow_session")
    agent, _ = _agent_with(conn, tmp_path, [{"text": "x"}])
    approve = agent._make_approver("sess_1")  # noqa: SLF001
    req = ApprovalRequest(call_id="c1", tool="write_file", title="write_file", kind="edit")

    assert await approve(req) is True
    assert await approve(req) is True
    assert len(conn.permission_calls) == 1  # 只问了一次

    # 另一个会话不继承这个记忆
    other = agent._make_approver("sess_2")  # noqa: SLF001
    assert await other(req) is True
    assert len(conn.permission_calls) == 2


async def test_acp_permission_failure_is_denied(tmp_path):
    """客户端没实现 session/request_permission 时不能放行。"""
    conn = _FakeConn()
    conn.request_permission = _raise  # type: ignore[assignment]
    agent, _ = _agent_with(conn, tmp_path, [{"text": "x"}])
    approve = agent._make_approver("sess_1")  # noqa: SLF001

    assert await approve(ApprovalRequest(call_id="c1", tool="edit", title="edit", kind="edit")) is False


async def _raise(*_a, **_kw):
    raise RuntimeError("client does not support session/request_permission")


async def test_acp_prompt_end_to_end_with_native_tool(tmp_path):
    """走真的 prompt()：工具 start 通知的 kind 必须是 ACP 合法值，且 stdout 干净。

    这里直接用假 conn，不起子进程 —— 子进程版本在 test_acp.py 里用 script 后端覆盖。
    """
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    conn = _FakeConn()
    agent, _ = _agent_with(
        conn,
        tmp_path,
        [
            {"tool_calls": [{"name": "glob", "arguments": {"pattern": "*.txt"}}]},
            {"text": "找到了"},
        ],
    )
    new_session = await agent.new_session(cwd=str(tmp_path))
    session_id = new_session.session_id  # new_session 返回的是响应模型，不是裸 id

    from acp.schema import TextContentBlock

    resp = await agent.prompt(session_id, [TextContentBlock(type="text", text="列出 txt")])
    assert resp.stop_reason == "end_turn"

    kinds = [getattr(u, "kind", None) for _, u in conn.updates]
    assert "search" in kinds  # glob ⇒ search，而不是以前写死的 execute
    assert all(k is None or k in _ACP_KIND.values() for k in kinds)

    outputs = [getattr(u, "raw_output", None) for _, u in conn.updates]
    assert any(o and "hello.txt" in o for o in outputs)
