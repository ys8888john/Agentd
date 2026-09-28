"""session/load 恢复 + 工具 diff 输出测试。

两块：
1. **ACP session/load**（本仓库对标准 ACP 客户端的兼容面）：
   - new_session 声明可用模式（single/agent），current 默认 agent；
   - load_session 给"重启后"的旧会话重绑 cwd/mcpServers —— 用相对路径的
     工具调用命中恢复后的 cwd 来证明绑定真的生效；
   - set_session_mode 后 modes.current_mode_id 跟着变；
   - 不存在的会话 load 抛 ValueError（SDK 转 JSON-RPC error）。
2. **write_file / edit 的 diff 输出**：行首标记 ---/+++/@@/-/+ 是前端 diff
   着色的解析契约（见 GUI 的 tool-out render），这里锁住输出形状。
"""

from __future__ import annotations

import json

from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.store import InMemorySessionStore
from agentd.kernel.tools import NativeToolbox
from agentd.transports.acp_stdio import AgentdAcpAgent


# ---------------------------------------------------------------------------
# 1) session/load 与模式声明
# ---------------------------------------------------------------------------


class PlayLLM(LLM):
    def __init__(self, scripts: list[list]) -> None:
        self._scripts = list(scripts)

    async def stream_events(self, messages, *, system=None, tools=None):
        for event in (self._scripts.pop(0) if self._scripts else [LLMText("（脚本耗尽）")]):
            yield event


def _agent(kernel: AgentKernel) -> AgentdAcpAgent:
    agent = AgentdAcpAgent(kernel)
    agent.on_connect(object())  # new_session/load_session 不用 conn，塞个占位
    return agent


async def test_new_session_declares_available_modes(tmp_path) -> None:
    agent = _agent(AgentKernel(llm=PlayLLM([]), store=InMemorySessionStore()))

    resp = await agent.new_session(cwd=str(tmp_path))

    assert resp.session_id.startswith("sess_")
    modes = resp.modes
    assert modes is not None
    assert [m.id for m in modes.available_modes] == ["agent", "single"]
    assert modes.current_mode_id == "agent"
    # 只声明存在的模式，且带人话描述
    by_id = {m.id: m for m in modes.available_modes}
    assert by_id["agent"].name and by_id["single"].description


async def test_load_session_rebinds_cwd_for_tools(tmp_path) -> None:
    """重启后的会话 load：历史在库里，cwd 要重新绑 —— 相对路径工具必须命中它。"""
    dir_a = tmp_path / "project-a"
    dir_a.mkdir()
    (dir_a / "note.txt").write_text("A 的内容", encoding="utf-8")

    kernel = AgentKernel(
        llm=PlayLLM(
            [
                [LLMToolCall(id="c1", name="glob", arguments='{"pattern":"note.txt"}')],
                [LLMText("找到了")],
                [LLMToolCall(id="c2", name="glob", arguments='{"pattern":"note.txt"}')],
                [LLMText("又找到了")],
            ]
        ),
        store=InMemorySessionStore(),
    )
    sid = await kernel.create_session(cwd=str(dir_a))
    async for _ in kernel.handle(sid, "第一轮", mode="agent"):
        pass

    agent = _agent(kernel)
    resp = await agent.load_session(sid, cwd=str(dir_a), mcp_servers=[])
    assert resp.modes is not None and resp.modes.current_mode_id == "agent"

    # 恢复后的会话跑第二轮：glob 用相对路径，只有绑定对了 cwd 才能命中
    events = [e async for e in kernel.handle(sid, "第二轮找文件", mode="agent")]
    done = [e for e in events if type(e).__name__ == "ToolCallDone"]
    assert done and "note.txt" in done[0].output


async def test_load_session_reflects_mode_overrides(tmp_path) -> None:
    kernel = AgentKernel(llm=PlayLLM([]), store=InMemorySessionStore())
    sid = await kernel.create_session(cwd=str(tmp_path))
    agent = _agent(kernel)
    await agent.set_session_mode(sid, "single")

    resp = await agent.load_session(sid, cwd=str(tmp_path))
    assert resp.modes.current_mode_id == "single"


async def test_load_unknown_session_raises(tmp_path) -> None:
    agent = _agent(AgentKernel(llm=PlayLLM([]), store=InMemorySessionStore()))
    try:
        await agent.load_session("sess_missing", cwd=str(tmp_path))
    except ValueError:
        pass  # SDK 会把它转成 JSON-RPC error —— 这是要的行为
    else:
        raise AssertionError("不存在的会话应该抛错，而不是悄悄新建")


class _RecordConn:
    """带记录的 conn 占位：验证 set_session_mode 的标准通知广播。"""

    def __init__(self) -> None:
        self.updates: list = []

    async def session_update(self, session_id, update) -> None:
        self.updates.append((session_id, update))


async def test_set_mode_broadcasts_current_mode_update(tmp_path) -> None:
    """切模式随发一条标准 current_mode_update —— 多端同步的协议闭环。"""
    kernel = AgentKernel(llm=PlayLLM([]), store=InMemorySessionStore())
    sid = await kernel.create_session(cwd=str(tmp_path))
    agent = AgentdAcpAgent(kernel)
    conn = _RecordConn()
    agent.on_connect(conn)

    await agent.set_session_mode(sid, "single")

    assert any(
        getattr(u, "session_update", "") == "current_mode_update"
        and getattr(u, "current_mode_id", None) == "single"
        for _, u in conn.updates
    )


# ---------------------------------------------------------------------------
# 2) 工具输出的 diff 块
# ---------------------------------------------------------------------------


def _tb(tmp_path) -> NativeToolbox:
    return NativeToolbox(cwd=tmp_path)


async def test_write_new_file_output_contains_added_lines(tmp_path) -> None:
    out = await _tb(tmp_path).call(
        "write_file", json.dumps({"path": "a.txt", "content": "第1行\n第2行\n"})
    )
    assert out.startswith("已新建")
    assert any(line.startswith("+") for line in out.splitlines())
    assert "+++ " in out


async def test_write_over_file_output_has_minus_and_plus(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("旧内容\n", encoding="utf-8")
    out = await _tb(tmp_path).call(
        "write_file", json.dumps({"path": "a.txt", "content": "新内容\n"})
    )
    assert out.startswith("已覆盖")
    lines = out.splitlines()
    assert any(line.startswith("-") for line in lines)
    assert any(line.startswith("+") for line in lines)
    assert "旧内容" in out and "新内容" in out


async def test_edit_output_shows_replaced_hunk(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("第一行\n第二行\n第三行\n", encoding="utf-8")
    out = await _tb(tmp_path).call(
        "edit",
        json.dumps({"path": "a.txt", "old_string": "第二行", "new_string": "改过的第二行"}),
    )
    assert out.startswith("已修改")
    assert any(line.startswith("-第二行") or line.startswith("-") for line in out.splitlines())
    assert "改过的第二行" in out


async def test_edit_without_change_has_no_diff_block(tmp_path) -> None:
    text = "same\n"
    (tmp_path / "a.txt").write_text(text, encoding="utf-8")
    out = await _tb(tmp_path).call(
        "edit", json.dumps({"path": "a.txt", "old_string": "same", "new_string": "same"})
    )
    assert out.startswith("已修改")
    assert not any(line.startswith(("-", "+")) for line in out.splitlines()[1:])
