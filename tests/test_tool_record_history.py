"""工具记录（role="tool_record"）落库与过滤测试。

要锁住的行为：
1. 每张完成的工具卡片整体落一条 tool_record 行，顺序在 user 之后、
   assistant 最终回复之前 —— 客户端续聊时能按序重放；
2. 被拒绝的调用落 cancelled 卡片（"点了拒绝"也是历史的一部分）；
3. **过滤器**：喂给 LLM 的上下文永远不含 tool_record 行（线格式安全），
   但 user / assistant 照常在；
4. SQLite 的 payload 权威性：新开连接（"重启"）后卡片完整还原。

LLM 用带记录钩子的假件（同 test_mcp_agent.ReplayLLM 思路），确定性强。
"""

from __future__ import annotations

import json
from pathlib import Path

from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.models import Message
from agentd.kernel.store import InMemorySessionStore, SqliteSessionStore


class ReplayLLM(LLM):
    """每轮按剧本产事件；把收到的完整 messages 快照进 seen 供断言。"""

    def __init__(self, scripts: list[list]) -> None:
        self._scripts = list(scripts)
        self.seen: list[list[Message]] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.seen.append(list(messages))
        script = self._scripts.pop(0) if self._scripts else [LLMText("（脚本耗尽）")]
        for event in script:
            yield event


async def test_agent_turn_persists_tool_record(tmp_path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("你好", encoding="utf-8")
    llm = ReplayLLM(
        [
            [LLMToolCall(id="c1", name="read_file", arguments=json.dumps({"path": str(target)}))],
            [LLMText("读到了")],
        ]
    )
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    sid = await kernel.create_session(cwd=str(tmp_path))
    async for _ in kernel.handle(sid, "帮我读文件", mode="agent"):
        pass

    history = await kernel.history(sid)
    assert [m.role for m in history] == ["user", "tool_record", "assistant"]

    card = history[1].tool_record
    assert card is not None
    assert card.call_id == "c1"
    assert card.title == "read_file"
    assert card.kind == "read"
    assert card.status == "completed"
    assert "你好" in card.output  # read_file 的真实输出（带行号）进了卡片
    # 冗余列 content 同步可读：sqlite3 直查也能看出是哪张卡
    assert history[1].content == "read_file"
    # assistant 结论在卡片之后
    assert history[2].content == "读到了"


async def test_denied_tool_persists_cancelled_record(tmp_path) -> None:
    llm = ReplayLLM(
        [
            [LLMToolCall(id="c2", name="write_file", arguments=json.dumps({"path": str(tmp_path / "x.txt"), "content": "hi"}))],
            [LLMText("好的")],
        ]
    )

    async def deny(req) -> bool:
        return False

    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    sid = await kernel.create_session(cwd=str(tmp_path))
    async for _ in kernel.handle(sid, "写个文件", mode="agent", approve=deny):
        pass

    history = await kernel.history(sid)
    card = history[1].tool_record
    assert card is not None
    assert card.status == "cancelled"
    assert "用户拒绝" in card.output
    assert (tmp_path / "x.txt").exists() is False  # 拒绝就该真的没写


async def test_tool_records_never_reach_llm_context(tmp_path) -> None:
    target = tmp_path / "hello.txt"
    target.write_text("你好", encoding="utf-8")
    llm = ReplayLLM(
        [
            [LLMToolCall(id="c1", name="read_file", arguments=json.dumps({"path": str(target)}))],
            [LLMText("读到了")],
            [LLMText("接着聊")],
        ]
    )
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    sid = await kernel.create_session(cwd=str(tmp_path))
    async for _ in kernel.handle(sid, "帮我读文件", mode="agent"):
        pass
    async for _ in kernel.handle(sid, "继续", mode="agent"):
        pass

    # 第二轮 handle 喂给 LLM 的上下文：无 tool_record，但 user/assistant 全在
    ctx = llm.seen[-1]
    assert all(m.role != "tool_record" for m in ctx)
    assert any(m.role == "user" and m.content == "帮我读文件" for m in ctx)
    assert any(m.role == "assistant" and m.content == "读到了" for m in ctx)


async def test_sqlite_payload_round_trip(tmp_path) -> None:
    """新开连接（等价于进程重启）后，tool_record 从 payload 完整还原。"""
    db = tmp_path / "sessions.db"
    target = tmp_path / "hello.txt"
    target.write_text("你好", encoding="utf-8")
    llm = ReplayLLM(
        [
            [LLMToolCall(id="c1", name="read_file", arguments=json.dumps({"path": str(target)}))],
            [LLMText("读到了")],
        ]
    )

    kernel = AgentKernel(llm=llm, store=SqliteSessionStore(db))
    sid = await kernel.create_session(cwd=str(tmp_path))
    async for _ in kernel.handle(sid, "帮我读文件", mode="agent"):
        pass
    kernel.store.close()  # type: ignore[attr-defined]

    reopened = AgentKernel(llm=llm, store=SqliteSessionStore(db))
    history = await reopened.history(sid)
    assert [m.role for m in history] == ["user", "tool_record", "assistant"]
    card = history[1].tool_record
    assert card is not None
    assert card.title == "read_file"
    assert "你好" in card.output
