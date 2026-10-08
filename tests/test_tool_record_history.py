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
    # 一次工具往返现在是**两套并存**：
    #   assistant(tool_calls) + role="tool"  —— 给模型（下一轮的上下文，见 test_context_history）
    #   tool_record                          —— 给 UI（历史回放的卡片）
    assert [m.role for m in history] == [
        "user",
        "assistant",
        "tool_record",
        "tool",
        "assistant",
    ]
    # assistant 举手那一条必须带着 tool_calls，否则后面的 tool 消息就是孤儿
    announce = history[1]
    assert [tc.id for tc in announce.tool_calls] == ["c1"]
    # role="tool" 带着真实输出，且 id 能对回上面那条 tool_call
    result = history[3]
    assert result.tool_call_id == "c1"
    assert "你好" in result.content

    cards = [m.tool_record for m in history if m.role == "tool_record"]
    assert len(cards) == 1
    card = cards[0]
    assert card is not None
    assert card.call_id == "c1"
    assert card.title == "read_file"
    assert card.kind == "read"
    assert card.status == "completed"
    assert "你好" in card.output  # read_file 的真实输出（带行号）进了卡片
    # 冗余列 content 同步可读：sqlite3 直查也能看出是哪张卡
    assert history[2].content == "read_file"
    # assistant 结论在最后
    assert history[-1].content == "读到了"


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
    cards = [m.tool_record for m in history if m.role == "tool_record"]
    assert len(cards) == 1
    card = cards[0]
    assert card is not None
    assert card.status == "cancelled"
    assert "用户拒绝" in card.output
    assert (tmp_path / "x.txt").exists() is False  # 拒绝就该真的没写
    # 拒绝也要有 role="tool" 回灌：模型得知道"用户没同意"，
    # 否则它下一轮还以为自己已经写好了
    tool_msgs = [m for m in history if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["c2"]
    assert "拒绝" in tool_msgs[0].content


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


async def test_next_turn_sees_the_previous_tool_roundtrip(tmp_path) -> None:
    """这是工具上下文的核心：续聊时模型必须看得见自己上轮调用过什么、读到了什么。

    此前落库的只有 user / assistant 结论两条，工具往返一结束就蒸发 —— 于是
    "刚才那个文件里写了什么？"这种追问，模型手里其实什么都没有，只能瞎编。
    """
    target = tmp_path / "hello.txt"
    target.write_text("内容只有这一句话", encoding="utf-8")
    llm = ReplayLLM(
        [
            [LLMToolCall(id="c1", name="read_file", arguments=json.dumps({"path": str(target)}))],
            [LLMText("读到了")],
            [LLMText("文件里写的是「内容只有这一句话」")],
        ]
    )
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    sid = await kernel.create_session(cwd=str(tmp_path))
    async for _ in kernel.handle(sid, "帮我读文件", mode="agent"):
        pass
    async for _ in kernel.handle(sid, "文件里写的什么？", mode="agent"):
        pass

    ctx = llm.seen[-1]
    # assistant(tool_calls) 与 role="tool" 都以 OpenAI 线格式原样回到了上下文里
    announce = [m for m in ctx if m.role == "assistant" and m.tool_calls]
    assert announce, "看不到上一轮的工具调用 —— 追问答不出来的根因复发了"
    results = [m for m in ctx if m.role == "tool"]
    assert "内容只有这一句话" in "".join(m.content for m in results)
    # 顺序必须正确：tool 紧跟在发起它的那条 assistant 之后（孤儿 tool 会招来 4xx）
    roles = [m.role for m in ctx]
    head = roles.index("assistant")
    assert roles[head: head + 2] == ["assistant", "tool"]


async def test_dangling_tool_call_is_not_replayed(tmp_path) -> None:
    """上一轮被叫停在工具执行中：那条没等到结果的 tool_call 必须整段丢掉。

    留着它的后果是端点 4xx（"each tool_call_id must have a response"），
    而且报错信息指向完全不同的会话历史，根本查不出是取消留下的。
    """
    target = tmp_path / "hello.txt"
    target.write_text("你好", encoding="utf-8")
    llm = ReplayLLM([[LLMText("还能聊")]])
    kernel = AgentKernel(llm=llm, store=InMemorySessionStore())
    sid = await kernel.create_session(cwd=str(tmp_path))
    # 手工造一条"发起了调用却没结果"的历史行：等价于上一轮在工具执行中被叫停
    await kernel.store.append(
        sid,
        Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "ghost", "name": "read_file", "arguments": "{}"}],  # type: ignore[arg-type]
        ),
    )
    await kernel.store.append(sid, Message.user("现在还能聊吗"))
    async for _ in kernel.handle(sid, "现在还能聊吗", mode="agent"):
        pass

    ctx = llm.seen[-1]
    assert [m.role for m in ctx] == ["user", "user"]


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
    assert [m.role for m in history] == [
        "user",
        "assistant",
        "tool_record",
        "tool",
        "assistant",
    ]
    card = history[2].tool_record
    assert card is not None
    assert card.title == "read_file"
    assert "你好" in card.output
    # 这条 tool 的 tool_call_id 必须能对回上一条 assistant 的 tool_call，
    # 否则表示 payload 序列化/反序列化把工具往返丢成了半截
    assert history[3].tool_call_id == "c1"
    assert [tc.id for tc in history[1].tool_calls] == ["c1"]
