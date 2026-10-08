"""自动压缩测试：到上下文预算的 80% 就该动手。

这一层要钉死的不是"压了没有"，而是四件更难看出对错的事：

1. **压完窗口真的缩小了**（不然这一趟纯粹在烧 token）；
2. **原始对话一条没删**（UI 回放与"搬运整段历史"都不能受影响）；
3. **摘要没压出来就不许移动窗口**（把没有备份的对话从上下文里抹掉，
   等于为了省地方删了文件才发现没备份）；
4. **压完之后不缺东西**：被切出去的部分在 system prompt 里有替代品，
   留下来的那一段也不能冒出孤儿 tool 消息。
"""

from __future__ import annotations

import pytest

from agentd.contracts import Done, Event, Notice
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText
from agentd.kernel.memory import COMPACT_PROMPT
from agentd.kernel.models import Message, ToolCall
from agentd.kernel.store import InMemorySessionStore, SqliteSessionStore


LONG_LINE = "这一段是为了凑够足够的长度，让它真的占掉不少 token。"  # 25 个中文字符


def _long(index: int, times: int = 12) -> str:
    return f"第{index}轮：" + LONG_LINE * times


class RecordingLLM(LLM):
    """把每一次"压缩"与"对话"请求都记下来。

    区分两者的依据是 system == COMPACT_PROMPT —— 这也是真实内核区分它们的方式，
    用别的方式标记会让测试通过但代码其实走错了路。
    """

    def __init__(self, compact_reply: str = "摘要：用户在给 context.py 写测试。", reply: str = "收到") -> None:
        self.compact_reply = compact_reply
        self.reply = reply
        self.flavors: list[str] = []
        self.systems: list[str | None] = []
        self.turns: list[list[Message]] = []
        self.fed_to_compaction: list[str] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        if system == COMPACT_PROMPT:
            self.flavors.append("compact")
            self.fed_to_compaction.append(messages[0].content if messages else "")
            yield LLMText(self.compact_reply)
            return
        self.flavors.append("chat")
        self.systems.append(system)
        self.turns.append(list(messages))
        yield LLMText(self.reply)

    @property
    def compacts(self) -> int:
        return sum(1 for f in self.flavors if f == "compact")


def _kernel(llm: LLM, **kw) -> AgentKernel:
    """测试里的内核默认**不开原生工具**。

    理由：工具 schema 本身也是上下文的固定开销（这一份就有两千多 token），
    带着它必须用几千以上的预算才压得动 —— 那是把被测对象和测试夹具的副作用
    缠在一起，断言会变成"到底是在测预算还是在测工具清单"。要测"工具往返不被切成两半"，往库里塞 tool 行就够了，
    不必真的挂上工具。
    """
    kw.setdefault("store", InMemorySessionStore())
    kw.setdefault("native_tools", "off")
    return AgentKernel(llm=llm, **kw)


async def _say(kernel: AgentKernel, sid: str, text: str) -> list[Event]:
    return [event async for event in kernel.handle(sid, text)]


def _notices(events: list[Event]) -> list[str]:
    return [e.text for e in events if isinstance(e, Notice)]


def _no_orphan_tools(messages: list[Message]) -> None:
    """喂给模型的序列里不能有没有着落的 role="tool"。

    见 context.split_blocks：孤儿 tool 消息会让 OpenAI 兼容端点直接 4xx。
    """
    for index, msg in enumerate(messages):
        if msg.role != "tool":
            continue
        prev = messages[index - 1]
        assert prev.role == "assistant", f"孤儿 tool 消息出现在下标 {index}"
        assert prev.tool_calls, f"下标 {index} 之前的 assistant 没有 tool_calls"
        assert any(tc.id == msg.tool_call_id for tc in prev.tool_calls)


def _still_carries(messages: list[Message], text: str) -> bool:
    """这条内容还在上下文里吗。

    用**内容**而不是"消息条数"当探针：条数这种相对量在压缩没生效时也能巧合对上
    （写这几个测试时就差点放过两个这样的假绿断言）。
    """
    return any(text in (m.content or "") for m in messages)


# ---------------------------------------------------------------- 触发


async def test_window_moves_once_the_trigger_line_is_crossed(tmp_path):
    llm = RecordingLLM()
    kernel = _kernel(
        llm,
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session()

    events: list[Event] = []
    for i in range(6):
        events = await _say(kernel, sid, _long(i))

    assert llm.compacts >= 1, "到 80% 了却没有压缩"

    # 1) 窗口真的小了：最开头的正文已经不在上下文里了（第一句会被抢救，用第二条当探针）
    fed = llm.turns[-1]
    rows = await kernel.store.history(sid)
    assert not _still_carries(fed, _long(1))

    # 2) 原始记录一条没删 —— 历史回放仍然完整
    assert len(rows) >= 12

    # 3) 用户要被告知：为什么压、占了多少
    text = " ".join(_notices(events))
    assert "压缩" in text
    assert "1000" in text


async def test_compaction_does_not_fire_without_a_budget(tmp_path):
    """没设预算就没有"到 80%"这件事 —— 一条都不许压。"""
    llm = RecordingLLM()
    kernel = _kernel(llm, context_budget=lambda: 0, summary_every=20, memory_file=tmp_path / "m.md")
    sid = await kernel.create_session()

    for i in range(8):
        await _say(kernel, sid, _long(i))

    assert llm.compacts == 0
    # 上下文一直在长 —— 没有预算就完全不动手，与此前的行为一致
    assert len(llm.turns[-1]) >= 14


async def test_it_gives_up_when_fixed_overhead_exceeds_the_trigger(capsys, tmp_path):
    """system + tools 的固定占用已经在触发线之上：再压也降不下来。

    这条守卫的意义是钱：认不出来就会每轮白跑一次压缩（多一次 LLM 调用），
    而窗口一点没变小 —— 而且这件事在界面上完全看不出来。
    """
    llm = RecordingLLM()
    kernel = _kernel(
        llm,
        native_tools="native",          # 故意把工具 schema 拉进来当固定开销
        context_budget=lambda: 1000,    # 但预算连工具清单都装不下
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session(cwd=str(tmp_path))

    for i in range(6):
        await _say(kernel, sid, _long(i))

    assert llm.compacts == 0
    # 但也不能完全无声 —— 配置错在哪，得在 stderr 里说得出来
    assert "AGENTD_MAX_CONTEXT_TOKENS" in capsys.readouterr().err


async def test_compaction_is_off_when_ratio_is_zero(tmp_path):
    llm = RecordingLLM()
    kernel = _kernel(
        llm,
        context_budget=lambda: 1000,
        compact_ratio=0,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session()

    for i in range(6):
        await _say(kernel, sid, _long(i))

    assert llm.compacts == 0


# ---------------------------------------------------------------- 信息去了哪里


async def test_the_compressed_part_reappears_as_a_summary(tmp_path):
    """压缩不是丢弃：被移出上下文的那一段必须在 system prompt 里有替代品。"""
    llm = RecordingLLM(compact_reply="摘要：我们在给 context.py 补自动压缩。")
    kernel = _kernel(
        llm,
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session()

    for i in range(6):
        await _say(kernel, sid, _long(i))

    assert "context.py 补自动压缩" in (llm.systems[-1] or "")


async def test_first_user_message_is_rescued_from_the_window(tmp_path):
    """任务声明通常写在第一句里，比摘要里被重写过的一句转述可靠得多。"""
    llm = RecordingLLM()
    kernel = _kernel(
        llm,
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session()

    task = "帮我把这个项目改成支持自动压缩的版本"
    await _say(kernel, sid, task)
    for i in range(1, 6):
        await _say(kernel, sid, _long(i))

    assert llm.compacts >= 1, "这一组数据的前提是要真的压过一次"
    fed = llm.turns[-1]
    # 第一句被抢救回来了，但它是窗口外唯一的幸存者 —— 第二条早就出了窗口
    assert fed[0].content == task
    assert not _still_carries(fed, _long(1))


async def test_window_survives_up_to_the_next_turn(tmp_path):
    """水位必须落库：只存在内存里的话，重启后窗口又从头开始，等于白压。"""
    llm = RecordingLLM()
    store = InMemorySessionStore()
    kernel = _kernel(
        llm,
        store=store,
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session()
    for i in range(6):
        await _say(kernel, sid, _long(i))
    assert not _still_carries(llm.turns[-1], _long(1))

    # 换一个内核接着聊（等价于进程重启）——
    fresh = _kernel(
        RecordingLLM(),
        store=store,
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    await _say(fresh, sid, "继续")

    rows = await store.history(sid)
    # 新内核没有历史包袱，却也不该把已经压过的东西重新灌回来
    assert not _still_carries(fresh.llm.turns[-1], _long(1))
    assert len(fresh.llm.turns[-1]) < len(rows)


async def test_window_survives_a_reopened_database(tmp_path):
    """上面那条用的是内存库；生产默认走 SQLite，得单独验一遍真的落了盘。"""
    db = tmp_path / "sessions.db"
    llm = RecordingLLM()
    first = _kernel(
        llm,
        store=SqliteSessionStore(db),
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await first.create_session()
    for i in range(6):
        await _say(first, sid, _long(i))
    assert not _still_carries(llm.turns[-1], _long(1))

    reopened = _kernel(
        RecordingLLM(),
        store=SqliteSessionStore(db),
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    await _say(reopened, sid, "继续")

    assert not _still_carries(reopened.llm.turns[-1], _long(1))
    # 摘要也得跟着回来，否则窗口之外的那段就是真空
    assert "context.py 写测试" in (reopened.llm.systems[-1] or "")


async def test_no_summary_means_the_window_stays_put(tmp_path):
    """模型交白卷时绝不移动窗口：那段对话还没留下备份。"""
    llm = RecordingLLM(compact_reply="抱歉，我做不到。")
    kernel = _kernel(
        llm,
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session()

    events: list[Event] = []
    for i in range(6):
        events = await _say(kernel, sid, _long(i))

    rows = await kernel.store.history(sid)
    fed = llm.turns[-1]
    # tool_record 之外的一条不落
    assert len([m for m in fed if m.role != "tool_record"]) >= len(
        [m for m in rows if m.role != "tool_record"]
    ) - 1
    assert any("没能产出摘要" in t for t in _notices(events))


# ---------------------------------------------------------------- 结构安全


async def test_compaction_never_orphans_a_tool_result(tmp_path):
    """一次工具往返要么整段留下，要么整段被压走 —— 绝不能从中间切开。"""
    llm = RecordingLLM()
    kernel = _kernel(
        llm,
        context_budget=lambda: 900,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session()

    # 直接往库里塞一段"读过文件"的历史（等价于上一轮跑完工具循环留下的行）
    await kernel.store.append(sid, Message.user("读一下 boot.py"))
    await kernel.store.append(
        sid,
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments='{"path":"boot.py"}')],
        ),
    )
    await kernel.store.append(sid, Message.tool(LONG_LINE * 12, tool_call_id="c1", name="read_file"))
    await kernel.store.append(sid, Message.assistant("读完了"))

    await _say(kernel, sid, _long(1))
    await _say(kernel, sid, _long(2))
    await _say(kernel, sid, _long(3))

    assert llm.compacts >= 1
    _no_orphan_tools(llm.turns[-1])


async def test_history_playback_is_unaffected(tmp_path):
    """GUI 回放走的是 store.history —— 压缩动的是"给模型看什么"，不是库里的内容。"""
    llm = RecordingLLM()
    kernel = _kernel(
        llm,
        context_budget=lambda: 1000,
        compact_ratio=80,
        summary_every=20,
        memory_file=tmp_path / "m.md",
    )
    sid = await kernel.create_session()

    for i in range(6):
        await _say(kernel, sid, _long(i))

    rows = await kernel.store.history(sid)
    assert [m.content for m in rows if m.role == "user"][0] == _long(0)
    assert any(isinstance(e, Done) for e in [e async for e in kernel.handle(sid, "最后一问")])


async def test_ratio_config_decides_when_it_fires(tmp_path):
    """触发线越低越早动手：50% 的线压的次数不该少于 90% 的线。"""
    early = RecordingLLM()
    late = RecordingLLM()
    for llm, r in ((early, 50), (late, 90)):
        kernel = _kernel(
            llm,
            context_budget=lambda: 1000,
            compact_ratio=r,
            summary_every=20,
            memory_file=tmp_path / f"m{r}.md",
        )
        sid = await kernel.create_session()
        for i in range(6):
            await _say(kernel, sid, _long(i))

    assert early.compacts >= 1
    assert early.compacts >= late.compacts
