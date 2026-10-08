"""跨会话记忆测试：解析层的容错 + 事实层的去重 + 内核层的触发时机。

这一层最容易被写坏的地方是**容错**：压缩调用的是一个本来就勉强能干这活的
模型（本机小模型），它爱加寒暄、爱写错标点、爱只答一半。每种畸形输出都要有
对应的断言，否则"没开成记忆"会伪装成"记忆开了但模型不配合"。
"""

from __future__ import annotations

import pytest

from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText
from agentd.kernel.memory import (
    COMPACT_PROMPT,
    MAX_FACTS,
    Compaction,
    MemoryFile,
    compact,
    parse_compaction,
    render_memory_block,
    render_transcript,
)
from agentd.kernel.models import Message, ToolCall
from agentd.kernel.store import InMemorySessionStore


# ---------------------------------------------------------------- parse_compaction


def test_parse_full_output():
    raw = (
        "摘要：用户在给 agentd 接跨会话记忆，已完成摘要层与事实层，"
        "下一步要接 GUI 的 token 配置。\n"
        "事实：\n"
        "- 偏好简体中文\n"
        "- 本机 Ollama 模型是 qwen3.5:9b\n"
    )
    got = parse_compaction(raw)

    assert "agentd" in got.summary
    assert got.facts == ("偏好简体中文", "本机 Ollama 模型是 qwen3.5:9b")


def test_parse_tolerates_preamble_and_ascii_colon():
    """模型最爱做的两件事：先来句寒暄、把全角冒号写成半角。

    两条都不能让解析交白卷 —— 内容都在，只是位置不对。
    """
    raw = "好的，我整理好了：\n\n摘要: 我们决定用 SQLite。\n事实:\n- 项目叫 agentd\n"
    got = parse_compaction(raw)

    assert "SQLite" in got.summary
    assert got.facts == ("项目叫 agentd",)


def test_parse_first_fact_may_sit_on_the_heading_line():
    got = parse_compaction("摘要：聊了存储选型。\n事实：- 存储用 SQLite\n- 并开了 WAL\n")

    assert got.facts == ("存储用 SQLite", "并开了 WAL")


def test_parse_summary_may_span_several_lines():
    got = parse_compaction("摘要：第一句。\n第二句。\n第三句。")

    assert got.summary == "第一句。 第二句。 第三句。"


def test_parse_without_facts_section():
    got = parse_compaction("摘要：只说了摘要。")

    assert "只说了摘要" in got.summary
    assert got.facts == ()


def test_parse_without_summary_section():
    got = parse_compaction("事实：\n- 只有事实\n")

    assert got.summary == ""
    assert got.facts == ("只有事实",)


def test_parse_english_headings_too():
    """小模型偶尔把标题译走；认不出来等于这一轮的压缩白做了。"""
    got = parse_compaction("Summary: decided on SQLite.\nFacts:\n- prefers Chinese\n")

    assert "SQLite" in got.summary
    assert got.facts == ("prefers Chinese",)


@pytest.mark.parametrize("raw", ["", "   ", "抱歉，我无法完成这个任务。"])
def test_parse_junk_returns_empty(raw):
    """垃圾输入 = 没有记忆，不能是异常 —— 压不动就该安静地什么都不做。"""
    assert parse_compaction(raw) == Compaction()


def test_parse_handles_bullet_variants():
    got = parse_compaction("事实：\n- 短横\n* 星号\n• 圆点\n")

    assert got.facts == ("短横", "星号", "圆点")


def test_parse_does_not_confuse_colon_inside_a_fact():
    """事实里带冒号很常见（"本机是 qwen3.5:9b"），不能被当成标题切开。"""
    got = parse_compaction("事实：\n- 本机是 qwen3.5:9b\n")

    assert got.facts == ("本机是 qwen3.5:9b",)


# ---------------------------------------------------------------- render_transcript


def test_transcript_labels_roles():
    text = render_transcript(
        [
            Message.user("读一下 boot.py"),
            Message.assistant("好的"),
        ]
    )
    assert text.splitlines() == ["用户：读一下 boot.py", "助手：好的"]


def test_transcript_shows_tool_activity_when_assistant_says_nothing():
    """只举手调用工具、没说话的 assistant content 是空的 —— 跳过它就等于丢掉了
    "这一步是靠工具走的"，摘要里会变成"模型凭空知道文件内容"。
    """
    announce = Message(
        role="assistant", content="", tool_calls=[ToolCall(id="c1", name="read_file")]
    )
    result = Message.tool("print('hi')", tool_call_id="c1", name="read_file")

    text = render_transcript([announce, result])

    assert "助手：（调用工具：read_file）" in text
    assert "工具：print('hi')" in text


def test_transcript_skips_blank_content():
    assert render_transcript([Message.assistant("")]) == ""


def test_transcript_truncates_from_the_head():
    """超限丢**最早的**，保住最近的进展 —— 反过来的话，摘要写完，
    用户最后问的那句反倒没进去，那这一轮的记忆就是残缺的。
    """
    msgs = [Message.user(f"第{i}轮的输入，内容足够长以便凑够字符数。" * 3) for i in range(40)]
    text = render_transcript(msgs, max_chars=500)

    assert "第39轮" in text
    assert "第0轮" not in text
    # 按行截而不是按字符切：不会出现半行残句
    for line in text.splitlines():
        assert line.startswith("用户：")
        assert line.endswith("。")


# ---------------------------------------------------------------- MemoryFile


async def test_memory_file_read_missing_file_returns_empty(tmp_path):
    assert await MemoryFile(tmp_path / "nope.md").read() == ""


async def test_memory_file_dedupes_repeated_facts(tmp_path):
    mem = MemoryFile(tmp_path / "memory.md")

    await mem.append_facts(("偏好简体中文",))
    await mem.append_facts(("偏好简体中文", "本机模型是 qwen3.5"))
    await mem.append_facts(("本机模型是 qwen3.5",))

    body = await mem.read()
    assert body.count("偏好简体中文") == 1
    assert body.count("本机模型是 qwen3.5") == 1


async def test_memory_file_dedupe_is_case_and_bullet_insensitive(tmp_path):
    mem = MemoryFile(tmp_path / "memory.md")
    await mem.append_facts(("Prefers Chinese",))
    await mem.append_facts(("- prefers chinese",))

    assert (await mem.read()).lower().count("prefers chinese") == 1


async def test_memory_file_caps_facts_keeping_the_newest(tmp_path):
    """这个文件每轮都进 system prompt —— 让它无限增长等于给每次回答悄悄加税。"""
    mem = MemoryFile(tmp_path / "memory.md")
    facts = tuple(f"f{i:03d}" for i in range(MAX_FACTS + 5))

    await mem.append_facts(facts)

    body = await mem.read()
    assert "- f044" in body                       # 最新的一定在
    assert "- f000" not in body                   # 最老的被挤掉了


async def test_memory_file_survives_unwritable_path():
    """写到不可能存在的路径只能打日志，不能把一轮对话带崩。

    理由：事实层是"写了更好"的东西，为它把对话挂掉属于典型的因小失大。
    """
    mem = MemoryFile("D:/nope/definitely/not/here/memory.md")
    await mem.append_facts(("x",))


# ---------------------------------------------------------------- render_memory_block


def test_render_empty_when_nothing_to_say():
    """返回空串而不是"暂无记忆"占位 —— 那是在每轮都要付的 system prompt 里
    放零信息字符。"""
    assert render_memory_block("", []) == ""


def test_render_extracts_bullets_and_drops_headings():
    text = "# agentd 长期记忆\n\n<!-- agentd:facts -->\n- 偏好简体中文\n- 项目是 ACP 架构\n"
    got = render_memory_block(text, [])

    assert "偏好简体中文" in got
    assert "# agentd 长期记忆" not in got        # 标题是给人看的
    assert "agentd:facts" not in got             # 标记只是给我自己看的


def test_render_summaries_only():
    # 元素顺序与 store.recent_summaries() 一致：(session_id, summary)
    got = render_memory_block("", [("s1", "上次做到 token 裁剪"), ("s2", "")])

    assert "上次做到 token 裁剪" in got
    assert got.strip().startswith("## 更早的会话摘要")


def test_render_puts_facts_before_summaries():
    """永远相关的事实要在按需相关的摘要之前 —— 模型对开头的注意力最足。"""
    got = render_memory_block("- 偏好简体中文", [("s1", "某次会话")])

    assert got.index("关于用户") < got.index("更早的会话摘要")


# ---------------------------------------------------------------- compact


class EchoBackLLM(LLM):
    """把收到的东西记下来，再吐一段固定的压缩输出回去。"""

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.seen: list[list[Message]] = []
        self.systems: list[str | None] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.seen.append(list(messages))
        self.systems.append(system)
        if self.reply:
            yield LLMText(self.reply)


class BoomLLM(LLM):
    async def stream_events(self, messages, *, system=None, tools=None):
        raise RuntimeError("模型挂了")
        yield  # pragma: no cover - 让解释器知道这仍是个生成器


async def test_compact_sends_one_flattened_user_message():
    """压缩请求必须是**一条 user 消息**。

    发原始 messages 会踩两个坑：切片可能从工具往返中间开始（孤儿 tool 消息），
    端点直接 4xx；工具输出的大段 JSON 也白占 token。
    """
    llm = EchoBackLLM("摘要：压缩好了。\n事实：\n- 喜欢短回复\n")

    got = await compact(
        llm,
        [
            Message.user("读一下 boot.py"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", name="read_file")],
            ),
            Message.tool("print('hi')", tool_call_id="c1", name="read_file"),
        ],
    )

    assert len(llm.seen[0]) == 1
    assert llm.seen[0][0].role == "user"
    assert llm.systems[0] == COMPACT_PROMPT
    assert "print('hi')" in llm.seen[0][0].content
    assert got.facts == ("喜欢短回复",)


async def test_compact_on_empty_history_does_not_call_llm():
    llm = EchoBackLLM()
    got = await compact(llm, [])

    assert got == Compaction()
    assert llm.seen == []


async def test_compact_failure_is_swallowed(tmp_path):
    got = await compact(BoomLLM(), [Message.user("hi")], MemoryFile(tmp_path / "m.md"))

    assert got == Compaction()


async def test_compact_writes_facts_to_the_file(tmp_path):
    mem = MemoryFile(tmp_path / "memory.md")
    llm = EchoBackLLM("摘要：定了 SQLite。\n事实：\n- 存储用 SQLite\n")

    await compact(llm, [Message.user("hi")], mem)

    assert "存储用 SQLite" in await mem.read()


# ---------------------------------------------------------------- 内核接线


class RecorderLLM(LLM):
    """区分"压缩请求"与"普通请求"：前者回压缩文本并记账，后者记录 system。"""

    def __init__(self, reply: str = "收到", compact_reply: str = "") -> None:
        self.reply = reply
        self.compact_reply = compact_reply
        self.flavors: list[str] = []
        self.systems: list[str | None] = []
        self.fed: list[str] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        if system == COMPACT_PROMPT:
            self.flavors.append("compact")
            self.fed.append(messages[0].content if messages else "")
            yield LLMText(self.compact_reply)
            return
        self.flavors.append("chat")
        self.systems.append(system)
        yield LLMText(self.reply)

    @property
    def compacts(self) -> int:
        return sum(1 for f in self.flavors if f == "compact")


def _kernel(llm: LLM, **kw) -> AgentKernel:
    return AgentKernel(llm=llm, store=InMemorySessionStore(), **kw)


async def _say(kernel: AgentKernel, sid: str, text: str) -> None:
    async for _ in kernel.handle(sid, text):
        pass


async def test_memory_is_off_when_summary_every_is_zero(tmp_path):
    """summary_every=0 必须把读和压一起关掉 —— 半开关（还在往 system 里塞旧摘要）
    比全关更难解释。"""
    llm = RecorderLLM()
    kernel = _kernel(llm, summary_every=0, memory_file=tmp_path / "m.md")
    sid = await kernel.create_session()

    for i in range(10):
        await _say(kernel, sid, f"第{i}句")

    assert llm.compacts == 0
    assert llm.systems == [None] * 10  # system 里一点记忆都没有


async def test_summary_is_written_once_the_session_is_long_enough(tmp_path):
    llm = RecorderLLM(
        compact_reply="摘要：用户在做 token 预算。\n事实：\n- 偏好简体中文\n"
    )
    kernel = _kernel(
        llm, summary_every=4, summary_recall=6, memory_file=tmp_path / "memory.md"
    )
    sid = await kernel.create_session()

    # 每轮开始时库里有 (2k-1) 条待压消息：第 1/2 轮不够，第 3 轮的 5 条才过线
    await _say(kernel, sid, "第一句")
    assert llm.compacts == 0
    await _say(kernel, sid, "第二句")
    assert llm.compacts == 0
    await _say(kernel, sid, "第三句")
    assert llm.compacts == 1

    summaries = await kernel.store.recent_summaries(8)
    assert [text for _, text in summaries] == ["用户在做 token 预算。"]
    assert "偏好简体中文" in await MemoryFile(tmp_path / "memory.md").read()


async def test_compaction_only_covers_the_new_slice(tmp_path):
    """再压一次时只喂新增部分。全文重压会让压缩成本随会话长度平方增长 ——
    那正是"为了省上下文而反被上下文吃掉"的反面教材。
    """
    llm = RecorderLLM(compact_reply="摘要：又一段。")
    kernel = _kernel(llm, summary_every=4, memory_file=tmp_path / "m.md")
    sid = await kernel.create_session()

    for i in range(5):  # 第 3 轮压一次（第 5 条开始），第 5 轮再压一次（第 5 条起）
        await _say(kernel, sid, f"第{i}句")

    assert llm.compacts == 2
    assert "第4句" in llm.fed[-1]      # 新的段落进去了
    assert "第0句" not in llm.fed[-1]  # 已经压过的不重复付费


async def test_new_session_recalls_the_previous_one(tmp_path):
    """跨会话记忆的验收标准：换一个会话，模型得知道上次发生过什么。"""
    llm = RecorderLLM(compact_reply="摘要：上次在给 memory.py 写测试。")
    kernel = _kernel(llm, summary_every=2, summary_recall=6, memory_file=tmp_path / "m.md")

    old = await kernel.create_session()
    await _say(kernel, old, "我们开始吧")
    await _say(kernel, old, "继续")     # 第 3 条：够线，压一次
    assert llm.compacts == 1

    fresh = await kernel.create_session()
    await _say(kernel, fresh, "接着上次")

    assert "上次在给 memory.py 写测试" in (llm.systems[-1] or "")


async def test_session_does_not_recall_its_own_summary(tmp_path):
    """自己的摘要不进自己的 system prompt：那些内容本来就在上下文里，
    重复一遍只会把"更早的会话讲了什么"挤出去。"""
    llm = RecorderLLM(compact_reply="摘要：自己的会话摘要。")
    kernel = _kernel(llm, summary_every=2, memory_file=tmp_path / "m.md")
    sid = await kernel.create_session()

    await _say(kernel, sid, "第一句")
    await _say(kernel, sid, "第二句")   # 压出摘要，但本轮就该看不到它
    await _say(kernel, sid, "第三句")   # 摘要确实已经躺在库里了

    assert "自己的会话摘要" not in (llm.systems[-1] or "")
    assert (await kernel.store.recent_summaries(8)) != []


async def test_facts_reach_the_system_prompt(tmp_path):
    mem = MemoryFile(tmp_path / "memory.md")
    await mem.append_facts(("偏好简体中文",))

    llm = RecorderLLM()
    kernel = _kernel(llm, summary_every=20, memory_file=tmp_path / "memory.md")
    sid = await kernel.create_session()
    await _say(kernel, sid, "你好")

    assert "偏好简体中文" in (llm.systems[-1] or "")
