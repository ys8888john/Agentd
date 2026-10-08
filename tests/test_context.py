"""context.py 单元测试：token 估算与上下文裁剪。

这一层的价值全在边界上 —— 估得对不对、裁了之后会不会产出孤儿 tool 消息、
用户的提问会不会被挤掉。**这些都是一旦出错就会变成"模型突然变笨"的静默故障**，
所以必须逐条钉死。
"""

from dataclasses import asdict

import pytest

from agentd.boot import LiveLLM, Settings, _build_settings, build_llm
from agentd.kernel.context import (
    NO_TRIM,
    describe_compact,
    describe_trim,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    plan_compaction,
    sanitize_history,
    split_blocks,
    trim_history,
)
from agentd.kernel.llm import LLMNotice, LLMText
from agentd.kernel.models import Message, ToolCall


def _user(text: str) -> Message:
    return Message.user(text)


def _assistant(text: str) -> Message:
    return Message.assistant(text)


def _tool(text: str, call_id: str = "call_1", name: str = "read_file") -> Message:
    return Message.tool(text, tool_call_id=call_id, name=name)


def _assistant_with_calls(text: str, *call_ids: str) -> Message:
    return Message(
        role="assistant",
        content=text,
        tool_calls=[ToolCall(id=cid, name="read_file", arguments='{"path":"a.py"}') for cid in call_ids],
    )


# ---- 估算 ----

def test_estimate_tokens_empty_is_zero():
    assert estimate_tokens("") == 0


def test_estimate_tokens_chinese_counts_per_char():
    # 中文一个字大约一个 token（+10% 安全余量，所以 10 字约 11）
    ten = estimate_tokens("你好世界这是一个测试")
    assert ten == 11


def test_estimate_tokens_ascii_is_roughly_a_quarter():
    # 40 个 ASCII 字符 → 约 11 token
    assert estimate_tokens("a" * 40) == 11


def test_estimate_tokens_is_monotonic_and_overestimates():
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world, this is a longer sentence with more words")
    assert short < long
    # 估必须偏大：估小了会溢出窗口，估大了只是少带一点上下文
    assert estimate_tokens("abc") >= 1


def test_estimate_message_tokens_counts_tool_arguments():
    plain = estimate_message_tokens(_tool("short"))
    bulky = estimate_message_tokens(_tool("x" * 4000))
    # 工具输出常常是几十 KB，只算 content 会严重低估预算
    assert bulky > plain * 100


def test_estimate_message_tokens_counts_tool_calls():
    plain = _assistant("hi")
    calling = _assistant_with_calls("", "c1", "c2")
    assert estimate_message_tokens(calling) > estimate_message_tokens(plain)


def test_estimate_messages_tokens_sums_up():
    messages = [_user("你好"), _assistant("好的")]
    assert estimate_messages_tokens(messages) == sum(
        estimate_message_tokens(m) for m in messages
    )


# ---- 分块 ----

def test_split_blocks_groups_tool_roundtrip():
    msgs = [
        _user("读一下 a.py"),
        _assistant_with_calls("", "c1"),
        _tool("文件内容"),
        _assistant("读完了。"),
    ]
    blocks = split_blocks(msgs)
    # [user] [assistant+tool] [assistant] —— 工具往返粘成一块
    assert [len(b) for b in blocks] == [1, 2, 1]
    assert blocks[1][0].tool_calls is not None
    assert blocks[1][1].role == "tool"


def test_split_blocks_groups_multiple_results_of_one_call():
    msgs = [
        _assistant_with_calls("", "c1", "c2"),
        _tool("结果1", call_id="c1"),
        _tool("结果2", call_id="c2"),
        _user("继续"),
    ]
    blocks = split_blocks(msgs)
    assert [len(b) for b in blocks] == [3, 1]


# ---- 裁剪 ----

def test_trim_noop_when_budget_disabled():
    msgs = [_user("a"), _assistant("b")]
    kept, report = trim_history(msgs, 0)
    assert [m.content for m in kept] == ["a", "b"]
    assert report is NO_TRIM
    assert not report.happened


def test_trim_noop_when_within_budget():
    msgs = [_user("a"), _assistant("b")]
    kept, report = trim_history(msgs, 100000)
    assert [m.content for m in kept] == ["a", "b"]
    assert not report.happened


def test_trim_drops_oldest_keeps_newest():
    msgs = [_user(f"第{i}轮提问" * 50) for i in range(6)]
    kept, report = trim_history(msgs, 200)
    assert report.happened
    assert report.dropped > 0
    # 最后一条（本轮提问）必须还在 —— 丢掉它等于没问
    assert kept[-1].content == msgs[-1].content


def test_trim_keeps_first_user_when_it_falls_outside_window():
    first = _user("帮我重构这个项目，重点是 token 预算" * 10)
    middle = [_assistant(f"中间输出 {i}" * 200) for i in range(5)]
    last = _user("最后一句")
    kept, report = trim_history([first, *middle, last], 400)
    assert report.kept_first_user
    assert kept[0].content == first.content
    assert kept[-1].content == last.content


def test_trim_never_orphans_tool_messages():
    """这是最要命的一条：孤儿 role="tool" 会让 OpenAI 兼容端点直接 4xx。"""
    msgs = [
        _user("读两个文件"),
        _assistant_with_calls("", "c1"),
        _tool("第一个文件内容" * 300, call_id="c1"),
        _assistant_with_calls("", "c2"),
        _tool("第二个文件内容" * 300, call_id="c2"),
        _user("总结一下"),
    ]
    kept, report = trim_history(msgs, 500)
    assert report.happened

    for index, msg in enumerate(kept):
        if msg.role != "tool":
            continue
        # tool 前面必须紧跟着带 tool_calls 的 assistant，且 call_id 对得上
        prev = kept[index - 1]
        assert prev.role == "assistant", f"孤儿 tool 消息出现在下标 {index}"
        assert prev.tool_calls, f"下标 {index} 前的 assistant 没有 tool_calls"
        assert any(tc.id == msg.tool_call_id for tc in prev.tool_calls)


def test_trim_reports_overflow_when_last_block_alone_exceeds_budget():
    msgs = [_user("x" * 4000)]
    kept, report = trim_history(msgs, 10)
    # 只剩一块又超预算：如实标记，调用方据此告诉用户"我尽力了"
    assert report.overflowing
    # 但绝不把它裁成空 —— 用户的提问一个字也不能丢
    assert kept == msgs


def test_describe_trim_mentions_budget_and_count():
    report = type(NO_TRIM)(before=10, after=4, dropped=6, kept_first_user=True)
    text = describe_trim(report, budget=32768)
    assert "6" in text
    assert "32768" in text


# ---- 自动压缩的决策（80% 触发线）----


def _long(text: str, times: int = 60) -> Message:
    return _user(text * times)


def test_plan_does_nothing_when_there_is_no_budget():
    """没设预算就没有"到 80%"这件事 —— 连"占了多少"都不该算。"""
    plan = plan_compaction([_long("a"), _user("当前提问")], budget=0)
    assert not plan.should_compact
    assert plan.pressure.percent == 0


def test_plan_does_nothing_when_ratio_is_zero():
    """ratio=0 是显式关掉自动压缩（只剩溢出时的硬裁剪）。"""
    msgs = [_long("a"), _user("当前提问")]
    plan = plan_compaction(msgs, budget=1, ratio_pct=0)
    assert not plan.should_compact


def test_plan_does_nothing_below_the_trigger_line():
    msgs = [_long("a", 10) for _ in range(4)]
    plan = plan_compaction(msgs, budget=100000, ratio_pct=80)
    assert not plan.should_compact
    assert plan.pressure.percent < 80


def test_plan_compacts_once_the_line_is_crossed():
    msgs = [_long("第{}轮说的".format(i), 40) for i in range(6)]
    plan = plan_compaction(msgs, budget=1000, ratio_pct=80)

    assert plan.should_compact
    assert plan.pressure.over_trigger
    assert plan.pressure.percent >= 80
    assert 0 < plan.cut < len(msgs)


def test_plan_never_compacts_the_current_turn():
    """本轮提问所在的那一块永远不压 —— 压掉它等于假装没听见。

    边界情形：只有一块时宁可不动手（那时再压也没用，交给硬裁剪去兜）。
    """
    assert not plan_compaction([_user("就这一句")], budget=1, ratio_pct=80).should_compact

    two = [_user("上一轮"), _user("这一轮")]
    plan = plan_compaction(two, budget=1, ratio_pct=80)
    # 触发了，但只能压最老的那块：最后一块必须留下
    assert plan.cut == 1


def test_plan_cut_lands_on_a_block_boundary():
    """切点必须整块 —— 压出来的窗口要持久化到下一轮，切坏了要到下一轮才炸。"""
    msgs = [
        _user("读两个文件"),
        _assistant_with_calls("", "c1"),
        _tool("第一个文件的内容" * 300, call_id="c1"),
        _assistant_with_calls("", "c2"),
        _tool("第二个文件的内容" * 300, call_id="c2"),
        _user("总结一下"),
    ]
    plan = plan_compaction(msgs, budget=1200, ratio_pct=80)
    assert plan.should_compact

    survivors = msgs[plan.cut:]
    for index, msg in enumerate(survivors):
        if msg.role != "tool":
            continue
        prev = survivors[index - 1]
        assert prev.role == "assistant", f"孤儿 tool 消息出现在压缩后下标 {index}"
        assert any(tc.id == msg.tool_call_id for tc in prev.tool_calls or [])


def test_plan_leaves_room_so_the_next_turn_will_not_fire_again():
    """压完要留出滞后区间：每轮压一次的代价比偶尔多占一点预算高得多。"""
    msgs = [_long("第{}轮".format(i), 40) for i in range(10)]
    plan = plan_compaction(msgs, budget=4000, ratio_pct=80)

    keep = msgs[plan.cut:]
    after = plan_compaction(keep, budget=4000, ratio_pct=80)
    assert not after.should_compact, "压完立刻又触发 = 没有滞后区间，会每轮压一次"


def test_plan_counts_system_and_tools_schema():
    """system prompt 与 tools schema 在同一个窗口里占位置，不算进去会漏压。"""
    msgs = [_long("第{}轮".format(i), 40) for i in range(6)]
    budget = 3000
    quiet = plan_compaction(msgs, budget=budget, ratio_pct=80)

    loaded = plan_compaction(
        msgs,
        system="你是一个助手。" * 300,
        tools=[{"type": "function", "function": {"name": "read_file"}}] * 20,
        budget=budget,
        ratio_pct=80,
    )
    # 同样的对话，算上 system/tools 之后占用明显更高、压掉的东西更多
    assert loaded.pressure.used > quiet.pressure.used
    assert loaded.cut >= quiet.cut


def test_describe_compact_explains_what_happened():
    plan = plan_compaction(
        [_long("第{}轮".format(i), 40) for i in range(6)],
        budget=2000,
        ratio_pct=80,
    )
    ok = describe_compact(plan.pressure, dropped=7, summarized=True)
    bad = describe_compact(plan.pressure, dropped=0, summarized=False)

    assert "7" in ok
    assert str(plan.pressure.budget) in ok
    assert "没能产出摘要" in bad


def test_sanitize_drops_tool_call_without_response():
    """叫停在工具执行中留下的悬空 tool_call —— 留着必然招来端点 4xx。"""
    msgs = [
        _user("读个文件"),
        _assistant_with_calls("", "c1"),
        _user("换个问题"),
    ]
    clean = sanitize_history(msgs)
    assert [m.role for m in clean] == ["user", "user"]


def test_sanitize_drops_partially_answered_tool_calls():
    """三个调用只回了两个 —— OpenAI 要求每个 tool_call_id 都要有回应。"""
    msgs = [
        _user("批量读"),
        _assistant_with_calls("", "c1", "c2", "c3"),
        _tool("结果1", call_id="c1"),
        _tool("结果2", call_id="c2"),
        _assistant("读完了"),
    ]
    clean = sanitize_history(msgs)
    assert [m.role for m in clean] == ["user", "assistant"]


def test_sanitize_drops_orphan_tool_messages():
    msgs = [
        _user("问一句"),
        _tool("没人问我要这个", call_id="zz"),
        _assistant("好吧"),
    ]
    clean = sanitize_history(msgs)
    assert [m.role for m in clean] == ["user", "assistant"]


def test_sanitize_keeps_complete_roundtrip_and_orders_by_call():
    msgs = [
        _user("批量读"),
        _assistant_with_calls("", "c1", "c2"),
        _tool("结果2", call_id="c2"),   # 故意乱序到达
        _tool("结果1", call_id="c1"),
        _assistant("读完了"),
    ]
    clean = sanitize_history(msgs)
    assert [m.role for m in clean] == ["user", "assistant", "tool", "tool", "assistant"]
    # 输出按 tool_calls 声明的顺序排：跟端点要求的顺序一致，也更好读
    assert [m.tool_call_id for m in clean if m.role == "tool"] == ["c1", "c2"]


def test_sanitize_is_a_noop_for_plain_conversation():
    msgs = [_user("你好"), _assistant("在"), _user("聊点别的")]
    assert sanitize_history(msgs) == msgs


# ---- 与 LiveLLM 的接线 ----

def _settings(**over: object) -> Settings:
    """造一份 Settings：默认取**真实解析路径**（_build_settings），只覆盖 over。

    别手抄字段字典 —— 那等于让"加了新配置项"这件事以 TypeError 的形式炸在无关
    的测试里，而且每加一次都要重复付一次排查成本。抄漏的字段也没有任何提示
    告诉你"这个默认值没被覆盖到"。
    """
    base = asdict(_build_settings(lambda name, default: default))
    base.update(
        # backend 必须是 fake：这几个测试要的是"预算与裁剪的行为"，不能真去联网
        backend="fake",
        fake_reply="ok",
        store_backend="memory",
        db_path=":memory:",
        tools="off",
        tools_approve="none",
    )
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_live_llm_emits_notice_only_when_trimmed():
    """没裁就别出声 —— 每轮都刷一行系统提示会让人以为出了什么事。"""
    msgs = [_user(f"第{i}段" * 300) for i in range(6)]
    seen = LiveLLM(lambda: _settings(max_context_tokens=200))
    notices = [ev.text async for ev in seen.stream_events(msgs) if isinstance(ev, LLMNotice)]
    assert len(notices) == 1
    assert "上下文超出预算" in notices[0]

    calm = LiveLLM(lambda: _settings(max_context_tokens=0))
    quiet = [ev async for ev in calm.stream_events(msgs) if isinstance(ev, LLMNotice)]
    assert quiet == []


@pytest.mark.asyncio
async def test_live_llm_notice_precedes_body():
    msgs = [_user(f"第{i}段" * 300) for i in range(6)]
    llm = LiveLLM(lambda: _settings(max_context_tokens=200))
    events = [ev async for ev in llm.stream_events(msgs)]
    assert isinstance(events[0], LLMNotice)
    assert any(isinstance(ev, LLMText) for ev in events)


@pytest.mark.asyncio
async def test_live_llm_unlimited_budget_passes_everything():
    msgs = [_user("早问题"), _user("晚问题")]
    llm = LiveLLM(lambda: _settings(max_context_tokens=0))
    # 底层是 FakeLLM，不认 messages，所以只能断言"没有 Notice 且正文照出"
    texts = [ev.text async for ev in llm.stream_events(msgs) if isinstance(ev, LLMText)]
    assert "".join(texts) == "ok"


@pytest.mark.asyncio
async def test_live_llm_huge_tool_schema_leaves_headroom_or_shrinks_it():
    """tools schema 也吃窗口：预算被它吃光时不该再裁用户的提问。"""
    msgs = [_user("还看得见我吗")]
    tools = [{"type": "function", "function": {"name": f"tool_{i}", "description": "d" * 2000}}
             for i in range(40)]
    llm = LiveLLM(lambda: _settings(max_context_tokens=50))
    events = [ev async for ev in llm.stream_events(msgs, tools=tools) if isinstance(ev, LLMNotice)]
    # system+tools 已经超预算 → 不动 Message，也不发提示（发了也是假警报）
    assert events == []


# ---- build_llm 的输出上限 ----

def test_ollama_receives_num_predict():
    llm = build_llm(_settings(backend="ollama", ollama_model="qwen3", max_output_tokens=1024))
    assert llm.options == {"num_predict": 1024}


def test_ollama_gets_no_options_when_unlimited():
    llm = build_llm(_settings(backend="ollama", ollama_model="qwen3"))
    assert llm.options is None


def test_openai_compat_receives_max_tokens():
    llm = build_llm(_settings(backend="openai_compat", max_output_tokens=2048))
    body = llm._payload([_user("hi")], None)
    assert body["max_tokens"] == 2048


def test_openai_compat_omits_max_tokens_when_unlimited():
    llm = build_llm(_settings(backend="openai_compat"))
    assert "max_tokens" not in llm._payload([_user("hi")], None)
