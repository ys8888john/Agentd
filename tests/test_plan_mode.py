"""plan（计划）模式测试。

锁的行为：
1. 规划：一次不带工具的调用产出编号计划 → PlanUpdate 全量快照（先全 pending）；
2. 计划原文以 assistant 消息落库（续聊时模型看得到），但不外发；
3. 审批门：复用工具审批通道；拒绝 = 礼貌收尾不进执行；approve=none 无人值守时不问；
4. 逐步执行：每条 = 一轮完整 agent 循环（工具 / 审批 / 中断全沿用），条目状态
   真实翻转 pending → in_progress → completed；
5. 兜底：模型不按格式输出 → 退化成单步 + Notice；超 8 步截断 + Notice；
6. 取消：第 1 步前叫停 → 只落"（已手动停止）"，不执行任何步骤。

驱动方式同 test_cancel.py：剧本化假 LLM（每次 stream_events 调用消费一段剧本），
不靠 sleep 碰运气。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agentd.contracts import (
    Done,
    EventAdapter,
    MessageDone,
    Notice,
    PlanEntryData,
    PlanUpdate,
    ToolCallDone,
    ToolCallStart,
)
from agentd.kernel.kernel import AgentKernel, UnknownModeError
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.models import Message
from agentd.kernel.modes.plan import MAX_PLAN_ENTRIES, PlanMode, parse_plan
from agentd.kernel.routing import UI_MODES, normalize_mode
from agentd.kernel.store import InMemorySessionStore

_PLAN_TEXT = "好的，我来拆分：\n1. 列出工作目录的文件\n2. 读取 config.json 确认端口\n3. 汇报结论"


class ScriptedLLM(LLM):
    """按调用次序发剧本：第 1 次调用发 script[0]，第 2 次发 script[1]……

    规划调用不带 tools、每步的 agent 循环带 tools —— 两者消费的是同一条剧本队列，
    测试按真实调用次序排剧即可。
    """

    def __init__(self, script: list[list]) -> None:
        self._script = script
        self.calls = 0

    async def stream_events(self, messages, *, system=None, tools=None):
        idx = self.calls
        self.calls += 1
        for event in self._script[idx] if idx < len(self._script) else []:
            yield event


def _kernel(script: list[list], **kwargs) -> AgentKernel:
    return AgentKernel(llm=ScriptedLLM(script), store=InMemorySessionStore(), **kwargs)


async def _yes(req) -> bool:
    return True


async def _no(req) -> bool:
    return False


def _plan_updates(events: list) -> list[list[tuple[str, str]]]:
    """把事件流里的 PlanUpdate 折成 [(content, status), ...] 快照序列。"""
    return [
        [(e.content, e.status) for e in ev.entries]
        for ev in events
        if isinstance(ev, PlanUpdate)
    ]


# ---- 纯函数：计划解析 ----


def test_parse_plan_handles_common_markers():
    text = "前言不计入\n1. 第一步\n2、第二步\n3) 第三步\n- 第四步\n* 第五步\n• 第六步"
    assert parse_plan(text) == ["第一步", "第二步", "第三步", "第四步", "第五步", "第六步"]


def test_parse_plan_strips_emphasis_and_quotes():
    assert parse_plan('1. **加粗步骤**\n2. `代码步骤`\n3. "引号步骤"') == [
        "加粗步骤",
        "代码步骤",
        "引号步骤",
    ]


def test_parse_plan_caps_at_limit():
    text = "\n".join(f"{i}. 步骤{i}" for i in range(1, 13))
    out = parse_plan(text)
    assert len(out) == MAX_PLAN_ENTRIES == 8
    assert out[-1] == "步骤8"


def test_parse_plan_empty_when_no_markers():
    assert parse_plan("我直接说结论：先这样再那样。") == []
    assert parse_plan("") == []


# ---- 内核注册 ----


async def test_kernel_registers_plan_mode():
    kernel = _kernel([[LLMText("x")]])
    assert kernel.modes() == ["agent", "ask", "plan", "single"]
    sid = await kernel.create_session()
    await kernel.validate(sid, "plan")  # 不抛即通过
    with pytest.raises(UnknownModeError):
        await kernel.validate(sid, "planning")


# ---- 主流程 ----


async def test_plan_mode_emits_plan_then_runs_each_step():
    """审批通过：先发全 pending 的计划，再逐条 in_progress → completed。"""
    script = [
        [LLMText(_PLAN_TEXT)],                      # 规划调用
        [LLMText("文件已列出：a.py、b.py")],          # 步骤 1
        [LLMText("端口是 8080")],                     # 步骤 2
        [LLMText("结论：配置正常")],                  # 步骤 3
    ]
    kernel = _kernel(script)
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "看看这个项目", mode="plan", approve=_yes)]

    updates = _plan_updates(events)
    # 第一个快照：三条全 pending
    assert updates[0] == [
        ("列出工作目录的文件", "pending"),
        ("读取 config.json 确认端口", "pending"),
        ("汇报结论", "pending"),
    ]
    # 最后一个快照：三条全 completed
    assert updates[-1] == [
        ("列出工作目录的文件", "completed"),
        ("读取 config.json 确认端口", "completed"),
        ("汇报结论", "completed"),
    ]
    # 中途必须出现过"第 1 条 in_progress、其余 pending"的中间态（状态是真翻转的）
    assert any(
        snap[0][1] == "in_progress" and snap[1][1] == "pending" for snap in updates
    ), updates
    # 快照数量：建计划 1 + 每步 in_progress/completed 各 1 = 1 + 3*2 = 7
    assert len(updates) == 7

    # 每一步的回复都落库（assistant 消息 ×3 + 计划原文 1）
    history = await kernel.history(sid)
    roles = [m.role for m in history]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert history[1].content.startswith("我的计划：")
    assert "【计划 1/3】列出工作目录的文件" in history[2].content
    assert history[3].content == "文件已列出：a.py、b.py"
    assert "【计划 2/3】" in history[4].content
    assert history[7].content == "结论：配置正常"

    # 内部消息打标（跨仓库契约：GUI 历史回放按 payload.internal 过滤）。
    # 计划原文（assistant）和步骤指令（user）是给模型/落库看的，不是给人看的；
    # 用户请求和每步的答复不打标 —— 过滤只针对内部消息，不能吞掉真实对话。
    assert history[1].internal is True
    assert history[2].internal is True and history[4].internal is True
    assert history[6].internal is True
    assert history[0].internal is None
    assert history[3].internal is None and history[7].internal is None

    # 流的末尾仍然是标准 Done（协议层不受影响）
    assert isinstance(events[-1], Done)
    assert events[-1].stop_reason == "end_turn"


async def test_plan_mode_denied_never_executes():
    """审批拒绝：只发计划 + 提示，不跑任何步骤，历史里没有步骤消息。"""
    script = [[LLMText(_PLAN_TEXT)], [LLMText("不该被调用")]]
    kernel = _kernel(script)
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "看看这个项目", mode="plan", approve=_no)]

    updates = _plan_updates(events)
    assert len(updates) == 1  # 只有建计划那一次
    assert all(status == "pending" for _c, status in updates[0])
    assert any(isinstance(e, Notice) and "取消" in e.text for e in events)
    dones = [e for e in events if isinstance(e, MessageDone)]
    assert len(dones) == 1 and dones[0].text == "（计划未执行）"
    # 第二步的剧本没被消费（LLM 只被调用过 1 次 = 规划）
    assert kernel.llm.calls == 1  # type: ignore[attr-defined]

    # 历史 = 用户请求 + 计划原文 + 取消结果（MessageDone 也会落库，结果要留痕）
    history = await kernel.history(sid)
    assert [m.role for m in history] == ["user", "assistant", "assistant"]
    assert history[1].content.startswith("我的计划：")
    assert history[2].content == "（计划未执行）"


async def test_plan_mode_policy_none_skips_approval():
    """无人值守（AGENTD_TOOLS_APPROVE=none）：不问，直接执行。"""

    async def _boom(req):  # pragma: no cover - 被调用就是失败
        raise AssertionError("approve=none 时不该问用户")

    script = [
        [LLMText(_PLAN_TEXT)],
        [LLMText("步骤一完成")],
        [LLMText("步骤二完成")],
        [LLMText("步骤三完成")],
    ]
    kernel = _kernel(script, approval_policy="none")
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "干活", mode="plan", approve=_boom)]

    assert _plan_updates(events)[-1][-1][1] == "completed"
    history = await kernel.history(sid)
    assert history[-1].content == "步骤三完成"


async def test_plan_mode_fallback_single_entry_when_unparseable():
    """模型不按格式输出：退化成单步执行 + Notice，不报错不摆烂。"""
    script = [[LLMText("我直接说结论：先这样再那样。")], [LLMText("已按单步完成")]]
    kernel = _kernel(script)
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "帮我整理一下", mode="plan", approve=_yes)]

    updates = _plan_updates(events)
    assert len(updates[0]) == 1
    assert updates[-1][0][1] == "completed"
    assert any(isinstance(e, Notice) and "单步" in e.text for e in events)
    history = await kernel.history(sid)
    assert history[-1].content == "已按单步完成"
    # 兜底条目内容来自用户请求（多模态输入也能抠出文本）
    assert updates[0][0][0] == "帮我整理一下"


async def test_plan_mode_truncates_over_limit_with_notice():
    text = "\n".join(f"{i}. 步骤{i}" for i in range(1, 13))
    script = [[LLMText(text)]] + [[LLMText(f"完成{i}")] for i in range(1, 9)]
    kernel = _kernel(script)
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "大工程", mode="plan", approve=_yes)]

    updates = _plan_updates(events)
    assert len(updates[0]) == MAX_PLAN_ENTRIES
    assert any(isinstance(e, Notice) and "截断" in e.text for e in events)
    # 只执行前 8 步
    assert kernel.llm.calls == 1 + MAX_PLAN_ENTRIES  # type: ignore[attr-defined]


async def test_plan_mode_cancel_before_any_step():
    """第 1 步前叫停：不执行，落一条"（已手动停止）"。"""
    cancel = asyncio.Event()
    cancel.set()
    script = [[LLMText(_PLAN_TEXT)], [LLMText("不该执行")]]
    kernel = _kernel(script)
    sid = await kernel.create_session()

    events = [e async for e in kernel.handle(sid, "干活", mode="plan", approve=_yes, cancel=cancel)]

    dones = [e for e in events if isinstance(e, MessageDone)]
    assert len(dones) == 1 and dones[0].text == "（已手动停止）"
    assert _plan_updates(events) == []  # 计划都没来得及发
    assert kernel.llm.calls <= 1  # type: ignore[attr-defined]
    assert events[-1].stop_reason == "cancelled"
    history = await kernel.history(sid)
    assert [m.role for m in history] == ["user", "assistant"]


async def test_plan_mode_step_reuses_agent_tool_loop(tmp_path):
    """每一步是完整的 agent 循环：步骤里的工具调用真的执行、卡片事件真的外流。"""
    target = tmp_path / "hello.txt"
    target.write_text("你好，计划模式", encoding="utf-8")
    script = [
        [LLMText("1. 读取 hello.txt\n2. 汇报内容")],                    # 规划
        [LLMToolCall(id="c1", name="read_file", arguments=json.dumps({"path": str(target)})),
         LLMText("读到了：你好，计划模式")],                             # 步骤 1：真调工具
        [LLMText("内容就是一句问候")],                                   # 步骤 2
    ]
    kernel = _kernel(script)
    sid = await kernel.create_session(cwd=str(tmp_path))

    events = [e async for e in kernel.handle(sid, "读个文件", mode="plan", approve=_yes)]

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    dones_tool = [e for e in events if isinstance(e, ToolCallDone)]
    assert len(starts) == 1 and starts[0].kind == "read"
    assert len(dones_tool) == 1 and dones_tool[0].status == "completed"
    assert "你好，计划模式" in dones_tool[0].output
    # 工具记录落库（续聊能重放卡片）
    history = await kernel.history(sid)
    assert any(m.role == "tool_record" for m in history)
    # 计划最终全完成
    assert all(s == "completed" for _c, s in _plan_updates(events)[-1])


# ---- 契约：事件可序列化 / 可回解析 ----


def test_plan_update_event_adapter_roundtrip():
    raw = PlanUpdate(
        session_id="s1",
        run_id="r1",
        entries=[
            PlanEntryData(content="第一步"),
            PlanEntryData(content="第二步", priority="high", status="completed"),
        ],
    ).model_dump_json()
    back = EventAdapter.validate_json(raw)
    assert isinstance(back, PlanUpdate)
    assert back.entries[1].priority == "high"
    assert back.entries[1].status == "completed"


# ---- 模式解析：只归一化，不猜意图 ----


def test_normalize_mode_falls_back_to_default():
    assert normalize_mode(None) == "agent"
    assert normalize_mode("") == "agent"
    assert normalize_mode("   ") == "agent"
    # 客户端 / 配置文件里出现过的别名
    assert normalize_mode("auto") == "agent"
    assert normalize_mode("DEFAULT") == "agent"
    assert normalize_mode("chat") == "agent"
    assert normalize_mode("ask") == "ask"
    assert normalize_mode("read_only") == "ask"
    assert normalize_mode("planning") == "plan"
    assert normalize_mode("once") == "single"


def test_normalize_mode_trims_and_lowercases():
    assert normalize_mode(" Ask ") == "ask"
    assert normalize_mode("PLAN") == "plan"


def test_normalize_mode_passes_unknown_through():
    """认不出来的原样返回，交给内核 _get_mode 抛 UnknownModeError。

    静默回退成 agent 会让「客户端选了个不存在的模式」这类错误消失得无声无息。
    """
    assert normalize_mode("nope") == "nope"


def test_ui_modes_exposes_exactly_three():
    """界面上就是三档：single 是调试档位，不给用户。"""
    assert UI_MODES == ("agent", "plan", "ask")
    assert "single" not in UI_MODES


async def test_agent_mode_never_rewrites_a_planning_request():
    """回归（2026-10-13 用户实测）：没有关键词自动路由。

    "帮我规划一个7天的新疆旅行"曾经会被提升进 plan：一次计划审批 + 7 轮逐步执行，
    一个完整答复被拆成 7 段，历史里还留下 7 条"【计划 i/7】…"内部消息。
    用户原话："这个问题不应该识别为一个 plan 任务"。中间加过咨询类话题豁免止血，
    但豁免名单是打地鼠 —— 索性连自动路由一起去掉：想分步执行就自己点 Plan。
    """
    script = [[LLMText("D1 乌鲁木齐…D7 返回，完整行程如下")]]
    kernel = _kernel(script)
    sid = await kernel.create_session()

    events = [
        e
        async for e in kernel.handle(
            sid, "帮我规划一个7天的新疆旅行", mode="agent", approve=_yes
        )
    ]

    assert _plan_updates(events) == [], "agent 模式被自动改写成了 plan"
    assert not any(isinstance(e, Notice) and "自动按计划模式" in e.text for e in events)
    dones = [e for e in events if isinstance(e, MessageDone)]
    assert len(dones) == 1 and "完整行程" in dones[0].text
    # 一轮结束：历史里只有用户请求 + 一条完整答复（没有计划原文、没有步骤消息）
    history = await kernel.history(sid)
    assert [m.role for m in history] == ["user", "assistant"]
    assert kernel.llm.calls == 1  # type: ignore[attr-defined]


async def test_plan_mode_requires_explicit_selection():
    """回归：停在 agent 档时，执行向的规划请求也**不会**自动走 plan。"""
    script = [[LLMText("直接回答，不规划")]]
    kernel = _kernel(script)
    sid = await kernel.create_session()

    events = [
        e
        async for e in kernel.handle(sid, "把这个需求拆解一下", mode="agent", approve=_yes)
    ]

    assert _plan_updates(events) == [], "agent 模式不该自动出计划"
    assert not any(isinstance(e, Notice) and "自动按计划模式" in e.text for e in events)
    assert any(isinstance(e, MessageDone) for e in events)
