"""模型自主进入计划模式（enter_plan_mode）测试。

锁的行为（2026-10-14，参考 WorkBuddy EnterPlanMode/ExitPlanMode 落地）：
1. 工具声明：kind=switch_mode（ACP 合法值）、不弹审批、Ask/只读档不给；
2. agent 回合里模型自主调用 → 回合内转换：计划卡亮相 → 审批门 → 逐条执行，
   与用户手点 Plan 档共用同一批阶段函数（modes/plan.py）；
3. 审批拒绝：礼貌收尾，不跑任何步骤；
4. 计划执行中再调 enter_plan_mode → 被闸（防嵌套烧穿步数护栏）；
5. 空 plan 参数：错误结果，不出计划卡，回合正常收尾；
6. Ask 档幻觉调用：被只读闸拦（明确错误，不弹审批）；
7. 系统提示词只在工具真可用时注入计划入口指引。
"""

from __future__ import annotations

import json

import pytest

from agentd.contracts import (
    Done,
    MessageDone,
    Notice,
    PlanUpdate,
    ToolCallDone,
    ToolCallStart,
)
from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.modes.plan import PLAN_ENTRY_GUIDE
from agentd.kernel.store import InMemorySessionStore
from agentd.kernel.tools import TOOL_PROFILES, ToolRuntime, _SPECS


class ScriptedLLM(LLM):
    """按调用次序发剧本（同 test_plan_mode.py 的驱动方式）。"""

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


def _plan_call(call_id: str, plan: str) -> LLMToolCall:
    return LLMToolCall(
        id=call_id,
        name="enter_plan_mode",
        arguments=json.dumps({"plan": plan}),
    )


_PLAN_3 = "1. 读取 README 了解项目\n2. 跑一遍测试确认基线\n3. 汇报结论"


def _plan_updates(events: list) -> list[list[tuple[str, str]]]:
    return [
        [(e.content, e.status) for e in ev.entries]
        for ev in events
        if isinstance(ev, PlanUpdate)
    ]


# ---- 工具声明 ----


def test_enter_plan_mode_spec():
    """kind=switch_mode（模式切换不是读写执行）、不弹审批、plan 必填。"""
    spec = next(s for s in _SPECS if s.name == "enter_plan_mode")
    assert spec.kind == "switch_mode"
    assert spec.requires_permission is False
    assert spec.parameters["required"] == ["plan"]
    # 给模型看的判断依据必须在描述里（识别质量全押在这段话上）
    assert "计划" in spec.description and "确认" in spec.description
    # Ask / 只读档不给：只读档不该能开计划执行
    assert "enter_plan_mode" in TOOL_PROFILES["native"]
    assert "enter_plan_mode" not in TOOL_PROFILES["ask"]
    assert "enter_plan_mode" not in TOOL_PROFILES["read_only"]


async def test_enter_plan_mode_handler_rejects_empty_plan(tmp_path):
    """空 plan：错误文本（调用方据此不出计划卡）。"""
    spec = next(s for s in _SPECS if s.name == "enter_plan_mode")
    rt = ToolRuntime(cwd=tmp_path, root=tmp_path)
    out = await spec.handler({}, rt)
    assert out.startswith("[错误] ") and "plan 不能为空" in out
    out2 = await spec.handler({"plan": "   "}, rt)
    assert out2.startswith("[错误] ")
    ok = await spec.handler({"plan": _PLAN_3}, rt)
    assert "已收到计划" in ok and "确认" in ok


# ---- 主流程：agent 回合内转换 ----


async def test_agent_mode_model_enters_plan_flow(tmp_path):
    """模型自主进计划：工具卡 → 计划卡 → 审批 → 三步真实执行。"""
    script = [
        [_plan_call("c1", _PLAN_3)],        # 主回合：模型决定先出计划
        [LLMText("README 读完了")],          # 步骤 1
        [LLMText("测试全过")],               # 步骤 2
        [LLMText("一切正常")],               # 步骤 3
    ]
    kernel = _kernel(script)
    sid = await kernel.create_session(cwd=str(tmp_path))

    events = [
        e async for e in kernel.handle(sid, "帮我规划这个项目怎么改造", mode="agent", approve=_yes)
    ]

    # 工具卡：kind=switch_mode，正常完成（进计划本身不弹审批）
    # ToolCallStart 才有 title；ToolCallDone 只能按 call_id 配对
    starts = [e for e in events if isinstance(e, ToolCallStart) and e.title == "enter_plan_mode"]
    dones = [e for e in events if isinstance(e, ToolCallDone) and e.call_id == "c1"]
    assert len(starts) == 1 and starts[0].kind == "switch_mode"
    assert len(dones) == 1 and dones[0].status == "completed"

    # 计划卡：首帧全 pending、末帧全 completed、中途真实 in_progress
    updates = _plan_updates(events)
    assert updates[0] == [
        ("读取 README 了解项目", "pending"),
        ("跑一遍测试确认基线", "pending"),
        ("汇报结论", "pending"),
    ]
    assert all(status == "completed" for _c, status in updates[-1])
    assert any(
        snap[0][1] == "in_progress" and snap[1][1] == "pending" for snap in updates
    ), updates
    assert len(updates) == 1 + 3 * 2  # 建计划 1 + 每步 in_progress/completed 各 1

    # 三步都真跑了：LLM 调用 = 主回合 1 + 步骤 3
    assert kernel.llm.calls == 4  # type: ignore[attr-defined]

    # 历史：用户请求 → 模型举手 → 计划原文（internal）→ 每步指令（internal）+ 答复
    history = await kernel.history(sid)
    roles = [m.role for m in history]
    assert roles[0] == "user"
    plan_msgs = [m for m in history if m.content.startswith("我的计划：")]
    assert len(plan_msgs) == 1 and plan_msgs[0].internal is True
    step_replies = ["README 读完了", "测试全过", "一切正常"]
    for text in step_replies:
        assert any(m.content == text for m in history)
    # 步骤指令带【计划 i/N】且 internal（GUI 历史回放按它过滤）
    step_asks = [m for m in history if "【计划 " in m.content and "/3】" in m.content]
    assert len(step_asks) == 3 and all(m.internal is True for m in step_asks)

    # 协议层收尾照旧
    assert isinstance(events[-1], Done)


async def test_agent_mode_plan_denied_never_executes(tmp_path):
    """审批拒绝：只有计划卡和取消提示，不跑任何步骤。"""
    script = [
        [_plan_call("c1", _PLAN_3)],
        [LLMText("不该被调用")],
    ]
    kernel = _kernel(script)
    sid = await kernel.create_session(cwd=str(tmp_path))

    events = [
        e async for e in kernel.handle(sid, "帮我规划这个项目", mode="agent", approve=_no)
    ]

    updates = _plan_updates(events)
    assert len(updates) == 1  # 只有建计划那一次
    assert all(status == "pending" for _c, status in updates[0])
    assert any(isinstance(e, Notice) and "取消" in e.text for e in events)
    dones = [e for e in events if isinstance(e, MessageDone)]
    assert len(dones) == 1 and dones[0].text == "（计划未执行）"
    assert kernel.llm.calls == 1  # type: ignore[attr-defined]


async def test_nested_enter_plan_mode_rejected_in_step(tmp_path):
    """计划执行中再调 enter_plan_mode：被闸，且不产生第二组计划卡。"""
    script = [
        [_plan_call("c1", "1. 第一步\n2. 第二步")],
        [_plan_call("c2", "1. 嵌套计划")],   # 步骤 1 的循环里又调 → 被闸
        [LLMText("步骤一完成")],
        [LLMText("步骤二完成")],
    ]
    kernel = _kernel(script)
    sid = await kernel.create_session(cwd=str(tmp_path))

    events = [e async for e in kernel.handle(sid, "做个两步的计划", mode="agent", approve=_yes)]

    # 嵌套调用：cancelled + 明确错误（不走审批、不执行）
    nested = [
        e for e in events if isinstance(e, ToolCallDone) and e.call_id == "c2"
    ]
    assert len(nested) == 1
    assert nested[0].status == "cancelled"
    assert "不能再开启新的计划" in nested[0].output

    # 只有一组计划卡（2 步：1 + 2*2 = 5 帧），且正常跑完
    updates = _plan_updates(events)
    assert len(updates) == 5
    assert all(status == "completed" for _c, status in updates[-1])


async def test_empty_plan_arg_is_error_without_plan_card(tmp_path):
    """空 plan 参数：错误结果，不出计划卡，回合正常收尾。"""
    script = [
        [_plan_call("c1", "")],
        [LLMText("好的，我直接说结论")],
    ]
    kernel = _kernel(script)
    sid = await kernel.create_session(cwd=str(tmp_path))

    events = [e async for e in kernel.handle(sid, "帮我规划一下", mode="agent", approve=_yes)]

    failed = [
        e for e in events if isinstance(e, ToolCallDone) and e.call_id == "c1"
    ]
    assert len(failed) == 1 and failed[0].status == "failed"
    assert "plan 不能为空" in failed[0].output
    assert _plan_updates(events) == []  # 没有计划卡
    # 回合继续正常走完（模型拿到错误结果后给了结论）
    assert any(isinstance(e, MessageDone) and e.text == "好的，我直接说结论" for e in events)


async def test_ask_mode_blocks_enter_plan_mode(tmp_path):
    """Ask 档模型幻觉调用：只读闸明确拒绝（不弹审批、不出计划卡）。"""
    script = [
        [_plan_call("c1", _PLAN_3)],
        [LLMText("只读档不能执行，我只能描述")],
    ]
    kernel = _kernel(script)
    sid = await kernel.create_session(cwd=str(tmp_path))

    events = [e async for e in kernel.handle(sid, "这个项目结构怎么样", mode="ask", approve=_yes)]

    blocked = [
        e for e in events if isinstance(e, ToolCallDone) and e.call_id == "c1"
    ]
    assert len(blocked) == 1
    assert "Ask（只读）模式不允许执行" in blocked[0].output
    assert _plan_updates(events) == []


# ---- 系统提示词 ----


def test_system_prompt_guide_injected_only_when_tool_available():
    """计划入口指引只在 enter_plan_mode 真在工具箱里时注入。"""
    kernel = _kernel([])
    native = kernel._system_prompt({}, profile="native") or ""
    ask = kernel._system_prompt({}, profile="ask") or ""
    assert "计划模式（可自主进入）" in native
    assert "enter_plan_mode" in native
    assert "enter_plan_mode" not in ask


def test_plan_entry_guide_mentions_both_entry_paths():
    """指引要说明两条入口等价（自主调用 / 界面手动切），别让用户以为失灵了。"""
    assert "手动" in PLAN_ENTRY_GUIDE and "Plan" in PLAN_ENTRY_GUIDE
