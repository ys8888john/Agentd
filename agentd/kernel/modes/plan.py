"""计划模式：先出计划 → 用户确认 → 逐条执行。

与 single / agent 的分工：
    single  一次 LLM 调用，不调工具；
    agent   直接进工具循环，边干边看；
    plan    **先让模型把任务拆成步骤列出来**（一次不带工具的调用），经用户确认后，
           每一步单独跑一轮 agent 循环，步骤状态实时推给界面。

为什么步骤状态是真实的而不是演的：每一步就是一次完整的 AgentMode 运行
（工具、审批、中断、落库全沿用），条目在步骤开跑前置 in_progress、跑完置
completed —— 界面上的 ✓ 背后是真干完了，不是计时器。

计划文本会以 assistant 消息落库（续聊时模型看得到"我计划过什么"），但**不流式
外发**：界面上计划由 PlanUpdate 事件渲染成计划卡，正文区不会出现一份重复的
计划原文。落库的这两类内部消息（计划原文 + 每步的「【计划 i/N】…」指令）都带
``internal=True`` 标记 —— 跨仓库契约，GUI 的 sessions.get_history 按它过滤，
历史回放和侧栏预览才不会混进步骤提示词（2026-10-13 用户实测"界面很乱"的根因之一）。

审批门复用工具审批那条通道（ApprovalRequest → session/request_permission）：
不新增协议面，GUI 的审批弹窗直接就能用；拒绝时本轮礼貌收尾，不进工具循环。
无人值守（AGENTD_TOOLS_APPROVE=none）时跳过问话，直接执行。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import ClassVar

from ...contracts import (
    Event,
    MessageDone,
    Notice,
    PlanEntryData,
    PlanUpdate,
    new_call_id,
)
from ..llm import LLMText
from ..models import Message
from ..tools import ApprovalRequest
from .agent import AgentMode
from .base import Mode, ModeContext

# 计划条数上限。每条 = 一轮 agent 循环（每轮最多 16 次 LLM 调用），条数不设限
# 时一个"大计划"能烧掉几百次调用；8 步覆盖日常任务，超了截断并告知。
MAX_PLAN_ENTRIES = 8

# 规划指令：要的是**机器可解析**的编号列表，所以把格式钉死（序号 + 点 + 内容），
# 并明确禁止计划以外的输出 —— 模型一客套，解析就废了。
_PLAN_ASK = """请把上面这个请求拆成一个可执行的分步计划。要求：
- 共 3~8 步，每行一条，格式严格为「序号. 步骤内容」，例如：
  1. 列出工作目录里的文件
  2. 读取 config.json 并确认端口配置
- 每步是一个能独立执行完的动作，说清做什么、做到什么程度；
- 只输出计划本身：不要解释、不要客套、不要代码块、不要小标题。"""

# 每一步的执行指令。带进度（i/n）和计划原文都在历史里（落库的 assistant 消息），
# 这里只钉"只做这一步 + 汇报方式"。
_STEP_ASK = (
    "【计划 {i}/{n}】{content}\n"
    "只完成这一步，不要提前做后面的步骤。完成后用一两句话汇报这一步的结果。"
)

# 编号 / 列表标记行：「1. xxx」「1、xxx」「1) xxx」「- xxx」「* xxx」「• xxx」
# 全角句号＂。＂和右括号也要认 —— 模型在中文化输出里很常用。
_LINE_RX = re.compile(r"^\s*(?:\d+\s*[.、)．]|[-*•●])\s*(.+?)\s*$")
# 剥掉模型爱加的强调标记：**粗体**、__粗体__、`代码`、首尾引号
_EMPH_RX = re.compile(r"[*_`]")


def _clean_step(raw: str) -> str:
    """清掉一条步骤上的强调标记和包裹引号，剩下纯内容。"""
    text = _EMPH_RX.sub("", raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”‘’":
        text = text[1:-1].strip()
    return text


def parse_plan(text: str, limit: int = MAX_PLAN_ENTRIES) -> list[str]:
    """从模型输出里解析编号计划 → 步骤内容列表（纯函数，单测直接打）。

    解析不出任何一条时返回空列表 —— 由调用方决定兜底（见 PlanMode.run 的
    单条兜底），不在这里猜。
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        m = _LINE_RX.match(line)
        if not m:
            continue
        item = _clean_step(m.group(1))
        if item:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def _user_text(user_input: str | list[dict]) -> str:
    """多模态输入（内容块列表）里抠出纯文本，用于兜底条目和日志。"""
    if isinstance(user_input, str):
        return user_input.strip()
    parts = [
        str(b.get("text") or "")
        for b in user_input
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return " ".join(p for p in parts if p).strip()


def _snapshot(entries: list[PlanEntryData]) -> list[PlanEntryData]:
    """条目列表的**独立快照**。

    必须每次新建对象再发：entries 是模式内部持续改状态的那个列表，直接 yield
    它等于把内部状态泄露给事件流 —— 事后再看早先的快照（测试、调试口、任何
    收集了事件流的消费者）会看到被后续修改污染的终态。传输层每次映射时虽然
    会另建对象，但内核不能指望下游替自己擦屁股。
    """
    return [
        PlanEntryData(content=e.content, priority=e.priority, status=e.status)
        for e in entries
    ]


class PlanMode(Mode):
    name: ClassVar[str] = "plan"

    def __init__(self) -> None:
        # 每一步复用 agent 模式的完整循环（工具 / 审批 / 中断 / 落库），
        # 不在这里重写一遍 —— 重写的那天就是两边行为漂移的那天。
        self._agent = AgentMode()

    async def run(
        self, ctx: ModeContext, user_input: str | list[dict]
    ) -> AsyncIterator[Event | Message]:
        # ---- 1. 规划：一次不带工具的调用 ----
        # 规划指令走**临时**消息（不进 ctx.history、不落库）：它是给这一次调用的
        # 指令，不是对话内容；落库了下轮模型该照着执行了。
        ask = Message(role="user", content=_PLAN_ASK)
        plan_text = ""
        stream = ctx.llm.stream_events(
            [*ctx.history, ask], system=ctx.system, tools=None
        )
        try:
            async for event in stream:
                if isinstance(event, LLMText):
                    plan_text += event.text
                if ctx.cancelled():
                    break
        finally:
            await stream.aclose()

        if ctx.cancelled():
            yield MessageDone(
                session_id=ctx.session_id, run_id=ctx.run_id, text="（已手动停止）"
            )
            return

        steps = parse_plan(plan_text)
        truncated = False
        if len(steps) >= MAX_PLAN_ENTRIES:
            # 到顶了：模型可能还写了更多，告诉用户一声（截断是静默动作，必须留痕）
            truncated = True
        if not steps:
            # 解析失败（模型没按格式来）：退化成"一整步"—— 计划卡仍然亮出来，
            # 执行语义与 agent 模式等价，比直接报错有用。
            fallback = _user_text(user_input) or "完成用户的请求"
            steps = [fallback[:120]]
            yield Notice(
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                text="模型没有按格式输出计划，已按单步执行。",
            )
        if truncated:
            yield Notice(
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                text=f"计划超过 {MAX_PLAN_ENTRIES} 步，已截断到前 {MAX_PLAN_ENTRIES} 步。",
            )

        entries = [PlanEntryData(content=s) for s in steps]

        # 计划原文落库（assistant 消息）：续聊时模型看得到"我计划过什么"。
        # Message 会被内核吃掉持久化、不外发，所以界面上不会出现重复的计划原文。
        # internal=True：跨仓库契约，GUI 历史回放按它跳过（见 models.Message.internal）
        # —— 不标记的话，续聊/刷新时"我的计划：…"会以普通气泡混进对话。
        plan_md = "我的计划：\n" + "\n".join(
            f"{i + 1}. {e.content}" for i, e in enumerate(entries)
        )
        plan_msg = Message(role="assistant", content=plan_md, internal=True)
        ctx.history.append(plan_msg)
        yield plan_msg
        yield PlanUpdate(session_id=ctx.session_id, run_id=ctx.run_id, entries=_snapshot(entries))

        # ---- 2. 审批门：按这个计划执行吗 ----
        # 复用工具审批通道（kind=other：计划不是工具，徽章别冒充工具）。
        # 拒绝 = 礼貌收尾，不进工具循环；没人可问（TUI/单测）时 request_approval
        # 直接放行；无人值守策略（approve=none）连问都不问。
        if ctx.approval_policy != "none" and ctx.approve is not None:
            ok = await ctx.request_approval(
                ApprovalRequest(
                    call_id=new_call_id(),
                    tool="plan",
                    title=f"按计划执行（{len(entries)} 步）",
                    kind="other",
                    detail="\n".join(f"{i + 1}. {e.content}" for i, e in enumerate(entries)),
                )
            )
            if not ok:
                yield Notice(
                    session_id=ctx.session_id,
                    run_id=ctx.run_id,
                    text="用户取消了计划执行，未开始执行。",
                )
                yield MessageDone(
                    session_id=ctx.session_id, run_id=ctx.run_id, text="（计划未执行）"
                )
                return

        # ---- 3. 逐条执行 ----
        total = len(entries)
        for i, entry in enumerate(entries):
            if ctx.cancelled():
                break
            entries[i].status = "in_progress"
            yield PlanUpdate(
                session_id=ctx.session_id, run_id=ctx.run_id, entries=_snapshot(entries)
            )

            step_msg = Message(
                role="user",
                content=_STEP_ASK.format(i=i + 1, n=total, content=entry.content),
                internal=True,  # 跨仓库契约：GUI 历史回放跳过，不当用户气泡画
            )
            # 两处都要有：yield 让内核落库（续聊的上下文），append 让**本轮**
            # 后续步骤的 LLM 调用立刻看得到（内核只管落库，不管模式的内存列表）。
            ctx.history.append(step_msg)
            yield step_msg

            async for event in self._agent.run(ctx, entry.content):
                if isinstance(event, Message):
                    # AgentMode 内部的工具往返：同样两边都要有（理由同上）。
                    ctx.history.append(event)
                    yield event
                else:
                    yield event

            if ctx.cancelled():
                # 这一步的 agent 循环自己会收尾（已流出的文本照常落库），
                # 这里不再补 MessageDone，直接终止后续步骤。
                break

            entries[i].status = "completed"
            yield PlanUpdate(
                session_id=ctx.session_id, run_id=ctx.run_id, entries=_snapshot(entries)
            )

        # 一步都没跑成（第 1 步前就被叫停）：历史里不能只有用户消息，
        # 补一条手动停止的记录，续聊时看着像正常说过话。
        if all(e.status == "pending" for e in entries):
            yield MessageDone(
                session_id=ctx.session_id, run_id=ctx.run_id, text="（已手动停止）"
            )
