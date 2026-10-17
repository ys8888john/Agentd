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

**模型自主进入计划**（2026-10-14，参考 WorkBuddy 的 EnterPlanMode/ExitPlanMode
思路落地）：agent 回合里模型可以自己调 ``enter_plan_mode`` 工具把编号计划交出来，
AgentMode 随即接管下面这套阶段（本模块的 plan_entries / plan_history_message /
plan_approval / execute_steps）—— 两条入口（用户手点 Plan 档 / 模型自主调用）
共用同一套阶段函数，不复制第二份，否则两边迟早漂移。与 WorkBuddy 的差别：
这里**不切换持久模式**（没有权限模式体系，也不想让"进过一次计划"粘住后续
所有回合），计划流转是当回合内的行为，回合结束即回到用户选的那档。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

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
from .base import Mode, ModeContext

if TYPE_CHECKING:  # 运行时不需要：AgentMode 只在注解里出现（见 PlanMode.__init__）
    from .agent import AgentMode

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


# 计划入口指引：agent 档的系统提示词段（内核在 enter_plan_mode 真在工具箱里时
# 注入，见 kernel._system_prompt）。工具描述管"这一个工具怎么用"，这一段管
# "什么时候该想到它"—— 只靠工具描述，小模型常常整个回合都想不起来自己还能
# 自主进计划，用户看到的还是"直接开干"。
PLAN_ENTRY_GUIDE = """## 计划模式（可自主进入）

遇到需要"先规划、后执行"的请求 —— 需求范围不明或有多重理解、要 3 步以上才能
完成、或涉及写文件 / 改代码 / 跑命令等有外部影响的动作 —— 调用 enter_plan_mode
把编号计划交给用户确认，确认后再逐步执行。调用前可以做只读调研（读文件 /
搜索 / 抓取），但不要做任何修改性动作：计划的意义就是让用户在动手前有机会叫停。

纯问答、一句话能说清的小改动不需要走计划；用户明确要求"直接做"时也别走。
用户也可以在界面上手动切到 Plan 档，与你自主调用等价。"""


def plan_entries(plan_text: str, fallback: str) -> tuple[list[PlanEntryData], list[str]]:
    """计划文本 → (条目, 提示文本列表)。两条入口（用户手切 / 模型自主）共用。

    提示文本由调用方转成 Notice 事件 —— 这里不碰 session_id / run_id，
    保持是纯函数，单测直接打。
    """
    steps = parse_plan(plan_text)
    hints: list[str] = []
    if len(steps) >= MAX_PLAN_ENTRIES:
        hints.append(f"计划超过 {MAX_PLAN_ENTRIES} 步，已截断到前 {MAX_PLAN_ENTRIES} 步。")
    if not steps:
        # 解析失败（模型没按格式来）：退化成"一整步"—— 计划卡仍然亮出来，
        # 执行语义与 agent 模式等价，比直接报错有用。
        fallback = (fallback or "").strip() or "完成用户的请求"
        steps = [fallback[:120]]
        hints.append("模型没有按格式输出计划，已按单步执行。")
    return [PlanEntryData(content=s) for s in steps], hints


def plan_history_message(entries: list[PlanEntryData]) -> Message:
    """计划原文落库用的 assistant 消息（续聊时模型看得到"我计划过什么"）。

    Message 会被内核吃掉持久化、不外发，所以界面上不会出现重复的计划原文。
    internal=True：跨仓库契约，GUI 历史回放按它跳过（见 models.Message.internal）
    —— 不标记的话，续聊/刷新时"我的计划：…"会以普通气泡混进对话。
    """
    plan_md = "我的计划：\n" + "\n".join(
        f"{i + 1}. {e.content}" for i, e in enumerate(entries)
    )
    return Message(role="assistant", content=plan_md, internal=True)


async def plan_preamble(
    ctx: ModeContext, entries: list[PlanEntryData]
) -> AsyncIterator[Event | Message]:
    """计划卡亮相：先交计划原文（Message，调用方 append 进活上下文并转交内核
    落库），再推全量 pending 快照（PlanUpdate）。

    用生成器而不是"返回一个 tuple"：调用方的活上下文各不相同（PlanMode 用
    ctx.history，AgentMode 的回合内转换用本轮局部 messages），append 的时机
    必须由调用方在两条 yield 之间自己掌握。
    """
    yield plan_history_message(entries)
    yield PlanUpdate(
        session_id=ctx.session_id, run_id=ctx.run_id, entries=_snapshot(entries)
    )


async def plan_approval(ctx: ModeContext, entries: list[PlanEntryData]) -> bool:
    """审批门：按这个计划执行吗。False = 用户叫停，调用方礼貌收尾。

    复用工具审批通道（kind=other：计划不是工具，徽章别冒充工具）。
    没人可问（TUI/单测）时 request_approval 直接放行；无人值守策略
    （approve=none）连问都不问。
    """
    if ctx.approval_policy == "none" or ctx.approve is None:
        return True
    return await ctx.request_approval(
        ApprovalRequest(
            call_id=new_call_id(),
            tool="plan",
            title=f"按计划执行（{len(entries)} 步）",
            kind="other",
            detail="\n".join(f"{i + 1}. {e.content}" for i, e in enumerate(entries)),
        )
    )


async def execute_steps(
    agent: AgentMode, ctx: ModeContext, entries: list[PlanEntryData]
) -> AsyncIterator[Event | Message]:
    """逐条执行：每条 = 一轮完整 agent 循环，条目状态实时推给界面。

    刻意 yield Message 而不自己 append 进 ctx.history —— 两条入口的"活上下文"
    不是同一个列表（PlanMode 用 ctx.history 本身，AgentMode 的回合内转换用本轮
    的局部 messages），由调用方 append 才能保证步骤 i+1 的 agent 循环看得到
    步骤 i 的工具往返。yield 与 append 的先后由生成器挂起秩序保证：
    调用方 append 完这一条，本函数才恢复跑去跑下一步。

    取消：每一步开跑前查一次；第 1 步前就被叫停时只落"（已手动停止）"，
    不执行任何步骤（历史里不能只有用户消息）。
    """
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
        yield step_msg  # 调用方负责 append 进活上下文 + 交给内核落库

        async for event in agent.run(ctx, entry.content):
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


class PlanMode(Mode):
    name: ClassVar[str] = "plan"

    def __init__(self) -> None:
        # 延迟导入打破循环：agent.py 要在工具循环里复用本模块的阶段函数
        # （模型自主进计划），模块级互相 import 会当场 CircularImportError。
        # execute_steps 的 AgentMode 注解是字符串（future annotations），
        # 运行时不需要这个名字 —— 只有这里真要用到类本身。
        from .agent import AgentMode

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

        # ---- 2. 计划条目 + 提示（与模型自主入口共用）----
        entries, hints = plan_entries(plan_text, _user_text(user_input))
        for hint in hints:
            yield Notice(session_id=ctx.session_id, run_id=ctx.run_id, text=hint)

        async for event in plan_preamble(ctx, entries):
            if isinstance(event, Message):
                ctx.history.append(event)
            yield event

        # ---- 3. 审批门（与模型自主入口共用）----
        if not await plan_approval(ctx, entries):
            yield Notice(
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                text="用户取消了计划执行，未开始执行。",
            )
            yield MessageDone(
                session_id=ctx.session_id, run_id=ctx.run_id, text="（计划未执行）"
            )
            return

        # ---- 4. 逐条执行（与模型自主入口共用）----
        async for event in execute_steps(self._agent, ctx, entries):
            if isinstance(event, Message):
                # AgentMode 内部的工具往返：append 让**后续步骤**立刻看得到，
                # yield 让内核落库留给下一轮。
                ctx.history.append(event)
            yield event
