"""Ask（只读）模式：能读、能搜、能抓，但不许动任何东西。

与另外两档的分工（三档是给用户在界面上选的，见 routing.UI_MODES）：
    agent  工具循环，可读写、可跑命令（默认档）；
    plan   先列计划 → 用户确认 → 逐条执行；
    ask    **只读档** —— 跑与 agent 完全相同的工具循环，但工具面被收窄：
           原生工具只剩读/搜索/抓取（内核按 profile="ask" 造工具箱），
           MCP 工具按 kind 过滤（见 AgentMode 里的两道闸）。

为什么继承 AgentMode 而不是另写一份循环：唯一差别是"哪些工具能用"，
而循环本身（审批、中断、落库、步数护栏、软硬兜底）必须与 agent 逐字一致 ——
复制一份就必然漂移，且漂移只在只读档出现，最难发现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import ClassVar

from ...contracts import Event
from .agent import AgentMode
from .base import ModeContext


class AskMode(AgentMode):
    """只读档：与 AgentMode 同构，差别只有 ctx.read_only 这一个标记。"""

    name: ClassVar[str] = "ask"

    async def run(self, ctx: ModeContext, user_input: str) -> AsyncIterator[Event]:
        # 内核已按 mode 把原生工具箱收窄成 profile="ask"（只留 read_file /
        # list_dir / glob / grep / web_search / web_fetch）；这里补上 ctx.read_only，
        # 让 AgentMode 在合并 MCP schema 与派发执行时再把非只读的一并剔掉。
        narrowed = replace(ctx, read_only=True)
        async for event in super().run(narrowed, user_input):
            yield event
