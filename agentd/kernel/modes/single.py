"""单次对话模式：历史 + 用户输入 → 一次 LLM 调用 → 流式回吐。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar

from ...contracts import MessageDelta, MessageDone
from .base import Mode, ModeContext


class SingleMode(Mode):
    name: ClassVar[str] = "single"

    async def run(self, ctx: ModeContext, user_input: str) -> AsyncIterator[MessageDelta | MessageDone]:
        # ctx.history 里已经包含了本轮用户消息（内核在调用前就 append 了）
        chunks: list[str] = []

        async for chunk in ctx.llm.stream(ctx.history, system=ctx.system):
            if not chunk:
                continue
            chunks.append(chunk)
            yield MessageDelta(
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                text=chunk
            )

        # 最后给一份完整文本，前端不用自己拼变量
        yield MessageDelta(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            text="".join(chunk),
        )