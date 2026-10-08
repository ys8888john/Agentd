"""单次对话模式：历史 + 用户输入 → 一次 LLM 调用 → 流式回吐。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar

from ...contracts import MessageDelta, MessageDone, Notice
from ..llm import LLMNotice, LLMText
from .base import Mode, ModeContext


class SingleMode(Mode):
    name: ClassVar[str] = "single"

    async def run(
        self, ctx: ModeContext, user_input: str
    ) -> AsyncIterator[MessageDelta | MessageDone | Notice]:
        # ctx.history 里已经包含了本轮用户消息（内核在调用前就 append 了）
        chunks: list[str] = []

        # 用 stream_events 而不是 stream：后者只透传文本，会把 LLMNotice
        # （"上下文被裁了"这类系统提示）滤掉，用户就成了睁眼瞎。
        stream = ctx.llm.stream_events(ctx.history, system=ctx.system)
        try:
            async for event in stream:
                if isinstance(event, LLMNotice):
                    yield Notice(
                        session_id=ctx.session_id, run_id=ctx.run_id, text=event.text
                    )
                    continue
                if not isinstance(event, LLMText) or not event.text:
                    continue
                chunks.append(event.text)
                yield MessageDelta(
                    session_id=ctx.session_id,
                    run_id=ctx.run_id,
                    text=event.text
                )
                # 取消检查贴在每个 chunk 后面：停止请求最多推迟一个 chunk 生效，
                # 长回答里它是即时可感的。中途 break 必须随即收掉底层 HTTP 流，
                # 别把它留给 async generator 的 finalizer 兜底。
                if ctx.cancelled():
                    break
        finally:
            await stream.aclose()

        # 最后给一份完整文本。两个坑：
        # 1. 必须是 MessageDone，不能是 MessageDelta。流式上面已经把每个 chunk
        #    都发出去了，这里再用 MessageDelta 发整份，客户端就会收到两遍文本。
        #    传输层对 MessageDone 是 pass（客户端靠累积 chunk 自己拼），不会重复。
        # 2. 必须是 chunks（复数）。写成 chunk 是拿循环变量凑巧能跑，
        #    一旦流式有多个 chunk 就只拼到最后一个。
        # 内核靠 MessageDone 把 assistant 回复落库 —— 漏了它历史就永远写不进去。
        # 一个 chunk 都没流出来就叫停时，落一条"已手动停止"，
        # 否则历史里这轮 user 消息后面什么都没有，续聊时看着像丢了一句。
        yield MessageDone(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            text="".join(chunks) or ("（已手动停止）" if ctx.cancelled() else ""),
        )
