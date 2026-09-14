"""带工具调用的对话模式：LLM ↔ MCP 工具循环。

与 SingleMode 的区别：单次模式只调一次 LLM 就结束；本模式在一轮里反复
「调 LLM → 若它要调工具就执行 → 把结果回灌 → 再调 LLM」，直到模型给出纯文本。

持久化仍由内核统一负责（只存最终的 MessageDone），中间的工具往返是本轮内的
临时消息，不落库 —— 与现有内核约定一致。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar

from ...contracts import Event, MessageDelta, MessageDone, ToolCallDone, ToolCallStart
from ..llm import LLMText, LLMToolCall
from ..mcp import McpHub
from ..models import Message, ToolCall
from .base import Mode, ModeContext

# 工具循环上限：防模型来回调不停（每次循环至少一次 LLM 调用）。
MAX_STEPS = 12


class AgentMode(Mode):
    name: ClassVar[str] = "agent"

    async def run(self, ctx: ModeContext, user_input: str) -> AsyncIterator[Event]:
        messages = list(ctx.history)

        async with McpHub(ctx.mcp_servers, cwd=ctx.cwd) as hub:
            tools = hub.tool_schema() or None
            last_text = ""

            for _ in range(MAX_STEPS):
                text_parts: list[str] = []
                calls: list[LLMToolCall] = []

                async for event in ctx.llm.stream_events(messages, system=ctx.system, tools=tools):
                    if isinstance(event, LLMText):
                        text_parts.append(event.text)
                        yield MessageDelta(
                            session_id=ctx.session_id, run_id=ctx.run_id, text=event.text
                        )
                    elif isinstance(event, LLMToolCall):
                        calls.append(event)

                text = "".join(text_parts)
                last_text = text

                if not calls:
                    yield MessageDone(session_id=ctx.session_id, run_id=ctx.run_id, text=text)
                    return

                # assistant 的这轮"举手要调工具"要记进消息历史，模型下一轮才看得到
                messages.append(
                    Message(
                        role="assistant",
                        content=text,
                        tool_calls=[
                            ToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in calls
                        ],
                    )
                )

                for call in calls:
                    binding = hub.binding(call.name)
                    title = binding.tool if binding is not None else call.name
                    yield ToolCallStart(
                        session_id=ctx.session_id,
                        run_id=ctx.run_id,
                        call_id=call.id,
                        title=title,
                        kind="execute",
                    )
                    output = await hub.call(call.name, call.arguments)
                    yield ToolCallDone(
                        session_id=ctx.session_id,
                        run_id=ctx.run_id,
                        call_id=call.id,
                        status="failed" if output.startswith("[错误]") else "completed",
                        output=output,
                    )
                    messages.append(
                        Message.tool(output, tool_call_id=call.id, name=call.name)
                    )

            # 到上限还没收敛：把已有文本收尾，别把用户晾着
            yield MessageDone(
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                text=last_text or "（达到工具调用轮数上限）",
            )
