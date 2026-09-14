"""带工具调用的对话模式：LLM ↔ 工具循环。

与 SingleMode 的区别：单次模式只调一次 LLM 就结束；本模式在一轮里反复
「调 LLM → 若它要调工具就执行 → 把结果回灌 → 再调 LLM」，直到模型给出纯文本。

工具来源有两条，对模型完全透明：
1. **原生工具**（ctx.toolbox）—— 进程内的 read_file / glob / grep / write_file /
   edit / run_command，见 kernel/tools.py；
2. **MCP 工具**（ctx.mcp_servers → McpHub）—— 客户端在 session/new 里声明的
   server，每轮现开现关地连。
两者合并成同一个 tools 数组，靠名字路由回各自的执行器（MCP 名字带 `{server}__` 前缀，
所以不会撞名）。

审批：会改变外部状态的动作（写文件、改文件、跑命令）在真正执行前走
`ctx.request_approval()`。这是本模式唯一一处「停下来等外部输入」的地方 ——
走回调而不是事件，理由见 ModeContext.approve 的注释。

持久化仍由内核统一负责（只存最终的 MessageDone），中间的工具往返是本轮内的
临时消息，不落库 —— 与现有内核约定一致。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import ClassVar

from ...contracts import Event, MessageDelta, MessageDone, ToolCallDone, ToolCallStart
from ..llm import LLMText, LLMToolCall
from ..mcp import McpHub, ToolBinding
from ..models import Message, ToolCall
from ..tools import ERROR_PREFIX, ApprovalRequest, NativeTool, needs_approval
from .base import Mode, ModeContext

# 工具循环上限：防模型来回调不停（每次循环至少一次 LLM 调用）。
MAX_STEPS = 12

# 审批弹窗里给用户看的一行摘要：按这个顺序取第一个非空的字符串参数。
_DETAIL_KEYS = ("command", "path", "pattern")


def _detail_of(arguments: str) -> str:
    """从工具参数里抠出一行「用户一眼能判断要不要点允许」的摘要。

    解析失败就返回空串 —— 摘要缺失只是弹窗难看一点，不该影响调用本身。
    """
    try:
        args = json.loads(arguments) if arguments else {}
    except (ValueError, TypeError):
        return ""
    if not isinstance(args, dict):
        return ""
    for key in _DETAIL_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}: {value.strip()[:200]}"
    return ""


def _mcp_kind(binding: ToolBinding) -> str:
    """MCP 工具没有 kind 概念，只能从 annotations 反推。

    `read_only_hint=True` 才当只读（图标好看、不进审批）；其余一律 execute。
    None（服务端没声明）也走 execute —— 对不认识的东西保守一点。

    用 getattr 而不是直接取属性：测试里的假 binding 只实现真正被用到的字段，
    这里多要一个字段不该让假件崩掉。
    """
    return "read" if getattr(binding, "read_only", None) else "execute"


class AgentMode(Mode):
    name: ClassVar[str] = "agent"

    async def run(self, ctx: ModeContext, user_input: str) -> AsyncIterator[Event]:
        messages = list(ctx.history)

        async with McpHub(ctx.mcp_servers, cwd=ctx.cwd) as hub:
            tools = self._merged_schema(ctx, hub)
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
                    async for event in self._dispatch(ctx, hub, call):
                        if isinstance(event, Message):
                            messages.append(event)
                        else:
                            yield event

            # 到上限还没收敛：把已有文本收尾，别把用户晾着
            yield MessageDone(
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                text=last_text or "（达到工具调用轮数上限）",
            )

    # ---- 工具来源合并 ----

    @staticmethod
    def _merged_schema(ctx: ModeContext, hub: McpHub) -> list[dict] | None:
        """原生工具排前面。

        顺序不是随意的：小模型看 tools 数组是有先后的（越靠前越容易被选中），
        而原生工具是高频基础动作，MCP 是专项能力。两者都空时返回 None ——
        这正是 "没有工具时与 single 等价" 的开关。
        """
        merged: list[dict] = []
        seen: set[str] = set()
        if ctx.toolbox is not None:
            for item in ctx.toolbox.tool_schema():
                name = item["function"]["name"]
                merged.append(item)
                seen.add(name)
        for item in hub.tool_schema():
            if item["function"]["name"] not in seen:
                merged.append(item)
        return merged or None

    # ---- 单次工具调用 ----

    async def _dispatch(
        self, ctx: ModeContext, hub: McpHub, call: LLMToolCall
    ) -> AsyncIterator[Event | Message]:
        """执行一次工具调用，产出事件；最后 yield 一条 Message 供调用方回灌历史。

        把「事件」和「要回灌的消息」用同一个生成器吐出来，是为了让 run() 里的循环
        只有一处 `async for` —— 否则每个分支都要自己 append，很容易漏。
        """
        native: NativeTool | None = ctx.toolbox.binding(call.name) if ctx.toolbox else None
        mcp: ToolBinding | None = None if native is not None else hub.binding(call.name)

        if native is not None:
            title, kind = native.name, native.kind
            requires, destructive = native.requires_permission, False
        elif mcp is not None:
            title = mcp.tool
            kind = _mcp_kind(mcp)
            requires = False
            destructive = bool(getattr(mcp, "destructive", None))
        else:
            # 名字没对上任何来源：不执行，但要走完 start/done 让客户端把卡片收掉
            title, kind, requires, destructive = call.name, "other", False, False

        yield ToolCallStart(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            call_id=call.id,
            title=title,
            kind=kind,  # type: ignore[arg-type]
        )

        if needs_approval(
            requires=requires, kind=kind, destructive=destructive, policy=ctx.approval_policy
        ):
            approved = await ctx.request_approval(
                ApprovalRequest(
                    call_id=call.id,
                    tool=call.name,
                    title=title,
                    kind=kind,
                    detail=_detail_of(call.arguments),
                )
            )
            if not approved:
                output = f"{ERROR_PREFIX}用户拒绝执行 {title}"
                yield ToolCallDone(
                    session_id=ctx.session_id,
                    run_id=ctx.run_id,
                    call_id=call.id,
                    status="cancelled",
                    output=output,
                )
                yield Message.tool(output, tool_call_id=call.id, name=call.name)
                return

        output = await self._execute(ctx, hub, call, native, mcp)
        yield ToolCallDone(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            call_id=call.id,
            status="failed" if output.startswith(ERROR_PREFIX) else "completed",
            output=output,
        )
        yield Message.tool(output, tool_call_id=call.id, name=call.name)

    @staticmethod
    async def _execute(
        ctx: ModeContext,
        hub: McpHub,
        call: LLMToolCall,
        native: NativeTool | None,
        mcp: ToolBinding | None,
    ) -> str:
        if native is not None:
            assert ctx.toolbox is not None  # native 非空 ⇒ toolbox 非空
            return await ctx.toolbox.call(call.name, call.arguments)
        if mcp is not None:
            return await hub.call(call.name, call.arguments)
        return f"{ERROR_PREFIX}未知工具：{call.name}"
