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

持久化仍由内核统一负责：每张工具卡片整体落一条 role="tool_record" 行（历史
回放用），assistant 的最终回复落 MessageDone；给 LLM 的上下文会把 tool_record
行过滤掉（模式不用关心这件事）。
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import ClassVar

from ...contracts import Event, MessageDelta, MessageDone, Notice, ThoughtDelta, ToolCallDone, ToolCallStart
from ..blobstore import externalize
from ..llm import LLMNotice, LLMText, LLMThought, LLMToolCall
from ..mcp import McpHub, ToolBinding
from ..models import Message, ToolCall
from ..tools import ERROR_PREFIX, ApprovalRequest, NativeTool, needs_approval
from .base import Mode, ModeContext

# 工具循环上限：防模型来回调不停（每次循环至少一次 LLM 调用）。
# 12 步对"多来源搜索 + 整理成文件"这类任务偏紧（2026-10-09 实测航班任务
# 12 步烧完还在 read_file），提到 16；真跑不完还有下面两级护栏兜住。
MAX_STEPS = 16

# 软护栏：剩这么多步时往本轮上下文注入一条收敛指令（只进 messages、不 yield
# —— 不落库、界面看不到）。让模型在烧完之前自己规划"最后两步出交付物"。
_WARN_REMAINING = 2
_BUDGET_WARN = (
    "（系统提示，用户在界面上看不到这条）本轮剩余的工具调用步数只剩 2 步了。"
    "请立即停止扩大搜索/抓取面，开始收敛：马上用已经拿到的信息完成用户的任务"
    " —— 需要产出文件就现在调用 make_xlsx / write_file；信息不够就基于现有"
    "信息给出最好的结果，并明确说明还缺什么。不要把剩余步数花在继续收集上。"
)

# 硬护栏：步数耗尽后追加一轮**不带工具**的收尾调用。到这一步模型已经没有
# 工具可调，唯一正确的动作是基于已收集的信息给出最终答复 —— 无论任务完成
# 与否，用户一定能拿到一段有内容的结论，而不是"回复继续"。这层是
# 「突然停止没有输出」的 100% 兜底（2026-10-09 用户要求）。
_WRAPUP_ASK = (
    "（系统提示）工具调用步数已用完，本轮不允许再调用任何工具。请基于以上"
    "已获取的信息立即给出最终答复：能完成的部分直接给出结果（例如把整理好的"
    "数据写成表格/清单放进正文），没完成的部分如实说明缺了什么、给出已确认的"
    "部分。不要提「回复继续」，不要说「让我再试试」。"
)

# 到上限时给用户兜底的正文。**不能用最后一步的文本** —— 那通常是模型准备调下一个
# 工具的过场话（"让我尝试从另一个来源获取…"），拿它当答复就是"没给答案就中断"
# （2026-10-08 实测 sess_65a0584cc8f94dd6：12 轮全烧在反复 web_search/web_fetch 上，
# 最后一条停在"让我尝试从另一个来源获取更详细的航班信息"，用户看到的就是这个）。
# 注意：这只是收尾调用本身也失败/返回空时的最后兜底 —— 正常情况下硬护栏那轮
# 会给出真正的答复。
_LOOP_LIMIT_NOTE = (
    "（已达到本轮工具调用上限 {n} 步，未能收敛出完整结论。）\n\n"
    "上面已完成的搜索/抓取结果仍然有效，可以据此参考。若要继续，请直接回复"
    "「继续」，我会接着往下做。"
)


def _fallback_text(last_text: str) -> str:
    """工具循环耗尽时的收尾正文。

    优先用**非过场**的最后一段文本：只有当它不像"我准备再调一次工具"的台词时
    才用它（例如模型已经写出了一段小结）。否则退回 `_LOOP_LIMIT_NOTE` ——
    与其把一句"让我再试一个来源"当答案交给用户，不如老实说"到上限了，可以继续"。
    """
    text = (last_text or "").strip()
    if text and not _looks_like_preamble(text):
        return text
    return _LOOP_LIMIT_NOTE.format(n=MAX_STEPS)


# 过场话的判据：很短、且以"让我/我再/接下来我/下面我"这类**打算做什么**的措辞
# 开头，或者以冒号结尾（"让我试试这样："）。宁可漏判也不能误判 —— 把真正的小结
# 判成过场话会让用户丢掉已有的结论。
_PREAMBLE_HEAD_RX = re.compile(
    r"^\s*(?:让我|我再|接下来|下面我|我准备|我将|现在让我|先让我)"
)
_PREAMBLE_TAIL = ("：", ":", "…", "...", "，", ",")


def _looks_like_preamble(text: str | None) -> bool:
    """这段文本是不是"准备再调一次工具"的过场话（而非给用户的答复）。"""
    if not text:
        return False
    if len(text) > 120:
        # 写得长，多半已经在给结论了；宁可留着
        return False
    if text.endswith(_PREAMBLE_TAIL):
        return True
    return bool(_PREAMBLE_HEAD_RX.match(text))

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
            final_text = ""  # 收尾时定格的正文（正常走完=最后一步文本；被叫停=已流出的部分）
            stopped = False  # 本轮是被人叫停的，还是撞到步数上限的 —— 收尾话术不同

            for step in range(MAX_STEPS):
                if ctx.cancelled():
                    final_text = "（已手动停止）"
                    stopped = True
                    break

                text_parts: list[str] = []
                calls: list[LLMToolCall] = []

                stream = ctx.llm.stream_events(messages, system=ctx.system, tools=tools)
                try:
                    async for event in stream:
                        if isinstance(event, LLMText):
                            text_parts.append(event.text)
                            yield MessageDelta(
                                session_id=ctx.session_id, run_id=ctx.run_id, text=event.text
                            )
                        elif isinstance(event, LLMThought):
                            yield ThoughtDelta(
                                session_id=ctx.session_id, run_id=ctx.run_id, text=event.text
                            )
                        elif isinstance(event, LLMNotice):
                            # "上下文超预算已裁剪"之类：系统自己的话，不进正文、不落库
                            yield Notice(
                                session_id=ctx.session_id, run_id=ctx.run_id, text=event.text
                            )
                        elif isinstance(event, LLMToolCall):
                            calls.append(event)

                        # 取消检查贴在每个事件后面：流式时一个 chunk 一查，停止请求
                        # 最多推迟一个 chunk 生效。中途 break 要随即收掉底层 HTTP 流。
                        if ctx.cancelled():
                            break
                finally:
                    await stream.aclose()

                text = "".join(text_parts)
                last_text = text
                final_text = text

                if ctx.cancelled():
                    # 流到一半被叫停：落一份"已流出的文本"当本轮回复，别让历史空着
                    final_text = text or "（已手动停止）"
                    stopped = True
                    break

                if not calls:
                    yield MessageDone(session_id=ctx.session_id, run_id=ctx.run_id, text=text)
                    return

                # assistant 这一轮"举手要调工具"必须记进消息历史 ——
                # 而且要让内核落库（yield 出去即可，内核会吃掉并持久化）：
                # 续聊时重建上下文缺了它，后面那些 role="tool" 就成了孤儿，
                # OpenAI 兼容端点会因为"没有对应 tool_call 的 tool 消息"直接 4xx。
                announce = Message(
                    role="assistant",
                    content=text,
                    tool_calls=[
                        ToolCall(id=c.id, name=c.name, arguments=c.arguments) for c in calls
                    ],
                )
                messages.append(announce)
                yield announce

                for call in calls:
                    if ctx.cancelled():
                        break
                    async for event in self._dispatch(ctx, hub, call):
                        if isinstance(event, Message):
                            # 工具结果也是同样的两条路：append 进本地列表让本轮
                            # 的 LLM 立刻看到，yield 出去让内核落库留给下一轮。
                            messages.append(event)
                            yield event
                        else:
                            yield event

                if ctx.cancelled():
                    final_text = last_text or "（已手动停止）"
                    stopped = True
                    break

                # 软护栏：步数快烧完时注入收敛指令 —— 只 append 进本轮上下文，
                # 不 yield（不落库、界面看不到）。下一次 LLM 调用就会看到它，
                # 从而把最后两步花在"出交付物"而不是继续收集。
                remaining = MAX_STEPS - (step + 1)
                if remaining == _WARN_REMAINING and not ctx.cancelled():
                    messages.append(Message(role="user", content=_BUDGET_WARN))

            # 到上限还没收敛：**硬护栏** —— 追加一轮不带工具的收尾调用，
            # 强制模型基于已收集的信息给出最终答复。这是「突然停止没有输出」
            # 的兜底：无论任务完成与否，用户都能拿到一段有内容的结论。
            # 被叫停时不做收尾调用 —— 那不是"到上限"，用户自己按的停。
            if stopped:
                closing = final_text or last_text
            else:
                wrap_text = ""
                try:
                    messages.append(Message(role="user", content=_WRAPUP_ASK))
                    stream = ctx.llm.stream_events(messages, system=ctx.system, tools=None)
                    try:
                        async for event in stream:
                            if isinstance(event, LLMText):
                                wrap_text += event.text
                                yield MessageDelta(
                                    session_id=ctx.session_id, run_id=ctx.run_id, text=event.text
                                )
                            elif isinstance(event, LLMThought):
                                yield ThoughtDelta(
                                    session_id=ctx.session_id, run_id=ctx.run_id, text=event.text
                                )
                            elif isinstance(event, LLMNotice):
                                yield Notice(
                                    session_id=ctx.session_id, run_id=ctx.run_id, text=event.text
                                )
                            if ctx.cancelled():
                                break
                    finally:
                        await stream.aclose()
                except Exception:  # noqa: BLE001 - 收尾调用失败也不能把用户晾着
                    wrap_text = ""

                wrap_text = wrap_text.strip()
                if ctx.cancelled() and not wrap_text:
                    closing = final_text or last_text or "（已手动停止）"
                elif wrap_text and not _looks_like_preamble(wrap_text):
                    # 收尾调用给出了真正的答复 —— 它就是本轮的最终结论
                    closing = wrap_text
                else:
                    # 收尾调用也哑了（空文本 / 还在说"让我再试试"）：
                    # 退回兜底话术，至少把状况讲清楚
                    closing = _fallback_text(final_text or last_text)
            yield MessageDone(
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                text=closing,
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
            # 非只读的 MCP 工具也走审批：不再只信服务端的 destructiveHint。
            # 只读（read_only_hint=True → kind="read"）继续放行；
            # 其余 execute（browser 开页 / sqlite 写 / git 提交 / memory 写…）一律
            # require，让用户在"动真格"之前确认。policy=all/none 仍由 needs_approval 兜底。
            requires = kind != "read"
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
        failed = output.startswith(ERROR_PREFIX)
        # 大结果外置化（参考 WorkBuddy ToolResultBlobService）：超过阈值的
        # 工具输出全文落盘，模型与界面只拿「预览 + 文件路径」。这是
        # 2026-10-08「大帧打死 GUI 读帧任务」事故的治本项 —— GUI 侧放大
        # limit 只是止血，管道里的帧本就不该有几百 KB。
        # 失败输出不外置：错误信息必须完整可见，而且不能让外置包装把
        # failed 状态挤掉。
        if not failed:
            output = externalize(output, ctx.session_id, call.id)
        yield ToolCallDone(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            call_id=call.id,
            status="failed" if failed else "completed",
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
