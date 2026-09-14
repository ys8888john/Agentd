"""ACP over stdio 传输适配器 —— 主线出口。

把 kernel.handle() 产出的 Event 翻译成 ACP session/update 通知，
把客户端发来的 session/prompt 翻译成一次 handle()。

三条硬约束（违反即协议错误，客户端会静默卡死或解析失败）：
1. stdout 只能有 JSON-RPC 帧，日志走 stderr（tests/test_acp.py 断言 stdout 每行可 JSON 解析）；
2. 路径必须是绝对路径，行号 1-based；
3. 能力判断用 initialize 协商的 protocolVersion，不能看 SDK 版本号。
"""

from __future__ import annotations

import sys
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    start_tool_call,
    update_agent_message_text,
    update_agent_thought_text,
    update_tool_call,
)
# PermissionOption / ToolCallUpdate 没在 acp 顶层导出，只能从 schema 取
from acp.schema import PermissionOption, ToolCallUpdate

from ..contracts import (
    Done,
    ErrorEvent,
    MessageDelta,
    MessageDone,
    ToolCallDone,
    ToolCallStart,
)
from ..kernel.kernel import AgentKernel
from ..kernel.tools import ApprovalRequest, ApproveHandler

# ACP 的 stop_reason 取值是固定的五种，其中没有 "error"：
#   end_turn | max_tokens | max_turn_requests | refusal | cancelled
# 所以内核里的 Done(stop_reason="error") 不能直译，见 prompt() 里的处理。
_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "cancelled": "cancelled",
    # 执行出错：错误内容已经通过 thought 通道告知客户端，这里按正常收尾
    "error": "end_turn",
}

# 内核的 kind/status 取值与 ACP schema 不完全重合，映射一下才合法：
#   ACP ToolKind   = read|edit|delete|move|search|execute|think|fetch|switch_mode|other
#   ACP ToolStatus = pending|in_progress|completed|failed
# kernel 侧多一个 "generic"（"不知道是哪类"）—— 唯一需要额外译的就是它；
# 其余取值与 ACP 同名，逐条列出来是为了**改 ACP schema 时能一眼看出差集**，
# 而不是靠默认分支悄悄兜住（默认兜住的后果是客户端收到非法 kind 后静默卡死）。
_ACP_KIND = {
    "read": "read",
    "edit": "edit",
    "delete": "delete",
    "move": "move",
    "search": "search",
    "execute": "execute",
    "think": "think",
    "fetch": "fetch",
    "switch_mode": "switch_mode",
    "other": "other",
    "generic": "other",
}
_ACP_STATUS = {"completed": "completed", "failed": "failed", "cancelled": "failed"}

# 审批弹窗给用户的三个选项。option_id 是我们自己定的字符串，
# 客户端只负责把它原样回传（见 _make_approver 里的解析）。
_PERMISSION_OPTIONS = (
    PermissionOption(option_id="allow_once", name="允许一次", kind="allow_once"),
    PermissionOption(option_id="allow_session", name="本会话总是允许", kind="allow_always"),
    PermissionOption(option_id="reject", name="拒绝", kind="reject_once"),
)
_ALLOW_OPTION_IDS = frozenset({"allow_once", "allow_session"})


class AgentdAcpAgent(Agent):
    """把 AgentKernel 暴露成 ACP agent。

    状态只有两样：连接对象、每个会话的模式覆盖值。
    真正的会话历史在内核的 SessionStore 里 —— 这里不复制一份。
    """

    def __init__(self, kernel: AgentKernel) -> None:
        self._kernel = kernel
        self._conn: Any = None
        self._client_caps: Any = None
        self._session_modes: dict[str, str] = {}
        # 会话 → 已被"本会话总是允许"放行的工具名集合。
        # 审批记忆放在传输层而不是内核里：它是「客户端怎么问用户」的一部分，
        # 换个客户端（HTTP、TUI）记忆策略完全可以不一样，内核不该替它决定。
        self._allow_all: dict[str, set[str]] = {}

    @staticmethod
    def _log(msg: str) -> None:
        """唯一允许的日志出口 —— 必须是 stderr。"""
        print(msg, file=sys.stderr, flush=True)

    def on_connect(self, conn: Any) -> None:
        """SDK 建立连接后回调，拿到 conn 才能反向调用 session_update。

        必须是**同步** def，不能写成 async def：
        SDK 是 `on_connect(self)` 直接调用，不 await（见 acp/agent/connection.py:101-102）。
        写成 async 的话，调用只会生成一个从未被执行的协程，self._conn 永远是 None，
        直到 prompt() 里才炸 "'NoneType' object has no attribute 'session_update'"——
        报错点离病根十万八千里，极难定位。
        """
        self._conn = conn
        self._log("[agentd] ACP 连接已建立")

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        self._client_caps = client_capabilities
        if protocol_version != PROTOCOL_VERSION:
            self._log(
                f"[agentd] protocolVersion 与 SDK 不一致: "
                f"client={protocol_version} sdk={PROTOCOL_VERSION}，按客户端版本继续"
            )
        # 回显客户端的版本而不是 SDK 的 —— 这是握手协商的规矩
        # （agent_capabilities / auth_methods / agent_info 都有默认值，先不声明）
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = await self._kernel.create_session(cwd=cwd, mcp_servers=mcp_servers)
        self._log(f"[agentd] 新会话 {session_id}  cwd={cwd}")
        if mcp_servers:
            self._log(f"[agentd] 本会话接入 {len(mcp_servers)} 个 MCP server")
        # modes / config_options 也在这里声明 —— 等 C 方案落地时填上
        return NewSessionResponse(session_id=session_id)

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> Any:
        """记录该会话的模式覆盖值，下一次 prompt 时生效。"""
        self._session_modes[session_id] = mode_id
        self._log(f"[agentd] 会话 {session_id} 模式切换为 {mode_id}")
        return None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """目前只记日志 —— 真正的中断要让内核在 handle() 里响应取消信号。"""
        self._log(f"[agentd] 收到取消请求 session={session_id}（暂未实现中断）")

    # ---- 审批 ----

    def _make_approver(self, session_id: str) -> ApproveHandler:
        """造一个绑好 session 的审批回调，交给内核。

        踩过的坑：**客户端可能根本没实现 session/request_permission**（ACP 的
        clientCapabilities 里有这一项）。这时 request_permission 会一直等不到响应
        或者直接报错。所以这里三件事一起做：问之前先查"本会话总是允许"的记忆、
        出错就按拒绝处理并打日志、把结果原样翻译成 bool。
        """

        async def approve(req: ApprovalRequest) -> bool:
            granted = self._allow_all.setdefault(session_id, set())
            if req.tool in granted:
                return True  # 之前选过"本会话总是允许"，不再打扰用户

            tool_call = ToolCallUpdate(
                tool_call_id=req.call_id,
                kind=_ACP_KIND.get(req.kind, "other"),  # type: ignore[arg-type]
                title=req.title,
                raw_input={"detail": req.detail} if req.detail else None,
            )
            self._log(f"[agentd] 请求审批：{req.tool}（{req.detail or '无摘要'}）")
            try:
                resp = await self._conn.request_permission(
                    session_id, tool_call=tool_call, options=list(_PERMISSION_OPTIONS)
                )
            except Exception as exc:  # noqa: BLE001 - 客户端不支持时按拒绝处理
                self._log(f"[agentd] 审批请求失败（按拒绝处理）：{type(exc).__name__}: {exc}")
                return False

            outcome = getattr(resp, "outcome", None)
            verdict = getattr(outcome, "outcome", None)
            if verdict == "selected":
                option_id = str(getattr(outcome, "option_id", "") or "")
                if option_id == "allow_session":
                    granted.add(req.tool)
                return option_id in _ALLOW_OPTION_IDS
            # "cancelled"（用户关掉弹窗）或任何没见过的形状 —— 一律不放行。
            # 审批这种东西，拿不准的时候必须往"拒绝"倒，反过来就是安全漏洞。
            self._log(f"[agentd] 审批未通过：{req.tool} outcome={verdict!r}")
            return False

        return approve

    async def prompt(
        self, session_id: str, prompt: list[Any], **kwargs: Any
    ) -> PromptResponse:
        text = self._extract_text(prompt)
        # 默认走 agent 模式（无工具时与 single 等价，有 MCP server 时才会触发工具循环）
        mode = self._session_modes.get(session_id, "agent")

        # 先校验：这里抛出的异常会被 SDK 转成 JSON-RPC error，
        # 比开流之后再失败干净得多。
        await self._kernel.validate(session_id, mode)

        stop_reason = "end_turn"

        async for event in self._kernel.handle(
            session_id, text, mode=mode, approve=self._make_approver(session_id)
        ):
            if isinstance(event, MessageDelta):
                await self._conn.session_update(
                    session_id, update_agent_message_text(event.text)
                )

            elif isinstance(event, MessageDone):
                # ACP 没有"整条消息"事件，客户端靠累积 chunk 自己拼。
                # 我们多留一份 MessageDone 是给内核自己和未来的 HTTP 调试口用的。
                pass

            elif isinstance(event, ErrorEvent):
                # 没有 error 这个 stop_reason，把错误内容送进 thought 通道，
                # 客户端一般暗色渲染，用户能看到又不污染正文。
                self._log(f"[agentd] 执行出错: {event.message}")
                await self._conn.session_update(
                    session_id, update_agent_thought_text(f"[错误] {event.message}")
                )

            elif isinstance(event, ToolCallStart):
                await self._conn.session_update(
                    session_id,
                    start_tool_call(
                        event.call_id,
                        event.title,
                        kind=_ACP_KIND.get(event.kind, "other"),
                        status="in_progress",
                    ),
                )

            elif isinstance(event, ToolCallDone):
                await self._conn.session_update(
                    session_id,
                    update_tool_call(
                        event.call_id,
                        status=_ACP_STATUS.get(event.status, "completed"),
                        raw_output=event.output,
                    ),
                )

            elif isinstance(event, Done):
                stop_reason = _STOP_REASON_MAP.get(event.stop_reason, "end_turn")

            else:
                self._log(f"[agentd] 暂未映射的事件类型: {event.type}")

        return PromptResponse(stop_reason=stop_reason)

    def _extract_text(self, blocks: list[Any]) -> str:
        parts: list[str] = []
        for block in blocks:
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
            else:
                self._log(f"[agentd] 忽略非文本 block: {type(block).__name__}")
        joined = "".join(parts).strip()
        if not joined:
            raise ValueError("prompt 里没有任何文本内容")
        return joined


async def serve(kernel: AgentKernel | None = None) -> None:
    """启动 ACP agent，一直跑到 stdin 关闭。"""
    from acp import run_agent

    if kernel is None:
        from ..boot import build_kernel

        kernel = build_kernel()

    await run_agent(AgentdAcpAgent(kernel))
