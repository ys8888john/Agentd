"""ACP over stdio 传输适配器 —— 主线出口。

职责单一：把 kernel.handle() 产出的 Event 翻译成 ACP session/update 通知，
把客户端发来的 session/prompt 翻译成内核的一次 handle()。

三条硬约束（违反即协议错误，客户端会静默卡死或报解析失败）：
1. stdout 上只能有 JSON-RPC 帧。任何 print() / logging.StreamHandler 都会污染协议，
   日志一律走 stderr。这条不靠自觉，靠 tests/test_acp.py 断言 stdout 每行都能 JSON 解析。
2. 路径必须是绝对路径；行号 1-based。
3. 判断能力用 initialize 协商回来的 protocolVersion，不能看 SDK 版本号
   （本 SDK 是 0.12.1，wire version 是 1，两者不对应）。
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
    update_agent_message_text,
    update_agent_thought_text,
)

from ..contracts import Done, ErrorEvent, MessageDelta, MessageDone
from ..kernel.kernel import AgentKernel

# ACP 的 stop_reason 取值是固定的五种，其中没有 "error"：
#   end_turn | max_tokens | max_turn_requests | refusal | cancelled
# 所以内核里的 Done(stop_reason="error") 不能直译，见 prompt() 里的处理。
_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "cancelled": "cancelled",
    # 执行出错：错误内容已经通过 thought 通道告知客户端，这里按正常收尾
    "error": "end_turn",
}


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

    @staticmethod
    def _log(msg: str) -> None:
        """唯一允许的日志出口 —— 必须是 stderr。"""
        print(msg, file=sys.stderr, flush=True)

    async def on_connect(self, conn: Any) -> None:
        # SDK 建立连接后回调，拿到 conn 才能反向调用 session_update
        self._conn = conn
        self._log("[agentd] ACP 连接已建立")

    # ---- 生命周期 ----

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
        session_id = await self._kernel.create_session()
        self._log(f"[agentd] 新会话 {session_id}  cwd={cwd}")
        if mcp_servers:
            self._log(f"[agentd] 收到 {len(mcp_servers)} 个 MCP server，暂未接入")
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

    # ---- 核心 ----

    async def prompt(
        self, session_id: str, prompt: list[Any], **kwargs: Any
    ) -> PromptResponse:
        text = self._extract_text(prompt)
        mode = self._session_modes.get(session_id, "single")

        # 先校验：这里抛出的异常会被 SDK 转成 JSON-RPC error，
        # 比开流之后再失败干净得多。
        await self._kernel.validate(session_id, mode)

        stop_reason = "end_turn"

        async for event in self._kernel.handle(session_id, text, mode=mode):
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

            elif isinstance(event, Done):
                stop_reason = _STOP_REASON_MAP.get(event.stop_reason, "end_turn")

            else:
                # 工具调用类事件还没映射 —— 现在没有任何模式产出它们
                self._log(f"[agentd] 暂未映射的事件类型: {event.type}")

        return PromptResponse(stop_reason=stop_reason)

    # ---- 辅助 ----

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
