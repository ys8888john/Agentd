"""三个入口共享的会话运行时。

ACP(stdio) / Gateway(HTTP+SSE) / CLI(终端) 问用户的方式完全不同：ACP 走
``session/request_permission`` 反向请求，Gateway 把审批请求推进 SSE 流等客户端
POST 回来，CLI 直接在终端上提示。但下面三件事的语义必须**逐条一致**，否则同一份
配置从不同入口进去行为会漂：

    - 这个会话当前是什么模式（覆盖值，下一轮 prompt 生效）
    - 要不要把正在跑的一轮停下来
    - 哪些工具"本会话总是允许"（审批记忆）

所以把它们收在这里，三个入口共用一份实现（``run()`` 是那条主干）。

**审批记忆刻意留在传输层而不是内核**：它是「客户端怎么问用户」的一部分，
换个入口记忆策略本可以不一样；但既然要求三入口一致，就统一按
"本会话总是允许"记（与 ACP 的既有行为一致）。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from ..contracts import Event
from ..kernel.kernel import AgentKernel
from ..kernel.tools import ApprovalRequest, ApproveHandler

# 审批选项 id：ACP 的 option_id、CLI 的按键、Gateway 的回传值共用同一套字面量。
ALLOW_ONCE = "allow_once"
ALLOW_SESSION = "allow_session"
REJECT = "reject"
_ALLOW_IDS = frozenset({ALLOW_ONCE, ALLOW_SESSION})

# 默认模式。与 ACP 一致：有工具时走 agent 工具循环，没工具时与 single 等价。
DEFAULT_MODE = "agent"

# 问用户：给一次 ApprovalRequest，返回选中的 option_id。
# 拿不准 / 出错时**必须**返回 REJECT —— 反过来就是安全漏洞（详见 acp_stdio 同款注释）。
AskUser = Callable[[ApprovalRequest], Awaitable[str]]


class SessionRuntime:
    """会话级运行时状态：模式覆盖 / 取消信号 / 审批记忆 + 跑一轮的主干。"""

    def __init__(self, kernel: AgentKernel) -> None:
        self.kernel = kernel
        self._modes: dict[str, str] = {}
        self._cancels: dict[str, asyncio.Event] = {}
        self._allow_all: dict[str, set[str]] = {}

    # ---- 模式 ----

    def mode_of(self, session_id: str) -> str:
        return self._modes.get(session_id, DEFAULT_MODE)

    def set_mode(self, session_id: str, mode_id: str) -> None:
        self._modes[session_id] = mode_id

    def modes(self) -> list[str]:
        return self.kernel.modes()

    # ---- 取消 ----

    def cancel(self, session_id: str) -> bool:
        """叫停本会话正在跑的一轮。没有进行中的轮次时是 no-op（返回 False）。

        真中断发生在内核里：停止信号沿 chunk / 工具边界检查，当前这段 LLM 流或
        工具执行走完就收尾（Done.stop_reason="cancelled"）。这里只负责"递信号"。
        """
        event = self._cancels.get(session_id)
        if event is None or event.is_set():
            return False
        event.set()
        return True

    # ---- 审批记忆 ----

    def granted(self, session_id: str, tool: str) -> bool:
        return tool in self._allow_all.setdefault(session_id, set())

    def remember(self, session_id: str, tool: str) -> None:
        self._allow_all.setdefault(session_id, set()).add(tool)

    def approver(self, session_id: str, ask: AskUser | None) -> ApproveHandler | None:
        """造审批回调交给内核。ask=None 表示"没人可问"（内核按放行处理）。

        ACP 那边客户端可能压根没实现 ``session/request_permission``；Gateway/CLI
        也可能因为连接断了问不到人。所以这里三件事一起做：先查本会话记忆 →
        问用户 → 任何异常按拒绝 → 结果原样译成 bool。
        """

        if ask is None:
            return None  # 没人可问（CI / 纯放行）：内核按放行处理，不装门

        async def approve(req: ApprovalRequest) -> bool:
            if self.granted(session_id, req.tool):
                return True  # 之前选过"本会话总是允许"，不再打扰用户
            try:
                option_id = await ask(req)
            except Exception:  # noqa: BLE001 - 问不到就当拒绝，绝不放行
                return False
            if option_id == ALLOW_SESSION:
                self.remember(session_id, req.tool)
            return option_id in _ALLOW_IDS

        return approve

    # ---- 跑一轮 ----

    async def run(
        self,
        session_id: str,
        user_input: str,
        *,
        mode: str | None = None,
        ask: AskUser | None = None,
    ) -> AsyncIterator[Event]:
        """执行一轮对话，产出事件流 —— 三个入口共用的那条主干。

        约定与内核一致：末尾必发 Done；失败先 ErrorEvent 再 Done(error)。
        这里额外负责"取消信号的摘除"：无论正常结束还是抛异常，下一轮必须是干净
        的 —— 旧轮次的 cancel 残留会把（并发到的）下一轮误停。
        """
        m = mode or self.mode_of(session_id)
        # 先校验：handle() 是异步生成器，代码要等首次 __anext__ 才跑，届时
        # 响应头已发出 / 控制台已开动，无法再改成错误状态，故必须先调。
        await self.kernel.validate(session_id, m)

        cancel = asyncio.Event()
        self._cancels[session_id] = cancel
        try:
            async for event in self.kernel.handle(
                session_id,
                user_input,
                mode=m,
                approve=self.approver(session_id, ask),
                cancel=cancel,
            ):
                yield event
        finally:
            if self._cancels.get(session_id) is cancel:
                self._cancels.pop(session_id, None)