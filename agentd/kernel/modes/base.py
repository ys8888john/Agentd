"""模式（Mode）的接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..llm import LLM
from ..models import Message
from ..tools import ApprovalRequest, ApproveHandler, NativeToolbox
from ...contracts import Event


@dataclass
class ModeContext:
    """模式执行时拿到的全部上下文。

    故意不包含 store —— 持久化由内核统一负责，
    否则每种模式都要重复写一遍"把消息存起来"，而且容易漏。
    """

    session_id: str
    run_id: str
    llm: LLM
    history: list[Message]
    system: str | None = None
    # 工具调用用：本会话声明的 MCP server 配置（原样是 ACP 传进来的结构），
    # 以及 agent 工作目录（stdio server 的相对路径/CWD 兜底）。
    mcp_servers: list[Any] = field(default_factory=list)
    cwd: str | None = None
    # 进程内原生工具（read_file / glob / grep / write_file / edit / run_command）。
    # 由内核按会话 cwd 造好传进来；None 表示本会话禁用原生工具。
    toolbox: NativeToolbox | None = None
    # 审批回调。**故意做成回调而不是内核事件**：审批是"请求-应答"，
    # 而 Event 流是单向生成器，答案没法从流里回灌。传输层（ACP 的
    # session/request_permission、HTTP 的 SSE+POST）各自实现这个回调，
    # 内核只认这个签名，于是内核依旧不认识任何传输协议。
    # None = 无人可问 → 直接放行（TUI / 单测 / 无人值守）。
    approve: ApproveHandler | None = None
    # 审批策略：native（默认，只拦原生写/执行类）| all（非只读全拦）| none（全放行）
    approval_policy: str = "native"

    async def request_approval(self, req: ApprovalRequest) -> bool:
        if self.approve is None:
            return True
        try:
            return bool(await self.approve(req))
        except Exception:  # noqa: BLE001 - 审批通道坏了不该让整轮任务失败
            return False


class Mode(ABC):
    # ClassVar 而不是实例属性：模式是无状态的，name 属于"类"不属于"对象"
    name: ClassVar[str] = ""

    @abstractmethod
    async def run(self, ctx: ModeContext, user_input: str) -> AsyncIterator[Event]: ...
