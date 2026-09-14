"""模式（Mode）的接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..llm import LLM
from ..models import Message
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


class Mode(ABC):
    # ClassVar 而不是实例属性：模式是无状态的，name 属于"类"不属于"对象"
    name: ClassVar[str] = ""

    @abstractmethod
    async def run(self, ctx: ModeContext, user_input: str) -> AsyncIterator[Event]: ...
