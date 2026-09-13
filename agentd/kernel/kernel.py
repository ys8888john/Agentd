"""内核入口：把会话、存储、模式编排缝在一起，对外只暴露 Event 流。

传输层（HTTP/SSE、ACP）只跟 handle() 打交道。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..contracts import (
    Done,
    ErrorEvent,
    Event,
    MessageDone,
    new_run_id,
    new_session_id,
)
from .llm import LLM
from .models import Message
from .modes import Mode, ModeContext, SingleMode
from .store import InMemorySessionStore, SessionStore, UnknownSessionError

class UnknownModeError(KeyError):
    """请求的模式未注册。"""


@dataclass
class AgentKernel:
    llm: LLM
    store: SessionStore = field(default_factory=InMemorySessionStore)
    system: str | None = None

    def __post_init__(self) -> None:
        self._modes: dict[str, Mode] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.register(SingleMode()) # 第一步只有这一种

    # ---- 模式注册 ----

    def register(self, mode: Mode) -> None:
        if not mode.name:
            raise ValueError("Mode.name 不能为空")
        self._modes[mode.name] = mode

    def modes(self) -> list[str]:
        return sorted(self._modes)

    def _get_mode(self, name: str) -> Mode:
        try:
            return self._modes[name]
        except KeyError as exc:
            raise UnknownModeError(name) from exc

    # ---- 会话 ----

    async def create_session(self) -> str:
        session_id = new_session_id()
        await self.store.create(session_id)
        return session_id

    async def history(self, session_id: str) -> list[Message]:
        return await self.store.history(session_id)

    async def validate(self, session_id: str, mode: str) -> None:
        """提前校验参数，失败抛 UnknownSessionError / UnknownModeError。

        存在的理由：handle() 是异步生成器，函数体要等到第一次 __anext__ 才执行，
        那时 HTTP 响应头已经发出去了，没法再改成 404。传输层必须先调这个方法。
        """
        if not await self.store.exists(session_id):
            raise UnknownSessionError(session_id)
        self._get_mode(mode)
    
    # ---- 执行 ----

    async def handle(
        self,
        session_id: str,
        user_input: str,
        *,
        mode: str = "single",
    ) -> AsyncIterator[Event]:
        """执行一次对话，产出事件流。

        约定：
        1. 无论成功失败，最后一个事件一定是 Done；
        2. 失败时先发 ErrorEvent 再发 Done(stop_reason="error")；
        3. 同一个 session 的多次 handle 串行执行（history 是有序的，并行会写乱）。
        """

        if not await self.store.exists(session_id):
            raise UnknownSessionError(session_id)
        impl = self._get_mode(mode)

        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            run_id = new_run_id()

            # 用户消息先落库，这样传给模式的 history 里已经包含本轮输入
            await self.store.append(session_id, Message.user(user_input))
            history = await self.store.history(session_id)

            ctx = ModeContext(
                session_id=session_id,
                run_id=run_id,
                llm=self.llm,
                history=history,
                system=self.system,
            )

            try:
                async for event in impl.run(ctx, user_input):
                    # 持久化由内核统一负责：模式只管产事件
                    if isinstance(event, MessageDone):
                        await self.store.append(session_id, Message.assistant(event.text))
                    yield event
            except Exception as exc:  # noqa: BLE001 - 边界处统一转成事件
                yield ErrorEvent(
                    session_id=session_id, run_id=run_id, message=f"{type(exc).__name__}: {exc}"
                )
                yield Done(session_id=session_id, run_id=run_id, stop_reason="error")
                return

            yield Done(session_id=session_id, run_id=run_id, stop_reason="end_turn")