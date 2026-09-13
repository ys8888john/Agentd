"""会话历史存储。第一步用内存实现，接口按将来换 SQLite 设计。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Message


class UnknownSessionError(KeyError):
    """会话 ID 不存在。继承 KeyError 让 str(e) 直接给出 session_id。"""


class SessionStore(ABC):
    @abstractmethod
    async def create(self, session_id: str) -> None: ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool: ...

    @abstractmethod
    async def append(self, session_id: str, message: Message) -> None: ...

    @abstractmethod
    async def history(self, session_id: str) -> list[Message]: ...

    @abstractmethod
    async def clear(self, session_id: str) -> None: ...


class InMemorySessionStore(SessionStore):
    """进程内存储，重启即丢。history() 返回的是内部列表，调用方不要原地修改。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[Message]] = {}

    async def create(self, session_id: str) -> None:
        self._sessions.setdefault(session_id, [])

    async def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def append(self, session_id: str, message: Message) -> None:
        if not await self.exists(session_id):
            raise UnknownSessionError(session_id)
        self._sessions[session_id].append(message)

    async def history(self, session_id: str) -> list[Message]:
        if not await self.exists(session_id):
            raise UnknownSessionError(session_id)
        return self._sessions[session_id]

    async def clear(self, session_id: str) -> None:
        if not await self.exists(session_id):
            raise UnknownSessionError(session_id)
        self._sessions[session_id] = []
