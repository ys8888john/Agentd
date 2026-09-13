"""会话历史存储：内存版（重启即丢）与 SQLite 版（持久化）。

接口按"换后端不改调用方"设计：新增方法时两个实现一起加，
tests/test_store.py 用同一套契约测试对两个实现各跑一遍。

为什么要两个：
    InMemorySessionStore —— 单测、fake 后端、一次性脚本，不该往磁盘写东西。
    SqliteSessionStore   —— 真实运行，进程退出后还能接着上次的会话聊。
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

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

    @abstractmethod
    async def list_sessions(self) -> list[str]:
        """已有会话 ID，按创建时间升序。给"接着上次聊"的 UI 和运维脚本用。"""


class InMemorySessionStore(SessionStore):
    """进程内存储，重启即丢。"""

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
        # 返回副本而不是内部列表：SQLite 版必然是副本，
        # 两边行为一致才不会出现"换个后端就莫名被改到"的鬼故事。
        return list(self._sessions[session_id])

    async def clear(self, session_id: str) -> None:
        if not await self.exists(session_id):
            raise UnknownSessionError(session_id)
        self._sessions[session_id] = []

    async def list_sessions(self) -> list[str]:
        return list(self._sessions)


# ---- SQLite ----

DEFAULT_DB_FILENAME = "sessions.db"

# role/content/name 三列是**冗余**的：读的时候一律用 payload 反序列化。
# 它们存在的唯一理由是让人能用 sqlite3 直接看库、用 SQL 统计，
# 而不是每次都要写个 Python 脚本解 JSON。
# payload 才是权威数据 —— 将来给 Message 加字段不用迁移表。
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    -- AUTOINCREMENT 不能省：普通 INTEGER PRIMARY KEY 会复用被删掉的 rowid，
    -- clear() 之后再 append，新消息的 seq 可能比老的小，历史就乱序了。
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    name       TEXT,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages (session_id, seq);
"""


def default_db_path() -> Path:
    """默认库位置：~/.agentd/sessions.db。

    放用户目录而不是 CWD —— CWD 随启动方式变（VSCode / 终端 / 桌面图标各不同），
    放那儿会出现"从 VSCode 启动看不到从终端聊过的记录"这种灵异现象。
    """
    return Path.home() / ".agentd" / DEFAULT_DB_FILENAME


class SqliteSessionStore(SessionStore):
    """SQLite 持久化存储。path 传 ':memory:' 就是纯内存库（测试用）。

    为什么不用 aiosqlite：
        所有调用都经 asyncio.to_thread 丢进线程池，事件循环本来就不会被阻塞，
        为此多引一个依赖不划算。

    为什么是单连接 + threading.Lock：
        sqlite3 连接默认 check_same_thread=True，而 to_thread 每次可能落到不同线程。
        开 check_same_thread=False 再自己串行化，比"每线程一个连接"更可控 ——
        后者写同一库要处理 SQLITE_BUSY，单连接串行写根本没这问题。
    """

    def __init__(self, path: str | Path | None = None, *, busy_timeout_ms: int = 5000) -> None:
        self.path = ":memory:" if path is None else str(path)
        if self.path != ":memory:":
            parent = Path(self.path).parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        # WAL：读写不互相阻塞。synchronous=NORMAL：掉电最多丢最后一两个事务，
        # 对聊天记录这个量级的价值来说，换来的写入速度是值的。
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- 同步实现：全部在锁内跑，被 to_thread 包一层对外就是 async --

    def _call(self, fn, *args):
        """把一个同步操作串行化 —— 连接只有一条，绝不能并发用。"""
        with self._lock:
            return fn(*args)

    def _create_sync(self, session_id: str) -> None:
        with self._conn:  # 事务：异常自动回滚
            # INSERT OR IGNORE —— create 语义是"确保存在"，
            # 重复 create 不能把已有历史清掉（内存版用的 setdefault 也是这个语义）。
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
                (session_id, time.time()),
            )

    def _exists_sync(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def _append_sync(self, session_id: str, message: Message) -> None:
        if not self._exists_sync(session_id):
            raise UnknownSessionError(session_id)
        with self._conn:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, name, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    message.role,
                    message.content,
                    message.name,
                    message.model_dump_json(),
                    time.time(),
                ),
            )

    def _history_sync(self, session_id: str) -> list[Message]:
        if not self._exists_sync(session_id):
            raise UnknownSessionError(session_id)
        rows = self._conn.execute(
            "SELECT payload FROM messages WHERE session_id = ? ORDER BY seq",
            (session_id,),
        ).fetchall()
        return [Message.model_validate_json(row[0]) for row in rows]

    def _clear_sync(self, session_id: str) -> None:
        if not self._exists_sync(session_id):
            raise UnknownSessionError(session_id)
        with self._conn:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    def _list_sessions_sync(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT id FROM sessions ORDER BY created_at, id"
        ).fetchall()
        return [row[0] for row in rows]

    # -- SessionStore 接口 --

    async def create(self, session_id: str) -> None:
        await asyncio.to_thread(self._call, self._create_sync, session_id)

    async def exists(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._call, self._exists_sync, session_id)

    async def append(self, session_id: str, message: Message) -> None:
        await asyncio.to_thread(self._call, self._append_sync, session_id, message)

    async def history(self, session_id: str) -> list[Message]:
        return await asyncio.to_thread(self._call, self._history_sync, session_id)

    async def clear(self, session_id: str) -> None:
        await asyncio.to_thread(self._call, self._clear_sync, session_id)

    async def list_sessions(self) -> list[str]:
        return await asyncio.to_thread(self._call, self._list_sessions_sync)

    # -- 生命周期 --

    def close(self) -> None:
        """关连接。不关也能靠 GC 回收，但 WAL 下显式关能把 -wal 文件正常收尾。"""
        with self._lock:
            self._conn.close()
