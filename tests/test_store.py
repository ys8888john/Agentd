"""存储测试：同一套契约断言对 InMemory / SQLite 各跑一遍，再加 SQLite 专有的持久化测试。

契约测试存在的意义：换存储后端是迟早的事（SQLite → Postgres → 远端），
"两个实现行为一致"这件事不能靠人记着，得由测试钉死。
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import FakeLLM
from agentd.kernel.models import Message
from agentd.kernel.store import (
    InMemorySessionStore,
    SqliteSessionStore,
    UnknownSessionError,
    default_db_path,
)


# ---------------------------------------------------------------- 契约测试


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        s: object = InMemorySessionStore()
    else:
        s = SqliteSessionStore(tmp_path / "sessions.db")
    yield s
    close = getattr(s, "close", None)
    if close is not None:
        close()


async def test_new_session_has_no_history(store):
    await store.create("s1")
    assert await store.history("s1") == []


async def test_exists_is_false_before_create(store):
    assert await store.exists("nope") is False
    await store.create("nope")
    assert await store.exists("nope") is True


async def test_append_roundtrip(store):
    await store.create("s1")
    await store.append("s1", Message.user("你好"))
    await store.append("s1", Message.assistant("在的"))

    history = await store.history("s1")
    assert [(m.role, m.content) for m in history] == [
        ("user", "你好"),
        ("assistant", "在的"),
    ]


async def test_history_preserves_insertion_order(store):
    await store.create("s1")
    for i in range(10):
        await store.append("s1", Message.user(f"第{i}条"))

    assert [m.content for m in await store.history("s1")] == [f"第{i}条" for i in range(10)]


async def test_tool_message_keeps_name(store):
    await store.create("s1")
    await store.append("s1", Message(role="tool", content="42", name="calculator"))

    m = (await store.history("s1"))[0]
    assert (m.role, m.name, m.content) == ("tool", "calculator", "42")


async def test_unicode_and_newlines_survive(store):
    text = "第一行\n第二行\t带 tab 🥟 emoji 与 '引号' \"双引\""
    await store.create("s1")
    await store.append("s1", Message.user(text))

    assert (await store.history("s1"))[0].content == text


async def test_clear_empties_history_but_keeps_session(store):
    await store.create("s1")
    await store.append("s1", Message.user("x"))
    await store.clear("s1")

    assert await store.history("s1") == []
    assert await store.exists("s1") is True


@pytest.mark.parametrize("op", ["append", "history", "clear"])
async def test_unknown_session_raises(store, op):
    """三个方法对不存在的会话都要炸，而且是同一个 KeyError 子类。"""
    call = {
        "append": lambda: store.append("ghost", Message.user("x")),
        "history": lambda: store.history("ghost"),
        "clear": lambda: store.clear("ghost"),
    }[op]

    with pytest.raises(UnknownSessionError):
        await call()


async def test_create_is_idempotent(store):
    """重复 create 不能把已有历史清掉 —— 语义是"确保存在"不是"重建"。"""
    await store.create("s1")
    await store.append("s1", Message.user("第一条"))
    await store.create("s1")
    await store.create("s1")

    assert [m.content for m in await store.history("s1")] == ["第一条"]


async def test_list_sessions(store):
    assert await store.list_sessions() == []

    for sid in ("s3", "s1", "s2"):
        await store.create(sid)

    assert sorted(await store.list_sessions()) == ["s1", "s2", "s3"]


async def test_sessions_are_isolated(store):
    await store.create("a")
    await store.create("b")
    await store.append("a", Message.user("只给 a"))

    assert [m.content for m in await store.history("a")] == ["只给 a"]
    assert await store.history("b") == []


# ---------------------------------------------------------------- SQLite 专有


def test_default_db_path_lives_in_home():
    p = default_db_path()
    assert p.parent == p.home() / ".agentd"
    assert p.name == "sessions.db"


async def test_sqlite_creates_db_file_on_disk(tmp_path):
    db = tmp_path / "nested" / "sessions.db"
    store = SqliteSessionStore(db)
    await store.create("s1")
    store.close()

    assert db.is_file()


async def test_data_survives_reopen(tmp_path):
    """持久化的全部意义就在这一条：关掉再开，上次的对话还在。"""
    db = tmp_path / "sessions.db"

    first = SqliteSessionStore(db)
    await first.create("s1")
    await first.append("s1", Message.user("你记住我了吗"))
    await first.append("s1", Message.assistant("记住了"))
    first.close()

    second = SqliteSessionStore(db)
    try:
        assert await second.exists("s1") is True
        assert [(m.role, m.content) for m in await second.history("s1")] == [
            ("user", "你记住我了吗"),
            ("assistant", "记住了"),
        ]
        assert await second.list_sessions() == ["s1"]
    finally:
        second.close()


async def test_payload_columns_are_queryable_by_sql(tmp_path):
    """role/content/name 冗余列的意义：不写 Python 也能看库。"""
    db = tmp_path / "sessions.db"
    store = SqliteSessionStore(db)
    await store.create("s1")
    await store.append("s1", Message.user("你好"))
    await store.append("s1", Message(role="tool", content="42", name="calculator"))
    store.close()

    raw = sqlite3.connect(db)
    try:
        rows = raw.execute(
            "SELECT role, content, name FROM messages ORDER BY seq"
        ).fetchall()
    finally:
        raw.close()

    assert rows == [("user", "你好", None), ("tool", "42", "calculator")]


async def test_clear_does_not_reuse_seq(tmp_path):
    """AUTOINCREMENT 守护的就是这条：清完再加，新消息必须排在后面而不是复用旧 rowid。"""
    db = tmp_path / "sessions.db"
    store = SqliteSessionStore(db)

    def seqs() -> list[int]:
        raw = sqlite3.connect(db)
        try:
            return [r[0] for r in raw.execute("SELECT seq FROM messages ORDER BY seq")]
        finally:
            raw.close()

    await store.create("s1")
    await store.append("s1", Message.user("旧"))
    before = seqs()

    await store.clear("s1")
    await store.append("s1", Message.user("新"))
    after = seqs()
    assert [m.content for m in await store.history("s1")] == ["新"]
    store.close()

    assert len(after) == 1
    # 关键的不是数值，而是"新 seq 严格大于被删掉那条"—— 复用 rowid 会让 ORDER BY seq 乱序
    assert after[0] > before[0]


async def test_concurrent_appends_lose_nothing(tmp_path):
    """单连接 + 锁：并发写不能丢消息、不能抛 'database is locked'。"""
    store = SqliteSessionStore(tmp_path / "sessions.db")
    await store.create("s1")
    await asyncio.gather(
        *[store.append("s1", Message.user(f"m{i}")) for i in range(50)]
    )

    history = await store.history("s1")
    store.close()

    assert len(history) == 50
    assert {m.content for m in history} == {f"m{i}" for i in range(50)}


async def test_deleting_session_cascades_messages(tmp_path):
    db = tmp_path / "sessions.db"
    store = SqliteSessionStore(db)
    await store.create("s1")
    await store.append("s1", Message.user("x"))
    store.close()

    raw = sqlite3.connect(db)
    try:
        raw.execute("PRAGMA foreign_keys=ON")
        raw.execute("DELETE FROM sessions WHERE id = 's1'")
        raw.commit()
        left = raw.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        raw.close()

    assert left == 0


async def test_in_memory_path_never_touches_disk(tmp_path):
    store = SqliteSessionStore(":memory:")
    await store.create("s1")
    await store.append("s1", Message.user("只在内存"))
    store.close()

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- 内核端到端


async def test_kernel_reply_is_persisted_and_readable_after_restart(tmp_path):
    """真正的验收：聊一句 → 换一个内核（模拟重启）→ 历史里要有 user 和 assistant。"""
    db = tmp_path / "sessions.db"

    k1 = AgentKernel(llm=FakeLLM(reply="我记住了"), store=SqliteSessionStore(db))
    sid = await k1.create_session()
    async for _ in k1.handle(sid, "记住这句话"):
        pass
    k1.store.close()  # type: ignore[attr-defined]

    k2 = AgentKernel(llm=FakeLLM(reply="随便"), store=SqliteSessionStore(db))
    try:
        history = await k2.history(sid)
        assert [(m.role, m.content) for m in history] == [
            ("user", "记住这句话"),
            ("assistant", "我记住了"),
        ]
        # 第二个内核还能接着同一个会话聊，说明会话确实是"活"的
        await k2.validate(sid, "single")
    finally:
        k2.store.close()  # type: ignore[attr-defined]


async def test_kernel_on_memory_store_still_works():
    """InMemory 不能被改坏 —— 单测和 fake 后端都靠它。"""
    k = AgentKernel(llm=FakeLLM(reply="ok"), store=InMemorySessionStore())
    sid = await k.create_session()
    async for _ in k.handle(sid, "hi"):
        pass

    assert [m.role for m in await k.history(sid)] == ["user", "assistant"]
