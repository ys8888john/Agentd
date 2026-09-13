"""会话库查看器：python scripts/sessions.py list | show <id> | clear <id>

持久化了却看不见，等于没持久化。这个脚本就是把"记忆"翻出来给人看的，
也是排查"上次聊的天到底存没存进去"的第一站。

它只碰 SQLite 库、不启 LLM，所以跑起来是秒开、也不会联网。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentd.kernel.store import SqliteSessionStore, default_db_path  # noqa: E402

PREVIEW = 400  # 单条消息默认预览长度，超了要 --full


def _preview(text: str, full: bool) -> str:
    body = text if full else text[:PREVIEW]
    if not full and len(text) > PREVIEW:
        body += f"  …（共 {len(text)} 字，--full 看全部）"
    return body.replace("\r\n", "\n")


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.split("\n"))


async def cmd_list(store: SqliteSessionStore) -> None:
    ids = await store.list_sessions()
    if not ids:
        print(f"（空）{store.path} 里还没有任何会话")
        return

    print(f"{store.path}   共 {len(ids)} 个会话\n")
    for sid in ids:
        history = await store.history(sid)
        last = history[-1].content.split("\n")[0][:40] if history else "（无消息）"
        print(f"  {sid}   {len(history):>3} 条   {last}")
    print("\n  看内容：python scripts/sessions.py show <会话ID>")


async def cmd_show(store: SqliteSessionStore, session_id: str, full: bool) -> None:
    history = await store.history(session_id)
    print(f"会话 {session_id}   共 {len(history)} 条   库：{store.path}\n")
    for i, m in enumerate(history, 1):
        tag = f"{m.role}:{m.name}" if m.name else m.role
        print(f"[{i}] {tag}")
        print(_indent(_preview(m.content, full)))
        print()


async def cmd_clear(store: SqliteSessionStore, session_id: str) -> None:
    before = len(await store.history(session_id))
    await store.clear(session_id)
    print(f"已清空 {session_id}（原 {before} 条）。会话本身还在，可继续往里聊。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="查看 / 清理 agentd 的 SQLite 会话库")
    p.add_argument(
        "--db",
        default=str(default_db_path()),
        help=f"会话库路径（默认 {default_db_path()}）",
    )

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="列出所有会话")

    show = sub.add_parser("show", help="打印某个会话的完整历史")
    show.add_argument("session_id")
    show.add_argument("--full", action="store_true", help="不截断长消息")

    clear = sub.add_parser("clear", help="清空某个会话的消息")
    clear.add_argument("session_id")
    return p


async def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    path = Path(args.db)
    if not path.is_file():
        print(f"找不到会话库：{path}\n（还没用 SQLite 存储跑过 agentd？默认位置见 --db）")
        raise SystemExit(1)

    store = SqliteSessionStore(path)
    try:
        if args.cmd == "list":
            await cmd_list(store)
        elif args.cmd == "show":
            await cmd_show(store, args.session_id, args.full)
        elif args.cmd == "clear":
            await cmd_clear(store, args.session_id)
    finally:
        store.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except KeyError as exc:
        # UnknownSessionError 是 KeyError，str(e) 就是 session_id
        print(f"没有这个会话：{exc}")
        raise SystemExit(1) from None
