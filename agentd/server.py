"""入口：python -m agentd.server

跑起来后该进程就是一个 ACP agent，等待客户端通过 stdio 说话。
正常情况你在终端里什么都看不到 —— 它在等 JSON-RPC 帧，不是给人看的。
"""

from __future__ import annotations

import asyncio

from .transports.acp_stdio import serve


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
