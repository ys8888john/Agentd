"""统一入口：``python -m agentd [acp|gateway|cli]``

三个入口共用同一个 AgentKernel，能力等价（建会话 / 续聊 / 流式对话 / 模式 /
取消 / 工具审批 / 历史），差别只在"怎么和外面说话"：

    acp       （默认）ACP over stdio —— 等客户端用 JSON-RPC 说话（Zed、ForgeAgent-GUI 等）
    gateway   HTTP + SSE 网关 —— 浏览器 / curl / 任何能读 SSE 的客户端
    cli       终端交互 —— 直接在终端里聊天

``acp`` 保持默认、不带子命令，是为了不打断既有的
``python -m agentd.server`` / ``FORGEAGENT_AGENT_CMD`` 用法。
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    sub = args[0] if args else "acp"

    if sub == "acp":
        from .server import main as acp_main

        acp_main()
    elif sub == "gateway":
        from .transports.gateway import main as gateway_main

        gateway_main(args[1:])
    elif sub == "cli":
        from .transports.cli import main as cli_main

        cli_main(args[1:])
    else:
        print(f"未知子命令 {sub!r}；可用：acp | gateway | cli", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()