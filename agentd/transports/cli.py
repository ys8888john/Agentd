"""CLI 入口：终端里直接对话，与 ACP / Gateway 功能等价。

和另外两个入口共用 :mod:`agentd.transports.runtime` 的会话语义（模式 / 取消 /
审批记忆）。工具审批在终端上 y/a/n 提示，选项 id 与 ACP、Gateway 同一套。

命令（行首 ``:``）::

    :quit/:q  退出          :cancel   叫停当前生成的一轮
    :new      开新会话      :load <id> 续聊
    :mode <m> 切模式        :modes    列出模式
    :history  打印历史      :help     帮助

设计取舍：一轮跑完才问下一句（不并发读输入），这样审批提示不会和主输入抢终端；
想打断正在生成的那一轮按 Ctrl+C —— 捕获后只递停止信号给内核，由它在下一个
chunk/工具边界收尾（Done.stop_reason="cancelled"）。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..contracts import (
    Done,
    ErrorEvent,
    MessageDelta,
    MessageDone,
    Notice,
    ThoughtDelta,
    ToolCallDone,
    ToolCallStart,
)
from ..kernel.kernel import AgentKernel
from .runtime import ALLOW_ONCE, ALLOW_SESSION, REJECT, SessionRuntime

_HELP = """\
可用命令：
  :quit / :q      退出
  :cancel         叫停当前生成的一轮（或按 Ctrl+C）
  :new            开一个新会话
  :load <id>      续聊指定会话
  :mode <name>    切换会话模式
  :modes          列出可用模式
  :history        打印当前会话历史
  :help           显示本帮助
直接输入文字即可发送。"""


class CliRunner:
    def __init__(self, kernel: AgentKernel, *, cwd: str | None = None,
                 mcp_servers: list[Any] | None = None, mode: str | None = None) -> None:
        self.runtime = SessionRuntime(kernel)
        self.cwd = cwd
        self.mcp_servers = list(mcp_servers or [])
        self.mode = mode
        self.sid: str | None = None

    async def _new_session(self) -> None:
        self.sid = await self.runtime.kernel.create_session(cwd=self.cwd, mcp_servers=self.mcp_servers)
        if self.mode:
            self.runtime.set_mode(self.sid, self.mode)

    # ---- 发一轮 ----

    async def _ask(self, req: Any) -> str:
        """审批提示。Ctrl+C / 空回车都按拒绝 —— 拿不准时必须往拒绝倒。"""
        prompt = (
            f"\n  [需要审批] {req.title}（{req.tool}）"
            f"{'：' + req.detail if req.detail else ''}\n"
            "  y=允许一次  a=本会话总是允许  n/回车=拒绝 > "
        )
        try:
            ans = (await asyncio.to_thread(input, prompt)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return REJECT
        if ans in ("y", "yes"):
            return ALLOW_ONCE
        if ans in ("a", "all"):
            return ALLOW_SESSION
        return REJECT  # n / no / 其它 / 空

    async def _turn(self, text: str) -> None:
        assert self.sid is not None
        async for event in self.runtime.run(self.sid, text, ask=self._ask):
            if isinstance(event, MessageDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ThoughtDelta):
                print(f"[思考] {event.text}", end="", flush=True)
            elif isinstance(event, Notice):
                # 换行而不是接着正文后头：它说的是"系统干了什么"，混进回答里读不清
                print(f"\n[提示] {event.text}", flush=True)
            elif isinstance(event, MessageDone):
                pass  # 正文靠 MessageDelta 逐字打印，Done 只负责收尾换行
            elif isinstance(event, ToolCallStart):
                print(f"\n  [工具] {event.title}（{event.kind}）…", flush=True)
            elif isinstance(event, ToolCallDone):
                head = f"  → {event.status}"
                if event.output:
                    head += f"：{event.output.splitlines()[0][:80]}"
                print(head, flush=True)
            elif isinstance(event, ErrorEvent):
                print(f"\n[错误] {event.message}", flush=True)
            elif isinstance(event, Done):
                tail = {"cancelled": "[已停止]", "error": "[出错]"}.get(event.stop_reason, "")
                print(tail, flush=True)
        print()

    # ---- 命令 ----

    async def _cmd(self, line: str) -> bool:
        """处理一条 : 命令。返回 True 表示要退出主循环。"""
        assert self.sid is not None
        parts = line[1:].split()
        name = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        if name in ("quit", "q", "exit"):
            return True
        if name == "help":
            print(_HELP)
        elif name == "cancel":
            print("已递停止信号" if self.runtime.cancel(self.sid) else "当前没有进行中的轮次")
        elif name == "new":
            await self._new_session()
            print(f"新会话 {self.sid}")
        elif name == "load":
            if not arg:
                print("用法：:load <session_id>")
            else:
                ok = await self.runtime.kernel.adopt_session(arg, cwd=self.cwd, mcp_servers=self.mcp_servers)
                if ok:
                    self.sid = arg
                    print(f"已续聊 {arg}")
                else:
                    print(f"会话不存在：{arg}")
        elif name == "mode":
            if not arg:
                print("用法：:mode <name>；可用：", ", ".join(self.runtime.modes()))
            elif arg not in self.runtime.modes():
                print(f"未知模式：{arg}")
            else:
                self.runtime.set_mode(self.sid, arg)
                print(f"模式切换为 {arg}（下一轮生效）")
        elif name == "modes":
            print("可用模式：", ", ".join(self.runtime.modes()), "  当前：", self.runtime.mode_of(self.sid))
        elif name == "history":
            for m in await self.runtime.kernel.history(self.sid):
                print(f"  [{m.role}] {m.content}")
        else:
            print(f"未知命令：:{name}（:help 看帮助）")
        return False

    # ---- 主循环 ----

    async def run(self) -> None:
        await self._new_session()
        modes = ", ".join(self.runtime.modes())
        print(f"[agentd] CLI 会话 {self.sid}  cwd={self.cwd}  "
              f"模式 [{self.runtime.mode_of(self.sid)}]（可用：{modes}）")
        print("输入 :help 看命令，Ctrl+C 打断当前生成，:quit 退出。\n")
        while True:
            try:
                line = await asyncio.to_thread(input, "你> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                if await self._cmd(line):
                    break
                continue
            task = asyncio.create_task(self._turn(line))
            try:
                await task
            except (KeyboardInterrupt, asyncio.CancelledError):
                # 打断当前轮次：只递停止信号，工具循环会在下个边界收尾。
                self.runtime.cancel(self.sid)
                try:
                    await task
                except BaseException:  # noqa: BLE001 - 收尾路径不该再抛
                    pass
        print("再见。")


async def serve(kernel: AgentKernel | None = None, *, cwd: str | None = None,
                mcp_servers: list[Any] | None = None, mode: str | None = None) -> None:
    if kernel is None:
        from ..boot import build_kernel

        kernel = build_kernel()
    await CliRunner(kernel, cwd=cwd, mcp_servers=mcp_servers, mode=mode).run()


def main(argv: list[str] | None = None) -> None:
    """console_scripts / ``python -m agentd.transports.cli`` 入口。"""
    import argparse

    parser = argparse.ArgumentParser(prog="agentd-cli", description="agentd 终端入口")
    parser.add_argument("--cwd", default=None, help="会话工作目录（工具的根）")
    parser.add_argument("--mode", default=None, help="初始会话模式（agent / single）")
    args = parser.parse_args(argv)
    asyncio.run(serve(cwd=args.cwd, mode=args.mode))