"""本地联调脚本：手动发一条 prompt 给 Agentd，通过真实 stdio ACP 看流式回复。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# 方法名优先从 SDK 常量表取，取不到再退回字面量
try:
    from acp import AGENT_METHODS, PROTOCOL_VERSION

    def _method(name: str, fallback: str) -> str:
        try:
            return AGENT_METHODS[name] if isinstance(AGENT_METHODS, dict) else getattr(AGENT_METHODS, name)
        except Exception:
            return fallback

    M_INIT = _method("initialize", "initialize")
    M_NEW = _method("new_session", "session/new")
    M_PROMPT = _method("prompt", "session/prompt")
except Exception:  # SDK 没装也能跑，只是方法名用默认值
    PROTOCOL_VERSION = 1
    M_INIT, M_NEW, M_PROMPT = "initialize", "session/new", "session/prompt"


def collect_text(node: Any) -> str:
    """从任意嵌套结构里捞所有 text 字段 —— 不依赖 SDK 的 model 形状。"""
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "text" and isinstance(v, str):
                out.append(v)
            else:
                out.append(collect_text(v))
    elif isinstance(node, (list, tuple)):
        for v in node:
            out.append(collect_text(v))
    elif hasattr(node, "model_dump"):
        out.append(collect_text(node.model_dump(exclude_none=True)))
    return "".join(out)


class Agent:
    """把手写 JSON-RPC 收发包一层。"""

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self.session_id = ""

    async def start(self) -> None:
        env = dict(os.environ)
        env.setdefault("AGENTD_LLM_BACKEND", "fake")
        # stderr 不重定向 —— 让 agentd 的日志直接打到当前终端，方便排错
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "agentd.server",
            cwd=ROOT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=env,
        )
        assert self.proc.stdin and self.proc.stdout
        await self._call(M_INIT, {"protocolVersion": PROTOCOL_VERSION})
        resp = await self._call(M_NEW, {"cwd": str(ROOT), "mcpServers": []})
        self.session_id = resp["result"]["sessionId"]

    async def _call(self, method: str, params: dict) -> dict:
        assert self.proc and self.proc.stdin and self.proc.stdout
        self._next_id += 1
        rid = self._next_id
        frame = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        self.proc.stdin.write(frame.encode() + b"\n")
        await self.proc.stdin.drain()

        while True:
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=300)
            if not line:
                raise RuntimeError("agentd 关闭了 stdout —— 多半崩了，看上面 stderr 的日志")
            msg = json.loads(line)  # stdout 必须是纯 JSON
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"agentd 返回错误: {json.dumps(msg['error'], ensure_ascii=False)}")
                return msg
            # 不是这次的响应 → 当成流式通知，直接打印
            text = collect_text(msg.get("params", {}))
            if text:
                print(text, end="", flush=True)

    async def ask(self, text: str) -> str:
        resp = await self._call(
            M_PROMPT,
            {"sessionId": self.session_id, "prompt": [{"type": "text", "text": text}]},
        )
        return resp["result"].get("stopReason", "?")

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()


async def main() -> None:
    agent = Agent()
    await agent.start()
    print(f"agentd 已连接  会话 {agent.session_id}", flush=True)
    print("直接回车退出。\n", flush=True)

    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            text = line.strip()
            if not text:
                break

            print("\033[36m你\033[0m  ", text, flush=True)
            print("\033[32magentd\033[0m  ", end="", flush=True)
            stop = await agent.ask(text)
            print(f"\n  [stopReason={stop}]\n", flush=True)
    finally:
        await agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
