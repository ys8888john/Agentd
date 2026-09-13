"""ACP(stdio) 端到端测试：起真实子进程验证 JSON-RPC 帧、会话链路与错误转译。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from acp import AGENT_METHODS, PROTOCOL_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPLY = "端到端测试回复"


def _method(name: str) -> str:
    """从 SDK 的常量表里取 wire 方法名，不硬编码字符串。"""
    if isinstance(AGENT_METHODS, dict):
        return AGENT_METHODS[name]
    return getattr(AGENT_METHODS, name)


def _collect_text(node: Any) -> str:
    """从任意嵌套结构里捞出所有 text 字段。

    不依赖 AgentMessageChunk 的确切形状 —— SDK 一升级测试就得跟着改，那样太脆。
    """
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "text" and isinstance(v, str):
                out.append(v)
            else:
                out.append(_collect_text(v))
    elif isinstance(node, (list, tuple)):
        for v in node:
            out.append(_collect_text(v))
    elif hasattr(node, "model_dump"):
        out.append(_collect_text(node.model_dump(exclude_none=True)))
    return "".join(out)


class _Proc:
    """包一层，省得每个测试都写一遍清理逻辑。"""

    def __init__(self) -> None:
        self.proc: asyncio.subprocess.Process | None = None

    async def start(self) -> asyncio.subprocess.Process:
        env = dict(os.environ, AGENTD_LLM_BACKEND="fake", AGENTD_FAKE_REPLY=REPLY)
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "agentd.server",
            cwd=PROJECT_ROOT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        assert self.proc.stdin and self.proc.stdout
        return self.proc

    async def call(
        self, rid: int, method: str, params: dict
    ) -> tuple[dict, list[dict]]:
        """发一帧请求，收集途经的所有通知，返回 (响应, 通知列表)。"""
        assert self.proc and self.proc.stdin and self.proc.stdout
        frame = json.dumps(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        )
        self.proc.stdin.write(frame.encode() + b"\n")
        await self.proc.stdin.drain()

        notifications: list[dict] = []
        while True:
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=30)
            assert line, "子进程关闭了 stdout（多半是崩了，看 stderr）"
            # 关键断言：stdout 上必须是纯 JSON-RPC，任何一个 print() 都会让这行炸
            msg = json.loads(line)
            if msg.get("id") == rid:
                return msg, notifications
            notifications.append(msg)

    async def drain_stderr(self) -> str:
        if not self.proc or not self.proc.stderr:
            return ""
        try:
            data = await asyncio.wait_for(self.proc.stderr.read(65536), timeout=2)
        except TimeoutError:
            return ""
        return data.decode("utf-8", "replace")

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()


@pytest.fixture
async def agent():
    p = _Proc()
    await p.start()
    yield p
    await p.stop()


async def test_handshake(agent: _Proc):
    resp, _ = await agent.call(
        1, _method("initialize"), {"protocolVersion": PROTOCOL_VERSION}
    )
    assert "error" not in resp, resp
    assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION


async def test_full_prompt_flow(agent: _Proc):
    _, _ = await agent.call(
        1, _method("initialize"), {"protocolVersion": PROTOCOL_VERSION}
    )

    resp, _ = await agent.call(
        2, _method("new_session"), {"cwd": str(PROJECT_ROOT), "mcpServers": []}
    )
    assert "error" not in resp, resp
    session_id = resp["result"]["sessionId"]
    assert session_id

    resp, updates = await agent.call(
        3,
        _method("prompt"),
        {"sessionId": session_id, "prompt": [{"type": "text", "text": "你好"}]},
    )
    assert "error" not in resp, resp
    assert resp["result"]["stopReason"] == "end_turn"

    # 流式的 session/update 通知里应当能拼出完整回复
    assert _collect_text(updates) == REPLY


async def test_unknown_session_is_reported_as_error(agent: _Proc):
    await agent.call(1, _method("initialize"), {"protocolVersion": PROTOCOL_VERSION})

    resp, _ = await agent.call(
        2,
        _method("prompt"),
        {"sessionId": "sess_not_exist", "prompt": [{"type": "text", "text": "x"}]},
    )
    # 内核抛 UnknownSessionError，SDK 应当转成 JSON-RPC error，而不是静默成功
    assert "error" in resp, resp
