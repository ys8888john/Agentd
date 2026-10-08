"""Gateway 入口（HTTP + SSE）端到端测试。

起真的 asyncio HTTP 服务，用 httpx 打真的请求、读真的 SSE。重点验证与
ACP / CLI 对齐的那条能力：建会话、发消息、SSE 流式事件、工具审批的反向闭环
（permission_request 推进 SSE、客户端 POST 回答案）、模式、历史。

不依赖 Ollama：全部用假 LLM。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.store import InMemorySessionStore
from agentd.transports.gateway import Gateway


class ListLLM(LLM):
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    async def stream_events(self, messages, *, system=None, tools=None):
        for t in self._texts:
            yield LLMText(t)


async def _serve(kernel: AgentKernel):
    gw = Gateway(kernel)
    server = await asyncio.start_server(gw.handle_conn, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return gw, server, f"http://127.0.0.1:{port}"


async def _read_sse(client: httpx.AsyncClient, sid: str, on_event) -> None:
    """读一条 SSE 流；on_event(data) 返回 False 时停止。"""
    async with client.stream("GET", f"/v1/sessions/{sid}/stream") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:].strip())
            if on_event(data) is False:
                return


async def test_gateway_prompt_over_sse(tmp_path) -> None:
    kernel = AgentKernel(llm=ListLLM(["你", "好"]), store=InMemorySessionStore())
    gw, server, base = await _serve(kernel)
    async with server:
        async with httpx.AsyncClient(base_url=base, timeout=10) as c:
            r = await c.post("/v1/sessions", json={"cwd": str(tmp_path)})
            assert r.status_code == 200
            sid = r.json()["session_id"]
            assert "agent" in r.json()["modes"]

            events: list[dict] = []

            def on_event(d):
                events.append(d)
                return d.get("type") != "done"  # 读到 done 停

            rt = asyncio.create_task(_read_sse(c, sid, on_event))
            await asyncio.sleep(0.02)  # 等 SSE 连上

            pr = await c.post(f"/v1/sessions/{sid}/prompt", json={"text": "你好"})
            assert pr.status_code == 200
            await asyncio.wait_for(rt, timeout=5)

    types = [e["type"] for e in events]
    assert types.count("message_delta") == 2
    assert types[-1] == "done"
    joined = "".join(e.get("text", "") for e in events if e["type"] == "message_delta")
    assert joined == "你好"
    assert events[-1]["stop_reason"] == "end_turn"


async def test_gateway_history_after_prompt(tmp_path) -> None:
    kernel = AgentKernel(llm=ListLLM(["答"]), store=InMemorySessionStore())
    gw, server, base = await _serve(kernel)
    async with server:
        async with httpx.AsyncClient(base_url=base, timeout=10) as c:
            sid = (await c.post("/v1/sessions", json={"cwd": str(tmp_path)})).json()["session_id"]
            events: list[dict] = []

            def on_event(d):
                events.append(d)
                return d.get("type") != "done"

            rt = asyncio.create_task(_read_sse(c, sid, on_event))
            await asyncio.sleep(0.02)
            await c.post(f"/v1/sessions/{sid}/prompt", json={"text": "你好"})
            await asyncio.wait_for(rt, timeout=5)

            hist = (await c.get(f"/v1/sessions/{sid}/history")).json()["history"]
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[0]["content"] == "你好"
    assert hist[1]["content"] == "答"


async def test_gateway_approval_roundtrip(tmp_path) -> None:
    """审批反向闭环：SSE 推 permission_request → 客户端 POST 回答案 → 工具执行 → done。"""

    class ToolLLM(LLM):
        def __init__(self) -> None:
            self.calls = 0

        async def stream_events(self, messages, *, system=None, tools=None):
            self.calls += 1
            if self.calls == 1:
                yield LLMToolCall(id="c1", name="write_file",
                                  arguments='{"path":"a.txt","content":"x"}')
            else:
                yield LLMText("收尾")

    kernel = AgentKernel(llm=ToolLLM(), store=InMemorySessionStore())
    gw, server, base = await _serve(kernel)
    async with server:
        async with httpx.AsyncClient(base_url=base, timeout=10) as c:
            sid = (await c.post("/v1/sessions", json={"cwd": str(tmp_path)})).json()["session_id"]
            events: list[dict] = []
            got_perm = asyncio.Event()
            rid_holder: dict[str, str] = {}

            def on_event(d):
                events.append(d)
                if d.get("type") == "permission_request":
                    rid_holder["rid"] = d["request_id"]
                    got_perm.set()
                return d.get("type") != "done"

            rt = asyncio.create_task(_read_sse(c, sid, on_event))
            await asyncio.sleep(0.02)
            await c.post(f"/v1/sessions/{sid}/prompt", json={"text": "写个文件", "mode": "agent"})

            # 内核因 write_file 触发审批 → SSE 推出 permission_request
            await asyncio.wait_for(got_perm.wait(), timeout=5)
            ans = await c.post(
                f"/v1/sessions/{sid}/permission",
                json={"request_id": rid_holder["rid"], "option_id": "allow_once"},
            )
            assert ans.status_code == 200
            await asyncio.wait_for(rt, timeout=5)

    types = [e["type"] for e in events]
    assert "permission_request" in types
    assert "tool_call_start" in types
    assert "tool_call_done" in types
    assert types[-1] == "done"
    # 允许之后文件真的落盘（审批→执行闭环）
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"


async def test_gateway_mode_and_modes_endpoints(tmp_path) -> None:
    kernel = AgentKernel(llm=ListLLM(["x"]), store=InMemorySessionStore())
    gw, server, base = await _serve(kernel)
    async with server:
        async with httpx.AsyncClient(base_url=base, timeout=10) as c:
            sid = (await c.post("/v1/sessions", json={})).json()["session_id"]
            assert (await c.get("/v1/modes")).json()["modes"] == ["agent", "single"]
            bad = await c.post(f"/v1/sessions/{sid}/mode", json={"mode_id": "nope"})
            assert bad.status_code == 400
            ok = await c.post(f"/v1/sessions/{sid}/mode", json={"mode_id": "single"})
            assert ok.status_code == 200
            assert gw.runtime.mode_of(sid) == "single"


async def test_gateway_prompt_unknown_session_404(tmp_path) -> None:
    kernel = AgentKernel(llm=ListLLM(["x"]), store=InMemorySessionStore())
    gw, server, base = await _serve(kernel)
    async with server:
        async with httpx.AsyncClient(base_url=base, timeout=10) as c:
            r = await c.post("/v1/sessions/sess_ghost/prompt", json={"text": "hi"})
            assert r.status_code == 404


async def test_gateway_cancel_endpoint(tmp_path) -> None:
    kernel = AgentKernel(llm=ListLLM(["x"]), store=InMemorySessionStore())
    gw, server, base = await _serve(kernel)
    async with server:
        async with httpx.AsyncClient(base_url=base, timeout=10) as c:
            sid = (await c.post("/v1/sessions", json={})).json()["session_id"]
            # 没有进行中的轮次：cancel 是无害 no-op，不报错
            r = await c.post(f"/v1/sessions/{sid}/cancel", json={})
            assert r.status_code == 200
            assert r.json()["cancelled"] is False


if __name__ == "__main__":
    pytest.main([__file__])