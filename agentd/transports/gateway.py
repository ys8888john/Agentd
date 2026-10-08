"""Gateway 入口：HTTP + SSE 网关，与 ACP / CLI 功能等价。

把同一个 AgentKernel 暴露成 127.0.0.1 上的一小撮 HTTP 接口，事件流用 SSE 推。
和另外两个入口共用 :mod:`agentd.transports.runtime` 的会话语义（模式 / 取消 /
审批记忆），所以从任一入口进去行为一致。

路由（统一前缀 /v1）::

    POST /v1/sessions                     新会话      {session_id, modes}
    POST /v1/sessions/load                续聊        {session_id} （或 404）
    POST /v1/sessions/{sid}/prompt        发消息      {text, mode?} → 后台跑，事件进 SSE
    POST /v1/sessions/{sid}/cancel        叫停本轮    {cancelled}
    POST /v1/sessions/{sid}/mode          切模式      {mode_id}
    POST /v1/sessions/{sid}/permission    回审批      {request_id, option_id}
    GET  /v1/sessions/{sid}/history       历史        {history: [...]}
    GET  /v1/sessions/{sid}/stream        SSE 事件流  （长连接，多轮复用）
    GET  /v1/modes                        可用模式

设计要点：
- **SSE 必须是 GET**（浏览器 EventSource 只支持 GET），所以"发消息"用 POST 起
  一轮后台生成、事件推进 SSE；一条 stream 覆盖整个会话的多轮对话。
- **审批是反向的**：内核 handle 里要审批时，经 channel 把 permission_request
  推进 SSE，客户端 POST 回 option_id 才继续 —— 与 GUI 的 /api/permission 同构。
  "问不到人"（连接断了）一律按拒绝，绝不放行。
- 只绑 127.0.0.1；每条请求一条连接（Connection: close），换确定性。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..kernel.kernel import AgentKernel, UnknownModeError
from ..kernel.store import UnknownSessionError
from .runtime import ALLOW_ONCE, ALLOW_SESSION, REJECT, SessionRuntime

_REASON = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed",
           409: "Conflict", 500: "Internal Server Error"}

_SSE_HEAD = (
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream; charset=utf-8\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    "X-Accel-Buffering: no\r\n"
    "\r\n"
).encode("latin-1")


def _sse(event_type: str, data: str) -> bytes:
    return f"event: {event_type}\ndata: {data}\n\n".encode("utf-8")


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


async def _send_json(writer: asyncio.StreamWriter, status: int, obj: Any) -> None:
    body = _json_bytes(obj)
    head = (
        f"HTTP/1.1 {status} {_REASON.get(status, 'OK')}\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("latin-1")
    writer.write(head + body)
    await writer.drain()


class _Channel:
    """一个会话的事件广播通道：后台 prompt 往里推，SSE 单消费者取走。"""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.pending: dict[str, asyncio.Future] = {}  # request_id -> Future[option_id]
        self.streaming = False

    async def push(self, item: Any) -> None:
        await self.queue.put(item)

    async def ask(self, req: Any) -> str:
        """内核要审批时走这里：把请求推进 SSE，等客户端 POST 回来。"""
        request_id = f"perm_{uuid.uuid4().hex[:12]}"
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = fut
        await self.push({
            "type": "permission_request",
            "request_id": request_id,
            "tool": req.tool,
            "title": req.title,
            "kind": req.kind,
            "detail": req.detail,
            "options": [ALLOW_ONCE, ALLOW_SESSION, REJECT],
        })
        try:
            return await fut
        finally:
            self.pending.pop(request_id, None)


class Gateway:
    """把 AgentKernel 暴露成 HTTP + SSE 网关。"""

    def __init__(self, kernel: AgentKernel | None = None, *, host: str = "127.0.0.1",
                 port: int = 0, mcp_servers: list[Any] | None = None) -> None:
        if kernel is None:
            from ..boot import build_kernel

            kernel = build_kernel()
        self.runtime = SessionRuntime(kernel)
        self.host = host
        self._requested_port = port
        self.mcp_servers = list(mcp_servers or [])
        self._channels: dict[str, _Channel] = {}
        self._tasks: set[asyncio.Task] = set()

    def channel(self, sid: str) -> _Channel:
        ch = self._channels.get(sid)
        if ch is None:
            ch = _Channel()
            self._channels[sid] = ch
        return ch

    # ---- 后台跑一轮 ----

    async def _run_prompt(self, sid: str, text: str, mode: str | None) -> None:
        ch = self.channel(sid)
        try:
            async for event in self.runtime.run(sid, text, mode=mode, ask=ch.ask):
                await ch.push(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 前置校验等，统一转成流内 error
            await ch.push({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    def _spawn_prompt(self, sid: str, text: str, mode: str | None) -> None:
        task = asyncio.create_task(self._run_prompt(sid, text, mode))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ---- HTTP 连接 ----

    async def handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            method, target, body = await self._read_request(reader)
            if method is None:
                return
            await self._dispatch(method, target, body, reader, writer)
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as exc:  # noqa: BLE001 - 单连接不该拖垮整个服务
            try:
                await _send_json(writer, 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    async def _read_request(reader: asyncio.StreamReader) -> tuple[str | None, str, bytes]:
        line = await reader.readline()
        if not line:
            return None, "", b""
        try:
            method, target, _ = line.decode("latin-1").split()
        except ValueError:
            return None, "", b""
        headers: dict[str, str] = {}
        while True:
            raw = await reader.readline()
            if raw in (b"\r\n", b"\n", b""):
                break
            k, _, v = raw.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length") or 0)
        body = await reader.readexactly(n) if n else b""
        return method, target, body

    def _json_body(self, body: bytes) -> dict:
        if not body:
            return {}
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}

    # ---- 路由 ----

    async def _dispatch(self, method: str, target: str, body: bytes,
                        reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        u = urlparse(target)
        seg = [s for s in u.path.split("/") if s]
        parse_qs(u.query)  # query 预留（当前无必填 query）

        if not seg or seg[0] != "v1":
            return await _send_json(writer, 404, {"ok": False, "error": f"没有这个路径: {u.path}"})
        rest = seg[1:]

        # GET /v1/modes
        if rest == ["modes"]:
            if method != "GET":
                return await _send_json(writer, 405, {"ok": False, "error": "用 GET"})
            return await _send_json(writer, 200, {"ok": True, "modes": self.runtime.modes()})

        # POST /v1/sessions  （新会话）
        if rest == ["sessions"]:
            if method != "POST":
                return await _send_json(writer, 405, {"ok": False, "error": "用 POST"})
            return await self._new_session(self._json_body(body), writer)

        # POST /v1/sessions/load  （续聊）
        if rest == ["sessions", "load"]:
            if method != "POST":
                return await _send_json(writer, 405, {"ok": False, "error": "用 POST"})
            return await self._load_session(self._json_body(body), writer)

        # /v1/sessions/{sid}/{action}
        if len(rest) == 3 and rest[0] == "sessions":
            sid, action = rest[1], rest[2]
            if action == "stream" and method == "GET":
                return await self._stream(sid, reader, writer)
            if action == "history" and method == "GET":
                return await self._history(sid, writer)
            if method != "POST":
                return await _send_json(writer, 405, {"ok": False, "error": "用 POST"})
            data = self._json_body(body)
            if action == "prompt":
                return await self._prompt(sid, data, writer)
            if action == "cancel":
                return await _send_json(writer, 200, {"ok": True, "cancelled": self.runtime.cancel(sid)})
            if action == "mode":
                return await self._set_mode(sid, data, writer)
            if action == "permission":
                return await self._permission(sid, data, writer)

        return await _send_json(writer, 404, {"ok": False, "error": f"没有这个接口: {u.path}"})

    # ---- 各路由实现 ----

    async def _new_session(self, data: dict, writer: asyncio.StreamWriter) -> None:
        mcp = data.get("mcp_servers")
        mcp_list = mcp if isinstance(mcp, list) else self.mcp_servers
        sid = await self.runtime.kernel.create_session(
            cwd=data.get("cwd"),
            mcp_servers=mcp_list,
            additional_directories=list(data.get("additional_directories") or []),
        )
        self.channel(sid)  # 预建 channel，stream 时不必再等
        await _send_json(writer, 200, {"ok": True, "session_id": sid, "modes": self.runtime.modes()})

    async def _load_session(self, data: dict, writer: asyncio.StreamWriter) -> None:
        sid = str(data.get("session_id") or "")
        if not sid:
            return await _send_json(writer, 400, {"ok": False, "error": "缺少 session_id"})
        mcp = data.get("mcp_servers")
        mcp_list = mcp if isinstance(mcp, list) else self.mcp_servers
        adopted = await self.runtime.kernel.adopt_session(
            sid,
            cwd=data.get("cwd"),
            mcp_servers=mcp_list,
            additional_directories=list(data.get("additional_directories") or []),
        )
        if not adopted:
            return await _send_json(writer, 404, {"ok": False, "error": f"会话不存在: {sid}"})
        self.channel(sid)
        await _send_json(writer, 200, {"ok": True, "session_id": sid, "modes": self.runtime.modes()})

    async def _prompt(self, sid: str, data: dict, writer: asyncio.StreamWriter) -> None:
        text = str(data.get("text") or "").strip()
        if not text:
            return await _send_json(writer, 400, {"ok": False, "error": "缺少 text"})
        mode = data.get("mode") or self.runtime.mode_of(sid)
        # 先校验：这样会话不存在 / 模式未注册能返回正常的 HTTP 错误码，
        # 而不是变成流里一条看不懂的 error。
        try:
            await self.runtime.kernel.validate(sid, str(mode))
        except UnknownSessionError:
            return await _send_json(writer, 404, {"ok": False, "error": f"会话不存在: {sid}"})
        except UnknownModeError as exc:
            return await _send_json(writer, 400, {"ok": False, "error": f"未知模式: {exc}"})
        self._spawn_prompt(sid, text, str(mode))
        await _send_json(writer, 200, {"ok": True, "mode": mode})

    async def _set_mode(self, sid: str, data: dict, writer: asyncio.StreamWriter) -> None:
        mode_id = str(data.get("mode_id") or "")
        modes = self.runtime.modes()
        if mode_id not in modes:
            return await _send_json(writer, 400, {"ok": False, "error": f"未知模式: {mode_id}"})
        self.runtime.set_mode(sid, mode_id)
        await _send_json(writer, 200, {"ok": True, "mode_id": mode_id, "modes": modes})

    async def _permission(self, sid: str, data: dict, writer: asyncio.StreamWriter) -> None:
        rid = str(data.get("request_id") or "")
        option_id = str(data.get("option_id") or "")
        ch = self._channels.get(sid)
        fut = ch.pending.get(rid) if ch else None
        if fut is None or fut.done():
            return await _send_json(writer, 404, {"ok": False, "error": f"没有这个待审批请求: {rid}"})
        # 拿不准就当拒绝：只认三个已知 option_id。
        if option_id not in (ALLOW_ONCE, ALLOW_SESSION, REJECT):
            option_id = REJECT
        fut.set_result(option_id)
        await _send_json(writer, 200, {"ok": True, "request_id": rid, "option_id": option_id})

    async def _history(self, sid: str, writer: asyncio.StreamWriter) -> None:
        try:
            hist = await self.runtime.kernel.history(sid)
        except UnknownSessionError:
            return await _send_json(writer, 404, {"ok": False, "error": f"会话不存在: {sid}"})
        await _send_json(writer, 200, {"ok": True, "history": [m.model_dump() for m in hist]})

    async def _stream(self, sid: str, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        ch = self.channel(sid)
        if ch.streaming:
            return await _send_json(writer, 409, {"ok": False, "error": "该会话已有 SSE 连接"})
        ch.streaming = True

        disconnected: asyncio.Event = asyncio.Event()

        async def _watch() -> None:
            # SSE 是单向的：客户端不会发数据，它一断开 read 就返回 EOF。
            try:
                while await reader.read(4096):
                    pass
            except Exception:  # noqa: BLE001
                pass
            disconnected.set()

        watcher = asyncio.create_task(_watch())
        try:
            writer.write(_SSE_HEAD)
            await writer.drain()
            while not disconnected.is_set():
                getter = asyncio.create_task(ch.queue.get())
                done, _ = await asyncio.wait(
                    {getter, watcher}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter not in done:
                    getter.cancel()
                    break  # 客户端断了
                item = getter.result()
                # queue 里混着两类东西：内核的 Event（pydantic 模型）和
                # gateway 自己的控制消息（permission_request / error，dict）。
                # **不能用 isinstance(item, Event)** —— contracts.Event 是
                # Annotated[Union[...]] 的 TypeAlias，对它 isinstance 会抛
                # "Subscripted generics cannot be used with class and instance
                # checks"，第一帧就炸、客户端一个字节都收不到。按 dict 分流。
                if isinstance(item, dict):
                    writer.write(_sse(item["type"], json.dumps(item, ensure_ascii=False)))
                else:
                    writer.write(_sse(item.type, item.model_dump_json()))
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            ch.streaming = False
            watcher.cancel()
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass


async def serve(
    kernel: AgentKernel | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    mcp_servers: list[Any] | None = None,
) -> None:
    """启动 gateway，一直跑到进程被终止。"""
    gw = Gateway(kernel, host=host, port=port, mcp_servers=mcp_servers)
    server = await asyncio.start_server(gw.handle_conn, gw.host, gw._requested_port)
    addr = server.sockets[0].getsockname()
    print(f"[agentd] gateway 监听 http://{addr[0]}:{addr[1]}  （SSE: /v1/sessions/<id>/stream）",
          flush=True)
    async with server:
        await server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    """console_scripts / ``python -m agentd.gateway`` 入口。"""
    import argparse

    parser = argparse.ArgumentParser(prog="agentd-gateway", description="agentd HTTP+SSE 网关")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认只绑本机）")
    parser.add_argument("--port", type=int, default=0, help="监听端口，0=由内核随机分配")
    args = parser.parse_args(argv)
    asyncio.run(serve(host=args.host, port=args.port))