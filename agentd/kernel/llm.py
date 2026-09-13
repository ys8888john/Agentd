"""LLM 抽象层。

目标：把"怎么调一个模型"这件事收口到一层，内核只认 LLM 接口，不关心背后是
Ollama / OpenAI / 本地 fake。发往 LLM 的 messages 走 contracts.Message 那套
（role/content/...），协议细节由这里包掉。

设计点：
- 流式优先。run() 是一个 async generator，逐块吐 (Event, Message|None)。
  Message|None 是"这一步积攒下来的完整回复"——工具调用那步是 None，纯文本步才有值。
- 非流式（FakeLLM 这种）用 send() 一个方法，内部直接 build 好最终 Message。
- 两个"原生"后端（OllamaNativeLLM / OpenAICompatLLM）不依赖 openai 包，
  用 httpx 直连，避免给轻量内核塞重依赖。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, AsyncIterator as _AI  # noqa: F401

import httpx

from ..contracts import (
    Done,
    ErrorEvent,
    Event,
    MessageDelta,
    MessageDone,
    ToolCallDone,
    ToolCallStart,
    new_call_id,
    new_run_id,
)
from .models import Message, Role


@dataclass
class StreamChunk:
    """流式过程中吐出的一块：要么是事件，要么是"这一步积攒的回复快照"。"""

    event: Event | None = None
    message: Message | None = None


class LLM:
    """内核只认这个接口。子类实现 run()，提供 send() 兜底。"""

    async def run(
        self, system: str | None, history: list[Message], prompt: str
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError

    async def send(self, system: str | None, history: list[Message], prompt: str) -> Message:
        raise NotImplementedError


class OllamaNativeLLM(LLM):
    """直连 Ollama 的 /api/chat（原生协议，不用 openai 包）。"""

    def __init__(self, host: str, model: str, think: bool = False) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.think = think

    def _payload(self, system, history, prompt, stream):
        messages = [
            {"role": m.role, "content": m.content}
            | ({"name": m.name} if m.name else {})
            for m in history
        ]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # think 是 Ollama 原生参数；闭源模型没有就把这开关忽略即可
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": self.think,
        }

    async def run(self, system, history, prompt) -> _AI[StreamChunk]:
        run_id = new_run_id()
        sid = history[0].content if history else ""
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream(
                "POST", f"{self.host}/api/chat", json=self._payload(system, history, prompt, True)
            ) as resp:
                # Ollama 出错时 HTTP 状态码仍是 200，错误藏在 body 里，得读出来判断
                buffer = ""
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    buffer += line
                    try:
                        obj = json.loads(buffer)
                    except json.JSONDecodeError:
                        continue
                    buffer = ""
                    if obj.get("error"):
                        yield StreamChunk(event=ErrorEvent(session_id=sid, run_id=run_id, message=str(obj["error"])))
                        yield StreamChunk(event=Done(session_id=sid, run_id=run_id, stop_reason="error"))
                        return
                    msg = obj.get("message") or {}
                    text = msg.get("content") or ""
                    if text:
                        yield StreamChunk(event=MessageDelta(session_id=sid, run_id=run_id, text=text))
                    if obj.get("done"):
                        full = msg.get("content") or ""
                        yield StreamChunk(event=MessageDone(session_id=sid, run_id=run_id, text=full), message=Message(role="assistant", content=full))
                        yield StreamChunk(event=Done(session_id=sid, run_id=run_id, stop_reason="end_turn"))
                        return

    async def send(self, system, history, prompt) -> Message:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(
                f"{self.host}/api/chat", json=self._payload(system, history, prompt, False)
            )
            obj = resp.json()
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            return Message(role="assistant", content=obj["message"]["content"])


class OpenAICompatLLM(LLM):
    """打 OpenAI /v1/chat/completions 兼容端点（vLLM / LiteLLM / 本地网关都行）。"""

    def __init__(self, base_url: str, model: str, api_key: str = "ollama") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _payload(self, system, history, prompt, stream):
        messages = [
            {"role": m.role, "content": m.content}
            | ({"name": m.name} if m.name else {})
            for m in history
        ]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return {"model": self.model, "messages": messages, "stream": stream}

    async def run(self, system, history, prompt) -> _AI[StreamChunk]:
        run_id = new_run_id()
        sid = history[0].content if history else ""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions",
                json=self._payload(system, history, prompt, True), headers=headers,
            ) as resp:
                # 兼容端点是 SSE 不是纯 NDJSON，逐行读、只对 data: 行做 json.loads
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        yield StreamChunk(event=Done(session_id=sid, run_id=run_id, stop_reason="end_turn"))
                        return
                    obj = json.loads(data)
                    if obj.get("error"):
                        yield StreamChunk(event=ErrorEvent(session_id=sid, run_id=run_id, message=str(obj["error"])))
                        yield StreamChunk(event=Done(session_id=sid, run_id=run_id, stop_reason="error"))
                        return
                    choice = obj["choices"][0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        yield StreamChunk(event=MessageDelta(session_id=sid, run_id=run_id, text=text))
                    if choice.get("finish_reason"):
                        full = text
                        yield StreamChunk(event=MessageDone(session_id=sid, run_id=run_id, text=full), message=Message(role="assistant", content=full))

    async def send(self, system, history, prompt) -> Message:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=self._payload(system, history, prompt, False), headers=headers,
            )
            obj = resp.json()
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            return Message(role="assistant", content=obj["choices"][0]["message"]["content"])


class FakeLLM(LLM):
    """固定回复，仅供测试和本地联调。"""

    def __init__(self, reply: str = "这是 FakeLLM 的固定回复。") -> None:
        self.reply = reply

    async def run(self, system, history, prompt):
        run_id = new_run_id()
        sid = history[0].content if history else ""
        yield StreamChunk(event=MessageDelta(session_id=sid, run_id=run_id, text=self.reply))
        yield StreamChunk(event=MessageDone(session_id=sid, run_id=run_id, text=self.reply), message=Message(role="assistant", content=self.reply))
        yield StreamChunk(event=Done(session_id=sid, run_id=run_id, stop_reason="end_turn"))

    async def send(self, system, history, prompt) -> Message:
        return Message(role="assistant", content=self.reply)
