"""LLM 抽象层。

把"怎么调一个模型"收口到一层，内核只认 LLM 接口。两个原生后端
（OllamaNativeLLM / OpenAICompatLLM）用 httpx 直连，不依赖 openai 包。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .models import Message


class LLMError(RuntimeError):
    """LLM 调用失败（网络/协议/模型错误）统一成这个异常。"""


class LLM:
    """内核只认这个接口。子类实现 stream()，complete() 有默认实现基于 stream 拼接。"""

    async def stream(self, message: list[Message], *, system: str | None = None) -> AsyncIterator[str]:
        raise NotImplementedError

    async def complete(self, message: list[Message], *, system: str | None = None) -> str:
        parts: list[str] = []
        async for chunk in self.stream(message, system=system):
            parts.append(chunk)
        return "".join(parts)


def _with_system(messages: list[Message], system: str | None) -> list[dict]:
    """把 system 塞到最前；剔除 tool 消息里 name=None，避免发给 OpenAI 出现 "name": null。"""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        item: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name is not None:
            item["name"] = m.name
        out.append(item)
    return out


class OllamaNativeLLM(LLM):
    """直连 Ollama 的 /api/chat（原生协议，不用 openai 包）。"""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3",
        think: bool = False,
        options: dict | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.think = think
        self.options = options

    def _payload(self, messages: list[Message], system: str | None) -> dict:
        body = {
            "model": self.model,
            "messages": _with_system(messages, system),
            "stream": True,
            "think": self.think,
        }
        # think 是 Ollama 原生参数；闭源模型没有就忽略即可
        if self.options:
            body["options"] = self.options
        return body

    async def stream(self, messages: list[Message], *, system: str | None = None) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream(
                "POST", f"{self.host}/api/chat", json=self._payload(messages, system)
            ) as resp:
                # Ollama 出错时 HTTP 状态码仍是 200，错误藏在 body 里，得读出来判断
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise LLMError(str(obj["error"]))
                    text = (obj.get("message") or {}).get("content") or ""
                    if text:
                        yield text
                    if obj.get("done"):
                        return


class OpenAICompatLLM(LLM):
    """打 OpenAI /v1/chat/completions 兼容端点（vLLM / LiteLLM / 本地网关都行）。"""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen3", api_key: str = "ollama") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _payload(self, messages: list[Message], system: str | None) -> dict:
        return {
            "model": self.model,
            "messages": _with_system(messages, system),
            "stream": True,
        }

    async def stream(self, messages: list[Message], *, system: str | None = None) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=self._payload(messages, system),
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                # 兼容端点是 SSE（逐行 `data: {...}`），遇 [DONE] 收尾
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    obj = json.loads(data)
                    if obj.get("error"):
                        raise LLMError(str(obj["error"]))
                    delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        yield text


class FakeLLM(LLM):
    """固定回复，仅供测试和本地联调。"""

    def __init__(self, reply: str = "这是 FakeLLM 的固定回复", chunk_size: int = 8) -> None:
        self.reply = reply
        self.chunk_size = chunk_size

    async def stream(self, messages: list[Message], *, system: str | None = None) -> AsyncIterator[str]:
        text = self.reply
        for i in range(0, len(text), self.chunk_size):
            yield text[i: i + self.chunk_size]
