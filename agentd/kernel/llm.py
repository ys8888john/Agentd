"""LLM 抽象层。

把"怎么调一个模型"收口到一层，内核只认 LLM 接口。两个原生后端
（OllamaNativeLLM / OpenAICompatLLM）用 httpx 直连，不依赖 openai 包。
"""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .models import Message

# 模型名写成这个值就表示"别让我猜，去问 Ollama 本机装了啥"。
# 存在的理由：写死默认模型名几乎必然在某台机器上 404（默认 qwen3，但这台装的是
# qwen3.5:9b-text），而 404 的表现是"回复一片空白"，极难自查。
AUTO = "auto"

# 自动挑选时的偏好顺序：按子串匹配，先命中先选。
DEFAULT_PREFER = "qwen3.5,qwen3,glm,deepseek,llama3,llama,gemma,mistral"


def _log(msg: str) -> None:
    """LLM 层的日志出口 —— 必须是 stderr。

    跟 acp_stdio.py 同一个约束：agentd 的 stdout 是纯 JSON-RPC 通道，
    往那儿打一个字，客户端就解不出帧了。
    """
    print(msg, file=sys.stderr, flush=True)


class LLMError(RuntimeError):
    """LLM 调用失败（网络/协议/模型错误）统一成这个异常。"""


@dataclass
class LLMText:
    """流式文本增量。"""

    text: str


@dataclass
class LLMToolCall:
    """模型发起的一次工具调用（arguments 是 JSON 字符串）。"""

    id: str
    name: str
    arguments: str = "{}"


class LLM:
    """内核只认这个接口。

    子类实现 `stream_events()`（可同时产出文本增量和工具调用）；`stream()` 是只取
    文本的便捷封装，老调用方（TUI / single 模式 / 测试）的用法和返回值都不变。
    """

    async def stream_events(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMText | LLMToolCall]:
        raise NotImplementedError

    async def stream(self, messages: list[Message], *, system: str | None = None) -> AsyncIterator[str]:
        async for event in self.stream_events(messages, system=system):
            if isinstance(event, LLMText) and event.text:
                yield event.text

    async def complete(self, messages: list[Message], *, system: str | None = None) -> str:
        parts: list[str] = []
        async for chunk in self.stream(messages, system=system):
            parts.append(chunk)
        return "".join(parts)


def _as_args_obj(arguments: str) -> Any:
    """把 arguments JSON 字符串还原成对象（Ollama 原生端点要对象，不要字符串）。"""
    try:
        return json.loads(arguments) if arguments else {}
    except (ValueError, TypeError):
        return {}


def _with_system(messages: list[Message], system: str | None, *, ollama: bool = False) -> list[dict]:
    """把 system 塞到最前，并按线格式序列化工具相关字段。

    - 默认（OpenAI 兼容）：assistant 带 tool_calls（id/type/function，arguments 为 JSON 字符串）；
      role="tool" 带 tool_call_id。
    - ollama=True：tool_calls 的 arguments 还原成对象、不带 id/type；工具结果用 tool_name。
    - name=None 一律不写，避免出现 "name": null。
    """
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        item: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.role == "assistant" and m.tool_calls:
            if ollama:
                item["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": _as_args_obj(tc.arguments)}}
                    for tc in m.tool_calls
                ]
            else:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in m.tool_calls
                ]
        if m.role == "tool":
            if ollama:
                if m.name:
                    item["tool_name"] = m.name
            else:
                if m.tool_call_id:
                    item["tool_call_id"] = m.tool_call_id
                if m.name:
                    item["name"] = m.name
        elif m.name is not None:
            item["name"] = m.name
        out.append(item)
    return out


def pick_model(names: list[str], prefer: str = DEFAULT_PREFER) -> str:
    """按偏好顺序从模型名列表里挑一个，全都不匹配就选第一个。

    纯函数，不碰网络 —— 这样"挑得对不对"可以单独测，不用起 Ollama。
    """
    if not names:
        raise LLMError("模型列表为空，无法自动挑选")
    lowered = [(name, name.lower()) for name in names]
    for token in (t.strip().lower() for t in prefer.split(",")):
        if not token:
            continue
        for name, low in lowered:
            if token in low:
                return name
    return names[0]


async def list_ollama_models(host: str) -> list[str]:
    """GET /api/tags 拿本机可用模型名。

    只留能聊天的：Ollama 会给纯 embedding 模型标 capabilities=["embedding"]，
    拿它去 /api/chat 必然报错，所以这里先过滤掉。
    老版本 Ollama 不返回 capabilities 字段，那就都留着（宁可多，不能少）。
    """
    url = f"{host.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as exc:
        raise LLMError(
            f"连不上 Ollama（{url}）：{exc}。"
            f"确认 ollama serve 在跑；如果在 WSL2 里跑、从 Windows 访问，"
            f"需要在 %USERPROFILE%\\.wslconfig 里开 networkingMode=mirrored"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"查询 Ollama 模型列表失败：{exc}") from exc

    names: list[str] = []
    for item in data.get("models") or []:
        name = item.get("name") or item.get("model") or ""
        if not name:
            continue
        caps = item.get("capabilities") or []
        if caps and "completion" not in caps:
            continue
        names.append(name)
    return names


async def list_openai_models(base_url: str, api_key: str) -> list[str]:
    """GET /v1/models（Ollama 的兼容端点也实现了这个）。"""
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        raise LLMError(f"查询 /v1/models 失败：{exc}") from exc

    names: list[str] = []
    for item in data.get("data") or []:
        name = item.get("id") or item.get("name") or ""
        if name:
            names.append(name)
    return names


class OllamaNativeLLM(LLM):
    """直连 Ollama 的 /api/chat（原生协议，不用 openai 包）。

    model="auto" 时，首次调用会去 /api/tags 问本机有什么模型再挑一个 —— 见 AUTO。
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3",
        think: bool = False,
        options: dict | None = None,
        prefer: str = DEFAULT_PREFER,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.think = think
        self.options = options
        self.prefer = prefer
        self._resolved: str | None = None  # auto 解析结果，只解析一次

    async def resolve_model(self) -> str:
        """确定真正要用的模型名。显式指定就直接返回，auto 才去问 Ollama。"""
        if self._resolved:
            return self._resolved
        if self.model and self.model != AUTO:
            self._resolved = self.model
            return self._resolved

        names = await list_ollama_models(self.host)
        if not names:
            raise LLMError(
                "Ollama 上没有可用的聊天模型。"
                "先拉一个（例如 ollama pull qwen3.5:9b），或显式设置 AGENTD_OLLAMA_MODEL=<名字>"
            )
        self._resolved = pick_model(names, self.prefer)
        _log(f"[agentd] 自动选用 Ollama 模型: {self._resolved}（本机可选: {', '.join(names[:8])}）")
        return self._resolved

    def _payload(
        self,
        messages: list[Message],
        system: str | None,
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        body = {
            "model": model or self.model,
            "messages": _with_system(messages, system, ollama=True),
            "stream": True,
            "think": self.think,
        }
        # think 是 Ollama 原生参数；闭源模型没有就忽略即可
        if self.options:
            body["options"] = self.options
        if tools:
            body["tools"] = tools
        return body

    async def stream_events(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMText | LLMToolCall]:
        model = await self.resolve_model()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                async with client.stream(
                    "POST",
                    f"{self.host}/api/chat",
                    json=self._payload(messages, system, model, tools),
                ) as resp:
                    # Ollama 出错时 HTTP 状态码仍是 200，错误藏在 body 里，得读出来判断；
                    # 但 4xx/5xx（比如模型不存在返回 404）是例外，body 里那句
                    # "model 'xxx' not found" 比状态码本身有用得多，一定要带给用户。
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode("utf-8", "replace").strip()
                        raise LLMError(
                            f"Ollama HTTP {resp.status_code}：{detail[:300] or '（空响应体）'}"
                            f"（当前模型 {model}）"
                        )
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        obj = json.loads(line)
                        if obj.get("error"):
                            raise LLMError(str(obj["error"]))
                        msg = obj.get("message") or {}
                        text = msg.get("content") or ""
                        if text:
                            yield LLMText(text)
                        # Ollama 原生 tool_calls 只给 function.name / function.arguments（对象），
                        # 没有 id —— 我们自己补一个，好让 start/done 能配对。
                        for call in msg.get("tool_calls") or []:
                            fn = call.get("function") or {}
                            args = fn.get("arguments")
                            args_str = args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)
                            yield LLMToolCall(
                                id=f"call_{uuid.uuid4().hex[:12]}",
                                name=fn.get("name") or "",
                                arguments=args_str or "{}",
                            )
                        if obj.get("done"):
                            return
        except httpx.ConnectError as exc:
            raise LLMError(
                f"连不上 Ollama（{self.host}）：{exc}。"
                f"确认 ollama serve 在跑；WSL2 里跑、Windows 访问的话，"
                f"需要 .wslconfig 开 networkingMode=mirrored"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama 请求超时：{exc}") from exc


class OpenAICompatLLM(LLM):
    """打 OpenAI /v1/chat/completions 兼容端点（vLLM / LiteLLM / 本地网关都行）。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3",
        api_key: str = "ollama",
        prefer: str = DEFAULT_PREFER,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.prefer = prefer
        self._resolved: str | None = None

    async def resolve_model(self) -> str:
        if self._resolved:
            return self._resolved
        if self.model and self.model != AUTO:
            self._resolved = self.model
            return self._resolved
        names = await list_openai_models(self.base_url, self.api_key)
        if not names:
            raise LLMError(
                f"{self.base_url} 上没有可用模型。或显式设置 AGENTD_OPENAI_MODEL=<名字>"
            )
        self._resolved = pick_model(names, self.prefer)
        _log(f"[agentd] 自动选用 OpenAI 兼容模型: {self._resolved}")
        return self._resolved

    def _payload(
        self,
        messages: list[Message],
        system: str | None,
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> dict:
        body = {
            "model": model or self.model,
            "messages": _with_system(messages, system),
            "stream": True,
        }
        if tools:
            body["tools"] = tools
        return body

    async def stream_events(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMText | LLMToolCall]:
        model = await self.resolve_model()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # 工具调用在 SSE 里按 index **增量拼接**：先来 id/name，arguments 分片渐次到达。
        # 用一个 dict 累积，流结束时再一次性吐出去（见函数末尾）。
        pending: dict[int, dict[str, str]] = {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=self._payload(messages, system, model, tools),
                    headers=headers,
                ) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode("utf-8", "replace").strip()
                        raise LLMError(
                            f"OpenAI 兼容端点 HTTP {resp.status_code}："
                            f"{detail[:300] or '（空响应体）'}（当前模型 {model}）"
                        )
                    # 兼容端点是 SSE（逐行 `data: {...}`），遇 [DONE] 收尾
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        obj = json.loads(data)
                        if obj.get("error"):
                            raise LLMError(str(obj["error"]))
                        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                        text = delta.get("content") or ""
                        if text:
                            yield LLMText(text)
                        for call in delta.get("tool_calls") or []:
                            idx = call.get("index", 0)
                            slot = pending.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                            if call.get("id"):
                                slot["id"] = call["id"]
                            fn = call.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
        except httpx.ConnectError as exc:
            raise LLMError(f"连不上 {self.base_url}：{exc}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"OpenAI 兼容端点请求超时：{exc}") from exc

        for idx in sorted(pending):
            slot = pending[idx]
            yield LLMToolCall(
                id=slot["id"] or f"call_{uuid.uuid4().hex[:12]}",
                name=slot["name"],
                arguments=slot["arguments"] or "{}",
            )


class FakeLLM(LLM):
    """固定回复，仅供测试和本地联调。"""

    def __init__(self, reply: str = "这是 FakeLLM 的固定回复", chunk_size: int = 8) -> None:
        self.reply = reply
        self.chunk_size = chunk_size

    async def stream_events(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMText | LLMToolCall]:
        text = self.reply
        for i in range(0, len(text), self.chunk_size):
            yield LLMText(text[i: i + self.chunk_size])
