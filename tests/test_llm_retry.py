"""云端过载（HTTP 429 / 智谱 1305）自动重试的测试。

背景（2026-10-09 用户截图）：glm-4.7-flash 高峰期回
HTTP 429 {"code":"1305","message":"该模型当前访问量过大，请您稍后再试"}，
以前第一次就把 LLMError 抛给用户。现在默认重试 5 次（递增等待），全失败才抛。

用 httpx.MockTransport 模拟各形态的响应，真跑 OpenAICompatLLM.stream_events。
"""

from __future__ import annotations

import json

import httpx
import pytest

from agentd.kernel import llm as llm_mod
from agentd.kernel.llm import LLMError, LLMNotice, LLMText, OpenAICompatLLM
from agentd.kernel.models import Message


def _sse(*chunks: dict) -> bytes:
    """把若干 delta 对象拼成一条 SSE 响应体（data: 行 + [DONE]）。"""
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}" for c in chunks]
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def _text_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def _err_body(code: str, message: str) -> bytes:
    return json.dumps(
        {"error": {"code": code, "message": message}}, ensure_ascii=False
    ).encode()


class _ScriptedEndpoint:
    """按顺序回放响应的 MockTransport，并记录收到的请求次数。"""

    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self.responses = list(responses)
        self.requests = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        status, body = (
            self.responses.pop(0)
            if self.responses
            else (500, b"mock: response script exhausted")
        )
        headers = {"content-type": "text/event-stream"} if status < 400 else {}
        return httpx.Response(status, content=body, headers=headers)


@pytest.fixture
def fast_sleep(monkeypatch):
    """把重试等待换成立即返回（只记录时长），测试不用真等 59 秒。"""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(llm_mod.asyncio, "sleep", fake_sleep)
    return delays


def _wire(monkeypatch, endpoint: _ScriptedEndpoint) -> None:
    """让 stream_events 里 new 出来的 AsyncClient 走 mock 通道。"""
    real_client = httpx.AsyncClient

    def factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(endpoint.handler)
        return real_client(**kwargs)

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", factory)


def _llm() -> OpenAICompatLLM:
    return OpenAICompatLLM(
        base_url="http://mock.local/v1", model="glm-test", api_key="k"
    )


async def _collect(llm: OpenAICompatLLM, prompt: str = "你好") -> list:
    return [
        e
        async for e in llm.stream_events([Message.user(prompt)], system=None, tools=None)
    ]


async def test_429_retries_then_succeeds(monkeypatch, fast_sleep):
    """第一次 429（智谱 1305），第二次成功：中间要有一条 Notice，正文完整。"""
    endpoint = _ScriptedEndpoint(
        [
            (429, _err_body("1305", "该模型当前访问量过大，请您稍后再试")),
            (200, _sse(_text_chunk("你好"), _text_chunk("呀"))),
        ]
    )
    _wire(monkeypatch, endpoint)

    events = await _collect(_llm())

    notices = [e for e in events if isinstance(e, LLMNotice)]
    texts = [e.text for e in events if isinstance(e, LLMText)]
    assert endpoint.requests == 2
    assert len(notices) == 1
    assert "429" in notices[0].text and "1/5" in notices[0].text
    assert "".join(texts) == "你好呀"
    assert fast_sleep == [2.0]


async def test_exhausts_five_retries_then_raises(monkeypatch, fast_sleep):
    """6 次（1 首次 + 5 重试）全 429 才把错误抛出去，等待按 2/4/8/15/30 递增。"""
    err = _err_body("1305", "该模型当前访问量过大，请您稍后再试")
    endpoint = _ScriptedEndpoint([(429, err)] * 6)
    _wire(monkeypatch, endpoint)

    with pytest.raises(LLMError) as excinfo:
        await _collect(_llm())

    assert endpoint.requests == 6
    assert "HTTP 429" in str(excinfo.value)
    assert "1305" in str(excinfo.value)
    assert fast_sleep == [2.0, 4.0, 8.0, 15.0, 30.0]


async def test_4xx_config_error_raises_immediately(monkeypatch, fast_sleep):
    """401 是配置错误，重试到天荒地老也不会好 —— 必须第一次就抛。"""
    endpoint = _ScriptedEndpoint([(401, _err_body("1002", "API Key 无效"))])
    _wire(monkeypatch, endpoint)

    with pytest.raises(LLMError) as excinfo:
        await _collect(_llm())

    assert endpoint.requests == 1
    assert fast_sleep == []
    assert "HTTP 401" in str(excinfo.value)


async def test_5xx_gateway_errors_are_retryable(monkeypatch, fast_sleep):
    """502/503 这类网关抖动同样自动重试，不必麻烦用户。"""
    endpoint = _ScriptedEndpoint(
        [(502, b"bad gateway"), (503, b"overloaded"), (200, _sse(_text_chunk("成了")))]
    )
    _wire(monkeypatch, endpoint)

    events = await _collect(_llm())

    assert endpoint.requests == 3
    assert [e.text for e in events if isinstance(e, LLMText)] == ["成了"]
    assert len(fast_sleep) == 2


async def test_overload_body_error_in_200_stream_retries(monkeypatch, fast_sleep):
    """HTTP 200 但 body 里带 1305 错误对象（部分网关的形态）：同样要重试。"""
    body_err = b"data: " + _err_body("1305", "该模型当前访问量过大，请您稍后再试") + b"\n\n"
    endpoint = _ScriptedEndpoint([(200, body_err), (200, _sse(_text_chunk("好了")))])
    _wire(monkeypatch, endpoint)

    events = await _collect(_llm())

    assert endpoint.requests == 2
    assert [e.text for e in events if isinstance(e, LLMText)] == ["好了"]


async def test_no_retry_after_streaming_started(monkeypatch, fast_sleep):
    """已经吐过正文再失败：绝不能重试 —— 重发会把已输出的内容重复一遍。"""
    # 第一条响应：先给一段正文，再在流内给一个 1305 错误对象（无 [DONE]，模拟中断）
    first = (
        b"data: "
        + json.dumps(_text_chunk("已经说出口的话"), ensure_ascii=False).encode("utf-8")
        + b"\n\ndata: "
        + _err_body("1305", "该模型当前访问量过大，请您稍后再试")
        + b"\n\n"
    )
    endpoint = _ScriptedEndpoint([(200, first), (200, _sse(_text_chunk("重复")))])
    _wire(monkeypatch, endpoint)

    with pytest.raises(LLMError):
        await _collect(_llm())

    assert endpoint.requests == 1
    assert fast_sleep == []


async def test_retry_leaves_partial_tool_fragments_behind(monkeypatch, fast_sleep):
    """重试成功后，上一次尝试里没收完整的工具调用分片不能混进来。

    工具分片是先攒在 pending 里、流结束才一次性吐的；而 pending 每次尝试
    都从零建 —— 这条测试守住"重试 = 干净重来"这个约定。
    """
    partial_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "function": {"name": "web_search", "arguments": '{"q'},
                        }
                    ]
                }
            }
        ]
    }
    # 第一次：半个 tool_call 分片 + 流内 1305 错误（分片没吐出去 → 允许重试）。
    # 手拼、不带 [DONE] —— 分片后面必须真的跟着错误行。
    first = (
        b"data: "
        + json.dumps(partial_chunk, ensure_ascii=False).encode("utf-8")
        + b"\n\ndata: "
        + _err_body("1305", "该模型当前访问量过大，请您稍后再试")
        + b"\n\n"
    )
    endpoint = _ScriptedEndpoint([(200, first), (200, _sse(_text_chunk("好了")))])
    _wire(monkeypatch, endpoint)

    events = await _collect(_llm())

    assert endpoint.requests == 2
    assert [e.text for e in events if isinstance(e, LLMText)] == ["好了"]
    # 关键断言：失败的尝试里那半个 tool_call（id=c1）没有混进最终结果
    assert not [e for e in events if type(e).__name__ == "LLMToolCall"]
