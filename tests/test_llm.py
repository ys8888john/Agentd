"""llm.py 单元测试：stream()/complete()/错误处理，以及 Ollama / OpenAI 兼容后端的直连行为。"""

import json
from collections.abc import AsyncGenerator

import httpx
import pytest

from agentd.kernel.llm import (
    LLM,
    LLMError,
    FakeLLM,
    OllamaNativeLLM,
    OpenAICompatLLM,
    _with_system,
)
from agentd.kernel.models import Message


class _FakeStreamCtx:
    """模拟 client.stream() 返回的异步上下文管理器。

    llm.py 里用 `async with client.stream(...) as resp:` 拿到 resp，
    然后 `resp.raise_for_status()` + `async for line in resp.aiter_lines()`。
    所以这个对象需要同时提供 raise_for_status 和 aiter_lines。
    """

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeClient:
    """捕获 POST 参数并把预设 NDJSON 行喂回去的 httpx.AsyncClient 替身。

    captured 用类属性，方便测试直接以 FakeClient.captured 读取；
    每个测试用独立子类，彼此不串。
    """

    captured: dict = {}
    _lines: list = []

    def __init__(self, *a, **k):
        pass

    def stream(self, method, url, json=None, headers=None):
        type(self).captured = {
            "method": method,
            "url": url,
            "json": json,
            "headers": headers,
        }
        # 子类通过 _lines 提供返回内容
        return _FakeStreamCtx(type(self)._lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


# LLMError
def test_llm_error_is_runtime_error():
    err = LLMError("boom")
    assert isinstance(err, RuntimeError)
    assert str(err) == "boom"


# _with_system
def test_with_system_prepends_when_given():
    out = _with_system([Message.user("hi")], system="sys")
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_with_system_skips_when_none():
    out = _with_system([Message.user("hi")], system=None)
    assert out == [{"role": "user", "content": "hi"}]


def test_with_system_strips_none_name():
    # tool 消息带 name 时应保留；不带 name 时不应出现 "name": null
    tool = Message(role="tool", content="ok", name="search")
    plain = Message.user("hi")
    out = _with_system([tool, plain], system=None)
    assert out[0] == {"role": "tool", "content": "ok", "name": "search"}
    assert out[1] == {"role": "user", "content": "hi"}


# LLM.complete (默认实现基于 stream 拼接)
async def test_complete_concatenates_stream():
    llm = FakeLLM(reply="abcdefg", chunk_size=3)
    result = await llm.complete([Message.user("hi")])
    assert result == "abcdefg"


async def test_complete_passes_system():
    seen = {}

    class CaptureLLM(FakeLLM):
        async def stream(self, message, *, system=None):
            seen["system"] = system
            async for c in super().stream(message, system=system):
                yield c

    llm = CaptureLLM(reply="x")
    await llm.complete([Message.user("hi")], system="SYS")
    assert seen["system"] == "SYS"


# FakeLLM 流式切分
async def test_fakellm_yields_chunks():
    llm = FakeLLM(reply="abcdefgh", chunk_size=3)
    chunks = [c async for c in llm.stream([Message.user("hi")])]
    assert chunks == ["abc", "def", "gh"]


async def test_fakellm_chunk_size_one():
    llm = FakeLLM(reply="ab", chunk_size=1)
    chunks = [c async for c in llm.stream([Message.user("hi")])]
    assert chunks == ["a", "b"]


async def test_fakellm_default_reply():
    llm = FakeLLM()
    out = await llm.complete([Message.user("hi")])
    assert out == "这是 FakeLLM 的固定回复"


# OllamaNativeLLM —— 请求体构造 + NDJSON 解析
async def test_ollama_native_builds_body(monkeypatch):
    class FakeClient(_FakeClient):
        _lines = [
            json.dumps({"message": {"content": "你"}, "done": False}),
            json.dumps({"message": {"content": "好"}, "done": True}),
        ]

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    llm = OllamaNativeLLM(host="http://h:11434", model="qwen3", think=False)
    chunks = [c async for c in llm.stream([Message.user("hi")], system="SYS")]

    assert chunks == ["你", "好"]
    assert FakeClient.captured["method"] == "POST"
    assert FakeClient.captured["url"] == "http://h:11434/api/chat"
    body = FakeClient.captured["json"]
    assert body["model"] == "qwen3"
    assert body["stream"] is True
    assert body["think"] is False
    # system 被塞到 messages 最前
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}


async def test_ollama_native_merges_options(monkeypatch):
    class FakeClient(_FakeClient):
        _lines = [json.dumps({"message": {"content": "ok"}, "done": True})]

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    llm = OllamaNativeLLM(options={"temperature": 0.2, "num_ctx": 8192})
    async for _ in llm.stream([Message.user("hi")]):
        pass
    assert FakeClient.captured["json"]["options"] == {"temperature": 0.2, "num_ctx": 8192}


async def test_ollama_native_raises_on_error_body(monkeypatch):
    class FakeClient(_FakeClient):
        _lines = [
            # Ollama 出错时 HTTP 仍是 200，错误在 body 里
            json.dumps({"error": "model not found"}),
            json.dumps({"done": True}),
        ]

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    llm = OllamaNativeLLM()
    with pytest.raises(LLMError) as exc:
        async for _ in llm.stream([Message.user("hi")]):
            pass
    assert "model not found" in str(exc.value)


async def test_ollama_native_skips_empty_content(monkeypatch):
    class FakeClient(_FakeClient):
        _lines = [
            # 第一个 chunk content 为空，不应 yield
            json.dumps({"message": {"content": ""}, "done": False}),
            json.dumps({"message": {"content": "data"}, "done": True}),
        ]

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    chunks = [c async for c in OllamaNativeLLM().stream([Message.user("hi")])]
    assert chunks == ["data"]


# OpenAICompatLLM —— SSE 解析
async def test_openai_compat_parses_sse(monkeypatch):
    class FakeClient(_FakeClient):
        _lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "你"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "好"}}]}),
            "data: [DONE]",
        ]

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    llm = OpenAICompatLLM(base_url="http://h:9999/v1", model="gpt", api_key="k")
    chunks = [c async for c in llm.stream([Message.user("hi")], system="SYS")]

    assert chunks == ["你", "好"]
    assert FakeClient.captured["url"] == "http://h:9999/v1/chat/completions"
    assert FakeClient.captured["json"]["model"] == "gpt"
    assert FakeClient.captured["json"]["stream"] is True
    assert FakeClient.captured["headers"]["Authorization"] == "Bearer k"
    assert FakeClient.captured["json"]["messages"][0] == {"role": "system", "content": "SYS"}


async def test_openai_compat_skips_non_data_lines(monkeypatch):
    class FakeClient(_FakeClient):
        _lines = [
            ": keep-alive",
            "data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}),
            "data: [DONE]",
        ]

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    chunks = [c async for c in OpenAICompatLLM().stream([Message.user("hi")])]
    assert chunks == ["x"]


# 已知拼写坑的锁定测试（与 models.py 的 assistent 拼写坑同惯例）
def test_ollama_stream_return_annotation_is_asyncgenerator():
    """OllamaNativeLLM.stream 的返回注解必须是 AsyncIterator。

    原先 llm.py 顶部误导入 AsyncGenerator，注解里的 AsyncIterator 是未定义名字；
    因 `from __future__ import annotations` 注解不求值，运行时静默不报错。
    现已修正导入。这里继续锁定"注解字符串确实存在"。
    """
    from agentd.kernel.llm import OllamaNativeLLM

    ann = OllamaNativeLLM.stream.__annotations__.get("return")
    assert ann is not None
    assert "AsyncIterator" in str(ann)


def test_fake_stream_return_annotation_is_asynciterator():
    """llm.py FakeLLM.stream 的返回注解。

    原来锁的是拼错的 AsyncItrator（少一个 e），现已修正为 AsyncIterator，
    并且 llm.py 顶部也改成真正 import AsyncIterator（原先误导入 AsyncGenerator）。
    """
    from agentd.kernel.llm import AsyncIterator, FakeLLM

    ann = FakeLLM.stream.__annotations__.get("return")
    assert ann is not None
    assert "AsyncIterator" in str(ann)
    # 注解里引用的名字必须真的能解析到，否则 get_type_hints() 会炸
    assert AsyncIterator is not None
