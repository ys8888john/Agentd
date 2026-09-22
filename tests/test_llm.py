"""llm.py 单元测试：stream()/complete()/错误处理，以及 Ollama / OpenAI 兼容后端的直连行为。"""

import json
from collections.abc import AsyncGenerator

import httpx
import pytest

from agentd.kernel.llm import (
    AUTO,
    LLM,
    LLMThought,
    LLMText,
    LLMError,
    FakeLLM,
    OllamaNativeLLM,
    OpenAICompatLLM,
    list_ollama_models,
    pick_model,
    _with_system,
)
from agentd.kernel.models import Message


class _FakeStreamCtx:
    """模拟 client.stream() 返回的异步上下文管理器。

    llm.py 里用 `async with client.stream(...) as resp:` 拿到 resp，
    然后先看 `resp.status_code`，再 `async for line in resp.aiter_lines()`。
    所以这个对象需要同时提供 status_code / aread / aiter_lines。
    status_code 默认 200 —— 真实 httpx.Response 一定有这个属性，
    假对象缺了它，llm.py 里那句 `if resp.status_code >= 400` 会直接 AttributeError。
    """

    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    async def aread(self):
        return b""

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


async def test_mimo_reasoning_content_not_leaked_to_text(monkeypatch):
    """MiMo 是推理模型：流里会先来一堆 reasoning_content 增量。

    这些思考内容只能留在模型侧 —— 混进正文的话，客户端会把"让我想想…"
    当成回复渲染出来。锁定 content 与 reasoning_content 分轨的行为。
    """

    class FakeClient(_FakeClient):
        _lines = [
            "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "让我想想…"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "答"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": ""}}]}),
            "data: [DONE]",
        ]

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    llm = OpenAICompatLLM(base_url="https://api.xiaomimimo.com/v1", model="mimo-v2.5-pro", api_key="k")
    chunks = [c async for c in llm.stream([Message.user("hi")])]
    assert chunks == ["答"]


async def test_zhipu_reasoning_content_streams_as_thought(monkeypatch):
    """推理模型的 reasoning_content 应产出 LLMThought 增量（GUI 思考区可显示）。

    之前这里是被直接丢弃的——GLM/MiMo 每次回答前都先思考一大段，
    用户盯着空白屏幕十几秒只能干等。直播出来体验完全不同。
    """

    class FakeClient(_FakeClient):
        _lines = [
            "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "先想"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"reasoning_content": "一下"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "答案"}}]}),
            "data: [DONE]",
        ]

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    llm = OpenAICompatLLM(base_url="https://api.xiaomimimo.com/v1", model="mimo-v2.5-pro", api_key="k")
    events = [e async for e in llm.stream_events([Message.user("hi")])]

    thoughts = [e.text for e in events if isinstance(e, LLMThought)]
    texts = [e.text for e in events if isinstance(e, LLMText)]
    assert "".join(thoughts) == "先想一下"  # 思考完整、分轨保留
    assert texts == ["答案"]                # 正文不受影响


async def test_mimo_402_balance_error_is_surfaced(monkeypatch):
    """真实踩过的错误路径（2026-09-21）：key 有效但账户没余额。

    MiMo 返回 HTTP 402 + `{"error":{"code":"402","message":"Insufficient
    account balance",...}}`。这些信息必须出现在异常里 —— 吞掉的话前端
    又是"回复一片空白"，用户不知道要充值。
    """

    class Ctx(_FakeStreamCtx):
        status_code = 402

        async def aread(self):
            return (
                b'{"error":{"code":"402","message":"Insufficient account balance",'
                b'"type":"insufficient_balance"}}'
            )

    class C(_FakeClient):
        _lines = []

        def stream(self, method, url, json=None, headers=None):
            type(self).captured = {"url": url}
            return Ctx([])

    monkeypatch.setattr(httpx, "AsyncClient", C)

    llm = OpenAICompatLLM(
        base_url="https://api.xiaomimimo.com/v1", model="mimo-v2.5-pro", api_key="sk-t"
    )
    with pytest.raises(LLMError) as exc:
        async for _ in llm.stream([Message.user("hi")]):
            pass

    msg = str(exc.value)
    assert "402" in msg
    assert "Insufficient account balance" in msg


async def test_zhipu_401_auth_error_is_surfaced(monkeypatch):
    """真实踩过的错误路径（2026-09-21）：只用 Key 的 id 段（缺 .secret）调智谱。

    BigModel 返回 HTTP 401 + `{"error":{"code":"401","message":"令牌已过期
    或验证不正确"}}`。这个信息必须进异常——否则前端又是"回复一片空白"，
    用户不知道要去查 Key 是否完整。
    """

    class Ctx(_FakeStreamCtx):
        status_code = 401

        async def aread(self):
            return '{"error":{"code":"401","message":"令牌已过期或验证不正确"}}'.encode("utf-8")

    class C(_FakeClient):
        _lines = []

        def stream(self, method, url, json=None, headers=None):
            type(self).captured = {"url": url}
            return Ctx([])

    monkeypatch.setattr(httpx, "AsyncClient", C)

    llm = OpenAICompatLLM(
        base_url="https://open.bigmodel.cn/api/paas/v4", model="glm-4.5-air", api_key="id-only"
    )
    with pytest.raises(LLMError) as exc:
        async for _ in llm.stream([Message.user("hi")]):
            pass

    msg = str(exc.value)
    assert "401" in msg
    assert "令牌已过期或验证不正确" in msg


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


# ---------------------------------------------------------------------------
# 模型自动探测（AGENTD_OLLAMA_MODEL=auto）
#
# 这一段存在的理由：默认模型名写死成 qwen3，但这台机器上装的是 qwen3.5:9b-text，
# Ollama 返回 404，而 404 的表现是"回复一片空白、stop_reason 还是 end_turn"——
# 用户完全无从下手。auto 就是为了让"换台机器也能开箱即通"。
# ---------------------------------------------------------------------------

class _FakeResponse:
    """给 list_ollama_models 用的 GET 响应替身。"""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeGetClient:
    """只实现 get() 的 AsyncClient 替身，用来喂 /api/tags 的结果。"""

    payload: dict = {}

    def __init__(self, *a, **k):
        pass

    async def get(self, url, **k):
        type(self).captured_url = url
        return _FakeResponse(type(self).payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _tags(*names, caps=("completion",)):
    return {
        "models": [
            {"name": n, "capabilities": list(caps)} for n in names
        ]
    }


def test_pick_model_prefers_earlier_token():
    assert pick_model(["llama3:8b", "qwen3.5:9b-text"], "qwen3.5,llama") == "qwen3.5:9b-text"


def test_pick_model_falls_back_to_first_when_no_match():
    assert pick_model(["mistral:7b", "phi:latest"], "qwen3.5,qwen3") == "mistral:7b"


def test_pick_model_prefer_is_case_insensitive():
    assert pick_model(["Qwen3.5:9b"], "QWEN3.5") == "Qwen3.5:9b"


def test_pick_model_on_empty_list_raises():
    with pytest.raises(LLMError):
        pick_model([])


async def test_list_ollama_models_filters_embedding_only(monkeypatch):
    class C(_FakeGetClient):
        payload = {
            "models": [
                {"name": "bge-m3", "capabilities": ["embedding"]},
                {"name": "qwen3.5:9b-text", "capabilities": ["completion"]},
            ]
        }

    monkeypatch.setattr(httpx, "AsyncClient", C)
    assert await list_ollama_models("http://h:11434") == ["qwen3.5:9b-text"]


async def test_list_ollama_models_keeps_all_when_no_capabilities_field(monkeypatch):
    """老版本 Ollama 不返回 capabilities —— 那就都留着，宁可多不能少。"""

    class C(_FakeGetClient):
        payload = {"models": [{"name": "qwen3:8b"}, {"name": "llama3:8b"}]}

    monkeypatch.setattr(httpx, "AsyncClient", C)
    assert await list_ollama_models("http://h:11434") == ["qwen3:8b", "llama3:8b"]


async def test_auto_model_resolves_once_and_is_used_in_body(monkeypatch):
    """auto 时：查一次 /api/tags，挑中的模型进请求体，并且只查一次。"""

    calls = {"n": 0}

    class GetClient(_FakeGetClient):
        payload = _tags("qwen3.5:9b-text", "qwen3:8b")

        async def get(self, url, **k):
            calls["n"] += 1
            return await super().get(url, **k)

    class StreamClient(_FakeClient):
        _lines = [json.dumps({"message": {"content": "ok"}, "done": True})]

    # 同一个替身要同时支持 get 和 stream
    class Both(GetClient, StreamClient):
        pass

    monkeypatch.setattr(httpx, "AsyncClient", Both)

    llm = OllamaNativeLLM(model=AUTO)
    chunks = [c async for c in llm.stream([Message.user("hi")])]
    # 再来一次，验证解析结果被缓存、不会重复查询
    chunks += [c async for c in llm.stream([Message.user("hi")])]

    assert chunks == ["ok", "ok"]
    assert calls["n"] == 1
    assert llm._resolved == "qwen3.5:9b-text"
    # captured 记在 type(self) 上，也就是 Both
    assert Both.captured["json"]["model"] == "qwen3.5:9b-text"


async def test_explicit_model_skips_tagging(monkeypatch):
    """显式指定了模型就别去问 Ollama —— 少一次网络往返，也避免依赖 tags 端点。"""
    calls = {"n": 0}

    class Both(_FakeGetClient, _FakeClient):
        payload = _tags("qwen3.5:9b-text")
        _lines = [json.dumps({"message": {"content": "ok"}, "done": True})]

        async def get(self, url, **k):
            calls["n"] += 1
            return await super().get(url, **k)

    monkeypatch.setattr(httpx, "AsyncClient", Both)

    llm = OllamaNativeLLM(model="my-model")
    async for _ in llm.stream([Message.user("hi")]):
        pass

    assert calls["n"] == 0
    assert Both.captured["json"]["model"] == "my-model"


async def test_httpx_404_body_is_surfaced(monkeypatch):
    """模型不存在时，Ollama 那句 'model not found' 必须出现在异常里。

    以前这里只抛裸的 HTTPStatusError，传输出去后前端看到的是一片空白。
    """

    class Ctx(_FakeStreamCtx):
        status_code = 404

        async def aread(self):
            return b'{"error":"model \'qwen3\' not found"}'

    class C(_FakeClient):
        _lines = []

        def stream(self, method, url, json=None, headers=None):
            type(self).captured = {"json": json}
            return Ctx([])

    monkeypatch.setattr(httpx, "AsyncClient", C)

    llm = OllamaNativeLLM(model="qwen3")
    with pytest.raises(LLMError) as exc:
        async for _ in llm.stream([Message.user("hi")]):
            pass

    msg = str(exc.value)
    assert "404" in msg
    assert "not found" in msg
    assert "qwen3" in msg


async def test_connect_error_gives_actionable_hint(monkeypatch):
    """连不上 Ollama 时，提示要指向具体动作，而不是甩一个 ConnectError。"""

    class C(_FakeClient):
        def stream(self, method, url, json=None, headers=None):
            raise httpx.ConnectError("[Errno 10061] 连不上")

    monkeypatch.setattr(httpx, "AsyncClient", C)

    llm = OllamaNativeLLM(model="qwen3")
    with pytest.raises(LLMError) as exc:
        async for _ in llm.stream([Message.user("hi")]):
            pass

    assert "ollama serve" in str(exc.value)
