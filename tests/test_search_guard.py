"""搜索缓存 / 搜狗节流熔断 / 实体反转义 / 智谱首选后端 测试。

背景（2026-10-08 事故二）：模型一轮对同一 query 连打 6 次 web_search，快速
连打第 2 次起搜狗就弹验证码，兜底整段失效 —— 修法是缓存 + 节流 + 熔断。
智谱搜索 API 上线为首选后端（key 在场时），Bing/搜狗降为兜底。
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

import agentd.kernel.tools as tools_mod
from agentd.kernel.tools import (
    NativeToolbox,
    _parse_sogou,
    _search_cache_get,
    _search_cache_put,
    _search_sogou,
)
from pathlib import Path


# ---- 缓存 ----

def test_search_cache_roundtrip_and_ttl(monkeypatch):
    key = ("query 甲", 5)
    assert _search_cache_get(key) is None
    _search_cache_put(key, "结果")
    assert _search_cache_get(key) == "结果"

    # TTL 过期后取不到
    future = time.monotonic() + tools_mod._SEARCH_CACHE_TTL + 1
    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: future)
    assert _search_cache_get(key) is None
    monkeypatch.setattr(time, "monotonic", real_monotonic)


def test_search_cache_evicts_when_full():
    tools_mod._search_cache.clear()
    for i in range(tools_mod._SEARCH_CACHE_MAX + 4):
        _search_cache_put((f"q{i}", 5), f"v{i}")
    # 容量有上限，不会无限涨
    assert len(tools_mod._search_cache) <= tools_mod._SEARCH_CACHE_MAX
    tools_mod._search_cache.clear()


# ---- 搜狗熔断 ----

@pytest.fixture(autouse=True)
def _reset_sogou_state():
    tools_mod._sogou_state = {"last_request": 0.0, "cooldown_until": 0.0}
    yield
    tools_mod._sogou_state = {"last_request": 0.0, "cooldown_until": 0.0}


def test_sogou_captcha_sets_cooldown(monkeypatch):
    """触发验证码 → 进冷却期；冷却期内的第二次调用不发任何网络请求。"""

    class _FakeResp:
        status_code = 200
        text = "<html>请输入验证码 antispider</html>"

        def raise_for_status(self):
            pass

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            _FakeClient.requests += 1
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, params=None):
            return _FakeResp()

    _FakeClient.requests = 0
    monkeypatch.setattr(tools_mod.httpx, "AsyncClient", _FakeClient)

    with pytest.raises(tools_mod._SearchError, match="验证码"):
        asyncio.run(_search_sogou("q", 5))
    # 必须经 tools_mod 取：fixture 是整体重绑 _sogou_state，函数看的是新 dict
    assert tools_mod._sogou_state["cooldown_until"] > time.monotonic()

    # 冷却期内：直接拒绝，request 数不涨
    with pytest.raises(tools_mod._SearchError, match="冷却中"):
        asyncio.run(_search_sogou("q", 5))
    assert _FakeClient.requests == 1


def test_sogou_throttle_enforces_min_interval(monkeypatch):
    """上次请求刚发过：先睡满最小间隔再发请求（用不可达端点收尾即可）。"""
    tools_mod._sogou_state["last_request"] = time.monotonic()
    monkeypatch.setattr(tools_mod, "_SOGOU_THROTTLE_SECS", 0.2)
    # 指向必然拒绝连接的地址：只验证"等了间隔"这件事
    monkeypatch.setenv("AGENTD_SOGOU_ENDPOINT", "http://127.0.0.1:9/web")

    started = time.monotonic()
    with pytest.raises(tools_mod._SearchError):
        asyncio.run(_search_sogou("q", 5))
    assert time.monotonic() - started >= 0.19


# ---- 搜狗页面实体反转义 ----

def test_parse_sogou_unescapes_entities():
    page = (
        '<div class="vrwrap"><h3 class="vr-title">'
        '<a href="/link?url=A1&amp;cmd=2">标题&amp;测试</a></h3>'
        '<div class="str-text-info">摘要一&amp;二，长度足够被采纳的内容文本</div></div>'
    )
    results = _parse_sogou(page, 5)
    assert results, "至少解析出一条"
    assert results[0]["title"] == "标题&测试"
    assert "&amp;" not in results[0]["url"]
    assert "amp" not in results[0]["snippet"] or "&" in results[0]["snippet"]


# ---- 智谱搜索 API（首选后端）----

@pytest.fixture(autouse=True)
def _clean_search_state():
    tools_mod._search_cache.clear()
    yield
    tools_mod._search_cache.clear()


async def _start_fake_api(payload: dict | None = None, status: int = 200):
    """本地假智谱 web_search API：记录请求次数，返回固定 JSON。"""
    state = {"requests": 0}
    body = json.dumps(payload if payload is not None else {
        "search_result": [
            {"title": "智谱结果一", "link": "https://example.com/z1",
             "content": "成都到北京航班号 3U8881 11:30 起飞", "media": "Example"},
            {"title": "智谱结果二", "link": "https://example.com/z2",
             "content": "国航 CA4115 每日一班", "media": "Example"},
        ]
    }).encode()

    async def handle(reader, writer):
        state["requests"] += 1
        # 先把请求读完再应答：Windows 上没读就关会触发 RST，客户端报 ReadError
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
        writer.write(
            b"HTTP/1.1 %d OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\nConnection: close\r\n\r\n"
            % (status, len(body))
        )
        writer.write(body)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}/v4/web_search", state


def test_zhipu_is_primary_backend(monkeypatch):
    """key 在场时智谱必须是首选：结果带「来源：智谱搜索 API」，且不碰 Bing/搜狗。"""
    async def run():
        server, url, state = await _start_fake_api()
        async with server:
            monkeypatch.setenv("AGENTD_ZHIPU_API_KEY", "test-key")
            monkeypatch.setenv("AGENTD_ZHIPU_SEARCH_ENDPOINT", url)
            monkeypatch.delenv("AGENTD_SEARCH_ENDPOINT", raising=False)
            box = NativeToolbox(cwd=Path.cwd(), profile="native")
            out = await box.call("web_search", json.dumps({"query": "成都到北京航班", "count": 5}))
        return out, state

    out, state = asyncio.run(run())
    assert "来源：智谱搜索 API" in out
    assert "智谱结果一" in out
    assert "3U8881" in out
    assert state["requests"] == 1  # 只打了假 API，Bing/搜狗没被碰


def test_zhipu_failure_falls_through_with_reason(monkeypatch):
    """智谱挂了（HTTP 401）→ 落到 Bing/搜狗，且失败原因透出在最终错误里。"""
    async def run():
        server, url, state = await _start_fake_api(status=401)
        async with server:
            monkeypatch.setenv("AGENTD_ZHIPU_API_KEY", "test-key")
            monkeypatch.setenv("AGENTD_ZHIPU_SEARCH_ENDPOINT", url)
            monkeypatch.delenv("AGENTD_SEARCH_ENDPOINT", raising=False)
            monkeypatch.setenv("AGENTD_SOGOU_ENDPOINT", "http://127.0.0.1:9/web")
            # Bing 也指向不可达地址：整条链路都离线，断言错误信息里三层原因齐全
            monkeypatch.setenv("AGENTD_SEARCH_ENDPOINT", "")  # 空串=没设，但需真连 Bing → 改为直接断言智谱失败后回落
            box = NativeToolbox(cwd=Path.cwd(), profile="native")
            # 为了不碰真网：把 Bing 整个替换掉
            async def fake_bing(query, limit):
                raise tools_mod._SearchError("Bing 不可达（假）")
            monkeypatch.setattr(tools_mod, "_search_bing", fake_bing)
            tools_mod._sogou_state = {"last_request": 0.0, "cooldown_until": 0.0}
            out = await box.call("web_search", json.dumps({"query": "q", "count": 5}))
        return out, state

    out, _state = asyncio.run(run())
    assert out.startswith("[错误]"), out
    assert "智谱搜索不可用" in out
    assert "Bing 不可达（假）" in out
    assert "搜狗也不可用" in out
