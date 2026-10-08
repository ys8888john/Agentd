"""搜索缓存 / 搜狗节流熔断 / 实体反转义 测试。

背景（2026-10-08 事故二）：模型一轮对同一 query 连打 6 次 web_search，快速
连打第 2 次起搜狗就弹验证码，兜底整段失效 —— 修法是缓存 + 节流 + 熔断。
"""

from __future__ import annotations

import asyncio
import time

import pytest

import agentd.kernel.tools as tools_mod
from agentd.kernel.tools import (
    _parse_sogou,
    _search_cache_get,
    _search_cache_put,
    _search_sogou,
)


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
