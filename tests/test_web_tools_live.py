"""联网工具的真实联网测试 —— **默认跳过**，要显式开 AGENTD_LIVE_WEB=1 才跑。

为什么默认跳过：它依赖外网和第三方搜索页面的结构，放进日常回归只会随机变红。
但它必须存在 —— web_search 的解析逻辑整个建立在「Bing 的结果页长这样」这个假设上，
离线单测用的是自己手写的样本（自己写的样本当然符合自己的假设），
**只有真跑一次才能证明假设没失效**。

    # 项目 venv 里跑（PowerShell）
    $env:AGENTD_LIVE_WEB=1
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_web_tools_live.py -v -s

跑之前确认网络通。这台机器上的可达性实测（2026-09-14）：
Bing 通；DuckDuckGo / 公共 SearXNG 被挡；Baidu 返回反爬页。详见 tools.py 的注释。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from agentd.kernel.tools import NativeToolbox

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENTD_LIVE_WEB") != "1",
    reason="联网测试：需要显式设 AGENTD_LIVE_WEB=1",
)


def _call(name: str, args: dict) -> str:
    box = NativeToolbox(cwd=Path.cwd(), profile="native")
    return asyncio.run(box.call(name, json.dumps(args)))


def test_live_search_returns_real_results():
    """真打一次 Bing，必须解析出带 URL 的条目（顺带打出来给人看）。"""
    out = _call("web_search", {"query": "python asyncio", "count": 3})
    print("\n" + out)
    assert not out.startswith("[错误]"), out
    assert "搜索到" in out
    assert "https://" in out


def test_live_search_handles_chinese():
    out = _call("web_search", {"query": "上海 天气 预报", "count": 2})
    print("\n" + out)
    assert not out.startswith("[错误]"), out
    assert "http" in out


def test_live_fetch_reads_plain_page():
    """web_fetch 要能把 example.com 折成干净正文（剥掉标签）。"""
    out = _call("web_fetch", {"url": "https://example.com"})
    print("\n" + out)
    assert not out.startswith("[错误]"), out
    assert "Example Domain" in out
    assert "<p>" not in out  # 标签必须已经剥掉


def test_live_fetch_reports_dns_failure():
    """解析不了的主机必须回 [错误]，而不是抛异常。

    用 `.invalid` 顶级域是刻意的：RFC 2606 保留，永远解析不成功，
    所以这条断言不受"某个站点今天挂了"影响。
    """
    out = _call("web_fetch", {"url": "https://no-such-host-xyz-123.invalid/"})
    print("\n" + out)
    assert out.startswith("[错误]"), out
