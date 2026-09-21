"""MiMo 真实端点验证，默认跳过 —— 要显式开：

    AGENTD_LIVE_MIMO=1 AGENTD_MIMO_API_KEY=sk-xxxx pytest tests/test_mimo_live.py -v

存在的原因跟 test_web_tools_live.py 一样：把出网断言塞进默认套件等于给 CI 埋
随机红。它验的是离线套件钉不住的另一半：MiMo 的 /v1 端点现在还认不认我们
发的请求形状（Bearer 认证、/v1/models 列表、chat 的错误可读性）。

Key 没余额（HTTP 402）不等于套件红 —— 列模型和错误可读性这两条是身份验证
级别的断言，402 只是"这个 key 当前不能 chat"，本身也是一次有效的真实验证。
"""

from __future__ import annotations

import os

import pytest

from agentd.boot import load_settings
from agentd.kernel.llm import LLMError, list_openai_models
from agentd.kernel.models import Message


pytestmark = pytest.mark.skipif(
    not os.getenv("AGENTD_LIVE_MIMO"),
    reason="真出网测试：AGENTD_LIVE_MIMO=1 且配好 AGENTD_MIMO_API_KEY 才跑",
)


async def test_live_mimo_lists_models():
    """Bearer key + /v1/models：能列出模型就说明认证与端点全对。"""
    settings = load_settings()
    api_key = settings.mimo_api_key
    assert api_key, "AGENTD_LIVE_MIMO=1 时必须同时给 AGENTD_MIMO_API_KEY"

    names = await list_openai_models(settings.mimo_base_url, api_key)
    assert any(n.startswith("mimo-") for n in names), names


async def test_live_mimo_chat_error_is_readable():
    """chat 层的错误必须可读（402 余额不足是最常见的一种）。"""
    from agentd.boot import build_llm

    settings = load_settings()
    assert settings.mimo_api_key, "AGENTD_LIVE_MIMO=1 时必须同时给 AGENTD_MIMO_API_KEY"
    llm = build_llm(settings)

    try:
        text = await llm.complete([Message.user("只回复：ok")])
        assert isinstance(text, str)  # 有余额时：正常回复
    except LLMError as exc:
        # 无余额时：错误要带上可辨认的信息，不能是一段裸 HTTP 噪音
        assert "402" in str(exc) or "Insufficient" in str(exc), str(exc)
