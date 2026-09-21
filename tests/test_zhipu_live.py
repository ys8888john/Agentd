"""智谱 BigModel 真实端点验证，默认跳过 —— 要显式开：

    AGENTD_LIVE_ZHIPU=1 AGENTD_ZHIPU_API_KEY=xxxx.yyyy pytest tests/test_zhipu_live.py -v

跟 test_mimo_live.py 一个套路：出网断言不进默认套件。这里最值钱的断言是
chat 往返完整走通；401（Key 只拼了 id 段、缺 secret）也是有效验证：
错误必须可读，不能吞成"回复空白"。
"""

from __future__ import annotations

import os

import pytest

from agentd.boot import load_settings
from agentd.kernel.llm import LLMError
from agentd.kernel.models import Message


pytestmark = pytest.mark.skipif(
    not os.getenv("AGENTD_LIVE_ZHIPU"),
    reason="真出网测试：AGENTD_LIVE_ZHIPU=1 且配好 AGENTD_ZHIPU_API_KEY 才跑",
)


async def test_live_zhipu_chat_roundtrip():
    from agentd.boot import build_llm

    settings = load_settings()
    assert settings.zhipu_api_key, "AGENTD_LIVE_ZHIPU=1 时必须同时给 AGENTD_ZHIPU_API_KEY"
    llm = build_llm(settings)
    text = await llm.complete(
        [Message.user("只回复两个字：通了")], system="你是测试助手，回答尽量短"
    )
    assert isinstance(text, str) and text.strip(), f"回复为空: {text!r}"


async def test_live_zhipu_bad_key_error_is_readable():
    """Key 缺 secret 段的认证错误必须可读（阻止"空白回复"回归）。"""
    from agentd.boot import build_llm

    settings = load_settings()
    # 故意把 Key 截成 id 段暴露 401 路径 —— 稳定复现认证失败，不烧配额
    bad = type(settings)(**{**settings.__dict__, "zhipu_api_key": "id-segment-only"})
    llm = build_llm(bad)

    with pytest.raises(LLMError) as exc:
        await llm.complete([Message.user("hi")])

    assert "401" in str(exc.value)
