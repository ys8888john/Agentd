"""多模态（图片识别）链路回归 —— agentd 侧。

两段各测各的：

1. 传输层 AcpClient prompt -> _extract_content：
   纯文本回 str（旧行为不变）、image block 翻成 OpenAI image_url 内容块、
   data 为空时 file:// uri 读文件兜底、纯图无文放行、全空报错。
2. 序列化 llm._with_system：
   OpenAI 兼容端点 image_url 原样透传；Ollama 端点翻成 content 文本 +
   images=[裸 base64]。Message.content_text() 的占位行为也在这里钉住。
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from agentd.kernel.llm import _with_system
from agentd.kernel.models import Message
from agentd.transports.acp_stdio import AgentdAcpAgent


@pytest.fixture()
def agent():
    # _extract_content 不碰 kernel，塞个 object 占位即可
    return AgentdAcpAgent(kernel=object())  # type: ignore[arg-type]


# ---- 1. _extract_content ----

def test_extract_content_text_only_returns_str(agent):
    blocks = [SimpleNamespace(type="text", text="你好")]
    assert agent._extract_content(blocks) == "你好"


def test_extract_content_with_image_returns_block_list(agent):
    blocks = [
        SimpleNamespace(type="text", text="这是什么"),
        SimpleNamespace(type="image", data="QUJD", mime_type="image/png", uri=""),
    ]
    got = agent._extract_content(blocks)
    assert isinstance(got, list)
    assert got[0] == {"type": "text", "text": "这是什么"}
    assert got[1]["type"] == "image_url"
    url = got[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert url.endswith("QUJD")


def test_extract_content_reads_file_uri_when_data_empty(agent, tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(b"IMG")
    blocks = [SimpleNamespace(type="image", data="", mime_type="image/png", uri=f"file://{p}")]
    got = agent._extract_content(blocks)
    url = got[0]["image_url"]["url"]
    assert url == f"data:image/png;base64,{base64.b64encode(b'IMG').decode('ascii')}"


def test_extract_content_image_only_allowed(agent):
    blocks = [SimpleNamespace(type="image", data="QUJD", mime_type="image/jpeg", uri="")]
    got = agent._extract_content(blocks)
    assert [p["type"] for p in got] == ["image_url"]  # 无正文也放行


def test_extract_content_all_empty_raises(agent):
    with pytest.raises(ValueError):
        agent._extract_content([])


# ---- 2. Message / _with_system ----

def _multimodal_user() -> Message:
    return Message.user([
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ])


def test_message_content_text_replaces_images_with_placeholder():
    assert _multimodal_user().content_text() == "看图\n[图片]"


def test_with_system_openai_passthrough_image_blocks():
    out = _with_system([_multimodal_user()], system=None)
    assert out[0]["role"] == "user"
    assert out[0]["content"][1]["type"] == "image_url"  # 原样透传


def test_with_system_ollama_translates_to_images_array():
    out = _with_system([_multimodal_user()], system=None, ollama=True)
    item = out[0]
    assert item["content"] == "看图"
    assert item["images"] == ["QUJD"]  # data URI 前缀剥掉，只剩裸 base64
