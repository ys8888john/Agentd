"""kernel/models.py 的单元测试。

运行：.venv/bin/python -m pytest -q

覆盖范围：
- Role 字面量取值约束
- Message 字段：role 必填且受 Role 约束、content 默认值、name 可空语义
- model_dump / model_dump(exclude_none=True) 与 OpenAI 格式的契合
- 工厂方法 user / system / assistent（注：assistent 目前有拼写 bug，见末尾）
"""

import pytest
from pydantic import ValidationError

from agentd.kernel.models import Message, Role


# ---------------------------------------------------------------------------
# Role 字面量
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_role_accepts_valid(role):
    m = Message(role=role, content="x")
    assert m.role == role


@pytest.mark.parametrize("bad", ["SYSTEM", "assistent", "human", "", None, 1])
def test_role_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        Message(role=bad, content="x")


# ---------------------------------------------------------------------------
# Message 基础字段
# ---------------------------------------------------------------------------

def test_message_requires_role():
    with pytest.raises(ValidationError):
        Message(content="hi")  # 缺 role


def test_content_defaults_to_empty_string():
    m = Message(role="user")
    assert m.content == ""


def test_content_accepts_string():
    m = Message(role="user", content="你好")
    assert m.content == "你好"


# ---------------------------------------------------------------------------
# name 字段：str 或 None，默认 None
# ---------------------------------------------------------------------------

def test_name_defaults_to_none():
    m = Message(role="user")
    assert m.name is None


def test_name_accepts_string():
    m = Message(role="tool", content="x", name="calc")
    assert m.name == "calc"


def test_name_rejects_non_str_non_none():
    with pytest.raises(ValidationError):
        Message(role="tool", content="x", name=123)


# ---------------------------------------------------------------------------
# 序列化：对齐 OpenAI /chat/completions 格式
# ---------------------------------------------------------------------------

def test_model_dump_openai_shape():
    m = Message(role="user", content="hi")
    dumped = m.model_dump()
    assert dumped == {"role": "user", "content": "hi", "name": None}


def test_exclude_none_drops_name_field():
    # name 默认 None 时，exclude_none=True 应把它整个剔掉，
    # 避免发给 OpenAI 端点出现多余的 "name": null
    m = Message(role="user", content="hi")
    dumped = m.model_dump(exclude_none=True)
    assert "name" not in dumped
    assert dumped == {"role": "user", "content": "hi"}


def test_exclude_none_keeps_real_name():
    m = Message(role="tool", content="x", name="calc")
    dumped = m.model_dump(exclude_none=True)
    assert dumped == {"role": "tool", "content": "x", "name": "calc"}


def test_roundtrip_via_json():
    m = Message(role="assistant", content="答", name=None)
    raw = m.model_dump_json()
    m2 = Message.model_validate_json(raw)
    assert m2.role == "assistant"
    assert m2.content == "答"
    assert m2.name is None


# ---------------------------------------------------------------------------
# 工厂方法
# ---------------------------------------------------------------------------

def test_factory_user():
    m = Message.user("hi")
    assert m.role == "user"
    assert m.content == "hi"
    assert m.name is None


def test_factory_system():
    m = Message.system("sys")
    assert m.role == "system"
    assert m.content == "sys"


def test_factory_assistent_is_broken_due_to_typo():
    """已知 bug：assistent() 内部把 role 写成 "assistent"（应为 "assistant"），
    会触发 Role 的 Literal 校验失败。此测试用于锁定该回归，修复后应改为 test_factory_assistant。

    修复方式：把 models.py 第 40-41 行的 role="assistent" 改为 "assistant"，
    并建议把方法名也修正为 assistant。
    """
    with pytest.raises(ValidationError):
        Message.assistent("hi")
