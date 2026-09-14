"""models.py 单元测试：Message 工厂、Literal 约束、model_dump 兼容 OpenAI、历史追加与 tool 消息序列化。"""

import pytest
from pydantic import ValidationError

from agentd.kernel.models import Message, Role


# Role 字面量

@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_role_accepts_valid(role):
    m = Message(role=role, content="x")
    assert m.role == role


@pytest.mark.parametrize("bad", ["SYSTEM", "assistent", "human", "", None, 1])
def test_role_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        Message(role=bad, content="x")


# Message 基础字段

def test_message_requires_role():
    with pytest.raises(ValidationError):
        Message(content="hi")  # 缺 role


def test_content_defaults_to_empty_string():
    m = Message(role="user")
    assert m.content == ""


def test_content_accepts_string():
    m = Message(role="user", content="你好")
    assert m.content == "你好"


# name 字段：str 或 None，默认 None

def test_name_defaults_to_none():
    m = Message(role="user")
    assert m.name is None


def test_name_accepts_string():
    m = Message(role="tool", content="x", name="calc")
    assert m.name == "calc"


def test_name_rejects_non_str_non_none():
    with pytest.raises(ValidationError):
        Message(role="tool", content="x", name=123)


# 序列化：对齐 OpenAI /chat/completions 格式

def test_model_dump_openai_shape():
    m = Message(role="user", content="hi")
    dumped = m.model_dump()
    assert dumped == {
        "role": "user",
        "content": "hi",
        "name": None,
        "tool_calls": None,
        "tool_call_id": None,
    }


# 工具调用字段（tool_calls / tool_call_id）

def test_message_defaults_no_tool_fields():
    m = Message(role="assistant", content="x")
    assert m.tool_calls is None
    assert m.tool_call_id is None


def test_assistant_carries_tool_calls():
    from agentd.kernel.models import ToolCall

    tc = ToolCall(id="call_1", name="echo", arguments='{"text":"hi"}')
    m = Message(role="assistant", content="", tool_calls=[tc])
    assert m.tool_calls[0].name == "echo"
    assert m.tool_calls[0].arguments == '{"text":"hi"}'


def test_tool_factory_sets_call_id():
    m = Message.tool("结果", tool_call_id="call_1", name="echo")
    assert m.role == "tool"
    assert m.content == "结果"
    assert m.tool_call_id == "call_1"
    assert m.name == "echo"


def test_roundtrip_tool_calls_via_json():
    from agentd.kernel.models import ToolCall

    m = Message(role="assistant", content="", tool_calls=[ToolCall(id="c1", name="t", arguments="{}")])
    m2 = Message.model_validate_json(m.model_dump_json())
    assert m2.tool_calls is not None
    assert m2.tool_calls[0].id == "c1"


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


# 工厂方法

def test_factory_user():
    m = Message.user("hi")
    assert m.role == "user"
    assert m.content == "hi"
    assert m.name is None


def test_factory_system():
    m = Message.system("sys")
    assert m.role == "system"
    assert m.content == "sys"


def test_factory_assistant():
    """原来这里锁的是 assistent 拼写 bug（方法名和 role 都写成了 "assistent"），
    现已修复 —— 方法名改回 assistant，role 也改回合法的 "assistant"。
    """
    m = Message.assistant("hi")
    assert m.role == "assistant"
    assert m.content == "hi"

    # role 拼错仍然必须被 Literal 拦住
    with pytest.raises(ValidationError):
        Message(role="assistent", content="hi")
