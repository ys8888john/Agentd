"""contracts.py 单元测试：Event 联合类型序列化、SSE 帧格式、session/run id 必填约束。"""

import json

import pytest
from pydantic import ValidationError

from agentd.contracts import (
    Done,
    ErrorEvent,
    EventAdapter,
    MessageDelta,
    MessageDone,
    ToolCallDone,
    ToolCallStart,
    new_call_id,
    new_run_id,
    new_session_id,
    to_sse,
)


# 公共基类：session_id / run_id 必填

def test_event_base_requires_session_and_run_id():
    with pytest.raises(ValidationError):
        MessageDelta(text="hi")  # 缺 session_id, run_id
    with pytest.raises(ValidationError):
        MessageDelta(session_id="s", text="hi")  # 缺 run_id


# MessageDelta

def test_message_delta_accepts_only_its_type():
    m = MessageDelta(session_id="s1", run_id="r1", text="你好")
    assert m.type == "message_delta"
    assert m.text == "你好"


def test_message_delta_rejects_wrong_type():
    with pytest.raises(ValidationError) as exc:
        MessageDelta(session_id="s1", run_id="r1", type="oops", text="x")
    assert exc.value.errors()[0]["type"] == "literal_error"


def test_message_delta_type_is_defaulted():
    m = MessageDelta(session_id="s1", run_id="r1", text="hi")
    assert m.model_dump()["type"] == "message_delta"


# MessageDone

def test_message_done_roundtrip():
    m = MessageDone(session_id="s1", run_id="r1", text="完整文本")
    assert m.type == "message_done"
    assert m.text == "完整文本"


# ToolCallStart：多值 Literal 的 kind

@pytest.mark.parametrize("kind", ["read", "edit", "execute", "generic"])
def test_tool_call_start_accepts_all_kinds(kind):
    t = ToolCallStart(session_id="s1", run_id="r1", call_id="c1", title="t", kind=kind)
    assert t.kind == kind


@pytest.mark.parametrize("bad", ["READ", "write", "", "read ", None, 1, True])
def test_tool_call_start_rejects_invalid_kind(bad):
    with pytest.raises(ValidationError):
        ToolCallStart(session_id="s1", run_id="r1", call_id="c1", title="t", kind=bad)


def test_tool_call_start_kind_defaults_to_generic():
    t = ToolCallStart(session_id="s1", run_id="r1", call_id="c1", title="t")
    assert t.kind == "generic"


# ToolCallDone：status 默认值 / output 默认值

def test_tool_call_done_defaults():
    d = ToolCallDone(session_id="s1", run_id="r1", call_id="c1")
    assert d.type == "tool_call_done"
    assert d.status == "completed"
    assert d.output == ""


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_tool_call_done_accepts_all_status(status):
    d = ToolCallDone(session_id="s1", run_id="r1", call_id="c1", status=status)
    assert d.status == status


def test_tool_call_done_rejects_invalid_status():
    with pytest.raises(ValidationError):
        ToolCallDone(session_id="s1", run_id="r1", call_id="c1", status="pending")


# ErrorEvent

def test_error_event_roundtrip():
    e = ErrorEvent(session_id="s1", run_id="r1", message="boom")
    assert e.type == "error"
    assert e.message == "boom"


# Done：stop_reason 默认值 + 取值

def test_done_defaults_to_end_turn():
    d = Done(session_id="s1", run_id="r1")
    assert d.type == "done"
    assert d.stop_reason == "end_turn"


@pytest.mark.parametrize("reason", ["end_turn", "cancelled", "error"])
def test_done_accepts_all_stop_reasons(reason):
    d = Done(session_id="s1", run_id="r1", stop_reason=reason)
    assert d.stop_reason == reason


def test_done_rejects_invalid_stop_reason():
    with pytest.raises(ValidationError):
        Done(session_id="s1", run_id="r1", stop_reason="timeout")


# 判别联合 EventAdapter：核心契约

def test_adapter_routes_by_type():
    cases = {
        '{"type":"message_delta","session_id":"s","run_id":"r","text":"hi"}': MessageDelta,
        '{"type":"message_done","session_id":"s","run_id":"r","text":"full"}': MessageDone,
        '{"type":"tool_call_start","session_id":"s","run_id":"r","call_id":"c","title":"t"}': ToolCallStart,
        '{"type":"tool_call_done","session_id":"s","run_id":"r","call_id":"c"}': ToolCallDone,
        '{"type":"error","session_id":"s","run_id":"r","message":"boom"}': ErrorEvent,
        '{"type":"done","session_id":"s","run_id":"r"}': Done,
    }
    for raw, expected_cls in cases.items():
        ev = EventAdapter.validate_json(raw)
        assert isinstance(ev, expected_cls)


def test_adapter_rejects_unknown_type():
    with pytest.raises(ValidationError) as exc:
        EventAdapter.validate_json('{"type":"unknown","text":"x"}')
    # 判别联合只产 1 条错误，且类型为 union_tag_invalid
    assert exc.value.error_count() == 1
    assert exc.value.errors()[0]["type"] == "union_tag_invalid"


def test_adapter_rejects_missing_type():
    with pytest.raises(ValidationError) as exc:
        EventAdapter.validate_json('{"text":"hi"}')
    assert exc.value.errors()[0]["type"] == "union_tag_not_found"


def test_adapter_roundtrip_preserves_type():
    m = MessageDelta(session_id="s", run_id="r", text="hi")
    raw = m.model_dump_json()
    ev = EventAdapter.validate_json(raw)
    assert isinstance(ev, MessageDelta)
    assert ev.text == "hi"


def test_adapter_validate_python_routes():
    ev = EventAdapter.validate_python({"type": "done", "session_id": "s", "run_id": "r"})
    assert isinstance(ev, Done)


# to_sse：SSE 帧格式

def test_to_sse_format():
    m = MessageDelta(session_id="s", run_id="r", text="hi")
    frame = to_sse(m)
    assert frame.startswith("event: message_delta\n")
    assert frame.endswith("\n\n")
    assert "data: " in frame


def test_to_sse_type_always_present():
    # type 恰好等于默认值，to_sse 用的是 model_dump_json（非 exclude_defaults），
    # 所以 type 必须出现在 data 里，前端才能路由。
    m = MessageDelta(session_id="s", run_id="r", text="hi")
    frame = to_sse(m)
    assert '"type":"message_delta"' in frame


def test_to_sse_carries_json_payload():
    m = MessageDelta(session_id="s", run_id="r", text="你好")
    frame = to_sse(m)
    data = frame.split("data: ", 1)[1].strip()
    payload = json.loads(data)
    assert payload["text"] == "你好"
    assert payload["type"] == "message_delta"


# new_*_id 工具函数

def test_new_session_id_prefix_and_length():
    sid = new_session_id()
    assert sid.startswith("sess_")
    assert len(sid) == len("sess_") + 16


def test_new_run_id_prefix_and_length():
    rid = new_run_id()
    assert rid.startswith("run_")
    assert len(rid) == len("run_") + 16


def test_new_call_id_prefix_and_length():
    cid = new_call_id()
    assert cid.startswith("call_")
    assert len(cid) == len("call_") + 12


def test_new_ids_are_unique():
    ids = {new_session_id() for _ in range(100)}
    assert len(ids) == 100


def test_new_ids_hex_only():
    for sid in (new_session_id(), new_run_id(), new_call_id()):
        body = sid.split("_", 1)[1]
        assert all(c in "0123456789abcdef" for c in body)
