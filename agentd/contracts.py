"""契约层：内核与传输层之间唯一的数据契约。

内核只产出 Event；传输层负责把它翻译成 SSE 帧或 ACP 通知。字段统一 snake_case，
前端要 camelCase 由传输层转换，不污染内核。
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, TypeAlias, Union
from pydantic import BaseModel, Field, TypeAdapter

# 工具调用的类别。取值对齐 ACP 的 ToolKind —— 客户端据此选图标、决定要不要弹审批。
# 多一个 "generic" 表示"不确定是哪类"，传输层会把它译成 ACP 的 "other"。
# 为什么不干脆用 str：kernel 产出的 kind 一旦拼错，ACP 那边是**校验失败**，
# 而失败的表现是客户端静默卡住。放在 Literal 里，拼错当场 ValidationError。
ToolKind = Literal[
    "read",
    "edit",
    "delete",
    "move",
    "search",
    "execute",
    "think",
    "fetch",
    "switch_mode",
    "other",
    "generic",
]

class EventBase(BaseModel):
    """所有事件的公共字段。run_id 区分同一次执行（一次 prompt 可能触发多次 LLM 调用）。"""

    session_id: str
    run_id: str


class MessageDelta(EventBase):
    """流式文本增量。"""

    type: Literal["message_delta"] = "message_delta"
    text: str


class MessageDone(EventBase):
    """一条完整消息结束。内核给出权威完整文本，前端落库/复制/渲染都用它。"""

    type: Literal["message_done"] = "message_done"
    text: str


class ToolCallStart(EventBase):
    """工具调用开始。call_id 用于把 start / done 配对（工具可能并发）。"""

    type: Literal["tool_call_start"] = "tool_call_start"
    call_id: str
    title: str
    # kind 决定前端用哪个图标；edit / execute 类通常会先弹权限确认
    kind: ToolKind = "generic"


class ToolCallDone(EventBase):
    """工具调用结束。"""

    type: Literal["tool_call_done"] = "tool_call_done"
    call_id: str
    status: Literal["completed", "failed", "cancelled"] = "completed"
    output: str = ""


class ErrorEvent(EventBase):
    """执行过程中的错误。不终止流，后面一定还跟一个 Done(stop_reason="error")。"""

    type: Literal["error"] = "error"
    message: str


class Done(EventBase):
    """本次执行结束，流的最后一个事件。"""

    type: Literal["done"] = "done"
    stop_reason: Literal["end_turn", "cancelled", "error"] = "end_turn"


# type 字段做判别：TypeAdapter 自动选类，match 之后类型收窄到具体子类
Event: TypeAlias = Annotated[
    Union[
        MessageDelta,
        MessageDone,
        ToolCallStart,
        ToolCallDone,
        ErrorEvent,
        Done,
    ],
    Field(discriminator="type"),
]


# 模块级建一次，别在循环里反复 new（TypeAdapter 首次构造有开销）
EventAdapter: TypeAdapter[Event] = TypeAdapter(Event)


def new_session_id() -> str:
    """带 sess_ 前缀的 ID，看日志一眼能分清类型，也避免字段填错。"""
    return f"sess_{uuid.uuid4().hex[:16]}"


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def to_sse(event: Event) -> str:
    """序列化成一条 SSE 帧（event: + data: 两路都发，前端按需监听）。"""
    payload = event.model_dump_json()
    return f"event: {event.type}\ndata: {payload}\n\n"
