"""契约层：内核与传输层之间唯一的数据契约。

内核只产出 Event；传输层只负责把 Event 翻译成 SSE 帧或 ACP session/update 通知。
这一层是整个项目里唯一"改起来会伤筋动骨"的东西，先定死，之后加模式、加传输都不用碰。

设计取舍：
- 用 Pydantic discriminated union（`type` 字段做判别），既能拿到运行时校验，
  也能让 pyright 在 match 分支里精确收窄类型 —— 这是换掉 Rust 编译期检查的主要补偿手段。
- 字段名统一 snake_case。前端要 camelCase 的话由传输层负责转换，不要污染内核。
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, TypeAlias, Union
from pydantic import BaseModel, Field, TypeAdapter

class EventBase(BaseModel):
    """所有事件的公共字段。

    run_id 区分"同一次执行"：一次 prompt 可能触发多次 LLM 调用（提示链、反思），
    前端靠 run_id 决定要不要开新的消息气泡 —— 同 run_id 的 delta 往一个气泡里追加，
    run_id 变了才另起一个。
    """

    session_id: str
    run_id: str


class MessageDelta(EventBase):
    """流式文本增量。一次 LLM 调用会产生几十到几百个。"""

    type: Literal["message_delta"] = "message_delta"
    text: str

class MessageDone(EventBase):
    """一条完整消息结束（增量拼接后的最终结果）。

    为什么 delta 之外还要 done：前端不应该自己去拼字符串 —— 拼错的、丢帧的、
    顺序乱的情况都得它处理。内核给一份权威的完整文本，前端拿去落库、复制、渲染都用它。
    """
    type: Literal["message_done"] = "message_done"
    text: str


class ToolCallStart(EventBase):
    """工具调用开始。call_id 用于把 start / done 配对 —— 工具是并发的，
    不能靠"上一个 start 配下一个 done"这种顺序假设。"""

    type: Literal["tool_call_start"] = "tool_call_start"
    call_id: str
    title: str
    # kind 决定前端用哪个图标；"execute" 类通常要触发权限确认
    kind: Literal["read", "edit", "execute", "generic"] = "generic"


class ToolCallDone(EventBase):
    """工具调用结束。"""

    type: Literal["tool_call_done"] = "tool_call_done"
    call_id: str
    status: Literal["completed", "failed", "cancelled"] = "completed"
    output: str = ""


class ErrorEvent(EventBase):
    """执行过程中的错误。

    不终止流 —— 后面一定还会跟一个 Done(stop_reason="error")。
    这样前端能同时拿到"出错原因"和"结束了"两个信息，而不用靠连接断开来猜。
    """

    type: Literal["error"] = "error"
    message: str


class Done(EventBase):
    """本次执行结束，流的最后一个事件。"""

    type: Literal["done"] = "done"
    stop_reason: Literal["end_turn", "cancelled", "error"] = "end_turn"


# 判别联合：type 字段的值决定具体是哪个类。
# 好处：TypeAdapter.validate_json() 能自动选对类，pyright 在 isinstance / match
# 之后能把类型收窄到具体子类，点 .text、.call_id 不会报"属性不存在"。
Event: TypeAlias = Annotated[
    Union[
        MessageDelta,
        MessageDone,
        ToolCallStart,
        ToolCallDone,
        ErrorEvent,
        Done
    ],
    Field(discriminator="type")
]


# 模块级建一次，别在循环里反复 new —— TypeAdapter 首次构造会做 schema 分析，有开销。
# 用途：测试里校验 JSON、以后 ACP 传输层反序列化对端发来的数据。
EventAdapter: TypeAdapter[Event] = TypeAdapter(Event)


def new_session_id() -> str:
    """带前缀的 ID。

    用 sess_ / run_ / call_ 前缀而不是纯 uuid，是为了看日志时一眼能分清是哪类 ID，
    也避免手滑把 session_id 填进 call_id 的字段里还看不出来。
    截 16 位十六进制够用（64 bit，碰撞概率可忽略），但比完整 uuid 短一半。
    """
    return f"sess_{uuid.uuid4().hex[:16]}"


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def to_sse(event: Event) -> str:
    """序列化成一条 SSE 帧。

    SSE 格式要求每个字段一行、字段间用 \\n 分隔、帧与帧之间用空行（\\n\\n）分隔。

    同时发 `event:` 和 `data:` 是有意的：
    - 前端可以用 addEventListener("message_delta", ...) 按类型精确监听；
    - 也可以只监听通用的 message 事件自己解析 JSON。
    两条路都留着，以后换前端实现不用改后端。
    """

    payload = event.model_dump_json()
    return f"event: {event.type}\ndata: {payload}\n\n"





