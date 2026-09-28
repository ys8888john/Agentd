"""内核内部的数据结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..contracts import ToolKind

# 用 Literal 而不是 str：拼错 role 在构造时即报 ValidationError。
# tool_record 是唯一"不喂给 LLM"的角色：它是工具调用的完整记录，
# 只服务客户端的历史回放（工具卡片）。内核在把 history 交给模式前会把它过滤掉。
Role = Literal["system", "user", "assistant", "tool", "tool_record"]


class ToolCall(BaseModel):
    """assistant 发起的一次工具调用。

    arguments 统一存成 **JSON 字符串**（OpenAI 线格式就是字符串）；要发给 Ollama
    原生端点时再反序列化成对象（见 llm._with_system 的 ollama 分支）。
    """

    id: str
    name: str
    arguments: str = "{}"


class ToolRecord(BaseModel):
    """一次工具调用的完整记录，落库供客户端续聊时重放工具卡片。

    和 role="tool"（OpenAI 线格式里的工具结果，必须喂给模型）不同，
    role="tool_record" 是纯粹的 UI 记录：内核把它过滤在 LLM 上下文之外 ——
    历史里的工具往返已经有"assistant 最终回复"这个结论兜底，把中间产物
    再喂回去只会占上下文、还可能让弱小模型被自己的工具输出带跑。
    """

    call_id: str
    title: str
    kind: ToolKind
    status: Literal["completed", "failed", "cancelled"]
    output: str = ""


class Message(BaseModel):
    """一条对话消息。Pydantic 一次定义，内存/JSON 序列化都免费拿到；
    model_dump() 直接是 OpenAI /chat/completions 的 {role, content} 格式。"""

    role: Role
    content: str = Field(default="")

    # 只给 role="tool" 用；允许 None 让 model_dump(exclude_none=True) 剔掉它，
    # 避免发给 OpenAI 兼容端点多出 "name": null。注意 `= None` 才表示可空且默认 None。
    name: str | None = None

    # 工具调用：assistant 侧带 tool_calls，tool 侧带 tool_call_id 指回对应的调用。
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    # 只给 role="tool_record" 用：一次工具调用的完整记录（历史回放用的卡片）。
    # 与 role="tool" 的分野见 ToolRecord 的 docstring。
    tool_record: ToolRecord | None = None

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str, name: str | None = None) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    @classmethod
    def from_tool_record(cls, record: ToolRecord) -> "Message":
        """把一次工具调用记录包装成一条历史消息。

        content 存 title（为的是 sqlite3 直查冗余列也能看懂是哪张卡），
        完整结构在 payload.tool_record 里 —— payload 才是权威数据，不迁移表。
        """
        return cls(role="tool_record", content=record.title, tool_record=record)
