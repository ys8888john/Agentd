"""内核内部的数据结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# 用 Literal 而不是 str：拼错 role 在构造时即报 ValidationError
Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """assistant 发起的一次工具调用。

    arguments 统一存成 **JSON 字符串**（OpenAI 线格式就是字符串）；要发给 Ollama
    原生端点时再反序列化成对象（见 llm._with_system 的 ollama 分支）。
    """

    id: str
    name: str
    arguments: str = "{}"


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
