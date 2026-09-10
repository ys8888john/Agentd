"""内核内部的数据结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# 用 Literal 而不是 str：拼错 role 会在构造时就报 ValidationError，
# 而不是等到发给 Ollama 才拿到 400。这是"把错误提前到最早可能的位置"。
Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """一条对话消息。

    用 Pydantic 而不是 dataclass，是因为这个对象要跨三个边界：
    内存（历史）→ JSON（发给 LLM）→ JSON（HTTP 响应给前端）。
    Pydantic 一次定义，两边序列化都免费拿到，dataclass 得自己写 asdict()。

    另一个关键点：model_dump() 的结果直接就是 OpenAI /chat/completions
    要的 {"role", "content"} 格式，所以 llm.py 里不用再做一次字段映射。
    这不是巧合，是故意对齐的 —— 以后换 LiteLLM / vLLM 也一样零改动。
    """


    role: Role
    content: str = Field(default="")

    # 只给 role="tool" 用：标识是哪个工具的返回结果。
    # 允许 None 是为了让 model_dump(exclude_none=True) 把它整个剔掉，
    # 这样发给 OpenAI 兼容端点时不会多出一个 "name": null 字段。
    # 注意：类型注解 `str | None` 只描述类型，不提供默认值；
    # 必须显式 `= None` 才会变成"可空且默认 None"，否则 Pydantic 会当必填字段。
    name: str | None = None

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistent(cls, content: str) -> "Message":
        return cls(role="assistent", content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

