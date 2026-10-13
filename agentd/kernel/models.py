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
    model_dump() 直接是 OpenAI /chat/completions 的 {role, content} 格式。

    content 两种形态：
      - str：普通文本消息（绝大多数）。
      - list[dict]：多模态内容（用户发了图片），OpenAI 视觉格式 ——
          [{"type":"text","text":"..."},
           {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
        llm._with_system 负责按后端（OpenAI 兼容 / Ollama）翻成线格式；
        其余消费方（token 估算、记忆压缩、历史回放）用 content_text() 取纯文本。
    """

    role: Role
    content: str | list[dict] = Field(default="")

    # 只给 role="tool" 用；允许 None 让 model_dump(exclude_none=True) 剔掉它，
    # 避免发给 OpenAI 兼容端点多出 "name": null。注意 `= None` 才表示可空且默认 None。
    name: str | None = None

    # 工具调用：assistant 侧带 tool_calls，tool 侧带 tool_call_id 指回对应的调用。
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    # 只给 role="tool_record" 用：一次工具调用的完整记录（历史回放用的卡片）。
    # 与 role="tool" 的分野见 ToolRecord 的 docstring。
    tool_record: ToolRecord | None = None

    # 内部消息标记（plan 模式用）：True = 这条消息是模式内部的指令/记录，
    # 要进模型上下文、要落库，但**不是给人看的** —— 客户端历史回放必须跳过。
    # 不打的后果（2026-10-13 实测）：plan 跑完一轮，历史里混进 1 条"我的计划：…"
    # 和 N 条"【计划 i/N】…只完成这一步…"，GUI 回放时全画成普通气泡，侧栏预览
    # 也变成步骤提示词 —— 用户原话"界面很乱"。
    # 默认 None（假）：model_dump(exclude_none=True) 时不出现，旧库数据兼容。
    # ⚠️ 跨仓库契约：GUI 的 sessions.get_history 按 payload.internal 过滤，
    #    两边对不上时最坏退化为"多画一个气泡"，不会崩。
    internal: bool | None = None

    @classmethod
    def user(cls, content: str | list[dict]) -> "Message":
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

    def content_text(self) -> str:
        """多形态 content 的纯文本视图（token 估算 / 展示 / 压缩都用它）。

        list 内容时拼接 text 块，图片块换成 [图片] 占位（base64 不该出现在
        记录压缩、日志这些文本场景里）。
        """
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for p in self.content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
            elif p.get("type") == "image_url":
                parts.append("[图片]")
        return "\n".join(x for x in parts if x)

    def content_images(self) -> list[dict]:
        """list 内容里的图片块原样取出（llm 序列化用）；str 内容返回空。"""
        if not isinstance(self.content, list):
            return []
        return [p for p in self.content if isinstance(p, dict) and p.get("type") == "image_url"]

    @classmethod
    def from_tool_record(cls, record: ToolRecord) -> "Message":
        """把一次工具调用记录包装成一条历史消息。

        content 存 title（为的是 sqlite3 直查冗余列也能看懂是哪张卡），
        完整结构在 payload.tool_record 里 —— payload 才是权威数据，不迁移表。
        """
        return cls(role="tool_record", content=record.title, tool_record=record)
