"""视觉转译层（kernel/vision.py）回归测试。

钉住的行为：
- 无图 / 未配置 → 原样放行，零副作用
- 有图 + 已配置 → image_url 块换成 [图片转述] 文本块，原 Message 不被改
- 同一张图第二轮命中缓存（不再调视觉模型）
- 转述失败折成说明文本，不外抛
- 内核接线：handle() 出去给模式的 history 里图片已被转述（Notice 同步带出）
"""

from __future__ import annotations

import asyncio

import pytest

from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, Message
from agentd.kernel.store import InMemorySessionStore
from agentd.kernel.vision import VisionInterpreter

_B64 = "QUJD"  # "ABC"
_IMG_PART = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_B64}"}}
_IMG2_B64 = "REVG"  # "DEF"
_IMG2_PART = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_IMG2_B64}"}}


def _texts(msg: Message) -> str:
    """消息里所有文本块的内容拼一起（断言用，不关心块的位置）。"""
    return "\n".join(
        str(p.get("text") or "")
        for p in (msg.content if isinstance(msg.content, list) else [])
        if isinstance(p, dict) and p.get("type") == "text"
    )


class FakeVisionLLM(LLM):
    """视觉模型的替身：complete() 返回固定转述，并记录调用次数。"""

    def __init__(self, reply: str = "图中是一只猫", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[list[Message]] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("vision endpoint down")
        yield LLMText(self.reply)


class MainLLM(LLM):
    """主模型替身：记录收到的 history，回复固定文本。"""

    def __init__(self) -> None:
        self.seen: list[Message] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.seen.extend(messages)
        yield LLMText("好的")


def _cfg(**over):
    table = {
        "AGENTD_VISION_BASE_URL": "http://vision.example/v1",
        "AGENTD_VISION_MODEL": "vl-model",
        "AGENTD_VISION_API_KEY": "k",
        "AGENTD_VISION_HOST": "",
    }
    table.update(over)
    return lambda name, default: table.get(name, default)


# ---- 单元：VisionInterpreter ----

def test_annotate_passthrough_without_images():
    it = VisionInterpreter(_cfg())
    msgs = [Message.user("纯文本"), Message.assistant("好")]
    out, note = asyncio.run(it.annotate(msgs))
    assert out == msgs  # 原对象原样
    assert note == ""
    # 未配置视觉模型时同样放行
    it2 = VisionInterpreter(lambda n, d: "")
    out2, note2 = asyncio.run(it2.annotate([Message.user([{"type": "text", "text": "x"}]), ]))
    assert out2[0].content == [{"type": "text", "text": "x"}]
    assert note2 == ""


def test_annotate_replaces_images_and_keeps_original(tmp_path):
    vision = FakeVisionLLM("图中是一只猫")
    it = VisionInterpreter(_cfg())
    it._resolve = lambda: vision  # 注入替身
    it._cache_dir = tmp_path

    orig = Message.user([
        {"type": "text", "text": "这是什么"},
        dict(_IMG_PART),
    ])
    out, note = asyncio.run(it.annotate([orig]))

    # 原 Message 未被改动
    assert orig.content[1] == _IMG_PART
    # 新消息里图片块变成了转述文本
    assert out[0].content[0] == {"type": "text", "text": "这是什么"}
    injected = out[0].content[1]["text"]
    assert out[0].content[1]["type"] == "text"
    assert "图中是一只猫" in injected
    assert "图片转述" in injected
    # 注入文本必须自带"系统代看"框架：主模型才知道这不是用户打的模板，
    # 不会回「我没有收到图片」（2026-10-11 用户截图实锤的症状）
    assert "系统注入" in injected
    assert "不要声称“没有收到图片”" in injected
    assert "1 张图片" in note
    # 视觉模型收到的请求：图片块 + 转述指令文本
    ask = vision.calls[0][0]
    assert any(p.get("type") == "image_url" for p in ask.content)
    assert any(p.get("type") == "text" for p in ask.content)


def test_annotate_second_round_hits_cache(tmp_path):
    vision = FakeVisionLLM("描述")
    it = VisionInterpreter(_cfg())
    it._resolve = lambda: vision
    it._cache_dir = tmp_path

    def _msg():
        return Message.user([dict(_IMG_PART)])

    asyncio.run(it.annotate([_msg()]))
    first_calls = len(vision.calls)
    assert first_calls == 1

    out2, note2 = asyncio.run(it.annotate([_msg()]))
    assert len(vision.calls) == first_calls  # 不再调视觉模型
    assert "描述" in _texts(out2[0])
    assert note2 == ""  # 全命中缓存的轮次不再刷 Notice（每轮重复提示=噪音）


def test_annotate_counts_cached_and_fresh(tmp_path):
    """计数口径修复：缓存命中的图也要计入总数。

    修复前命中缓存的 continue 排在 seen.add 之前，缓存轮会报
    「已用视觉模型转述 0 张图片」（2026-10-11 用户截图实锤）。
    """
    vision = FakeVisionLLM("新图描述")
    it = VisionInterpreter(_cfg())
    it._resolve = lambda: vision
    it._cache_dir = tmp_path

    asyncio.run(it.annotate([Message.user([dict(_IMG_PART)])]))  # 图 A 入缓存

    _out, note = asyncio.run(it.annotate([
        Message.user([dict(_IMG_PART)]),    # A：缓存
        Message.user([dict(_IMG2_PART)]),   # B：新转述
    ]))
    assert "2 张图片" in note
    assert "新转述 1 张" in note
    assert "命中缓存 1 张" in note
    assert len(vision.calls) == 2  # 只补了图 B


def test_failure_not_cached_next_round_retries(tmp_path):
    """转述失败不进缓存：端点恢复后下一轮重试还能成功。

    修复前失败文本也会被缓存（含落盘），这张图从此永远「转述失败」。
    """
    vision = FakeVisionLLM("修好了的描述", fail=True)
    it = VisionInterpreter(_cfg())
    it._resolve = lambda: vision
    it._cache_dir = tmp_path

    out1, _ = asyncio.run(it.annotate([Message.user([dict(_IMG_PART)])]))
    assert "图片转述失败" in _texts(out1[0])

    vision.fail = False
    out2, note2 = asyncio.run(it.annotate([Message.user([dict(_IMG_PART)])]))
    assert "修好了的描述" in _texts(out2[0])   # 重试成功
    assert "1 张图片" in note2                  # 这轮算新转述，有提示


def test_annotate_failure_becomes_note_not_exception(tmp_path):
    vision = FakeVisionLLM(fail=True)
    it = VisionInterpreter(_cfg())
    it._resolve = lambda: vision
    it._cache_dir = tmp_path

    out, _ = asyncio.run(it.annotate([Message.user([dict(_IMG_PART)])]))
    assert "图片转述失败" in _texts(out[0])


def test_disk_cache_survives_new_interpreter(tmp_path):
    it1 = VisionInterpreter(_cfg())
    it1._resolve = lambda: FakeVisionLLM("磁盘描述")
    it1._cache_dir = tmp_path
    asyncio.run(it1.annotate([Message.user([dict(_IMG_PART)])]))

    # 新实例（模拟进程重启），无内存缓存、视觉模型不可用也应能从磁盘回填
    it2 = VisionInterpreter(_cfg())
    it2._cache_dir = tmp_path
    vision2 = FakeVisionLLM("不该被调用")
    it2._resolve = lambda: vision2
    out, _ = asyncio.run(it2.annotate([Message.user([dict(_IMG_PART)])]))
    assert "磁盘描述" in _texts(out[0])
    assert vision2.calls == []


# ---- 集成：内核接线 ----

@pytest.mark.asyncio
async def test_kernel_handle_annotates_history(monkeypatch, tmp_path):
    """handle() 出去给模式的 history 里，图片块已被转述；Notice 一并带出。"""
    vision = FakeVisionLLM("转述内容ABC")
    kernel = AgentKernel(
        llm=MainLLM(),
        store=InMemorySessionStore(),
        vision_config_fn=_cfg(),
    )
    kernel._vision._resolve = lambda: vision  # type: ignore[union-attr]
    kernel._vision._cache_dir = tmp_path  # type: ignore[union-attr]

    sid = await kernel.create_session()

    events = []
    async for ev in kernel.handle(sid, [
        {"type": "text", "text": "看图说话"},
        dict(_IMG_PART),
    ]):
        events.append(ev)

    main_llm = kernel.llm
    assert any(
        isinstance(p, dict) and p.get("type") == "text" and "转述内容ABC" in str(p.get("text"))
        for m in main_llm.seen
        if isinstance(m.content, list)
        for p in m.content
    ), "主模型没收到转述文本"
    # Notice 提示也发了
    assert any(getattr(ev, "text", "") and "转述" in getattr(ev, "text", "") for ev in events)


@pytest.mark.asyncio
async def test_kernel_without_vision_config_untouched():
    """没配 vision_config_fn：内核行为与从前完全一致。"""
    kernel = AgentKernel(llm=MainLLM(), store=InMemorySessionStore())
    sid = await kernel.create_session()
    events = []
    async for ev in kernel.handle(sid, "纯文本问题"):
        events.append(ev)
    assert any(getattr(ev, "type", "") == "done" or hasattr(ev, "stop_reason") for ev in events)
