"""额外工作区（ACP additionalDirectories）测试。

链路：transport 的 new/load 收到的 additional_directories 会进内核会话配置，
原生工具箱把它解析成额外的越界白名单 —— 模型可以对那些目录用绝对路径做
读/搜。cwd 之外、额外根之外的路径仍然拒绝。

覆盖三层：
1. NativeToolbox：额外 root 内可读、外依旧拒、重复/自身 root 去重；
2. kernel：new_session 带 additional_directories 的整轮 agent 对话
   （glob 用 path 指到额外目录、read_file 读界外文件被拒）；
3. transport：load_session 把额外目录重新绑回去（对齐 GUI 重启场景）。
"""

from __future__ import annotations

import asyncio
import json

from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.store import InMemorySessionStore
from agentd.kernel.tools import NativeToolbox
from agentd.transports.acp_stdio import AgentdAcpAgent

_loop = asyncio.new_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


class PlayLLM(LLM):
    def __init__(self, scripts: list[list]) -> None:
        self._scripts = list(scripts)

    async def stream_events(self, messages, *, system=None, tools=None):
        for event in (self._scripts.pop(0) if self._scripts else [LLMText("（脚本耗尽）")]):
            yield event


# ---------------------------------------------------------------------------
# 1) NativeToolbox
# ---------------------------------------------------------------------------


def test_additional_root_allows_read_and_keeps_others_out(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "note.txt").write_text("额外目录内容", encoding="utf-8")
    # 真正的"界外"必须在会话 cwd（tmp_path）之外：pytest 的 tmp_path 本身
    # 是唯一目录，放同级即可（探针文件互不冲突）
    outside = tmp_path.parent / ("outside-" + tmp_path.name)
    outside.mkdir()
    (outside / "secret.txt").write_text("越界内容", encoding="utf-8")

    box = NativeToolbox(cwd=tmp_path, additional_roots=[extra])

    out = _run(box.call("read_file", json.dumps({"path": str(extra / "note.txt")})))
    assert "额外目录内容" in out

    out = _run(box.call("read_file", json.dumps({"path": str(outside / "secret.txt")})))
    assert out.startswith("[错误]")


def test_outside_without_additional_root_still_rejected(tmp_path):
    outside = tmp_path.parent / ("outside-" + tmp_path.name)
    outside.mkdir()
    (outside / "secret.txt").write_text("越界内容", encoding="utf-8")
    box = NativeToolbox(cwd=tmp_path)
    out = _run(box.call("read_file", json.dumps({"path": str(outside / "secret.txt")})))
    assert out.startswith("[错误]")


def test_duplicate_and_self_roots_are_deduped(tmp_path):
    box1 = NativeToolbox(cwd=tmp_path, additional_roots=[])
    box2 = NativeToolbox(cwd=tmp_path, additional_roots=[tmp_path, str(tmp_path) + "/sub"])
    assert box1.runtime.additional_roots == ()   # 只有 cwd：无额外根
    roots2 = box2.runtime.additional_roots
    assert len(roots2) == 1 and roots2[0].name == "sub"  # cwd 自身被去重


# ---------------------------------------------------------------------------
# 2) kernel 端到端：new_session 带额外目录
# ---------------------------------------------------------------------------


async def test_kernel_roundtrip_with_additional_directories(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "roadmap.md").write_text("把 sqlite 排进下月", encoding="utf-8")
    outside = tmp_path.parent / ("outside-" + tmp_path.name)
    outside.mkdir()
    (outside / "secret.md").write_text("不该看到", encoding="utf-8")

    kernel = AgentKernel(
        llm=PlayLLM(
            [
                [LLMToolCall(id="c1", name="glob", arguments=json.dumps({"path": str(extra), "pattern": "*.md"}))],
                [LLMText("找到了 roadmap")],
                [LLMToolCall(id="c2", name="read_file", arguments=json.dumps({"path": str(outside / "secret.md")}))],
                [LLMText("没有读到")],
            ]
        ),
        store=InMemorySessionStore(),
    )
    sid = await kernel.create_session(cwd=str(tmp_path), additional_directories=[str(extra)])

    # 第一轮：glob 命中额外目录
    gathered = [e async for e in kernel.handle(sid, "看看 roadmap", mode="agent")]
    dones = [e for e in gathered if type(e).__name__ == "ToolCallDone"]
    assert len(dones) == 1 and "roadmap.md" in dones[0].output

    # 第二轮：read_file 读 cwd 外、且不在任何额外根里的文件 —— 必须被拒
    gathered = [e async for e in kernel.handle(sid, "读 secret", mode="agent")]
    dones = [e for e in gathered if type(e).__name__ == "ToolCallDone"]
    assert len(dones) == 1
    assert dones[0].output.startswith("[错误]") and "越界" in dones[0].output


# ---------------------------------------------------------------------------
# 3) transport：load_session 重绑额外目录
# ---------------------------------------------------------------------------


async def test_load_session_rebinds_additional_directories(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "note.txt").write_text("恢复后的额外目录", encoding="utf-8")

    kernel = AgentKernel(
        llm=PlayLLM(
            [
                [LLMToolCall(id="c1", name="glob", arguments=json.dumps({"path": str(extra), "pattern": "*.txt"}))],
                [LLMText("第一轮")],
                [LLMToolCall(id="c2", name="glob", arguments=json.dumps({"path": str(extra), "pattern": "*.txt"}))],
                [LLMText("第二轮")],
            ]
        ),
        store=InMemorySessionStore(),
    )
    sid = await kernel.create_session(cwd=str(tmp_path), additional_directories=[str(extra)])
    async for _ in kernel.handle(sid, "第一轮", mode="agent"):
        pass

    # 模拟进程重启后配置丢失：先摘掉 opts，再经 load_session 恢复
    kernel._session_opts.pop(sid, None)
    agent = AgentdAcpAgent(kernel)
    agent.on_connect(object())
    resp = await agent.load_session(sid, cwd=str(tmp_path), additional_directories=[str(extra)])
    assert resp.modes is not None
    assert kernel._session_opts[sid]["additional_directories"] == [str(extra)]
