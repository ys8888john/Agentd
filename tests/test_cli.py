"""CLI 入口冒烟测试。

把 builtins.input 换成脚本，驱动 CliRunner 走完"发消息 → 流式打印 → 命令"。
验证 CLI 与 gateway 一样：同一份 kernel 上真的落库、模式命令生效、工具审批
按提示放行/拒绝。用假 LLM，不依赖 Ollama。
"""

from __future__ import annotations

import pytest

from agentd.kernel.kernel import AgentKernel
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.store import InMemorySessionStore
from agentd.transports.cli import CliRunner


class ListLLM(LLM):
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    async def stream_events(self, messages, *, system=None, tools=None):
        for t in self._texts:
            yield LLMText(t)


def _script_input(monkeypatch, lines: list[str]):
    it = iter(lines)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(it))


async def test_cli_turn_lands_in_history(monkeypatch, tmp_path) -> None:
    kernel = AgentKernel(llm=ListLLM(["你好呀"]), store=InMemorySessionStore())
    runner = CliRunner(kernel, cwd=str(tmp_path))
    _script_input(monkeypatch, ["你好", ":quit"])

    await runner.run()

    hist = await kernel.history(runner.sid)
    assert [m.role for m in hist] == ["user", "assistant"]
    assert [m.content for m in hist] == ["你好", "你好呀"]


async def test_cli_mode_command(monkeypatch, tmp_path) -> None:
    kernel = AgentKernel(llm=ListLLM(["x"]), store=InMemorySessionStore())
    runner = CliRunner(kernel, cwd=str(tmp_path))
    _script_input(monkeypatch, [":mode single", ":quit"])

    await runner.run()

    assert runner.runtime.mode_of(runner.sid) == "single"


async def test_cli_unknown_command_is_harmless(monkeypatch, tmp_path) -> None:
    kernel = AgentKernel(llm=ListLLM(["x"]), store=InMemorySessionStore())
    runner = CliRunner(kernel, cwd=str(tmp_path))
    _script_input(monkeypatch, [":nope", ":help", ":quit"])

    await runner.run()  # 不抛就算过


class WriteToolLLM(LLM):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_events(self, messages, *, system=None, tools=None):
        self.calls += 1
        if self.calls == 1:
            yield LLMToolCall(id="c1", name="write_file",
                              arguments='{"path":"a.txt","content":"x"}')
        else:
            yield LLMText("收尾")


async def test_cli_approval_reject_does_not_write(monkeypatch, tmp_path) -> None:
    """审批选 n：文件绝不落盘（与 gateway 的 allow_once 路径互为对照）。"""
    kernel = AgentKernel(llm=WriteToolLLM(), store=InMemorySessionStore())
    runner = CliRunner(kernel, cwd=str(tmp_path))
    _script_input(monkeypatch, ["写个文件", "n", ":quit"])

    await runner.run()

    assert not (tmp_path / "a.txt").exists()


async def test_cli_approval_allow_writes(monkeypatch, tmp_path) -> None:
    kernel = AgentKernel(llm=WriteToolLLM(), store=InMemorySessionStore())
    runner = CliRunner(kernel, cwd=str(tmp_path))
    _script_input(monkeypatch, ["写个文件", "y", ":quit"])

    await runner.run()

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "x"


if __name__ == "__main__":
    pytest.main([__file__])