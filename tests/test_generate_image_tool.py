"""generate_image（文生图原生工具）测试。

覆盖：注册与免审批语义、配置缺失/参数校验的错误路径、b64 与 url 两种响应、
HTTP 错误透传、配置读取的优先级（boot.env_value > 进程环境）、以及
AgentMode 端到端（工具被喂给 LLM、kind/status 正确、产物落盘、不弹审批）。

HTTP 全部 monkeypatch，一个真请求都不发；落盘目录钉在 tmp_path。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from agentd.contracts import ToolCallDone, ToolCallStart
from agentd.kernel.llm import LLM, LLMText, LLMToolCall
from agentd.kernel.modes.agent import AgentMode
from agentd.kernel.modes.base import ModeContext
from agentd.kernel.models import Message
from agentd.kernel.tools import (
    ALL_TOOL_NAMES,
    ApprovalRequest,
    NativeToolbox,
    _IMAGE_SIZES,
    needs_approval,
)
from agentd.transports.acp_stdio import _ACP_KIND

import agentd.kernel.tools as tools_mod


FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def box(tmp_path: Path, **kw) -> NativeToolbox:
    kw.setdefault("cwd", tmp_path)
    return NativeToolbox(**kw)


# 配置表：fixture 把 boot.env_value 指到它上面。测试要改配置就直接 mutate 这个
# dict（清空 = 模拟「哪儿都没配」，走进程环境兜底分支）。
_CONF_TABLE: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _isolate(no_live_credentials, monkeypatch, tmp_path):
    """配置读真的 _image_conf()，只把 boot.env_value 指到 _CONF_TABLE；
    落盘目录钉在 tmp_path，绝不碰真 ~/.agentd。"""
    _CONF_TABLE.clear()
    _CONF_TABLE.update(
        {
            "AGENTD_IMAGE_API_KEY": "test-key",
            "AGENTD_IMAGE_BASE_URL": "https://image.example/v1",
            "AGENTD_IMAGE_MODEL": "glm-image",
        }
    )
    monkeypatch.setattr(
        "agentd.boot.env_value",
        lambda name, default="": _CONF_TABLE.get(name, default),
    )
    monkeypatch.setattr(
        tools_mod, "tool_results_root", lambda: tmp_path / "tool_results"
    )
    monkeypatch.delenv("AGENTD_IMAGE_API_KEY", raising=False)
    monkeypatch.delenv("AGENTD_IMAGE_BASE_URL", raising=False)
    monkeypatch.delenv("AGENTD_IMAGE_MODEL", raising=False)
    yield


# ---------------------------------------------------------------------------
# 注册 / 审批语义
# ---------------------------------------------------------------------------


def test_generate_image_registered_and_no_approval(tmp_path):
    tb = box(tmp_path)
    assert "generate_image" in ALL_TOOL_NAMES
    assert tb.kind_of("generate_image") == "other"
    assert tb.requires_permission("generate_image") is False
    # kind=other + requires=False：native 策略下不弹审批（all 策略仍会弹，符合矩阵）
    assert (
        needs_approval(requires=False, kind="other", destructive=False, policy="native")
        is False
    )
    # kind 必须落在 ACP 的合法映射里，否则客户端静默卡死
    assert tb.kind_of("generate_image") in _ACP_KIND.values()


def test_generate_image_schema_fed_to_llm(tmp_path):
    tb = box(tmp_path)
    schema = tb.tool_schema()
    mine = [s for s in schema if s["function"]["name"] == "generate_image"]
    assert len(mine) == 1
    params = mine[0]["function"]["parameters"]
    assert params["required"] == ["prompt"]
    assert params["properties"]["size"]["enum"] == list(_IMAGE_SIZES)


# ---------------------------------------------------------------------------
# 参数与配置校验
# ---------------------------------------------------------------------------


async def test_empty_prompt_is_error(tmp_path):
    out = await box(tmp_path).call("generate_image", json.dumps({"prompt": "  "}))
    assert out.startswith("[错误] ")
    assert "提示词" in out


async def test_missing_config_is_actionable_error(tmp_path):
    _CONF_TABLE.update({"AGENTD_IMAGE_API_KEY": "", "AGENTD_IMAGE_MODEL": ""})
    out = await box(tmp_path).call("generate_image", json.dumps({"prompt": "一只鸭子"}))
    assert out.startswith("[错误] ")
    # 报错要指路：多媒体模型设置 + 环境变量名都要点出来
    assert "多媒体模型" in out and "AGENTD_IMAGE_MODEL" in out


async def test_bad_size_is_error(tmp_path):
    out = await box(tmp_path).call(
        "generate_image", json.dumps({"prompt": "鸭子", "size": "33x44"})
    )
    assert out.startswith("[错误] ")
    assert "1024x1024" in out


# ---------------------------------------------------------------------------
# 两种响应形态 + 落盘
# ---------------------------------------------------------------------------


async def test_b64_response_writes_file(tmp_path, monkeypatch):
    async def fake_post(url, payload, headers, timeout):
        assert url == "https://image.example/v1/images/generations"
        assert payload["model"] == "glm-image"
        assert payload["prompt"] == "一只游泳的小鸭子"
        assert headers["Authorization"] == "Bearer test-key"
        return 200, {"data": [{"b64_json": base64.b64encode(FAKE_PNG).decode()}]}

    monkeypatch.setattr(tools_mod, "_image_http_post", fake_post)
    out = await box(tmp_path).call(
        "generate_image", json.dumps({"prompt": "一只游泳的小鸭子"})
    )
    # 跨仓库契约：开头必须是「已新建 <路径>（」，GUI 靠它提取产物路径
    assert out.startswith("已新建 ")
    path = Path(out.split("（")[0].removeprefix("已新建 ").strip())
    assert path.parent.name == "imagegen"
    assert path.read_bytes() == FAKE_PNG
    assert "1024x1024" in out and "glm-image" in out


async def test_url_response_refetches_bytes(tmp_path, monkeypatch):
    async def fake_post(url, payload, headers, timeout):
        return 200, {"data": [{"url": "https://cdn.example/img/abc"}]}

    async def fake_get(url, headers, timeout):
        assert url == "https://cdn.example/img/abc"
        return b"fake-jpeg-bytes", "image/jpeg"

    monkeypatch.setattr(tools_mod, "_image_http_post", fake_post)
    monkeypatch.setattr(tools_mod, "_image_http_get_bytes", fake_get)
    out = await box(tmp_path).call("generate_image", json.dumps({"prompt": "海报"}))
    assert out.startswith("已新建 ")
    path = Path(out.split("（")[0].removeprefix("已新建 ").strip())
    assert path.suffix == ".jpg"
    assert path.read_bytes() == b"fake-jpeg-bytes"


# ---------------------------------------------------------------------------
# 错误透传
# ---------------------------------------------------------------------------


async def test_http_error_is_surfaced(tmp_path, monkeypatch):
    async def fake_post(url, payload, headers, timeout):
        return 400, {"error": {"message": "模型不存在"}}

    monkeypatch.setattr(tools_mod, "_image_http_post", fake_post)
    out = await box(tmp_path).call("generate_image", json.dumps({"prompt": "x"}))
    assert out.startswith("[错误] ")
    assert "400" in out and "模型不存在" in out


async def test_network_failure_is_surfaced(tmp_path, monkeypatch):
    async def fake_post(url, payload, headers, timeout):
        return 0, {"error": "ConnectError: timed out"}

    monkeypatch.setattr(tools_mod, "_image_http_post", fake_post)
    out = await box(tmp_path).call("generate_image", json.dumps({"prompt": "x"}))
    assert out.startswith("[错误] ")
    assert "请求失败" in out


async def test_empty_data_is_error(tmp_path, monkeypatch):
    async def fake_post(url, payload, headers, timeout):
        return 200, {"data": []}

    monkeypatch.setattr(tools_mod, "_image_http_post", fake_post)
    out = await box(tmp_path).call("generate_image", json.dumps({"prompt": "x"}))
    assert out.startswith("[错误] ")
    assert "没有返回图像数据" in out


# ---------------------------------------------------------------------------
# 配置读取：boot.env_value 优先，进程环境兜底
# ---------------------------------------------------------------------------


def test_conf_prefers_env_value_over_os_environ(monkeypatch):
    # env_value（热配置链）有的值优先于进程环境
    _CONF_TABLE["AGENTD_IMAGE_MODEL"] = "glm-image"
    monkeypatch.setenv("AGENTD_IMAGE_MODEL", "cogview-4")
    conf = tools_mod._image_conf()
    assert conf["model"] == "glm-image"


def test_conf_falls_back_to_os_environ(monkeypatch):
    # 热配置链没给（表清空 → env_value 返回默认 ""）时，回落进程环境
    _CONF_TABLE.clear()
    monkeypatch.setenv("AGENTD_IMAGE_API_KEY", "env-key")
    monkeypatch.setenv("AGENTD_IMAGE_BASE_URL", "http://env.example/v1")
    monkeypatch.setenv("AGENTD_IMAGE_MODEL", "cogview-4")
    conf = tools_mod._image_conf()
    assert conf == {
        "api_key": "env-key",
        "base_url": "http://env.example/v1",
        "model": "cogview-4",
    }


# ---------------------------------------------------------------------------
# AgentMode 端到端：喂给 LLM → 调用 → 产物落盘 → 不弹审批
# ---------------------------------------------------------------------------


class ScriptedLLM(LLM):
    """按轮产出事件；记录每次收到的 tools 便于断言。"""

    def __init__(self, rounds: list[list]) -> None:
        self._rounds = list(rounds)
        self.seen_tools: list[list[dict] | None] = []

    async def stream_events(self, messages, *, system=None, tools=None):
        self.seen_tools.append(tools)
        round_events = self._rounds.pop(0) if self._rounds else [LLMText("（脚本耗尽）")]
        for event in round_events:
            yield event


async def test_agent_mode_generate_image_end_to_end(tmp_path, monkeypatch):
    async def fake_post(url, payload, headers, timeout):
        return 200, {"data": [{"b64_json": base64.b64encode(FAKE_PNG).decode()}]}

    monkeypatch.setattr(tools_mod, "_image_http_post", fake_post)

    llm = ScriptedLLM(
        [
            [
                LLMToolCall(
                    id="c1",
                    name="generate_image",
                    arguments=json.dumps({"prompt": "一只游泳的小鸭子"}),
                )
            ],
            [LLMText("图片已生成好了。")],
        ]
    )

    approvals: list[ApprovalRequest] = []

    async def approve(req: ApprovalRequest) -> bool:
        approvals.append(req)
        return True

    ctx = ModeContext(
        session_id="s",
        run_id="r",
        llm=llm,
        history=[Message.user("帮我生成一张小鸭子游泳的照片")],
        cwd=str(tmp_path),
        toolbox=box(tmp_path),
        approve=approve,
    )
    events = [e async for e in AgentMode().run(ctx, "帮我生成一张小鸭子游泳的照片")]

    # 工具确实被喂给了 LLM
    assert any(
        item["function"]["name"] == "generate_image" for item in (llm.seen_tools[0] or [])
    )
    starts = [e for e in events if isinstance(e, ToolCallStart)]
    dones = [e for e in events if isinstance(e, ToolCallDone)]
    assert [s.kind for s in starts] == ["other"]
    assert [d.status for d in dones] == ["completed"]
    assert "已新建 " in dones[0].output
    # 全程零审批（对话里一句话直接出图，不打断）
    assert approvals == []
    # 产物真的落盘
    generated = list((tmp_path / "tool_results" / "imagegen").iterdir())
    assert len(generated) == 1
    assert generated[0].read_bytes() == FAKE_PNG
