"""「完全信任」开关（agentd/trust.py）测试。

盯四件事：
1. 标记的写 / 读 / 删 / 幂等；
2. 优先级：显式配置 > 信任标记 > 默认（effective_approval_policy）；
3. 坏文件（不存在 / 坏 JSON / 缺字段 / 非 dict）一律按"未开启" —— 宁可多问
   一句，不能因为一个坏文件把闸门全拆了；
4. boot._build_settings 的接线：没显式设 + 标记在 → tools_approve="none"；
   显式设了 → 配置赢，标记不盖它。
"""

from __future__ import annotations

import json

import pytest

from agentd import trust
from agentd.boot import _build_settings


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """把 trust_file 指到 tmp_path（真 home 不碰）。"""
    marker = tmp_path / "trust_all.json"
    monkeypatch.setattr(trust, "trust_file", lambda: marker)
    return marker


# ---- 1. 标记读写 ----


def test_enable_creates_marker_with_metadata(fake_home):
    path = trust.enable_trust()
    assert path == fake_home and path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["enabled"] is True
    assert data["since"]  # ISO 时间戳，人可读
    assert trust.trust_enabled() is True


def test_enable_is_idempotent(fake_home):
    trust.enable_trust()
    trust.enable_trust()  # 重复点只刷新时间戳，不炸、不写坏
    data = json.loads(fake_home.read_text(encoding="utf-8"))
    assert data["enabled"] is True
    assert trust.trust_enabled() is True


def test_disable_removes_marker_and_is_idempotent(fake_home):
    trust.enable_trust()
    trust.disable_trust()
    assert not fake_home.exists()
    assert trust.trust_enabled() is False
    trust.disable_trust()  # 本来就不在也算成功
    assert trust.trust_enabled() is False


def test_trust_file_lives_under_agentd_home():
    """默认位置必须是 ~/.agentd/trust_all.json —— 别的代码/文档都按这个路径指路。"""
    from pathlib import Path

    assert trust.trust_file() == Path.home() / ".agentd" / "trust_all.json"


# ---- 2. 优先级 ----


@pytest.mark.parametrize(
    ("policy", "explicit", "marker_on", "expected"),
    [
        ("native", True, False, "native"),   # 显式配置永远赢
        ("all", True, False, "all"),
        ("none", True, False, "none"),
        ("native", True, True, "native"),    # 显式配置连标记也压得住
        ("native", False, False, "native"),  # 没标记：原样
        ("native", False, True, "none"),     # 有标记：折成 none
        ("all", False, True, "none"),        # 标记把任何非显式策略统一折成 none
    ],
)
def test_effective_policy_matrix(fake_home, policy, explicit, marker_on, expected):
    if marker_on:
        trust.enable_trust()
    assert trust.effective_approval_policy(policy, explicit=explicit) == expected


def test_explicit_config_beats_marker(fake_home):
    """用户在 .env 里显式管这个行为时，点击记忆不许盖过去。"""
    trust.enable_trust()
    assert trust.effective_approval_policy("native", explicit=True) == "native"


# ---- 3. 坏文件一律按未开启 ----


@pytest.mark.parametrize(
    "content",
    [
        None,                      # 文件不存在
        "",                        # 空文件
        "{ 坏 json",               # 解析失败
        '["not", "a", "dict"]',    # 类型不对
        '{"enabled": false}',      # 显式关过的旧标记
        "{}",                      # 缺字段
    ],
)
def test_corrupt_marker_means_disabled(fake_home, content):
    if content is not None:
        fake_home.write_text(content, encoding="utf-8")
    assert trust.trust_enabled() is False
    assert trust.effective_approval_policy("native", explicit=False) == "native"


# ---- 4. boot 接线 ----


def _val_of(**overrides: str):
    """造一个只认白名单配置的 val（其余全走默认）—— 不碰真实环境。"""

    def val(name: str, default: str) -> str:
        return overrides.get(name, default)

    return val


def test_boot_marker_disables_approval(fake_home):
    trust.enable_trust()
    s = _build_settings(_val_of())
    assert s.tools_approve == "none"


def test_boot_explicit_env_beats_marker(fake_home):
    trust.enable_trust()
    s = _build_settings(_val_of(**{"AGENTD_TOOLS_APPROVE": "native"}))
    assert s.tools_approve == "native"


def test_boot_without_marker_keeps_default(fake_home):
    s = _build_settings(_val_of())
    assert s.tools_approve == "native"


def test_boot_explicit_none_still_none(fake_home):
    """显式 none 与标记 none 殊途同归，但不能因为标记不在就把它弄丢。"""
    s = _build_settings(_val_of(**{"AGENTD_TOOLS_APPROVE": "none"}))
    assert s.tools_approve == "none"
