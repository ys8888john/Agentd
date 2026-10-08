"""tests 公共夹具。

一条硬规矩：**离线用例绝不能继承开发者真实的凭据**。

仓库根的 ``.env`` 里躺着真的 ``AGENTD_ZHIPU_API_KEY``（还有 MiMo 的），而配置取数链
``boot.env_value`` / ``current_settings`` 的优先级是
``overrides > 热配置文件 > 启动环境变量 > .env > 默认`` —— 也就是 .env **会被读进来**。
于是任何"以为没配 key"的用例都可能真去打智谱 API：结果随网络波动、失败原因还随机，
而且悄悄烧额度、拖慢套件。

（2026-10-08：web_search 的智谱后端从裸 ``os.getenv`` 改成走 ``env_value`` 之后，
``test_web_search_reports_connection_failure`` 就从"两个后端都连不上"变成"智谱通了、
直接返回 50 条结果"，于是红变绿/绿变红——正是这条规矩没被守住。）

用法：
    def test_xxx(no_live_credentials):        # 声明"没配任何 key"
        ...
    def test_yyy(no_live_credentials, monkeypatch):   # 想验某 key 生效就自己盖
        monkeypatch.setenv("AGENTD_ZHIPU_API_KEY", "test-key")
"""

from __future__ import annotations

import pytest

# 会被 .env / 真实环境带进来、又会让离线用例出网的 provider 凭据。
_CREDENTIAL_NAMES = (
    "AGENTD_ZHIPU_API_KEY",
    "ZHIPU_API_KEY",
    "AGENTD_MIMO_API_KEY",
    "MIMO_API_KEY",
    "AGENTD_OPENAI_API_KEY",
)


@pytest.fixture
def no_live_credentials(monkeypatch, tmp_path):
    """清空 provider 凭据的**所有**来源：os.environ / 启动快照 / .env / hotenv / overrides。

    ``_INITIAL_ENV`` 也要清：它是 ``agentd.boot`` 首次 import 时的 os.environ 快照，
    谁先给 os.environ 塞了 key，快照就被永久污染（用例顺序一变结果就变）。
    """
    from agentd import boot

    saved_overrides = dict(boot.RUNTIME_CONFIG.overrides)
    boot.RUNTIME_CONFIG.clear()
    for name in _CREDENTIAL_NAMES:
        monkeypatch.delitem(boot._INITIAL_ENV, name, raising=False)
        monkeypatch.delenv(name, raising=False)
    # .env / hotenv 指到不存在的路径：这两档也是凭据会溜进来的地方
    monkeypatch.setenv("AGENTD_DOTENV", str(tmp_path / "none.env"))
    monkeypatch.setenv("AGENTD_HOTENV", str(tmp_path / "none.json"))
    try:
        yield
    finally:
        boot.RUNTIME_CONFIG.overrides.clear()
        boot.RUNTIME_CONFIG.overrides.update(saved_overrides)
