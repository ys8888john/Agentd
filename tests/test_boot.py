"""boot.py 单元测试：.env 解析与优先级。

.env 的语义是"只补缺、不覆盖"——这条要是反了，CI 里一份误提交的 .env
就能悄悄改掉整个后端行为，而且极难察觉。所以必须钉死。
"""

from __future__ import annotations

import os

import pytest

from agentd.boot import _parse_dotenv, build_kernel, build_store, load_dotenv, load_settings
from agentd.kernel.llm import AUTO
from agentd.kernel.store import InMemorySessionStore, SqliteSessionStore


# ---- _parse_dotenv ----

def test_parse_dotenv_basic():
    assert _parse_dotenv("A=1\nB=2") == {"A": "1", "B": "2"}


def test_parse_dotenv_strips_quotes():
    assert _parse_dotenv("A='x'\nB=\"y\"") == {"A": "x", "B": "y"}


def test_parse_dotenv_ignores_comments_and_blanks():
    text = "# 注释\n\nA=1\n  # 缩进注释\nB=2"
    assert _parse_dotenv(text) == {"A": "1", "B": "2"}


def test_parse_dotenv_keeps_spaces_inside_value():
    # 只剥两侧空白，值中间的空格得留着（比如提示词）
    assert _parse_dotenv("AGENTD_SYSTEM_PROMPT= 你是个 好助手 ") == {
        "AGENTD_SYSTEM_PROMPT": "你是个 好助手"
    }


def test_parse_dotenv_tolerates_garbage():
    assert _parse_dotenv("这不是配置\n=没键\n") == {}


# ---- load_dotenv 的优先级 ----

def test_load_dotenv_fills_missing_keys(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("AGENTD_LLM_BACKEND=fake\n", encoding="utf-8")
    monkeypatch.delenv("AGENTD_LLM_BACKEND", raising=False)

    applied = load_dotenv(f)
    assert applied == {"AGENTD_LLM_BACKEND": "fake"}
    assert os.environ["AGENTD_LLM_BACKEND"] == "fake"


def test_load_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    """真实环境变量优先 —— 这是本文件最重要的断言。"""
    f = tmp_path / ".env"
    f.write_text("AGENTD_LLM_BACKEND=fake\n", encoding="utf-8")
    monkeypatch.setenv("AGENTD_LLM_BACKEND", "ollama")

    applied = load_dotenv(f)
    assert applied == {}
    assert os.environ["AGENTD_LLM_BACKEND"] == "ollama"


def test_load_dotenv_ignores_unprefixed_keys(tmp_path, monkeypatch):
    """.env 里非本项目的键不能被塞进环境，避免污染其它进程。"""
    f = tmp_path / ".env"
    f.write_text("PATH=/hacked\nOTHER=1\n", encoding="utf-8")
    before = os.environ.get("PATH")

    applied = load_dotenv(f)
    assert applied == {}
    assert os.environ.get("PATH") == before
    assert "OTHER" not in os.environ


def test_load_dotenv_missing_file_is_silent(tmp_path, monkeypatch):
    # 没配 .env 是常态，不能报错
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_load_dotenv_honors_AGENTD_DOTENV(tmp_path, monkeypatch):
    f = tmp_path / "custom.env"
    f.write_text("AGENTD_OLLAMA_MODEL=my-model\n", encoding="utf-8")
    monkeypatch.setenv("AGENTD_DOTENV", str(f))
    monkeypatch.delenv("AGENTD_OLLAMA_MODEL", raising=False)

    load_dotenv()
    assert os.environ["AGENTD_OLLAMA_MODEL"] == "my-model"


# ---- load_settings ----

def test_settings_default_model_is_auto(monkeypatch):
    """默认必须是 auto。

    写死 qwen3 在这台机器上 404 过（装的是 qwen3.5），
    而 404 的表现是"回复空白"，用户无从下手。
    """
    monkeypatch.delenv("AGENTD_OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    assert load_settings().ollama_model == AUTO


def test_settings_reads_from_dotenv(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("AGENTD_OLLAMA_MODEL=qwen3.5:9b-text\n", encoding="utf-8")
    monkeypatch.setenv("AGENTD_DOTENV", str(f))
    monkeypatch.delenv("AGENTD_OLLAMA_MODEL", raising=False)

    assert load_settings().ollama_model == "qwen3.5:9b-text"


@pytest.mark.parametrize("value,expected", [("true", True), ("TRUE", True), ("false", False), ("", False)])
def test_settings_think_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("AGENTD_OLLAMA_THINK", value)
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    assert load_settings().ollama_think is expected


# ---- 存储接线 ----

def test_settings_default_store_is_sqlite(monkeypatch):
    """默认是持久化的 —— 想要"重启即丢"得显式声明，不能反过来。"""
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.delenv("AGENTD_STORE", raising=False)
    assert load_settings().store_backend == "sqlite"


def test_build_kernel_uses_sqlite_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_LLM_BACKEND", "fake")
    monkeypatch.setenv("AGENTD_DB_PATH", str(tmp_path / "sessions.db"))

    k = build_kernel()
    assert isinstance(k.store, SqliteSessionStore)
    k.store.close()  # type: ignore[attr-defined]


def test_build_store_memory_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_STORE", "memory")
    # 内存后端不该在磁盘上留下任何东西
    assert isinstance(build_store(), InMemorySessionStore)
    assert list(tmp_path.iterdir()) == []


def test_build_store_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_STORE", "redis")
    with pytest.raises(ValueError, match="未知存储后端"):
        build_store()


def test_build_store_reports_unusable_db_path(tmp_path, monkeypatch):
    """打不开库要报错 + 给出退路，不能静默退回内存（那会丢数据还查不出来）。"""
    blocker = tmp_path / "iam_a_file"
    blocker.write_text("", encoding="utf-8")

    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_STORE", "sqlite")
    # 让一个普通文件当父目录 —— mkdir 必然失败，且与权限无关
    monkeypatch.setenv("AGENTD_DB_PATH", str(blocker / "sessions.db"))

    with pytest.raises(RuntimeError, match="AGENTD_STORE=memory"):
        build_store()
