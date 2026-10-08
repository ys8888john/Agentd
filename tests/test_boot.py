"""boot.py 单元测试：.env 解析与优先级。

.env 的语义是"只补缺、不覆盖"——这条要是反了，CI 里一份误提交的 .env
就能悄悄改掉整个后端行为，而且极难察觉。所以必须钉死。
"""

from __future__ import annotations

import json
import os

import pytest

from agentd.boot import _parse_dotenv, build_llm, build_kernel, build_store, load_dotenv, load_settings
from agentd.kernel.llm import AUTO, OpenAICompatLLM
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


# ---- MiMo 后端 ----

def _mimo_env(monkeypatch) -> None:
    """把环境收干净，让每个测试从同一张白纸开始。"""
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_LLM_BACKEND", "mimo")
    monkeypatch.delenv("AGENTD_MIMO_API_KEY", raising=False)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("AGENTD_MIMO_MODEL", raising=False)
    monkeypatch.delenv("AGENTD_MIMO_BASE_URL", raising=False)


def test_build_llm_mimo_defaults(monkeypatch):
    _mimo_env(monkeypatch)
    monkeypatch.setenv("AGENTD_MIMO_API_KEY", "sk-test")

    llm = build_llm()
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.base_url == "https://api.xiaomimimo.com/v1"
    assert llm.model == "mimo-v2.5-pro"
    assert llm.api_key == "sk-test"


def test_build_llm_mimo_falls_back_to_mimo_api_key_env(monkeypatch):
    """官方习惯名 MIMO_API_KEY（真实环境变量）也要认 —— 但 .env 里写它不生效，见 boot.py。"""
    _mimo_env(monkeypatch)
    monkeypatch.setenv("MIMO_API_KEY", "sk-official-name")
    assert build_llm().api_key == "sk-official-name"


def test_build_llm_mimo_without_key_raises(monkeypatch):
    """缺 key 必须在启动时报，不能等用户发了第一条消息才炸。"""
    _mimo_env(monkeypatch)
    with pytest.raises(ValueError, match="AGENTD_MIMO_API_KEY"):
        build_llm()


# ---- 智谱 BigModel 后端 ----

def _zhipu_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_LLM_BACKEND", "zhipu")
    monkeypatch.delenv("AGENTD_ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("AGENTD_ZHIPU_MODEL", raising=False)
    monkeypatch.delenv("AGENTD_ZHIPU_BASE_URL", raising=False)


def test_build_llm_zhipu_defaults(monkeypatch):
    _zhipu_env(monkeypatch)
    monkeypatch.setenv("AGENTD_ZHIPU_API_KEY", "id.secret")

    llm = build_llm()
    assert isinstance(llm, OpenAICompatLLM)
    assert llm.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert llm.model == "glm-4.5-air"
    assert llm.api_key == "id.secret"


def test_build_llm_zhipu_falls_back_to_zhipu_api_key_env(monkeypatch):
    _zhipu_env(monkeypatch)
    monkeypatch.setenv("ZHIPU_API_KEY", "id.secret-from-env")
    assert build_llm().api_key == "id.secret-from-env"


def test_build_llm_zhipu_without_key_raises(monkeypatch):
    _zhipu_env(monkeypatch)
    with pytest.raises(ValueError, match="AGENTD_ZHIPU_API_KEY"):
        build_llm()


# ---- LiveLLM 热加载（provider / key / model / base_url 改了下一轮即用，不用重启）----


def test_live_llm_hot_reload(tmp_path, monkeypatch):
    """改 .env 或 RUNTIME_CONFIG.set 都能在下一轮对话生效；缺 key 抛错由内核转 ErrorEvent。"""

    from agentd.boot import LiveLLM, RUNTIME_CONFIG
    from agentd.kernel.llm import FakeLLM, OpenAICompatLLM

    env_file = tmp_path / ".env"
    env_file.write_text("AGENTD_LLM_BACKEND=fake\nAGENTD_FAKE_REPLY=hi\n", encoding="utf-8")
    monkeypatch.setenv("AGENTD_DOTENV", str(env_file))
    RUNTIME_CONFIG.clear()  # 隔离其它测试可能留下的覆盖层

    try:
        live = LiveLLM()

        # 启动：从 .env 读到 fake
        assert isinstance(live._resolve(), FakeLLM)

        # 程序化切换：RUNTIME_CONFIG.set -> 下一轮应是 openai_compat
        RUNTIME_CONFIG.set(
            backend="openai_compat",
            openai_base_url="http://x/v1",
            openai_model="q",
            openai_api_key="k",
        )
        swapped = live._resolve()
        assert isinstance(swapped, OpenAICompatLLM)
        assert swapped.base_url == "http://x/v1"
        # 同一轮内配置没变 -> 复用同一实例（Ollama 模型解析缓存等状态得以保留）
        assert live._resolve() is swapped

        # 改 .env 文件（不重启）：下一轮应自动热加载 zhipu
        env_file.write_text(
            "AGENTD_LLM_BACKEND=zhipu\nAGENTD_ZHIPU_API_KEY=id.secret\n",
            encoding="utf-8",
        )
        RUNTIME_CONFIG.clear()
        zhipu = live._resolve()
        assert isinstance(zhipu, OpenAICompatLLM)
        assert zhipu.base_url == "https://open.bigmodel.cn/api/paas/v4"

        # 缺 key 不应让进程崩，抛 ValueError 即可（内核 handle() 会转成 ErrorEvent）
        RUNTIME_CONFIG.set(backend="mimo")  # 不给 key
        with pytest.raises(ValueError, match="AGENTD_MIMO_API_KEY"):
            live._resolve()
    finally:
        RUNTIME_CONFIG.clear()


def test_build_kernel_returns_live_llm(monkeypatch):
    """build_kernel 现在返回热加载包装，provider 改了下一轮即用。"""
    from agentd.boot import LiveLLM, build_kernel

    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_LLM_BACKEND", "fake")
    assert isinstance(build_kernel().llm, LiveLLM)


def test_hotenv_carries_the_token_budget(tmp_path, monkeypatch):
    """token 预算必须能跟着 GUI 的 profile 一起热切换 —— 这是把预算做成
    **per-profile** 之后唯一真正重要的验收点。

    GUI 里每个模型配置各带一份 AGENTD_MAX_*_TOKENS，切模型时整组写进 hotenv.json；
    agentd 每轮 current_settings() 从那儿读。要是这条通路断了，症状是
    "切了本地小模型但预算还是云端那个 128k"，而且完全不会有报错。
    """
    from agentd.boot import RUNTIME_CONFIG, current_settings

    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    for key in ("AGENTD_MAX_CONTEXT_TOKENS", "AGENTD_MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    RUNTIME_CONFIG.clear()

    hot = tmp_path / "hotenv.json"
    hot.write_text(
        json.dumps({
            "AGENTD_LLM_BACKEND": "fake",
            "AGENTD_MAX_CONTEXT_TOKENS": "8192",
            "AGENTD_MAX_OUTPUT_TOKENS": "1024",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTD_HOTENV", str(hot))

    try:
        s = current_settings()
        assert s.max_context_tokens == 8192
        assert s.max_output_tokens == 1024
    finally:
        RUNTIME_CONFIG.clear()


def test_bad_token_budget_falls_back_instead_of_crashing(monkeypatch):
    """预算配错（写成 "8k" 这种）不该让 agent 起不来 —— 兜回 0（不限）并留日志。

    理由：预算是"优化项"，不是"能不能用"的前提。用户更该看到
    "预算没生效"（可查），而不是"程序打不开"。
    """
    from agentd.boot import RUNTIME_CONFIG, current_settings

    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_MAX_CONTEXT_TOKENS", "8k")
    monkeypatch.setenv("AGENTD_MAX_OUTPUT_TOKENS", "-5")
    RUNTIME_CONFIG.clear()

    s = current_settings()
    assert s.max_context_tokens == 0
    assert s.max_output_tokens == 0


def test_compact_ratio_defaults_to_80_and_is_clamped(monkeypatch):
    """默认 80%（留出压缩这一步本身要花的余量），越界的值要夹住而不是静默失效。"""
    from agentd.boot import RUNTIME_CONFIG, current_settings

    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    RUNTIME_CONFIG.clear()
    assert current_settings().compact_ratio == 80

    # 热切换这类值走 RUNTIME_CONFIG（真实环境变量是**启动时**的快照，改不动）
    RUNTIME_CONFIG.set(compact_ratio="300")
    assert current_settings().compact_ratio == 100  # 夹住：写 300 等于永远不触发
    RUNTIME_CONFIG.clear()


def test_kernel_reads_the_budget_lazily(tmp_path, monkeypatch):
    """自动压缩的预算必须**每轮重新问**，不能用启动时的死值。

    GUI 的 token 上限是按模型 profile 热写的（本地 8b 与云端 GLM 差三倍以上），
    定死了会变成"切到 128k 的模型、却还按 8k 的窗口在压缩"。
    """
    from agentd.boot import RUNTIME_CONFIG, build_kernel

    hot = tmp_path / "hotenv.json"
    hot.write_text(json.dumps({"AGENTD_MAX_CONTEXT_TOKENS": "4096"}), encoding="utf-8")
    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_HOTENV", str(hot))
    monkeypatch.setenv("AGENTD_STORE", "memory")
    RUNTIME_CONFIG.clear()
    try:
        kernel = build_kernel()
        assert kernel.context_budget() == 4096
        # GUI 切到另一个 profile：同一个内核对象下一轮就必须看到新值
        RUNTIME_CONFIG.set(max_context_tokens="16384")
        assert kernel.context_budget() == 16384
    finally:
        RUNTIME_CONFIG.clear()


def test_summary_settings_have_sane_defaults(monkeypatch):
    from agentd.boot import RUNTIME_CONFIG, current_settings

    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    for key in ("AGENTD_SUMMARY_EVERY", "AGENTD_SUMMARY_RECALL"):
        monkeypatch.delenv(key, raising=False)
    RUNTIME_CONFIG.clear()

    s = current_settings()
    # 默认开着：用户要的就是"换个会话还认识我"，默认关等于没做
    assert s.summary_every > 0
    assert s.summary_recall > 0


def test_current_settings_reads_hotenv_file(tmp_path, monkeypatch):
    """GUI 写入的热配置文件是跨进程切换 provider/模型的主通道：

    current_settings() 每轮都读它，所以改了下一轮即生效、不用重启 agentd 子进程。
    缺失的键回落到 .env / 默认（不会因热文件只写了部分字段就炸）。
    """
    from agentd.boot import RUNTIME_CONFIG, current_settings

    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    # 清掉所有可能干扰的真实环境变量
    for key in (
        "AGENTD_LLM_BACKEND", "AGENTD_FAKE_REPLY", "AGENTD_ZHIPU_API_KEY",
        "AGENTD_OLLAMA_MODEL", "ZHIPU_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    RUNTIME_CONFIG.clear()  # 隔离其它测试可能留下的覆盖层

    hot = tmp_path / "hotenv.json"
    hot.write_text(
        json.dumps({"AGENTD_LLM_BACKEND": "fake", "AGENTD_FAKE_REPLY": "from-hot-file"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTD_HOTENV", str(hot))

    try:
        s = current_settings()
        assert s.backend == "fake"
        assert s.fake_reply == "from-hot-file"
        # 热文件只写了部分字段：其余回落到默认值（不报错）
        assert s.ollama_model == AUTO

        # 改写热文件（不重启进程）：下一轮读取应反映新值
        hot.write_text(
            json.dumps({"AGENTD_LLM_BACKEND": "zhipu", "AGENTD_ZHIPU_API_KEY": "id.secret"}),
            encoding="utf-8",
        )
        s2 = current_settings()
        assert s2.backend == "zhipu"
        assert s2.zhipu_api_key == "id.secret"
        # 热文件没写的 zhipu model -> 回落默认 glm-4.5-air
        assert s2.zhipu_model == "glm-4.5-air"

        # 删掉热文件：回落到默认 backend（不再有热覆盖）
        hot.unlink()
        assert current_settings().backend == "ollama"
    finally:
        RUNTIME_CONFIG.clear()


def test_current_settings_hotenv_beats_initial_env(tmp_path, monkeypatch):
    """热配置文件档位高于启动时的真实环境变量（GUI 切换应覆盖启动时注入的配置）。"""
    from agentd.boot import RUNTIME_CONFIG, current_settings

    monkeypatch.setenv("AGENTD_DOTENV", "__nonexistent__")
    monkeypatch.setenv("AGENTD_LLM_BACKEND", "ollama")  # 启动环境里是 ollama
    RUNTIME_CONFIG.clear()

    hot = tmp_path / "hotenv.json"
    hot.write_text(json.dumps({"AGENTD_LLM_BACKEND": "zhipu"}), encoding="utf-8")
    monkeypatch.setenv("AGENTD_HOTENV", str(hot))

    try:
        assert current_settings().backend == "zhipu"  # 热文件盖过启动环境
    finally:
        RUNTIME_CONFIG.clear()
