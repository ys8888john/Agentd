"""配置读取与依赖组装。

配置优先级（后者盖前者）：
    代码默认值  <  项目根的 .env  <  真实环境变量

.env 只是"少敲几个 export"的便利，**不覆盖**真实环境变量——
否则 CI 里被一份误提交的 .env 悄悄改掉行为，排查起来很痛苦。
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .kernel.kernel import AgentKernel
from .kernel.llm import DEFAULT_PREFER, LLM, OllamaNativeLLM, OpenAICompatLLM
from .kernel.store import (
    InMemorySessionStore,
    SessionStore,
    SqliteSessionStore,
    default_db_path,
)

ENV_PREFIXES = ("AGENTD_", "FORGEAGENT_")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _parse_dotenv(text: str) -> dict[str, str]:
    """极简 .env 解析：KEY=VALUE，# 开头是注释，值两侧引号会被剥掉。

    刻意不引 python-dotenv —— 二十行能搞定的事不值得多一个依赖，
    而且这版的语义（只补缺、不覆盖）跟标准库的 override 行为不一样，自己写更清楚。
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """读 .env 并写进 os.environ（只补缺，不覆盖）。返回实际写入的项。

    查找顺序：显式传入的路径 → 环境变量 AGENTD_DOTENV → 当前目录 .env。
    找不到就安静返回空字典，不报错——没配 .env 是常态。
    """
    candidate = Path(path) if path else Path(os.getenv("AGENTD_DOTENV") or Path.cwd() / ".env")
    try:
        if not candidate.is_file():
            return {}
        parsed = _parse_dotenv(candidate.read_text(encoding="utf-8"))
    except OSError:
        return {}

    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if key.startswith(ENV_PREFIXES) and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


@dataclass(frozen=True)
class Settings:
    backend: str              # ollama | openai_compat | fake
    ollama_host: str
    ollama_model: str
    ollama_think: bool
    ollama_prefer: str
    openai_base_url: str
    openai_model: str
    openai_api_key: str
    # 小米 MiMo 开放平台（OpenAI 兼容端点，backend=mimo 时用）
    mimo_base_url: str
    mimo_model: str
    mimo_api_key: str
    fake_reply: str
    script_file: str           # backend=script：脚本文件路径（与 script_json 二选一）
    script_json: str           # backend=script：内联脚本 JSON（优先于 script_file）
    system_prompt: str | None
    store_backend: str          # sqlite | memory
    db_path: str
    # 原生工具（进程内 read_file/glob/grep/write_file/edit/run_command）
    tools: str                  # native | read_only | off
    tools_allow_outside: bool   # 是否允许碰 cwd 之外的路径
    tools_timeout: float        # run_command 超时（秒）
    tools_approve: str          # native | all | none


def load_settings() -> Settings:
    load_dotenv()  # 副作用：把 .env 里缺失的项补进 os.environ
    return Settings(
        backend=_env("AGENTD_LLM_BACKEND", "ollama"),
        ollama_host=_env("AGENTD_OLLAMA_HOST", "http://localhost:11434"),
        # 默认 auto：写死模型名在这台机器上 404 过一次（装的是 qwen3.5 不是 qwen3），
        # 而 404 的表现是"回复空白"，用户根本无从下手。改成让程序自己去 /api/tags 问。
        ollama_model=_env("AGENTD_OLLAMA_MODEL", "auto"),
        ollama_think=_env("AGENTD_OLLAMA_THINK", "false").lower() == "true",
        ollama_prefer=_env("AGENTD_OLLAMA_PREFER", DEFAULT_PREFER),
        openai_base_url=_env("AGENTD_OPENAI_BASE_URL", "http://localhost:11434/v1"),
        openai_model=_env("AGENTD_OPENAI_MODEL", "qwen3"),
        openai_api_key=_env("AGENTD_OPENAI_API_KEY", "ollama"),
        # MiMo 的 key 官方习惯叫 MIMO_API_KEY，但它不带 AGENTD_ 前缀，
        # 写进 .env 不会被 load_dotenv 读进来 —— 所以主名用 AGENTD_MIMO_API_KEY，
        # MIMO_API_KEY 作为兜底（直接 export 它时照常生效）。
        mimo_base_url=_env("AGENTD_MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
        mimo_model=_env("AGENTD_MIMO_MODEL", "mimo-v2.5-pro"),
        mimo_api_key=_env("AGENTD_MIMO_API_KEY", "") or os.getenv("MIMO_API_KEY", ""),
        fake_reply=_env("AGENTD_FAKE_REPLY", "这是 FakeLLM 的固定回复。"),
        script_file=_env("AGENTD_SCRIPT_FILE", ""),
        script_json=os.getenv("AGENTD_SCRIPT_JSON", ""),
        system_prompt=os.getenv("AGENTD_SYSTEM_PROMPT") or None,
        store_backend=_env("AGENTD_STORE", "sqlite").lower(),
        db_path=_env("AGENTD_DB_PATH", str(default_db_path())),
        tools=_env("AGENTD_TOOLS", "native").lower(),
        # 默认**不允许**越出 cwd：agent 的工作目录就是它的世界。
        # 越界读 .ssh / 系统配置这种事，出一次就够吓人了，放开要显式声明。
        tools_allow_outside=_env("AGENTD_TOOLS_ALLOW_OUTSIDE", "false").lower() == "true",
        tools_timeout=float(_env("AGENTD_TOOLS_TIMEOUT", "30")),
        # 默认只拦原生写/执行类（+ MCP 自己标了 destructive 的工具），见 tools.needs_approval
        tools_approve=_env("AGENTD_TOOLS_APPROVE", "native").lower(),
    )


def build_llm(settings: Settings | None = None) -> LLM:
    s = settings or load_settings()
    if s.backend == "ollama":
        return OllamaNativeLLM(
            host=s.ollama_host,
            model=s.ollama_model,
            think=s.ollama_think,
            prefer=s.ollama_prefer,
        )
    if s.backend == "openai_compat":
        return OpenAICompatLLM(
            base_url=s.openai_base_url,
            model=s.openai_model,
            api_key=s.openai_api_key,
            prefer=s.ollama_prefer,
        )
    if s.backend == "mimo":
        # 小米 MiMo 开放平台：与 OpenAI 兼容的 /v1 端点（chat/completions + models）。
        # 按量付费 sk- 开头的 Key；缺 key 在启动时报，别等第一次对话才炸。
        if not s.mimo_api_key:
            raise ValueError(
                "backend=mimo 需要 AGENTD_MIMO_API_KEY（.env / 环境变量均可，"
                "或直接 export MIMO_API_KEY=...）"
            )
        return OpenAICompatLLM(
            base_url=s.mimo_base_url,
            model=s.mimo_model,
            api_key=s.mimo_api_key,
        )
    if s.backend == "fake":
        from .kernel.llm import FakeLLM

        return FakeLLM(reply=s.fake_reply)
    if s.backend == "script":
        # 回放脚本：端到端验证 MCP 工具循环用，不依赖任何真实模型。
        from .kernel.llm import ScriptLLM

        if s.script_json.strip():
            return ScriptLLM(s.script_json)
        if s.script_file:
            try:
                raw = Path(s.script_file).read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(f"读不到脚本文件 {s.script_file}: {exc}") from exc
            return ScriptLLM(raw)
        raise ValueError("backend=script 需要 AGENTD_SCRIPT_JSON 或 AGENTD_SCRIPT_FILE")
    raise ValueError(f"未知 LLM 后端: {s.backend}")


def build_store(settings: Settings | None = None) -> SessionStore:
    """按配置造存储。默认 SQLite —— 持久化是默认行为，不持久化才要显式声明。"""
    s = settings or load_settings()
    if s.store_backend == "sqlite":
        try:
            return SqliteSessionStore(s.db_path)
        except (sqlite3.Error, OSError) as exc:
            # 不静默退回内存：那会让"我明明聊过，重启后没了"变成一个查不出来的问题。
            # 把退路写进报错里，让人自己选。
            raise RuntimeError(
                f"打不开会话库 {s.db_path}: {exc}\n"
                f"  可设 AGENTD_DB_PATH 换个位置，或 AGENTD_STORE=memory 退回内存存储（重启即丢）。"
            ) from exc
    if s.store_backend == "memory":
        return InMemorySessionStore()
    raise ValueError(f"未知存储后端: {s.store_backend}")


def build_kernel(settings: Settings | None = None) -> AgentKernel:
    s = settings or load_settings()
    return AgentKernel(
        llm=build_llm(s),
        store=build_store(s),
        system=s.system_prompt,
        native_tools=s.tools,
        tools_allow_outside=s.tools_allow_outside,
        tools_timeout=s.tools_timeout,
        approval_policy=s.tools_approve,
    )
