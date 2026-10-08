"""配置读取与依赖组装。

配置优先级（后者盖前者）：
    代码默认值  <  项目根的 .env  <  真实环境变量

.env 只是"少敲几个 export"的便利，**不覆盖**真实环境变量——
否则 CI 里被一份误提交的 .env 悄悄改掉行为，排查起来很痛苦。
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .kernel.context import (
    DEFAULT_COMPACT_RATIO,
    NO_TRIM,
    TrimReport,
    describe_trim,
    estimate_messages_tokens,
    estimate_tokens,
    estimate_tools_tokens,
    trim_history,
)
from .kernel.kernel import AgentKernel
from .kernel.llm import (
    DEFAULT_PREFER,
    LLM,
    LLMNotice,
    LLMText,
    LLMToolCall,
    Message,
    OllamaNativeLLM,
    OpenAICompatLLM,
)
from .kernel.store import (
    InMemorySessionStore,
    SessionStore,
    SqliteSessionStore,
    default_db_path,
)

ENV_PREFIXES = ("AGENTD_", "FORGEAGENT_")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(val: Callable[[str, str], str], name: str, default: int) -> int:
    """读一个整数配置项（token 预算这类）。

    写错时**不抛异常**：预算配错不该让整个 agent 起不来。兜回默认值并在 stderr
    留一句 —— 用户看到"预算怎么没生效"比看到"程序打不开"好查得多。
    负值一并夹到 0（0 表示"不限"），免得下游拿负数去做大小比较。
    """
    raw = val(name, str(default)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        print(
            f"[agentd] {name}={raw!r} 不是合法整数，按 {default} 处理",
            file=sys.stderr,
            flush=True,
        )
        return default
    return max(parsed, 0)


def _percent(val: Callable[[str, str], str], name: str, default: int) -> int:
    """读一个百分比配置项（自动压缩的触发线）。

    夹在 0~100 之间：触发线写 200 等于永远不触发（还让人以为设对了），
    写负数则无意中关掉整个功能 —— 两种都比报错更难查。
    """
    parsed = _int(val, name, default)
    return min(max(parsed, 0), 100)


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
    backend: str              # ollama | openai_compat | mimo | zhipu | fake | script
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
    # 智谱 BigModel 开放平台（GLM 系列，OpenAI 兼容端点，backend=zhipu 时用）
    zhipu_base_url: str
    zhipu_model: str
    zhipu_api_key: str
    fake_reply: str
    script_file: str           # backend=script：脚本文件路径（与 script_json 二选一）
    script_json: str           # backend=script：内联脚本 JSON（优先于 script_file）
    system_prompt: str | None
    # ---- token 预算 ----
    # max_context_tokens：喂给模型的上下文上限（**只数 historical message，不含
    #    system prompt 和 tools schema**）。0 = 不限，等同加入这一层之前的行为。
    #   必须按所选模型的真实窗口设 —— 本地 40k 的模型和云端 128k 的模型差 3 倍，
    #   写死一个全局值必然"一边浪费一边溢出"，所以默认值是 0（不限），
    #   由使用者（GUI 的 profile 或 .env）按模型给。
    max_context_tokens: int
    # max_output_tokens：单次回复的上限，塞进 num_predict / max_tokens。
    #   0 = 交给后端默认（大多数情况下就是它的最大输出）。
    max_output_tokens: int
    # ---- 跨会话记忆 ----
    # summary_every：累积这么多条消息压一次摘要 + 抽一次事实。0 = 完全关掉记忆
    #   （连"读历史摘要"也一起关，新会话就是干净的第一次见面）。
    # summary_recall：新会话的 system prompt 里放几条历史摘要。
    summary_every: int
    summary_recall: int
    # 上下文占用到预算的百分之多少就自动压一次（0 = 只保留溢出时的硬裁剪）。
    # 默认 80：留两成给"本轮提问 + 工具往返 + 模型回答"，因为压缩本身还要额外
    # 调一次模型 —— 等到满格再压，就只能硬丢了。
    compact_ratio: int
    # 事实层文件位置；空串 = memory.DEFAULT_MEMORY_FILE（~/.agentd/memory.md）
    memory_path: str
    store_backend: str          # sqlite | memory
    db_path: str
    # 原生工具（进程内 read_file/glob/grep/write_file/edit/run_command）
    tools: str                  # native | read_only | off
    tools_allow_outside: bool   # 是否允许碰 cwd 之外的路径
    tools_timeout: float        # run_command 超时（秒）
    tools_approve: str          # native | all | none


def _build_settings(val: Callable[[str, str], str]) -> Settings:
    """按给定的取数函数装配 Settings。val(name, default) 决定每个字段从哪来——

    - load_settings 传 _env（读 os.environ，含 load_dotenv 补进来的 .env 项）；
    - current_settings 传 _hot_val（程序化 override > 启动真实环境变量 > 重新解析的 .env > 默认），
      用来热加载。
    """
    return Settings(
        backend=val("AGENTD_LLM_BACKEND", "ollama"),
        ollama_host=val("AGENTD_OLLAMA_HOST", "http://localhost:11434"),
        # 默认 auto：写死模型名在这台机器上 404 过一次（装的是 qwen3.5 不是 qwen3），
        # 而 404 的表现是"回复空白"，用户根本无从下手。改成让程序自己去 /api/tags 问。
        ollama_model=val("AGENTD_OLLAMA_MODEL", "auto"),
        ollama_think=val("AGENTD_OLLAMA_THINK", "false").lower() == "true",
        ollama_prefer=val("AGENTD_OLLAMA_PREFER", DEFAULT_PREFER),
        openai_base_url=val("AGENTD_OPENAI_BASE_URL", "http://localhost:11434/v1"),
        openai_model=val("AGENTD_OPENAI_MODEL", "qwen3"),
        openai_api_key=val("AGENTD_OPENAI_API_KEY", "ollama"),
        # MiMo 的 key 官方习惯叫 MIMO_API_KEY，但它不带 AGENTD_ 前缀，
        # 写进 .env 不会被 load_dotenv 读进来 —— 所以主名用 AGENTD_MIMO_API_KEY，
        # MIMO_API_KEY 作为兜底（直接 export 它时照常生效）。
        mimo_base_url=val("AGENTD_MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
        mimo_model=val("AGENTD_MIMO_MODEL", "mimo-v2.5-pro"),
        mimo_api_key=val("AGENTD_MIMO_API_KEY", "") or os.getenv("MIMO_API_KEY", ""),
        # 智谱：AGENTD_ZHIPU_API_KEY 为主（.env 能生效），ZHIPU_API_KEY 兜底
        zhipu_base_url=val("AGENTD_ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        zhipu_model=val("AGENTD_ZHIPU_MODEL", "glm-4.5-air"),
        zhipu_api_key=val("AGENTD_ZHIPU_API_KEY", "") or os.getenv("ZHIPU_API_KEY", ""),
        fake_reply=val("AGENTD_FAKE_REPLY", "这是 FakeLLM 的固定回复。"),
        script_file=val("AGENTD_SCRIPT_FILE", ""),
        script_json=os.getenv("AGENTD_SCRIPT_JSON", ""),
        system_prompt=os.getenv("AGENTD_SYSTEM_PROMPT") or None,
        max_context_tokens=_int(val, "AGENTD_MAX_CONTEXT_TOKENS", 0),
        max_output_tokens=_int(val, "AGENTD_MAX_OUTPUT_TOKENS", 0),
        summary_every=_int(val, "AGENTD_SUMMARY_EVERY", 20),
        summary_recall=_int(val, "AGENTD_SUMMARY_RECALL", 6),
        compact_ratio=_percent(val, "AGENTD_COMPACT_RATIO", DEFAULT_COMPACT_RATIO),
        memory_path=val("AGENTD_MEMORY_FILE", ""),
        store_backend=val("AGENTD_STORE", "sqlite").lower(),
        db_path=val("AGENTD_DB_PATH", str(default_db_path())),
        tools=val("AGENTD_TOOLS", "native").lower(),
        # 默认**不允许**越出 cwd：agent 的工作目录就是它的世界。
        # 越界读 .ssh / 系统配置这种事，出一次就够吓人了，放开要显式声明。
        tools_allow_outside=val("AGENTD_TOOLS_ALLOW_OUTSIDE", "false").lower() == "true",
        tools_timeout=float(val("AGENTD_TOOLS_TIMEOUT", "30")),
        # 默认只拦原生写/执行类（+ MCP 自己标了 destructive 的工具），见 tools.needs_approval
        tools_approve=val("AGENTD_TOOLS_APPROVE", "native").lower(),
    )


def load_settings() -> Settings:
    load_dotenv()  # 副作用：把 .env 里缺失的项补进 os.environ
    return _build_settings(_env)


def _ollama_options(max_output_tokens: int) -> dict | None:
    """把 max_output_tokens 翻译成 Ollama 的 ``options.num_predict``。

    刻意**不**顺手设 ``num_ctx``：那是上下文窗口大小，只有使用者知道自己拉的
    模型是 8k 还是 40k，替他猜一个必然错 —— 而且设小了 Ollama 会真的按这个值
    截断输入，症状跟"机器人变笨"一模一样，极难自查。要控输入量就该用
    max_context_tokens 明确地裁：裁了几条、为什么裁，都能向用户讲清楚。

    返回 None 而不是空 dict：给 options 传空对象，某些 Ollama 版本会告警。
    """
    if max_output_tokens <= 0:
        return None
    return {"num_predict": max_output_tokens}


def build_llm(settings: Settings | None = None) -> LLM:
    s = settings or load_settings()
    if s.backend == "ollama":
        return OllamaNativeLLM(
            host=s.ollama_host,
            model=s.ollama_model,
            think=s.ollama_think,
            options=_ollama_options(s.max_output_tokens),
            prefer=s.ollama_prefer,
        )
    if s.backend == "openai_compat":
        return OpenAICompatLLM(
            base_url=s.openai_base_url,
            model=s.openai_model,
            api_key=s.openai_api_key,
            prefer=s.ollama_prefer,
            max_tokens=s.max_output_tokens,
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
            max_tokens=s.max_output_tokens,
        )
    if s.backend == "zhipu":
        # 智谱 BigModel：与 OpenAI 兼容的 /api/paas/v4 端点（glm-4.x 系列）。
        # 缺 key 在启动时报，别等第一次对话才炸。
        if not s.zhipu_api_key:
            raise ValueError(
                "backend=zhipu 需要 AGENTD_ZHIPU_API_KEY（.env / 环境变量均可，"
                "或直接 export ZHIPU_API_KEY=...）"
            )
        return OpenAICompatLLM(
            base_url=s.zhipu_base_url,
            model=s.zhipu_model,
            api_key=s.zhipu_api_key,
            max_tokens=s.max_output_tokens,
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


# 进程启动时的真实环境变量快照：在 load_dotenv 往 os.environ 里塞 .env 项之前抓取，
# 作为热加载层里"最高优先级且不可变"的一档。这样直接改 .env 不会卡在启动时被填进
# os.environ 的旧值上，而真正在进程外 export 的环境变量（CI 场景）依旧最优先、且无法被
# .env 覆盖——保留了原先"真实环境变量不被 .env 悄悄改掉"的设计意图。
_INITIAL_ENV: dict[str, str] = dict(os.environ)


@dataclass
class RuntimeConfig:
    """运行时可变配置（仅 LLM provider 相关字段热加载）。

    优先级（current_settings 每轮读取时逐档回落）：
        RUNTIME_CONFIG.overrides  >  GUI 热配置文件  >  启动时的真实环境变量  >  重新解析的 .env  >  代码默认

    - ``overrides``：代码路径（admin 命令 / 未来扩展）调用 ``.set(...)`` 即时切换，
      例如 ``RUNTIME_CONFIG.set(backend="mimo", mimo_api_key="sk-...")``；
    - **GUI 热配置文件**（``AGENTD_HOTENV`` 指向的 JSON）：GUI 切换模型/provider 时把
      选中的 profile 环境变量写进去，运行中的 agentd **下一轮对话即生效、不用重启子进程**；
      这是 GUI 前端管理 provider、会话中随时切模型的主通道（见 ForgeAgent-GUI）；
    - 真实环境变量：进程启动时固定；
    - ``.env`` 文件：每次都重新解析，所以**直接改 .env 下一轮对话即生效，无需重启**。
    """

    overrides: dict[str, str] = field(default_factory=dict)

    def set(self, **fields: str) -> None:
        # 空字符串视为"清回默认"，不写入覆盖层。
        # 短名（backend / mimo_api_key …）自动映射到 AGENTD_* 环境变量名；
        # 直接传全大写 AGENTD_* 也接受。
        for key, value in fields.items():
            if value in (None, ""):
                continue
            env = _SHORT_TO_ENV.get(key, key if key.isupper() else key.upper())
            self.overrides[env] = value

    def clear(self) -> None:
        self.overrides.clear()


# 全局单例：LiveLLM 每轮读取它拿到的"当前"配置。改 provider 即改这里（或改 .env）。
RUNTIME_CONFIG = RuntimeConfig()

# set() 的短名字 -> 真实环境变量名，省得调用方记全大写前缀。
_SHORT_TO_ENV = {
    "backend": "AGENTD_LLM_BACKEND",
    "ollama_host": "AGENTD_OLLAMA_HOST",
    "ollama_model": "AGENTD_OLLAMA_MODEL",
    "ollama_think": "AGENTD_OLLAMA_THINK",
    "ollama_prefer": "AGENTD_OLLAMA_PREFER",
    "openai_base_url": "AGENTD_OPENAI_BASE_URL",
    "openai_model": "AGENTD_OPENAI_MODEL",
    "openai_api_key": "AGENTD_OPENAI_API_KEY",
    "mimo_base_url": "AGENTD_MIMO_BASE_URL",
    "mimo_model": "AGENTD_MIMO_MODEL",
    "mimo_api_key": "AGENTD_MIMO_API_KEY",
    "zhipu_base_url": "AGENTD_ZHIPU_BASE_URL",
    "zhipu_model": "AGENTD_ZHIPU_MODEL",
    "zhipu_api_key": "AGENTD_ZHIPU_API_KEY",
    # token 预算：GUI 按模型 profile 热改写（见 ForgeAgent-GUI 的模型弹窗）
    "max_context_tokens": "AGENTD_MAX_CONTEXT_TOKENS",
    "max_output_tokens": "AGENTD_MAX_OUTPUT_TOKENS",
    # 记忆开关也可以热改：想临时"这一轮当新会话聊"就把 summary_every 设 0
    "summary_every": "AGENTD_SUMMARY_EVERY",
    # 自动压缩的触发线：0 = 关掉（只剩溢出时的硬裁剪）
    "compact_ratio": "AGENTD_COMPACT_RATIO",
    "summary_recall": "AGENTD_SUMMARY_RECALL",
}


def _read_dotenv_fresh() -> dict[str, str]:
    """重新解析 .env（只读、不改 os.environ），让文件改动热生效。"""
    candidate = Path(os.getenv("AGENTD_DOTENV") or Path.cwd() / ".env")
    try:
        if candidate.is_file():
            return _parse_dotenv(candidate.read_text(encoding="utf-8"))
    except OSError:
        return {}
    return {}


def _read_hotenv() -> dict[str, str]:
    """读 GUI 写入的热配置文件 —— 跨进程切换 provider/模型的主通道。

    GUI（ForgeAgent-GUI）切换模型时，把选中的 profile 环境变量写进这个 JSON
    文件；运行中的 agentd 每轮对话重新读它，于是**下一轮即用、无需重启子进程**。

    路径取环境变量 ``AGENTD_HOTENV``（GUI 启动 agentd 时设好，指向它自己管理的
    ``~/.agentd/gui/hotenv.json``）；缺省回落到 ``~/.agentd/hotenv.json``
    （非 GUI 场景，一般不存在 = 等于没有热覆盖）。

    文件缺失 / 损坏 / 非 dict 一律返回空 dict —— 等于"没有热覆盖"，回落到下一档
    （启动环境变量 / .env / 默认）。只读不写，且只收 ``AGENTD_/FORGEAGENT_`` 前缀的键。
    """
    path = os.getenv("AGENTD_HOTENV")
    if not path:
        path = str(Path.home() / ".agentd" / "hotenv.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        k: str(v)
        for k, v in data.items()
        if isinstance(k, str) and k.startswith(("AGENTD_", "FORGEAGENT_"))
    }


def current_settings() -> Settings:
    """当前生效的配置，供热加载后端每轮读取。

    优先级（逐档回落）：
        RUNTIME_CONFIG.overrides  >  GUI 热配置文件  >  启动时的真实环境变量  >  重新解析的 .env  >  默认
    """
    dotenv = _read_dotenv_fresh()
    initial = _INITIAL_ENV
    hot = _read_hotenv()

    def val(name: str, default: str) -> str:
        if name in RUNTIME_CONFIG.overrides:
            return RUNTIME_CONFIG.overrides[name]
        if name in hot:
            return hot[name]
        if name in initial:
            return initial[name]
        return dotenv.get(name, default)

    return _build_settings(val)


class LiveLLM(LLM):
    """热加载 LLM 后端 —— provider / key / model / base_url 随时改、下一轮即用，不用重启。

    内核只认 LLM 接口；本包装在**每轮对话开始前**按 current_settings() 重建底层后端，
    于是改 ``AGENTD_LLM_BACKEND`` / key / model / base_url（改 .env 或 RUNTIME_CONFIG.set）
    在下一轮对话即生效。

    只有当 provider 相关配置签名变化时才真正重建；正常对话复用同一实例，
    Ollama 的模型解析缓存等状态得以保留。缺 key / 未知后端时 build_llm 抛 ValueError，
    由内核 handle() 统一转成 ErrorEvent，不会让整轮挂掉。
    """

    def __init__(self, settings_fn: Callable[[], Settings] = current_settings) -> None:
        self._settings_fn = settings_fn
        self._sig: tuple | None = None
        self._instance: LLM | None = None

    def _resolve(self) -> LLM:
        s = self._settings_fn()
        sig = (
            s.backend,
            s.ollama_host,
            s.ollama_model,
            s.ollama_think,
            s.ollama_prefer,
            s.openai_base_url,
            s.openai_model,
            s.openai_api_key,
            s.mimo_base_url,
            s.mimo_model,
            s.mimo_api_key,
            s.zhipu_base_url,
            s.zhipu_model,
            s.zhipu_api_key,
            # max_output_tokens 要进签名：它直接决定传给后端的参数，改了必须重建
            s.max_output_tokens,
        )
        if self._instance is None or self._sig != sig:
            self._sig = sig
            # 缺 key / 未知后端会在此抛 ValueError，由内核 handle() 转成 ErrorEvent
            self._instance = build_llm(s)
        return self._instance

    def _trim(
        self, messages: list[Message], system: str | None, tools: list[dict] | None
    ) -> tuple[list[Message], LLMNotice | None]:
        """按剩余的上下文预算裁剪消息序列；裁了就返回一行提示。

        为什么放在这一层而不是内核里：这里已经是"每次真实 LLM 调用"的唯一出口。
        多步工具模式下一次用户提问会触发 MAX_STEPS 次调用，消息列表一路增长，
        只有每次调用前都裁，才能管住中途塞进来的大段工具输出。
        """
        budget = self._settings_fn().max_context_tokens
        if budget <= 0:
            return messages, None  # 0 = 不限，与引入预算之前的行为完全一致

        # 预算是对"整个窗口"的，不能只数历史：system prompt 和 tools schema
        # 在同一个窗口里同样占位置（放着 25 个 MCP 工具的 schema 可不是小数）。
        headroom = budget - estimate_tokens(system or "") - _tools_tokens(tools)
        if headroom <= 0:
            # system + tools 已经把预算吃光了。此时再裁就会裁到用户刚说的那句话，
            # 那还不如让它超 —— 宁可模型截断回答，也不能假装没听见提问。
            return messages, None

        trimmed, report = trim_history(messages, headroom)
        if not report.happened:
            return trimmed, None
        return trimmed, LLMNotice(describe_trim(report, budget=budget))

    async def stream_events(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMText | LLMToolCall | LLMNotice]:
        llm = self._resolve()
        trimmed, notice = self._trim(messages, system, tools)
        if notice is not None:
            yield notice
        async for ev in llm.stream_events(trimmed, system=system, tools=tools):
            yield ev


_tools_tokens = estimate_tools_tokens  # 名字留着给读代码的人，实现在 context 里唯一一份


def build_kernel(settings: Settings | None = None) -> AgentKernel:
    s = settings or load_settings()
    kernel = AgentKernel(
        # 热加载：provider 改了下一轮即用，无需重启。s 仍用于 store / system / tools（这些保持启动定死）。
        llm=LiveLLM(),
        store=build_store(s),
        system=s.system_prompt,
        native_tools=s.tools,
        tools_allow_outside=s.tools_allow_outside,
        tools_timeout=s.tools_timeout,
        approval_policy=s.tools_approve,
        summary_every=s.summary_every,
        summary_recall=s.summary_recall,
        memory_file=s.memory_path or None,
        compact_ratio=s.compact_ratio,
        # 预算必须**每轮重新问**：GUI 的 token 上限是按模型 profile 热写的，
        # 在这里读一个启动时的死值，会变成"切到 128k 的模型却还按 8k 的窗口压缩"。
        context_budget=lambda: current_settings().max_context_tokens,
    )
    return kernel
