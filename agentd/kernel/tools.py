"""原生工具层：跑在 agentd 进程内的文件 / 命令工具。

为什么不全部交给 MCP？MCP 的 stdio 客户端内部是 anyio task group，**必须与 enter/exit
同 task**（见 mcp.py 开头的生命周期说明），所以现在每轮 run 都要现开现关一次子进程。
读文件、搜代码这类动作一次对话里要用几十次，全走 MCP 等于每轮重启一次 node/python，
代价完全不成比例。所以把「高频、无状态、纯本地」的动作放进进程里做；
MCP 留给「本来就是独立服务」的能力（fetch / git / 数据库 / 厂商托管）。

对外契约与 MCP 工具**完全一致**：tool_schema() 吐 OpenAI 形状的 tools 数组，
call(name, arguments) -> str，失败文本以 `[错误] ` 开头（沿用 McpHub 的约定，
AgentMode 靠这个前缀决定 ToolCallDone.status）。

比 MCP 多出来的是一个 kind —— ACP 的 ToolKind 决定客户端拿什么图标渲染、
要不要弹审批。MCP 协议里没有这个概念（只有 readOnlyHint/destructiveHint 这种
粗粒度提示），原生工具可以精确声明。
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import os
import re
import sys
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# ACP ToolKind 的子集：原生工具只用到这几个（其余 delete/move/think/fetch/switch_mode
# 留给以后，声明了也没工具用得上）。
Kind = Literal["read", "edit", "search", "execute", "other"]

ERROR_PREFIX = "[错误] "

# 审批策略（对应 AGENTD_TOOLS_APPROVE）
POLICIES = ("native", "all", "none")

# 搜索类工具默认跳过的目录：命中它们只会淹没有效结果
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
)


def _log(msg: str) -> None:
    """日志走 stderr —— agentd 的 stdout 是纯 JSON-RPC 通道。"""
    print(msg, file=sys.stderr, flush=True)


def err(msg: str) -> str:
    return ERROR_PREFIX + msg


# ---------------------------------------------------------------------------
# 运行时约束
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolRuntime:
    """一次 run 内所有原生工具共用的边界与限额。

    单独抽出来（而不是让每个 handler 自己去摸 os.getcwd()）有两个好处：
    边界只有一个地方定义，测试里也只需要构造一个假的 runtime。
    """

    cwd: Path          # 相对路径的基准目录
    root: Path         # 允许访问的根，默认等于 cwd
    allow_outside: bool = False
    max_bytes: int = 65536      # 单个工具返回文本的上限（防一口气灌爆上下文）
    timeout: float = 30.0       # run_command 默认超时（秒）
    max_results: int = 200      # glob / grep 默认条数上限


Handler = Callable[[dict, ToolRuntime], Awaitable[str]]


@dataclass(frozen=True)
class NativeTool:
    """一个原生工具的完整声明。`parameters` 是 JSON Schema（直接进 tools 数组）。"""

    name: str
    description: str
    parameters: dict
    handler: Handler
    kind: Kind = "other"
    # 是否属于「会改变外部状态」的动作。true 时在 policy=native 下就会弹审批。
    requires_permission: bool = False


# ---------------------------------------------------------------------------
# 小工具函数
# ---------------------------------------------------------------------------


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _clip(text: str, limit: int, *, what: str = "输出") -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…（{what}已截断：完整 {len(text)} 字符，上限 {limit}）"


def _resolve(raw: str, rt: ToolRuntime) -> Path | None:
    """把用户给的路径折成绝对路径；越界（不在 root 内）返回 None。

    不做 `strict=True` 的 resolve —— 写新文件时目标本来就不存在。
    但 resolve 会把 `..` 和符号链接都展开，所以 `../../etc/passwd` 这种
    绕法在这里就露馅了。
    """
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = rt.cwd / p
    try:
        p = p.resolve()
    except OSError:  # pragma: no cover - 取决于平台，正常路径走不到
        return None
    if rt.allow_outside:
        return p
    try:
        p.relative_to(rt.root)
    except ValueError:
        return None
    return p


def _display(path: Path, rt: ToolRuntime) -> str:
    """喂给模型看的路径：优先相对，短且稳定。"""
    for base in (rt.cwd, rt.root):
        with contextlib.suppress(ValueError):
            return path.relative_to(base).as_posix()
    return path.as_posix()


def _matches(path: Path, base: Path, pattern: str) -> bool:
    """pattern 既能匹配相对路径（`src/*.py`），也能匹配裸文件名（`*.py`）。"""
    if fnmatch.fnmatch(path.name, pattern):
        return True
    try:
        rel = path.relative_to(base).as_posix()
    except ValueError:  # pragma: no cover - base 一定是 path 的祖先
        rel = path.as_posix()
    return fnmatch.fnmatch(rel, pattern)


def _iter_files(base: Path, pattern: str | None) -> Iterable[Path]:
    """遍历 base 下的文件，跳过 _SKIP_DIRS。

    不用 `Path.rglob`：它没法在遍历中途剪掉整个目录，撞上 node_modules 就卡死了。
    """
    if base.is_file():
        yield base
        return
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            p = Path(root) / name
            if pattern is None or _matches(p, base, pattern):
                yield p


_OUT_OF_ROOT = "路径越界：{raw} 不在工作目录 {root} 内（要放开请设 AGENTD_TOOLS_ALLOW_OUTSIDE=1）"


# ---------------------------------------------------------------------------
# 六个工具
# ---------------------------------------------------------------------------


async def _read_file(args: dict, rt: ToolRuntime) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        return err("read_file 缺少 path 参数")
    path = _resolve(raw, rt)
    if path is None:
        return err(_OUT_OF_ROOT.format(raw=raw, root=rt.root.as_posix()))
    if not path.exists():
        return err(f"文件不存在：{raw}")
    if path.is_dir():
        return err(f"{raw} 是目录不是文件；列目录请用 glob")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return err(f"读 {raw} 失败：{type(exc).__name__}: {exc}")

    lines = text.splitlines()
    start = max(_as_int(args.get("offset"), 1) - 1, 0)
    limit = _as_int(args.get("limit"), 0)
    chosen = lines[start:] if limit <= 0 else lines[start : start + limit]
    if not chosen:
        return err(f"{raw} 在 offset 之后没有内容（该文件共 {len(lines)} 行）")
    body = "\n".join(f"{start + i + 1:>6}\t{line}" for i, line in enumerate(chosen))
    header = f"{raw}（共 {len(lines)} 行，本次显示第 {start + 1}-{start + len(chosen)} 行）\n"
    return _clip(header + body, rt.max_bytes, what="文件内容")


async def _glob(args: dict, rt: ToolRuntime) -> str:
    pattern = str(args.get("pattern") or "").strip() or "**/*"
    base_raw = str(args.get("path") or ".").strip() or "."
    base = _resolve(base_raw, rt)
    if base is None:
        return err(_OUT_OF_ROOT.format(raw=base_raw, root=rt.root.as_posix()))
    if not base.is_dir():
        return err(f"{base_raw} 不是目录")

    try:
        hits = [p for p in base.glob(pattern) if p.is_file()]
    except (NotImplementedError, ValueError, OSError) as exc:
        # Path.glob 不接受绝对/含盘符的 pattern，报的是 NotImplementedError
        return err(f"glob 模式不可用（不要带绝对路径或盘符）：{type(exc).__name__}: {exc}")
    if not hits:
        return f"没有匹配 {pattern} 的文件（基准目录 {_display(base, rt)}）"

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    hits.sort(key=_mtime, reverse=True)
    shown = hits[: rt.max_results]
    head = f"匹配 {len(hits)} 个文件"
    if len(shown) < len(hits):
        head += f"，按修改时间倒序显示前 {len(shown)} 个"
    else:
        head += "，按修改时间倒序"
    return _clip(head + "\n" + "\n".join(_display(p, rt) for p in shown), rt.max_bytes)


async def _grep(args: dict, rt: ToolRuntime) -> str:
    pattern = str(args.get("pattern") or "")
    if not pattern:
        return err("grep 缺少 pattern 参数")
    flags = re.IGNORECASE if _truthy(args.get("ignore_case")) else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        return err(f"正则表达式非法：{exc}")

    base_raw = str(args.get("path") or ".").strip() or "."
    base = _resolve(base_raw, rt)
    if base is None:
        return err(_OUT_OF_ROOT.format(raw=base_raw, root=rt.root.as_posix()))
    if not base.exists():
        return err(f"文件或目录不存在：{base_raw}")

    file_glob = str(args.get("glob") or "").strip() or None
    limit = _as_int(args.get("max_results"), 0) or rt.max_results
    hits: list[str] = []
    scanned = 0
    capped = False

    for file in _iter_files(base, file_glob):
        try:
            data = file.read_bytes()
        except OSError:
            continue
        scanned += 1
        if b"\0" in data[:8192]:  # 二进制文件直接跳过，行内容没意义
            continue
        text = data.decode("utf-8", "replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{_display(file, rt)}:{lineno}: {line.strip()[:300]}")
                if len(hits) >= limit:
                    capped = True
                    break
        if capped:
            break

    if not hits:
        return f"没有匹配 /{pattern}/ 的内容（扫描 {scanned} 个文件）"
    head = f"命中 {len(hits)} 处，扫描 {scanned} 个文件"
    if capped:
        head += f"（已达上限 {limit}，可能还有更多）"
    return _clip(head + "\n" + "\n".join(hits), rt.max_bytes)


async def _write_file(args: dict, rt: ToolRuntime) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        return err("write_file 缺少 path 参数")
    if "content" not in args or args.get("content") is None:
        return err("write_file 缺少 content 参数（写空文件请显式传空字符串）")
    content = str(args["content"])

    path = _resolve(raw, rt)
    if path is None:
        return err(_OUT_OF_ROOT.format(raw=raw, root=rt.root.as_posix()))
    existed = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return err(f"写 {raw} 失败：{type(exc).__name__}: {exc}")
    action = "覆盖" if existed else "新建"
    return f"已{action} {_display(path, rt)}（{len(content)} 字符，{content.count(chr(10)) + 1} 行）"


async def _edit(args: dict, rt: ToolRuntime) -> str:
    raw = str(args.get("path") or "").strip()
    if not raw:
        return err("edit 缺少 path 参数")
    old = args.get("old_string")
    new = args.get("new_string")
    if old is None or old == "":
        return err("edit 需要非空的 old_string（整文件重写请用 write_file）")
    if new is None:
        return err("edit 缺少 new_string（删除内容请传空字符串）")
    old, new = str(old), str(new)

    path = _resolve(raw, rt)
    if path is None:
        return err(_OUT_OF_ROOT.format(raw=raw, root=rt.root.as_posix()))
    if not path.is_file():
        return err(f"文件不存在：{raw}（新建请用 write_file）")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return err(f"读 {raw} 失败：{type(exc).__name__}: {exc}")

    count = text.count(old)
    if count == 0:
        return err(f"{raw} 里找不到 old_string —— 注意缩进、换行、引号必须逐字符一致")
    replace_all = _truthy(args.get("replace_all"))
    if count > 1 and not replace_all:
        return err(
            f"old_string 在 {raw} 里匹配到 {count} 处，不唯一。"
            f"请补足上下文让它唯一，或设 replace_all=true 全部替换"
        )

    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return err(f"写回 {raw} 失败：{type(exc).__name__}: {exc}")
    return f"已修改 {_display(path, rt)}：替换 {count if replace_all else 1} 处"


async def _run_command(args: dict, rt: ToolRuntime) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        return err("run_command 缺少 command 参数")
    timeout = _as_float(args.get("timeout"), rt.timeout) or rt.timeout

    workdir = rt.cwd
    if args.get("cwd"):
        cand = _resolve(str(args["cwd"]), rt)
        if cand is None:
            return err(_OUT_OF_ROOT.format(raw=args["cwd"], root=rt.root.as_posix()))
        if not cand.is_dir():
            return err(f"cwd 不是目录：{args['cwd']}")
        workdir = cand

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return err(f"起 shell 失败：{type(exc).__name__}: {exc}")

    try:
        out, out_err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:  # py3.11 起 asyncio.TimeoutError 就是内置 TimeoutError
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        return err(f"命令超时（超过 {timeout:g}s）已被终止：{command}")

    stdout = out.decode("utf-8", "replace")
    stderr = out_err.decode("utf-8", "replace")
    merged = stdout
    if stderr.strip():
        merged = (merged + ("\n" if merged and not merged.endswith("\n") else "") + "[stderr]\n" + stderr)
    body = _clip(merged, rt.max_bytes, what="命令输出") or "（无输出）"

    # 非零退出码按失败上报：AgentMode 靠 [错误] 前缀把 status 记成 failed
    if proc.returncode:
        return err(f"命令退出码 {proc.returncode}：{command}\n{body}")
    return f"exit=0：{command}\n{body}"


_SPECS: tuple[NativeTool, ...] = (
    NativeTool(
        name="read_file",
        description=(
            "读取一个文本文件，返回带行号的内容（行号 1-based）。"
            "大文件用 offset/limit 分段读。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径，相对工作目录或绝对路径"},
                "offset": {"type": "integer", "description": "从第几行开始读（1-based，默认 1）"},
                "limit": {"type": "integer", "description": "最多读几行（默认读到文件结尾）"},
            },
            "required": ["path"],
        },
        handler=_read_file,
        kind="read",
    ),
    NativeTool(
        name="glob",
        description=(
            "按文件名/通配符找文件，返回按修改时间倒序的路径列表。"
            "支持 `**` 递归（如 `**/*.py`、`src/**/*.ts`）。要按内容找请用 grep。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "通配符，如 **/*.py；默认 **/*"},
                "path": {"type": "string", "description": "搜索基准目录，默认工作目录"},
            },
            "required": ["pattern"],
        },
        handler=_glob,
        kind="search",
    ),
    NativeTool(
        name="grep",
        description=(
            "在文件内容里做正则搜索，返回 `文件:行号: 内容`。"
            "适合定位符号定义、报错信息、配置项。默认跳过 .git/node_modules 等目录。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python 正则，如 `def handle_|class Kernel`"},
                "path": {"type": "string", "description": "文件或目录，默认工作目录"},
                "glob": {"type": "string", "description": "只搜这些文件，如 `*.py`"},
                "ignore_case": {"type": "boolean", "description": "忽略大小写，默认 false"},
                "max_results": {"type": "integer", "description": "最多返回多少处，默认 200"},
            },
            "required": ["pattern"],
        },
        handler=_grep,
        kind="search",
    ),
    NativeTool(
        name="write_file",
        description=(
            "把内容整体写入文件（已存在则覆盖，父目录自动创建）。"
            "只改文件里一小段请用 edit，别用 write_file 覆盖整个文件。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目标文件路径"},
                "content": {"type": "string", "description": "完整文件内容"},
            },
            "required": ["path", "content"],
        },
        handler=_write_file,
        kind="edit",
        requires_permission=True,
    ),
    NativeTool(
        name="edit",
        description=(
            "对文件做精确字符串替换：把 old_string 换成 new_string。"
            "old_string 在文件里必须唯一（否则报错并要求补上下文，或设 replace_all=true）。"
            "改动前请先 read_file 看清原文，缩进和换行要逐字符一致。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目标文件路径"},
                "old_string": {"type": "string", "description": "要被替换掉的原文（必须唯一）"},
                "new_string": {"type": "string", "description": "替换成的内容，传空字符串表示删除"},
                "replace_all": {"type": "boolean", "description": "匹配多处时是否全部替换，默认 false"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=_edit,
        kind="edit",
        requires_permission=True,
    ),
    NativeTool(
        name="run_command",
        description=(
            "在工作目录里执行一条 shell 命令，返回退出码与 stdout/stderr。"
            "默认 30 秒超时；非零退出码会以失败上报。适合编译、跑测试、git 查询。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令（走 shell，支持管道）"},
                "cwd": {"type": "string", "description": "工作目录，默认会话的 cwd"},
                "timeout": {"type": "number", "description": "超时秒数，默认 30"},
            },
            "required": ["command"],
        },
        handler=_run_command,
        kind="execute",
        requires_permission=True,
    ),
)

ALL_TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in _SPECS)

# AGENTD_TOOLS 的取值 → 实际启用的工具集合
TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    "native": ALL_TOOL_NAMES,
    "all": ALL_TOOL_NAMES,
    "read_only": ("read_file", "glob", "grep"),
    "ro": ("read_file", "glob", "grep"),
    "off": (),
    "none": (),
}


def needs_approval(*, requires: bool, kind: str, destructive: bool, policy: str) -> bool:
    """统一的审批判定（原生工具与 MCP 工具共用一套规则）。

    - `none`  —— 一律放行（CI / 无人值守用）
    - `all`   —— 只要不是只读就弹审批
    - `native`（默认）—— 原生工具按各自声明的 requires；MCP 工具只看 destructiveHint
      （MCP 协议没有 kind 概念，能拿到的信号只有 annotations 里那几个 hint，
       所以对 MCP 只信它自己声明的 destructive，不替它猜）

    只读动作（read / search）在任何策略下都不弹 —— 每个 ls 都弹窗，用户三分钟就会
    学会无脑点"允许"，审批本身也就废了。
    """
    if policy not in POLICIES:
        policy = "native"
    if policy == "none":
        return False
    if kind in ("read", "search"):
        return False
    if policy == "all":
        return True
    return requires or destructive


@dataclass(frozen=True)
class ApprovalRequest:
    """一次待审批的工具调用。传输层拿它去弹窗（ACP 走 session/request_permission）。

    `tool` 是**稳定的工具名**（含 MCP 前缀），传输层用它做"本会话总是允许"的记忆键；
    `title` 只是给人看的短名，可能重名，不能当键。
    """

    call_id: str
    tool: str
    title: str
    kind: str
    detail: str = ""


ApproveHandler = Callable[[ApprovalRequest], Awaitable[bool]]


class NativeToolbox:
    """原生工具注册表。每个 run 造一个 —— 构造很便宜，而 cwd 是会话级的。

    对外三个方法刻意与 McpHub 同名（tool_schema / binding / call），
    这样 AgentMode 里两条工具来源的代码形状是一样的。
    """

    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        allow_outside: bool = False,
        profile: str = "native",
        max_bytes: int = 65536,
        timeout: float = 30.0,
        max_results: int = 200,
    ) -> None:
        base = Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()
        self.runtime = ToolRuntime(
            cwd=base,
            root=base,
            allow_outside=allow_outside,
            max_bytes=max_bytes,
            timeout=timeout,
            max_results=max_results,
        )
        enabled = TOOL_PROFILES.get(profile, ALL_TOOL_NAMES)
        self._tools: dict[str, NativeTool] = {
            spec.name: spec for spec in _SPECS if spec.name in enabled
        }
        self.profile = profile

    # ---- 对外（与 McpHub 同形） ----

    @property
    def has_tools(self) -> bool:
        return bool(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def tool_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    def binding(self, name: str) -> NativeTool | None:
        return self._tools.get(name)

    def kind_of(self, name: str) -> str:
        spec = self._tools.get(name)
        return spec.kind if spec else "other"

    def requires_permission(self, name: str) -> bool:
        spec = self._tools.get(name)
        return bool(spec and spec.requires_permission)

    async def call(self, name: str, arguments: str) -> str:
        spec = self._tools.get(name)
        if spec is None:
            return err(f"未知工具：{name}")
        try:
            args = json.loads(arguments) if arguments else {}
        except (ValueError, TypeError):
            return err(f"{name} 的 arguments 不是合法 JSON：{str(arguments)[:200]}")
        if not isinstance(args, dict):
            return err(f"{name} 的 arguments 必须是 JSON 对象，收到 {type(args).__name__}")
        try:
            return await spec.handler(args, self.runtime)
        except Exception as exc:  # noqa: BLE001 - 工具是外部输入边界，不能让整轮崩掉
            _log(f"[agentd] 原生工具 {name} 抛异常：{type(exc).__name__}: {exc}")
            return err(f"{name} 执行失败：{type(exc).__name__}: {exc}")
