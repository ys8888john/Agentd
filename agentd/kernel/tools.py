"""原生工具层：跑在 agentd 进程内的文件 / 命令 / 联网工具。

为什么不全部交给 MCP？MCP 的 stdio 客户端内部是 anyio task group，**必须与 enter/exit
同 task**（见 mcp.py 开头的生命周期说明），所以现在每轮 run 都要现开现关一次子进程。
读文件、搜代码这类动作一次对话里要用几十次，全走 MCP 等于每轮重启一次 node/python，
代价完全不成比例。所以把「高频、无状态、纯本地」的动作放进进程里做；
MCP 留给「本来就是独立服务」的能力（数据库 / 厂商托管 / 要 API key 的服务）。

联网检索（web_search / web_fetch）也放在这里，理由更直接：本机能用的搜索后端只有
Bing，而 MCP 生态里现成的搜索 server 要么是 Node 实现（本机 npx 被沙箱拦死）、
要么要 API key、要么走被墙的 DuckDuckGo —— 都不是「装一个包就能用」。
自己用 httpx + 正则做，零依赖、零配置。详见下面「联网工具」一节。

对外契约与 MCP 工具**完全一致**：tool_schema() 吐 OpenAI 形状的 tools 数组，
call(name, arguments) -> str，失败文本以 `[错误] ` 开头（沿用 McpHub 的约定，
AgentMode 靠这个前缀决定 ToolCallDone.status）。

比 MCP 多出来的是一个 kind —— ACP 的 ToolKind 决定客户端拿什么图标渲染、
要不要弹审批。MCP 协议里没有这个概念（只有 readOnlyHint/destructiveHint 这种
粗粒度提示），原生工具可以精确声明。
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import difflib
import fnmatch
import html
import json
import os
import re
import sys
import urllib.parse
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

# ACP ToolKind 的子集：原生工具只用到这几个
# （其余 delete/move/think/switch_mode 留给以后，声明了也没工具用得上）。
Kind = Literal["read", "edit", "search", "execute", "fetch", "other"]

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
    # 额外允许访问的根（ACP additionalDirectories）：解析成绝对路径的元组
    additional_roots: tuple[Path, ...] = ()
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
    # 会话 cwd 与 additionalDirectories 里声明的每个根都算"界内"；任何一个
    # 命中即放行（相对路径天然落在 cwd 里，额外的根用绝对路径访问）。
    for root in (rt.root, *rt.additional_roots):
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    return None  # 任何一个根都不含 → 越界（这句写错成 return p 就是安全洞）


def _display(path: Path, rt: ToolRuntime) -> str:
    """喂给模型看的路径：优先相对，短且稳定。"""
    for base in (rt.cwd, rt.root, *rt.additional_roots):
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
# 联网工具（web_search / web_fetch）
# ---------------------------------------------------------------------------

# 搜索引擎主后端：Bing。**这不是偷懒，是实测筛出来的**（2026-09-14）：
#   - DuckDuckGo（lite. / html.duckduckgo.com）：走隧道全是 502，验不了 ——
#     写一个自己验不了的解析器等于埋雷，所以宁可不写；（2026-10-08 复测仍连不上）
#   - 公共 SearXNG 实例：同样被挡；
#   - Baidu：返回"百度安全验证"反爬页，不是结果页（2026-10-08 复测这次通了，
#     但结果 URL 全是 baidu.com/link 跳转，反爬时好时坏，不做主后端）；
#   - Bing（cn.bing.com / www.bing.com）：通，中英文都能稳定拿到 10 条，
#     而且结果是目标站**直链**（不用解跳转包装）。
# 要加后端的话形状很固定：拉 HTML → 解析成 {title, url, snippet}，
# 在 _parse_bing 旁边照写一个就行。
#
# **Bing 有个修不掉的毛病**（2026-10-08 实测，见 _looks_degraded）：带出行/票务
# 意图的 query（机票/高铁票…）会被砍成只搜第一个词——「成都飞北京机票价格」
# 返回的其实是搜「成都」的结果。cookie 预热、mkt、format=rss、换新 UA、补全
# 浏览器头、引号强查，全试过无效，是 Bing 对非交互客户端的处理。
# 所以主后端之外再备一个**搜狗**（实测对同一 query 返回真实机票结果）：
# Bing 结果做相关性抽查，答非所问就换搜狗重试。
_SEARCH_ENDPOINT = "https://cn.bing.com/search"
_SOGOU_ENDPOINT = "https://www.sogou.com/web"


def _search_endpoint() -> str:
    """搜索后端地址，**每次调用时**读一次，好让它能被覆盖。

    AGENTD_SEARCH_ENDPOINT 的用处有两个，都是实的：
      - 联网链路的端到端验证指向本地假后端，从而离线、可复现（见 ForgeAgent-GUI
        的 scripts/native_tools_e2e.py --scenario web）；
      - cn.bing.com 在有些网络里不通，换个镜像不用改代码。
    """
    return os.getenv("AGENTD_SEARCH_ENDPOINT") or _SEARCH_ENDPOINT


def _sogou_endpoint() -> str:
    """搜狗后端地址，同上可被 AGENTD_SOGOU_ENDPOINT 覆盖（离线单测用）。"""
    return os.getenv("AGENTD_SOGOU_ENDPOINT") or _SOGOU_ENDPOINT

# 必须伪装成浏览器：默认 UA（httpx/urllib）会被 Bing 直接挡掉。
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_WEB_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 联网单独一套超时：read 给得比 connect 长，搜索/抓取本来就可能慢。
_WEB_TIMEOUT = httpx.Timeout(connect=6.0, read=20.0, write=10.0, pool=6.0)

# Bing 一页固定回 10 条 —— 实测 count 传 20 也只回 10 个 b_algo 块。
# 所以不做翻页，count 直接按 10 封顶；要更多结果就打几个不同的 query。
_MAX_SEARCH_RESULTS = 10

# web_fetch 最多读这么多 HTML：防一个巨大页面把内存和上下文一起灌爆。
_MAX_HTML_BYTES = 2 * 1024 * 1024

_COMMENT_RX = re.compile(r"<!--.*?-->", re.S)
# script/style 里的内容全是代码，不剥掉会污染正文
_SCRIPT_RX = re.compile(r"<(script|style|noscript|template)\b.*?</\1\s*>", re.S | re.I)
# 块级标签折成换行，否则整页会挤成一行
_BLOCK_RX = re.compile(
    r"</?(?:p|div|br|li|ul|ol|tr|td|th|table|h[1-6]|section|article|header|footer"
    r"|blockquote|pre|form|nav|aside|main)\b[^>]*>",
    re.I,
)
_TAG_RX = re.compile(r"<[^>]+>")

# Bing 结果页的形状（实测）：结果都在 `<li class="b_algo">` 块里，
# 标题+链接是块内第一个 h2>a，摘要是块内第一个 <p>。
_BING_BLOCK_SPLIT = re.compile(r'<li class="b_algo"')
_BING_ANCHOR_RX = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_BING_SNIPPET_RX = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_BING_COUNT_RX = re.compile(r'<span class="sb_count"[^>]*>(.*?)</span>', re.S)

# 搜狗结果页的形状（2026-10-08 实测）：结果在 `<div class="vrwrap">` 块里，
# 标题+链接是块内 `h3.vr-title > a`（标题里的 <em> 高亮和 <!--red_beg--> 注释要剥掉）；
# 摘要没有统一容器（text-layout / space-txt / star-wiki 都见过），取块内第一段
# 长度够的文本凑合。`/link?url=` 不是 302，是个 200 小中转页，真实地址写在
# meta refresh 和 window.location.replace 里（见 _resolve_sogou_links）。
_SOGOU_BLOCK_SPLIT = re.compile(r'<div class="vrwrap"')
_SOGOU_ANCHOR_RX = re.compile(
    r'<h3[^>]*class="[^"]*vr-title[^"]*"[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SOGOU_SNIPPET_RX = re.compile(
    r'<(?:p|div)[^>]*class="[^"]*(?:text-layout|space-txt|star-wiki)[^"]*"[^>]*>(.*?)</(?:p|div)>', re.S)
_SOGOU_PLAIN_P_RX = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_SOGOU_LINK_JS_RX = re.compile(r'window\.location\.replace\("([^"]+)"\)')
_SOGOU_LINK_META_RX = re.compile(r'<meta[^>]*url=("|\')?([^"\'>\s]+)', re.I)
_SOGOU_ANTISPIDER_RX = re.compile(r"验证码|antispider", re.I)

# content-type 里出现这些片段就当作"能当文本看"
_TEXTUAL_HINTS = ("text/", "json", "xml", "javascript", "x-www-form-urlencoded")


def _text_of(fragment: str) -> str:
    """剥标签 + 还原实体，并折成单行。"""
    return " ".join(html.unescape(_TAG_RX.sub("", fragment)).split())


def _looks_textual(content_type: str) -> bool:
    """content-type 看起来是不是文本。

    匹配刻意宽松：不少站点把 HTML 标成 application/octet-stream，
    宁可多试一次，也别因为一个 header 就把整页内容判死。
    """
    lowered = (content_type or "").lower()
    if not lowered:
        return True  # 没声明就当文本试一次
    return any(hint in lowered for hint in _TEXTUAL_HINTS)


def _decode(body: bytes, content_type: str) -> str:
    """按 content-type 里声明的 charset 解码，没写就按 utf-8 兜底。"""
    lowered = (content_type or "").lower()
    charset = ""
    if "charset=" in lowered:
        charset = lowered.split("charset=", 1)[1].split(";")[0].strip().strip("\"'")
    try:
        return body.decode(charset or "utf-8", "replace")
    except LookupError:  # 声明了一个 Python 不认识的编码名
        return body.decode("utf-8", "replace")


def _unwrap_bing_url(url: str) -> str:
    """Bing 偶尔把结果包成 `https://www.bing.com/ck/a?...&u=a1<base64url>`。

    实测抓到的那十条都是目标站直链，但结果里混进包装链接是老问题
    （官网/首页卡片尤其容易），所以顺手解开；解不开就原样返回。
    """
    if "bing.com/ck/a" not in url:
        return url
    params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    raw = (params.get("u") or [""])[0]
    if not raw:
        return url
    if raw.startswith("a1"):  # a1 是"原始 URL"的前缀标记
        raw = raw[2:]
    try:
        padded = raw + "=" * (-len(raw) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8", "replace")
    except ValueError:  # binascii.Error 是 ValueError 的子类；非 ASCII 也会走这里
        return url
    # 只在解出来的东西确实像个 URL 时才采用 —— b64 解码会「宽容地」忽略非法字符，
    # 于是垃圾输入会安静地解成空串或乱码，那种情况必须退回原值。
    return decoded if decoded.startswith(("http://", "https://")) else url


def _parse_bing(page: str, limit: int) -> tuple[list[dict], str]:
    """解析 Bing 结果页，返回 (结果列表, 结果条数提示)。

    ⚠️ 已知限制：**Bing 查不到东西时不给"无结果"标记**，而是塞一批不相关结果
    （实测乱码 query 返回了一屏"抖音"）；页面上也没有可靠的空结果标志
    （`b_no` 在有结果的页面上同样出现，是别的用途）。所以这里不做"无结果"判定，
    只能靠把 query 写具体。这类"答非所问"现在由 _looks_degraded 抽查 +
    搜狗兜底来兜（见 _web_search），解析层不做判定。
    """
    results: list[dict] = []
    for block in _BING_BLOCK_SPLIT.split(page)[1:]:
        anchor = _BING_ANCHOR_RX.search(block)
        if anchor is None:
            continue
        title = _text_of(anchor.group(2))
        url = _unwrap_bing_url(anchor.group(1))
        if not title or not url:
            continue
        snippet = _BING_SNIPPET_RX.search(block)
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": _text_of(snippet.group(1)) if snippet else "",
            }
        )
        if len(results) >= limit:
            break
    count_hit = _BING_COUNT_RX.search(page)
    return results, _text_of(count_hit.group(1)) if count_hit else ""


def _html_to_text(raw: str) -> str:
    """把 HTML 折成可读纯文本。

    刻意不引 bs4 / lxml —— agentd 的依赖表不该为这一个功能多一项，而这种粗提取
    （去脚本样式 → 块级标签转换行 → 剥标签 → 解实体）用正则够用。
    代价是表格和复杂排版会走形；在"喂给模型读"的场景里无所谓。
    """
    text = _COMMENT_RX.sub("", raw)
    text = _SCRIPT_RX.sub(" ", text)
    text = _BLOCK_RX.sub("\n", text)
    text = _TAG_RX.sub("", text)
    text = html.unescape(text)
    lines: list[str] = []
    blanks = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            blanks += 1
            if blanks > 1:  # 连续空行压成一个
                continue
        else:
            blanks = 0
        lines.append(line)
    return "\n".join(lines).strip()


class _SearchError(Exception):
    """某个搜索后端不可用（网络 / HTTP 状态 / 反爬 / 解析不出条目）。"""


def _query_fingerprint(query: str) -> list[str]:
    """query 的内容指纹：中文段拆 2-gram，英文/数字段整段小写。

    不做真分词（不引依赖），2-gram 足以回答"结果里有没有出现 query 的词"。
    「成都飞北京机票价格 2024年」→ 成都/都飞/飞北/北京/京机/机票/票价/价格/2024/年。
    """
    grams: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]{2,}", query):
        if seg.isascii():
            grams.append(seg.lower())
        elif len(seg) == 1:
            grams.append(seg)
        else:
            grams.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return grams


def _looks_degraded(query: str, results: list[dict]) -> bool:
    """Bing 是否在答非所问（结果几乎不含 query 的内容词）。

    实测（2026-10-08）：「成都飞北京机票价格」被降级成搜「成都」，top5 的
    标题+摘要只能命中「成都」一两个 2-gram；正常结果至少命中机票/价格/北京等
    好几个。阈值取指纹的一半（至少 2）：真结果过线很宽裕，降级结果够不着。
    指纹凑不齐 3 个（query 本身太短）时不判定 —— 分不出来，宁可信 Bing。
    """
    grams = set(_query_fingerprint(query))
    if len(grams) < 3:
        return False
    corpus = " ".join(f"{it['title']} {it['snippet']}" for it in results).lower()
    hits = sum(1 for g in grams if g in corpus)
    return hits < max(2, len(grams) // 2)


def _parse_sogou(page: str, limit: int) -> list[dict]:
    """解析搜狗结果页，形状见 _SOGOU_* 正则上方的注释。"""
    results: list[dict] = []
    for block in _SOGOU_BLOCK_SPLIT.split(page)[1:]:
        anchor = _SOGOU_ANCHOR_RX.search(block)
        if anchor is None:
            continue
        title = _text_of(_COMMENT_RX.sub("", anchor.group(2)))
        url = urllib.parse.urljoin("https://www.sogou.com/", anchor.group(1))
        if not title or not url:
            continue
        snippet = ""
        for rx in (_SOGOU_SNIPPET_RX, _SOGOU_PLAIN_P_RX):
            for frag in rx.findall(block):
                text = _text_of(_COMMENT_RX.sub("", frag))
                if len(text) >= 15:  # 更短的多是日期/标签，不是摘要
                    snippet = text
                    break
            if snippet:
                break
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


async def _resolve_sogou_links(results: list[dict], client: httpx.AsyncClient) -> None:
    """把 `/link?url=` 中转链接换成真实地址；解不出的保持原样。

    中转页只有 200 来字节（meta refresh + location.replace 双保险），
    用 stream 限读 16KB，防止个别链接真的 302 到大文章时把整页拖下来。
    """
    for item in results:
        if "/link?url=" not in item["url"]:
            continue
        try:
            async with client.stream("GET", item["url"]) as resp:
                buf = bytearray()
                async for chunk in resp.aiter_bytes(4096):
                    buf.extend(chunk)
                    if len(buf) >= 16384:
                        break
                body = bytes(buf).decode("utf-8", "replace")
                final_url = str(resp.url)
        except httpx.HTTPError:
            continue
        # 真 302 了就直接用落点；还停在中转页就从页面里抠目标地址
        if final_url != item["url"] and "sogou.com/link" not in final_url \
                and final_url.startswith(("http://", "https://")):
            item["url"] = final_url
            continue
        m = _SOGOU_LINK_JS_RX.search(body)
        real = m.group(1) if m else ""
        if not real:
            m = _SOGOU_LINK_META_RX.search(body)
            real = html.unescape(m.group(2)) if m else ""
        if real.startswith(("http://", "https://")):
            item["url"] = real


async def _search_bing(query: str, limit: int) -> tuple[list[dict], str]:
    """Bing 主后端：返回 (结果, 结果数提示)；任何失败抛 _SearchError。"""
    try:
        async with httpx.AsyncClient(
            headers=_WEB_HEADERS, follow_redirects=True, timeout=_WEB_TIMEOUT
        ) as client:
            resp = await client.get(
                _search_endpoint(),
                params={"q": query, "count": str(limit), "setlang": "zh-CN"},
            )
            resp.raise_for_status()
            page = resp.text
    except httpx.HTTPStatusError as exc:
        raise _SearchError(
            f"搜索引擎返回 HTTP {exc.response.status_code}（多半被限流，稍后再试）"
        ) from exc
    except httpx.HTTPError as exc:
        raise _SearchError(f"{type(exc).__name__}: {exc}") from exc
    results, count_hint = _parse_bing(page, limit)
    if not results:
        raise _SearchError(
            f"没能从结果页解析出条目（页面 {len(page)} 字符；可能被限流或页面结构变了）"
        )
    return results, count_hint


async def _search_sogou(query: str, limit: int) -> tuple[list[dict], str]:
    """搜狗备用后端：返回 (结果, "")；任何失败抛 _SearchError。"""
    try:
        async with httpx.AsyncClient(
            headers=_WEB_HEADERS, follow_redirects=True, timeout=_WEB_TIMEOUT
        ) as client:
            resp = await client.get(_sogou_endpoint(), params={"query": query})
            resp.raise_for_status()
            page = resp.text
            if _SOGOU_ANTISPIDER_RX.search(page):
                raise _SearchError("搜狗要求验证码（触发反爬）")
            results = _parse_sogou(page, limit)
            await _resolve_sogou_links(results, client)
    except httpx.HTTPError as exc:
        raise _SearchError(f"{type(exc).__name__}: {exc}") from exc
    return results, ""


def _format_search_results(query: str, results: list[dict], count_hint: str,
                           max_bytes: int, source_note: str) -> str:
    head = f"「{query}」搜索到 {len(results)} 条"
    if count_hint:
        head += f"（{count_hint}）"
    lines = [head]
    if source_note:
        lines.append(source_note)
    for i, item in enumerate(results, 1):
        lines.append(f"{i}. {item['title']}\n   {item['url']}")
        if item["snippet"]:
            lines.append(f"   {item['snippet']}")
    lines.append("要看正文可以用 web_fetch 打开上面的链接。")
    return _clip("\n".join(lines), max_bytes, what="搜索结果")


async def _web_search(args: dict, rt: ToolRuntime) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return err("web_search 缺少 query 参数")
    limit = min(max(_as_int(args.get("count"), 5), 1), _MAX_SEARCH_RESULTS)
    # AGENTD_SEARCH_ENDPOINT 指了自定义后端（离线 e2e 的假后端 / 内网镜像）时，
    # 行为跟旧版完全一致：不做相关性抽查、不碰搜狗 —— 假后端的结果和 query
    # 本来就零词面重叠，抽查必然误判；用户显式指定的后端也不该被二次猜疑。
    custom_backend = bool(os.getenv("AGENTD_SEARCH_ENDPOINT"))

    bing: tuple[list[dict], str] | None = None
    bing_err = ""
    try:
        bing = await _search_bing(query, limit)
    except _SearchError as exc:
        bing_err = str(exc)

    if bing is not None and (custom_backend or not _looks_degraded(query, bing[0])):
        return _format_search_results(query, bing[0], bing[1], rt.max_bytes, "")

    # 走到这：Bing 失败，或结果答非所问（仅默认后端）→ 搜狗兜底
    if custom_backend:
        return err(f"web_search 失败：{bing_err}")

    sogou_note = ""
    try:
        sogou = await _search_sogou(query, limit)
        if sogou[0]:
            reason = ("Bing 返回的结果与 query 不相关（被降级）" if bing is not None
                      else f"Bing 不可用（{bing_err}）")
            return _format_search_results(
                query, sogou[0], "", rt.max_bytes, f"来源：搜狗（{reason}）")
        sogou_note = "搜狗没解析出条目"
    except _SearchError as exc:
        sogou_note = f"搜狗也不可用（{exc}）"

    if bing is not None:
        # Bing 被降级但搜狗没顶上：结果给出去但明说不可靠，模型可以换个说法重搜
        return _format_search_results(
            query, bing[0], bing[1], rt.max_bytes,
            "注意：以下结果可能与 query 不相关（Bing 降级且搜狗兜底失败），建议换个说法重搜")
    return err(f"web_search 失败：Bing（{bing_err}）；{sogou_note}")


async def _web_fetch(args: dict, rt: ToolRuntime) -> str:
    raw = str(args.get("url") or "").strip()
    if not raw:
        return err("web_fetch 缺少 url 参数")
    if not raw.lower().startswith(("http://", "https://")):
        return err(f"只支持 http/https 的 URL，收到：{raw[:100]}")

    limit = min(max(_as_int(args.get("max_bytes"), rt.max_bytes) or rt.max_bytes, 500), _MAX_HTML_BYTES)

    try:
        async with httpx.AsyncClient(
            headers=_WEB_HEADERS, follow_redirects=True, timeout=_WEB_TIMEOUT
        ) as client:
            async with client.stream("GET", raw) as resp:
                resp.raise_for_status()
                status = resp.status_code
                ctype = resp.headers.get("content-type", "")
                final_url = str(resp.url)
                chunks: list[bytes] = []
                size = 0
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= _MAX_HTML_BYTES:  # 到顶就停，别把超大页面读进内存
                        break
                body = b"".join(chunks)
    except httpx.HTTPStatusError as exc:
        return err(f"{raw} 返回 HTTP {exc.response.status_code}")
    except httpx.HTTPError as exc:
        return err(f"抓取 {raw} 失败：{type(exc).__name__}: {exc}")

    if not _looks_textual(ctype):
        return err(
            f"{final_url} 不是文本内容（content-type: {ctype or '未知'}，{len(body)} 字节）"
        )

    text = _html_to_text(_decode(body, ctype))
    if not text:
        return err(f"{final_url} 抓回 {len(body)} 字节，但剥掉标签后没有正文")
    head = f"{final_url}（HTTP {status}，{ctype or '未知类型'}，{len(body)} 字节）"
    return _clip(head + "\n\n" + text, limit, what="网页正文")


# ---------------------------------------------------------------------------
# 九个工具
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
        return err(f"{raw} 是目录不是文件；列它下面有什么请用 list_dir")
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


_MAX_LIST_DEPTH = 3     # list_dir 递归深度上限：再深就是整棵树了，没有信息量
_MAX_LIST_ENTRIES = 300  # 单层最多列出多少条目

_HUMAN_UNITS = ("B", "K", "M", "G")


def _human_size(size: int) -> str:
    """把字节数压成 '1.2M' 这种短写法 —— 列目录时比原始字节数好扫一眼。"""
    value = float(size)
    unit = 0
    while value >= 1024 and unit < len(_HUMAN_UNITS) - 1:
        value /= 1024
        unit += 1
    digits = 0 if unit == 0 or value >= 100 else 1
    return f"{value:.{digits}f}{_HUMAN_UNITS[unit]}"


def _walk_tree(
    base: Path,
    *,
    depth: int,
    include_hidden: bool,
    skip_dirs: frozenset[str],
    limit: int,
) -> tuple[list[str], int, int, bool]:
    """按层的顺序铺出目录树，返回 (行文本, 目录数, 文件数, 是否撞到上限)。

    用 os.walk 的 topdown 剪枝而不是 Path.rglob：后者没法在中途砍掉整个子树，
    撞上一个塞满文件的 node_modules 就得先把几万个 path 对象都造出来。
    """
    lines: list[str] = []
    dirs_seen = 0
    files_seen = 0
    capped = False

    for root, dirnames, filenames in os.walk(base):
        here = Path(root)
        level = len(here.parts) - len(base.parts)
        if level >= depth:
            dirnames[:] = []  # 已经到倒数第二层，孙子辈不再进入
            continue

        dirnames[:] = sorted(
            d
            for d in dirnames
            if (include_hidden or not d.startswith(".")) and d not in skip_dirs
        )
        filenames = sorted(f for f in filenames if include_hidden or not f.startswith("."))

        # 缩进 = 层级：根的儿子在 0 缩进，孙子在 2 空格，以此类推。
        # 目录名由**父亲**在自己那一级列出（这样被上限截断时至少知道有这个目录），
        # 递归进来只补它的孩子 —— 再打一次自己的名字就会出现重名两行。
        pad = "  " * level
        # 目录优先：模型得先知道"有哪些子目录"，才好决定往哪儿钻
        lines.extend(f"{pad}{name}/" for name in dirnames)
        # 计数照实累加（哪怕马上要被上限截断）：尾巴被砍了，head 里的
        # "共 N 个子目录 / M 个文件"仍然是真的总数
        dirs_seen += len(dirnames)
        files_seen += len(filenames)
        lines.extend(f"{pad}{name}" for name in filenames)

        if len(lines) >= limit:
            lines = lines[:limit]
            capped = True
            break

    return lines, dirs_seen, files_seen, capped


def workspace_brief(root: Path, *, depth: int = 2, max_lines: int = 40) -> str:
    """一段"你现在站在哪儿"的说明，给 system prompt 用。

    为什么要加这一段：模型的 system prompt 里原来一行路径都没有，它开局是瞎的
    —— 要么先浪费一整轮去 list_dir 探路，要么干脆凭空猜文件名（猜错的代价是
    工具报错、用户看到一轮废话）。把浅层目录结构直接塞进 system prompt，
    它第一句话就能说到点子上。

    刻意做得**浅**（默认 2 层、40 行）：这是每一轮都要付的固定开销，再多点就是
    给每次回答悄悄加了一份房租。要深的结构，模型自己会调 list_dir。

    目录不存在 / 读不动就返回空串 —— 环境说明掉了不影响对话，绝不在这里抛。
    """
    if not root.is_dir():
        return ""
    try:
        lines, dirs_seen, files_seen, capped = _walk_tree(
            root,
            depth=depth,
            include_hidden=False,
            skip_dirs=_SKIP_DIRS,
            limit=max_lines + 2,  # 多取两行用来判断"是不是被截了"
        )
    except OSError:
        return ""

    head = f"当前工作目录：{root}"
    if not lines:
        return f"{head}\n（下面没有可见条目）"

    truncated = capped or len(lines) > max_lines
    lines = lines[:max_lines]
    body = "\n".join(lines)
    summary = f"（{dirs_seen} 个子目录 / {files_seen} 个文件，depth={depth}"
    summary += "，下面还有更多" if truncated else ""
    summary += "）"
    return f"{head}\n```\n{body}\n```\n{summary}\n附上这段 tree 只是让你开局有方位感；细节请调 list_dir / glob / grep 现查。"


async def _list_dir(args: dict, rt: ToolRuntime) -> str:
    raw = str(args.get("path") or "").strip() or "."
    base = _resolve(raw, rt)
    if base is None:
        return err(_OUT_OF_ROOT.format(raw=raw, root=rt.root.as_posix()))
    if not base.exists():
        return err(f"目录不存在：{raw}")
    if not base.is_dir():
        return err(f"{raw} 是文件不是目录；要看它的内容请用 read_file")

    depth = min(max(_as_int(args.get("depth"), 1), 1), _MAX_LIST_DEPTH)
    # 默认不显示隐藏条目、也不显示 node_modules / __pycache__ 这类依赖目录：
    # 它们一出场就把真正有用的几个名字挤到 _clip 的截断点外面去了。
    include_hidden = _truthy(args.get("include_hidden"))
    skip_dirs = frozenset() if _truthy(args.get("include_ignored")) else _SKIP_DIRS
    limit = min(max(_as_int(args.get("max_results"), 0) or rt.max_results, 1), _MAX_LIST_ENTRIES)

    lines, dirs_seen, files_seen, capped = _walk_tree(
        base, depth=depth, include_hidden=include_hidden, skip_dirs=skip_dirs, limit=limit
    )
    shown_root = _display(base, rt)
    if not lines:
        # 真的是空目录，还是"有东西但全被过滤掉了"—— 这两件事对模型来说下一步
        # 完全不同（前者该收工，后者该改参数重试），必须写清楚。
        why = "（它本身是空的）" if not any(base.iterdir()) else "（可能都是隐藏目录或依赖目录，可试 include_hidden / include_ignored）"
        return f"{shown_root} 下没有可显示的条目{why}"

    head = f"{shown_root}（{dirs_seen} 个子目录，{files_seen} 个文件，depth={depth}"
    if not include_hidden:
        head += "，跳过隐藏条目"
    if skip_dirs:
        head += "，跳过依赖/构建目录"
    head += "）"
    tail = ""
    if capped:
        tail = f"\n…（已达 {limit} 条上限，目录里可能还有更多）"
    return _clip(head + "\n" + "\n".join(lines) + tail, rt.max_bytes, what="目录树")


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
    before = ""
    if existed:
        try:
            before = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            before = ""  # 旧内容读不出来（二进制/权限）就不给 diff，照常写
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return err(f"写 {raw} 失败：{type(exc).__name__}: {exc}")
    action = "覆盖" if existed else "新建"
    head = f"已{action} {_display(path, rt)}（{len(content)} 字符，{content.count(chr(10)) + 1} 行）"
    diff = _diff_block(before, content, _display(path, rt))
    return f"{head}\n{diff}" if diff else head


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
    head = f"已修改 {_display(path, rt)}：替换 {count if replace_all else 1} 处"
    diff = _diff_block(text, updated, _display(path, rt))
    return f"{head}\n{diff}" if diff else head


def _diff_block(before: str, after: str, label: str, max_lines: int = 80) -> str:
    """before/after 的 unified diff，给客户端的工具卡片当"改动前后对比"。

    上下文 n=2：卡片里那么长的上下文没用；产出空 diff（内容没变）返回 ""，
    调用方就不追加。超长截断 —— 64KB 的输出上限不该被一个大 diff 吃光。
    行首标记（---/+++/@@/-/+）同时是前端 diff 着色的解析依据，别改名。
    """
    diff = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{label}（旧）",
            tofile=f"{label}（新）",
            lineterm="",
            n=2,
        )
    )
    if not diff:
        return ""
    shown = diff[:max_lines]
    tail = "" if len(diff) <= max_lines else f"\n…（diff 其余 {len(diff) - max_lines} 行略）"
    return "\n".join(shown) + tail


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
        name="list_dir",
        description=(
            "列出一个目录里的内容（子目录带 `/` 后缀，排在文件前面）。"
            "**回答「这里有什么 / 项目长什么样」时先调它**，再决定读哪个文件 —— "
            "glob 只能按名字找、而且看不见空目录。"
            "depth 可调（默认 1 层，最多 3 层）。默认过滤隐藏条目与 node_modules、"
            "__pycache__ 等依赖/构建目录，想看全就传 include_hidden / include_ignored。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "目录路径，默认工作目录"},
                "depth": {"type": "integer", "description": "递归几层，默认 1，最多 3"},
                "include_hidden": {"type": "boolean", "description": "是否显示 . 开头的条目，默认 false"},
                "include_ignored": {"type": "boolean", "description": "是否显示 node_modules 等依赖目录，默认 false"},
                "max_results": {"type": "integer", "description": "最多列多少条，默认 200"},
            },
            "required": [],
        },
        handler=_list_dir,
        kind="search",
    ),
    NativeTool(
        name="glob",
        description=(
            "按文件名/通配符找文件，返回按修改时间倒序的路径列表。"
            "支持 `**` 递归（如 `**/*.py`、`src/**/*.ts`）。要按内容找请用 grep，"
            "要看目录结构（包括空目录）请用 list_dir。"
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
    NativeTool(
        name="web_search",
        description=(
            "联网搜索（Bing），返回若干条【标题 / 链接 / 摘要】。"
            "query 尽量具体（中英文都可以）；摘要不够就看正文，用 web_fetch 打开链接。"
            "注意：查不到匹配内容时，Bing 会返回一批不相关的结果，而不是报「没有结果」。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索词，越具体越好"},
                "count": {"type": "integer", "description": "返回几条，默认 5，最多 10"},
            },
            "required": ["query"],
        },
        handler=_web_search,
        kind="search",
    ),
    NativeTool(
        name="web_fetch",
        description=(
            "抓取一个网页并转成纯文本（自动剥掉 HTML 标签、脚本、样式）。"
            "适合读文档、README、博客正文。"
            "注意：不执行 JavaScript，纯前端渲染的页面可能抓不到内容。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "完整 URL，必须以 http:// 或 https:// 开头"},
                "max_bytes": {"type": "integer", "description": "返回正文的字符上限，默认跟工具限额一致"},
            },
            "required": ["url"],
        },
        handler=_web_fetch,
        kind="fetch",
    ),
)

ALL_TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in _SPECS)

# AGENTD_TOOLS 的取值 → 实际启用的工具集合
TOOL_PROFILES: dict[str, tuple[str, ...]] = {
    "native": ALL_TOOL_NAMES,
    "all": ALL_TOOL_NAMES,
    "read_only": ("read_file", "list_dir", "glob", "grep"),
    "ro": ("read_file", "list_dir", "glob", "grep"),
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

    只读动作（read / search / fetch）在任何策略下都不弹 —— 每个 ls 都弹窗，用户三分钟
    就会学会无脑点"允许"，审批本身也就废了。

    联网工具（web_search / web_fetch）刻意也归到"不弹"这一档，尽管它们能把数据送出
    本机。理由：`read_file` 读到的内容本来就会进 LLM 的请求体，**出网通道早就存在**，
    再对搜索/抓取加审批拦不住什么，只会让"查个文档"变成一路点允许。
    真正需要把住的闸门是「改文件 / 执行命令」——那两个仍然是 must-approve。
    """
    if policy not in POLICIES:
        policy = "native"
    if policy == "none":
        return False
    if kind in ("read", "search", "fetch"):
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
        additional_roots: list[str | Path] | None = None,
        allow_outside: bool = False,
        profile: str = "native",
        max_bytes: int = 65536,
        timeout: float = 30.0,
        max_results: int = 200,
    ) -> None:
        base = Path(cwd).expanduser().resolve() if cwd else Path.cwd().resolve()
        roots: list[Path] = []
        for raw in additional_roots or []:
            try:
                p = Path(raw).expanduser().resolve()
            except OSError:  # pragma: no cover
                continue
            if p != base and p not in roots:
                roots.append(p)
        self.runtime = ToolRuntime(
            cwd=base,
            root=base,
            additional_roots=tuple(roots),
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
