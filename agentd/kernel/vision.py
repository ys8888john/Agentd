"""视觉转译层：把消息里的图片块翻译成文字，喂给不支持视觉的主模型。

为什么要有这一层
----------------
主模型是纯文本（本机 qwen3.5 这类）时，用户发来的图片块打过去只会吃端点 4xx。
与其要求用户换模型，不如在 agentd 内部做一次"代看"：

    含图消息 ──► 视觉模型 V（AGENTD_VISION_* 配置）──► 文字转述
                                                        │
    image_url 块 ──替换成──► [图片转述] … ◄─────────────┘
                                                        │
    纯文本消息流 ──► 主模型 X（它以为自己"看过"图）

三个设计决定：

1. **按图片内容 hash 缓存**。历史里同一张图每轮都在，不缓存就是每轮多一次
   LLM 往返。缓存键是图片 base64 的 sha256，内存 + 落盘（~/.agentd/vision_cache/）
   双层，进程重启后同图依旧秒回。
2. **配置走 boot.env_value 注入的 val 回调**，与主模型同一套优先级链
   （RUNTIME_CONFIG > GUI 热配置 > 启动环境变量 > .env），GUI 的 hotenv.json
   直接写 AGENTD_VISION_* 即可热切换。没配 = 功能关，annotate 原样放行，
   行为与没有这一层完全一致。
3. **转述失败不炸轮**。单张图转述失败就塞一句失败说明文本，主模型至少能
   把"图没看成"讲清楚，整轮对话照常继续。

诚实的边界：主模型看到的是**转述**不是原图 —— 精细布局、密集小字的还原度
取决于视觉模型；截图问代码、报错弹窗这类 OCR 场景效果最好。
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

from .llm import AUTO, LLM, OllamaNativeLLM, OpenAICompatLLM
from .models import Message

# 转译指令：截图/代码场景的核心诉求是 OCR，所以"逐字转写"必须放最前面压阵
_DESCRIBE_PROMPT = (
    "你是图片转述器。请用中文详尽描述这张图片，供一个无法看图的语言模型使用：\n"
    "1. 图中出现的所有文字【逐字转写】——包括代码、报错信息、表格内容、"
    "界面文案，尽量保持原有换行与结构；\n"
    "2. 若是图表，读出坐标轴、图例和关键数据；\n"
    "3. 描述整体布局与图中对象的关系。\n"
    "只输出转述本身，不要寒暄、不要评论。"
)

_PLACEHOLDER_OK_PREFIX = "[图片转述（由视觉模型生成）]"

# 成功转述的"使用说明书"。没有这段话，主模型会把注入的转述当成**用户自己打的
# 模板文本**（2026-10-11 用户截图实锤：模型回"我没有收到要描述的图片……您提到的
# '图片转述'似乎是一个模板"），一边说没图一边又按转述内容分析，自相矛盾。
# 必须在注入点把身份、来源、用法一次性讲清楚。
_VISION_NOTE_FRAME = (
    "（系统注入，非用户输入：用户在这条消息里附了图片，当前主模型无法直接接收图像，"
    "视觉模型已代为查看，下面这段转述就是这张图片的实际内容。"
    "请把它当作你亲眼所见的图片直接回答用户；不要声称“没有收到图片”，"
    "不要把它当成用户提供的文字描述或模板，也不要要求用户重新上传。）"
)


def _image_hash(part: dict) -> str:
    """图片块的缓存键：base64 内容的 sha256。取不到内容返回空串。"""
    url = str((part.get("image_url") or {}).get("url") or "")
    b64 = url.split(",", 1)[1] if url.startswith("data:") and "," in url else url
    if not b64:
        return ""
    return hashlib.sha256(b64.encode("ascii")).hexdigest()


def _is_failed_note(note: str) -> bool:
    """转述失败折出来的说明文本（不该进缓存、不该套'代看成功'的框架）。"""
    return note.startswith("[图片转述失败") or note.startswith("[视觉模型返回了空转述]")


def _log(msg: str) -> None:
    try:
        sys.stderr.write(f"[agentd/vision] {msg}\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - 日志失败不影响主流程
        pass


class VisionInterpreter:
    """按配置把消息里的图片块替换成文字转述。线程不设防：只在内核锁内跑。"""

    def __init__(self, val: Callable[[str, str], str]) -> None:
        self._val = val                      # 配置取数回调（boot.env_value，热加载）
        self._llm: LLM | None = None
        self._sig: tuple | None = None
        self._cache: dict[str, str] = {}     # sha256 -> 转述文本（进程内热缓存）
        self._cache_dir = Path.home() / ".agentd" / "vision_cache"

    # ---- 配置 ----

    def _resolve(self) -> LLM | None:
        """按当前配置返回视觉后端；没配置返回 None（功能关）。

        与 LiveLLM 同款签名比对：配置变了才重建，Ollama 的 auto 模型解析
        缓存等实例状态得以保留。
        """
        base_url = self._val("AGENTD_VISION_BASE_URL", "").strip()
        host = self._val("AGENTD_VISION_HOST", "").strip()
        model = self._val("AGENTD_VISION_MODEL", "").strip()
        api_key = self._val("AGENTD_VISION_API_KEY", "").strip()
        if not base_url and not host:
            return None
        sig = (base_url, host, model, api_key)
        if self._llm is not None and self._sig == sig:
            return self._llm
        if base_url:
            self._llm = OpenAICompatLLM(
                base_url=base_url, model=model or AUTO, api_key=api_key or "ollama"
            )
        else:
            self._llm = OllamaNativeLLM(host=host, model=model or AUTO)
        self._sig = sig
        _log(f"启用视觉转译: {type(self._llm).__name__} model={model or AUTO}")
        return self._llm

    # ---- 缓存 ----

    def _cache_read(self, h: str) -> str | None:
        if h in self._cache:
            return self._cache[h]
        try:
            p = self._cache_dir / f"{h}.txt"
            if p.is_file():
                note = p.read_text(encoding="utf-8")
                self._cache[h] = note
                return note
        except OSError:
            pass
        return None

    def _cache_write(self, h: str, note: str) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            (self._cache_dir / f"{h}.txt").write_text(note, encoding="utf-8")
        except OSError as exc:
            _log(f"转述缓存写盘失败（不影响本轮）: {exc}")

    # ---- 主入口 ----

    async def annotate(self, messages: list[Message]) -> tuple[list[Message], str]:
        """把消息里的图片块换成文字转述。返回 (新消息列表, 给用户的提示)。

        无图 / 未配置视觉模型时原样返回（提示为空串）——调用方据此决定
        要不要给界面发 Notice。提示只在**本轮有新转述**时才发：历史里那张
        图每轮都在窗口里，全命中缓存的轮次再刷一遍只是思考通道的噪音。
        """
        # 1. 有没有图可转（没图直接放行，零开销）
        has_images = any(m.content_images() for m in messages)
        if not has_images:
            return messages, ""

        llm = self._resolve()
        if llm is None:
            return messages, ""

        # 2. 补齐缺失的转述。计数口径（2026-10-11 修正）：seen = 本轮窗口里
        #    去重后的全部图片，fresh = 新调视觉模型的，reused = 命中缓存的，
        #    恒等式 fresh + reused == len(seen)。修复前命中缓存的 continue 排在
        #    seen.add 之前，缓存轮会报「已用视觉模型转述 0 张图片」。
        fresh = 0
        reused = 0
        failed: dict[str, str] = {}   # 本轮转述失败的：替换要用，但不进缓存
        seen: set[str] = set()
        for m in messages:
            for part in m.content_images():
                h = _image_hash(part)
                if not h or h in seen:
                    continue
                seen.add(h)
                if self._cache_read(h) is not None:
                    reused += 1
                    continue
                note = await self._describe(llm, part)
                fresh += 1
                if _is_failed_note(note):
                    # 失败不进缓存（内存/磁盘都不进）：端点恢复后下一轮还有
                    # 图在窗口里，重试还有机会成功。缓存失败 = 这张图永远失败。
                    failed[h] = note
                else:
                    self._cache[h] = note
                    self._cache_write(h, note)

        # 3. 替换：image_url 块 -> [图片转述] 文本块（不改原 Message，copy 出新对象）
        out: list[Message] = []
        for m in messages:
            if not m.content_images():
                out.append(m)
                continue
            parts: list[dict] = []
            for p in m.content:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    h = _image_hash(p)
                    note = (
                        self._cache_read(h)
                        or failed.get(h)
                        or "[图片转述失败：无可用结果]"
                    )
                    if _is_failed_note(note):
                        # 失败文本自带 [图片转述失败：…] 标签，能自解释；
                        # 不套"代看成功"的框架去误导主模型。
                        parts.append({"type": "text", "text": note})
                    else:
                        parts.append({
                            "type": "text",
                            "text": f"{_PLACEHOLDER_OK_PREFIX}\n{_VISION_NOTE_FRAME}\n{note}",
                        })
                else:
                    parts.append(p)
            out.append(m.model_copy(update={"content": parts}))

        if not fresh:
            return out, ""   # 全是缓存回填：不发 Notice，别每轮都刷同一条

        bits = [f"已用视觉模型转述 {len(seen)} 张图片", f"新转述 {fresh} 张"]
        if reused:
            bits.append(f"命中缓存 {reused} 张")
        return out, "，".join(bits)

    async def _describe(self, llm: LLM, part: dict) -> str:
        """调视觉模型转述一张图。任何失败都折成说明文本，绝不外抛。"""
        ask = Message.user([part, {"type": "text", "text": _DESCRIBE_PROMPT}])
        try:
            text = (await llm.complete([ask])).strip()
        except Exception as exc:  # noqa: BLE001 - 转述失败不能拖垮整轮对话
            _log(f"转述失败: {type(exc).__name__}: {exc}")
            return f"[图片转述失败：{type(exc).__name__}: {exc}]"
        return text or "[视觉模型返回了空转述]"
