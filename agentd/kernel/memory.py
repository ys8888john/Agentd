"""跨会话记忆：摘要层 + 事实层。

两层为什么都要
--------------
只有摘要：聊过十次之后，摘要自己也有几万字，而且"用户是个什么样的开发者"这种
稳定事实每次都被重新总结一遍、次次措辞不同 —— 既费 token 又不稳定。
只有事实：对话过程全丢，"上次那个 token 预算改到哪儿了"照样答不上来。

所以分两层存：

    事实层（facts）  —— ~/.agentd/memory.md，稳定、长期、可直接手改的人读文本。
                        例如"偏好简体中文""本机 Ollama 拉的模型是 qwen3.5"。
    摘要层（summary）—— 会话库里的 summaries 表，按会话归档的进展（"上次做到哪一步"）。

新会话的 system prompt 里先给事实（永远相关），再给最近若干条摘要（按需相关）。

压缩由谁来调、什么时候调
------------------------
由内核在**下一轮对话开始前**判断要不要压（见 kernel._maybe_compact），不是轮结束时：
轮一结束，传输层拿到 Done 就停了，生成器体之后的代码不保证被跑到，
放在那里等于"记忆看心情"。放在下一轮开头，行为是确定的、也可单测。

压失败了怎么办
--------------
静默出错是最坏的结果（"它为什么像第一次见我"完全没有线索）。所以：失败只写日志、
不抛，会话照常进行 —— 记忆是增强不是依赖，一次压不动不影响这一轮回答。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from .llm import LLM
from .models import Message

# 事实文件的默认位置。放用户目录而不是 cwd —— cwd 随启动方式变，放那儿会出现
# "从 VSCode 启动就失忆"的灵异现象（同理见 store.default_db_path）。
DEFAULT_MEMORY_FILE = Path.home() / ".agentd" / "memory.md"

# 一次最多保留多少条事实。上限存在的理由：这个文件每轮都进 system prompt，
# 让它无限增长等于给每个回答悄悄加税。
MAX_FACTS = 40


def _strip_bullet(text: str) -> str:
    """抠掉行首的项目符号，剩下的是事实本体。

    必须双向清洗（对比时也清洗）：模型抽出来的事实常常自带 "- "，
    直接拿去过重会出现 "- - 偏好简体中文" 这种双符号垃圾。
    """
    return text.strip().lstrip("-*•").strip()


@dataclass(frozen=True)
class Compaction:
    """一次压缩的产物。"""

    summary: str = ""
    facts: tuple[str, ...] = ()


COMPACT_PROMPT = """你是 agentd 的记忆整理器。下面是一段已经结束的对话，把它压缩成两部分。

严格按下面的格式输出，**不要有多余的解释或寒暄**：

摘要：<关于这一段对话的进展与结论，3-5 句。写清楚"用户要做什么、做到哪一步、
下一步卡在哪儿"，而不是"我们讨论了 X"这种空话。>

事实：（每行一条，以 "- " 开头；没有值得长期记住的内容就整块省略）
- <关于用户本人 / 项目 / 环境的稳定事实。例：偏好简体中文、项目是 ACP 架构、
  本机 Ollama 模型是 qwen3.5:9b>

两条判据，拿不准就按它裁：
1. 事实必须是**跨会话仍然成立**的。一次性的决定、临时路径、当天的心情不写。
2. 摘要里不要出现"总结道""讨论了"这类自述行为词，只写内容本身。
""".strip()


_COLON_RE = re.compile(r"[：:]")


def _head_and_body(line: str) -> tuple[str, str]:
    """按行首的分隔符把一行拆成 (标题, 正文)。

    半角冒号必须跟全角一样认：模型写提示符时爱混着来，只认全角的话，
    "摘要: xxx" 整行的干货会被当成"标题行没有正文"直接丢掉 —— 症状是
    摘要恒为空，而日志里一切正常，极难查。
    """
    match = _COLON_RE.search(line)
    if match is None:
        return line.strip().lower(), ""
    return line[: match.start()].strip().lower(), line[match.end():].strip()


# 万一模型把中文标题译走了（小模型偶尔自作主张），也得接住。
_SUMMARY_HEADS = ("摘要", "summary")
_FACT_HEADS = ("事实", "fact")  # "fact" 顺带盖住 "facts"


def parse_compaction(raw: str) -> Compaction:
    """把模型的输出拆成摘要与事实。

    刻意用宽松解析而不是 JSON：给本机小模型（qwen3.5 之类）压东西
    本来就是它勉强能干的事，再要求它输出合法 JSON，失败率会高到不可用。
    缺一块、多一块、标号写错，都不影响另一块 —— 半份记忆远好过没有。
    """
    summary_parts: list[str] = []
    facts: list[str] = []
    target: list[str] | None = None

    for line in (raw or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        head, body = _head_and_body(stripped)

        if head.startswith(_SUMMARY_HEADS):
            target = summary_parts
            if body:
                summary_parts.append(body)
            continue
        if head.startswith(_FACT_HEADS):
            target = facts
            if not body:
                continue
            # "事实：- 项目叫 agentd" 这种把第一条写在标题行上的，别把它扔了
            stripped = body

        if target is facts:
            item = _strip_bullet(stripped)
            if item:
                facts.append(item)
        elif target is summary_parts:
            summary_parts.append(stripped)

    return Compaction(
        summary=" ".join(p for p in summary_parts if p).strip(), facts=tuple(facts)
    )


class MemoryFile:
    """~/.agentd/memory.md —— 人可读、可直接手改的事实层。

    为什么是 Markdown 而不是数据库：这份东西的价值就在于**用户能打开改**。
    "它记错了我的编辑器"这种事，让用户去写 SQL 是不人道的。
    """

    MARKER = "<!-- agentd:facts -->"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_MEMORY_FILE

    async def read(self) -> str:
        try:
            text = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        except OSError:
            return ""
        return text

    async def merge(self, new_facts: tuple[str, ...]) -> str:
        """把新事实并进文件，返回合并后的全文。

        按**纯文本内容**去重（忽略 "- " 前缀与大小写），所以同一个事实被重复抽取
        不会越堆越多。超出 MAX_FACTS 保留最新 —— 越新的观察通常越准。
        """
        existing = await self.read()
        lines = [
            line.rstrip()
            for line in existing.splitlines()
            if line.strip().startswith("-") and not line.strip().startswith("#")
        ]
        seen = {_strip_bullet(line).lower() for line in lines}
        merged = list(lines)
        for fact in new_facts:
            body = _strip_bullet(fact)
            key = body.lower()
            if not body or key in seen:
                continue
            seen.add(key)
            merged.append(f"- {body}")

        if len(merged) > MAX_FACTS:
            merged = merged[-MAX_FACTS:]

        body = "\n".join(merged)
        header = (
            "# agentd 长期记忆\n\n"
            "这个文件每轮对话都会进系统提示 —— 只写**跨会话仍然成立**的事实。\n"
            "手动删改这里的内容会立即生效；写错也不会让对话失败。\n\n"
            f"{self.MARKER}\n"
        )
        return header + body + "\n" if body else header

    async def append_facts(self, facts: tuple[str, ...]) -> None:
        if not facts:
            return
        merged = await self.merge(facts)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self.path.write_text, merged, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - 取决于权限
            print(f"[agentd] 写长期记忆失败：{exc}", flush=True)


# 一次喂给压缩模型的原文上限（字符）。压缩本身就是笔"为了省 token 而花 token"
# 的买卖，不封顶的话，一段带几个大文件读值的对话能把下一次压请求撑到几万字。
TRANSCRIPT_MAX_CHARS = 12000

# 扁平化时每个角色的名字。用中文是因为提示词和事实层本身就是中文的 ——
# 让模型在同一个语言环境里读和写，比让它"读英文产中文"稳。
_ROLE_LABELS = {
    "user": "用户",
    "assistant": "助手",
    "tool": "工具",
    "tool_record": "工具",
    "system": "系统",
}


def render_transcript(messages: list[Message], max_chars: int = TRANSCRIPT_MAX_CHARS) -> str:
    """把一段对话摊成给压缩模型看的纯文本。

    角色名保留（"用户/助手/工具"）而不是纯流水账：谁说的决定了这句话的价值 ——
    "助手说我们决定用 SQLite"和"用户说我们决定用 SQLite"完全是两件事。
    """
    lines: list[str] = []
    for m in messages:
        label = _ROLE_LABELS.get(m.role, m.role)
        body = (m.content or "").strip()
        if m.role == "assistant" and m.tool_calls and not body:
            # 只举手要调工具、没说话的 assistant：content 是空的，
            # 直接跳过会丢掉"这一步是靠工具走的"这个信息，补一句。
            names = ", ".join(tc.name for tc in m.tool_calls if tc.name)
            body = f"（调用工具：{names}）" if names else "（调用工具）"
        if not body:
            continue
        lines.append(f"{label}：{body}")

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    # 超限就**从头部丢**（保最近的进展）。
    # 按行截而不是按字符切：那一刀落在半句话上，模型会把这半句当成原话写进摘要。
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        if total + len(line) > max_chars:
            break
        kept.append(line)
        total += len(line) + 1
    kept.reverse()
    return "\n".join(kept)


async def compact(
    llm: LLM,
    messages: list[Message],
    memory: MemoryFile | None = None,
) -> Compaction:
    """调用一次 LLM 把这段对话压成 (摘要, 事实)，顺手把事实落盘。

    喂进去的不是原始 messages 而是一份**纯文本稿**（见 render_transcript）：
    要压的那一段是"上一次压完到第几条"切出来的，切口可能正好落在一次工具
    往返的中间 —— 那种序列里的 role="tool" 前面没有带 tool_calls 的 assistant，
    原样发给 OpenAI 兼容端点就是 400。扁平化没有这个风险，而且顺手把工具返回
    的大段 JSON 挤成一行可读性更高的文字。

    失败一律返回空 Compaction：记忆是"有了更好"的能力，压不动就让这一轮照常
    进行 —— 挂掉对话去换一份记忆，这笔买卖不划算。
    """
    transcript = render_transcript(messages)
    if not transcript.strip():
        return Compaction()
    try:
        raw = await llm.complete(
            [Message.user(f"对话记录：\n\n{transcript}")], system=COMPACT_PROMPT
        )
    except Exception as exc:  # noqa: BLE001 - 压缩是旁路，任何失败都不该冒泡
        print(f"[agentd] 记忆压缩失败：{type(exc).__name__}: {exc}", flush=True)
        return Compaction()

    result = parse_compaction(raw)
    if result.facts and memory is not None:
        await memory.append_facts(result.facts)
    return result


def render_memory_block(
    facts_text: str, summaries: list[tuple[str, str]], own: str = ""
) -> str:
    """把记忆拼成 system prompt 里的一整段。

    ``summaries`` 是 **store.recent_summaries() 的原样返回值**：每个元素是
    ``(session_id, summary)``。刻意不在这里换顺序 —— 一旦两处的约定对不上，
    摘要段里插的就是一串会话号，而它长得"很像一份摘要"，肉眼很难看出来。

    ``own`` 是**当前会话自己**被压掉的那一段的摘要。平时会话不会读到自己的摘要
    （那些内容本来就在上下文里），但窗口一旦移动，被移出去的部分就只剩这份摘要了
    —— 不给它，模型就真的"什么都不记得"。

    没有内容就返回空串（调用方按空段跳过），别写一句"暂无记忆"占位 ——
    那是在每轮都要付的 system prompt 里放一堆零信息的字符。
    """
    blocks: list[str] = []

    facts = [
        line.strip()
        for line in facts_text.splitlines()
        if line.strip().startswith("-") and not line.strip().startswith("#")
    ]
    if facts:
        # 抠掉标题和注释，只留 "- " 行的原因：标题是给人看的，
        # 模型拿到"# agentd 长期记忆"这种字样只会浪费注意力。
        blocks.append("## 关于用户（长期记忆）\n" + "\n".join(facts))

    own_text = (own or "").strip()
    if own_text:
        # 自己的过去排在别人的摘要之前：相关性更高
        blocks.append("## 本会话更早的进展（已压缩）\n" + own_text)

    earlier = [text.strip() for _session_id, text in summaries if (text or "").strip()]
    if earlier:
        blocks.append("## 更早的会话摘要（按时间从旧到新）\n" + "\n".join(f"- {t}" for t in earlier))

    return "\n\n".join(blocks)
