"""上下文预算：估 token、按预算裁剪。

为什么要有这一层
----------------
此前会话历史是**全量**喂给模型的（store 读全部行、kernel 不做长度过滤）。短会话
没问题；一旦累积超过模型窗口，表现不是"报错"，而是模型开始答非所问、或输出被静默
截断 —— 用户看到的是"它突然变傻了"，而日志里什么都没有。宁可我们自己在发请求前
按预算裁一刀，并把"裁掉了几轮"告诉用户。

为什么不引 tiktoken
--------------------
1. 依赖体积不值当：光词表就几十 MB，只为估算。
2. **算不准才是常态**：GLM / Qwen / Llama / MiMo 的词表互不相同，同一段中文在
   各家能差 20%~40%。既然只能估，就估保守一点。
3. 估**大**的代价是少放一两句历史；估**小**的代价是请求溢出、模型胡说。所以最终
   结果统一上浮安全余量 —— 宁可浪费一点预算。

估法（不看具体内容，只做字符分类计数）::

    CJK（汉 / 日文假名 / 韩文音节）  ≈ 1 字符 / token
    其余（ASCII / 数字 / 标点 / emoji）≈ 4 字符 / token

这个 Ratio 对英文略保守、对中文恰好；偏差由上面的安全余量统一兜住。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from .models import Message

# CJK 字符类：中文阅读习惯下这些字符基本一个字一个 token。
# 范围说明（避免有人以为是随手抄的魔数）：
#   3040-30FF  日文平假名 / 片假名
#   3400-4DBF  CJK 扩展 A
#   4E00-9FFF  CJK 基本区（日常汉字全在这里）
#   F900-FAFF  CJK 兼容表意文字
#   AC00-D7AF  韩文音节
_CJK_RE = re.compile(
    "["
    "\u3040-\u30ff"
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\uf900-\ufaff"
    "\uac00-\ud7af"
    "]"
)

_CJK_CHARS_PER_TOKEN = 1.0    # CJK：1 字 1 token
_ASCII_CHARS_PER_TOKEN = 4.0  # 其余：4 字符 1 token（官方口径 3~4，取 4 偏保守）
_SAFETY_MARGIN = 1.1          # 上浮 10%：估大不估小

# 每条消息的固定开销（role 字段 + JSON 包裹；OpenAI 官方口径每条约 3~4 token）
_PER_MESSAGE = 4
# 一次 tool_call 的包裹开销：{"id":"..","type":"function","function":{"name":"..","arguments":".."}}
_PER_TOOL_CALL = 10


def estimate_tokens(text: str) -> int:
    """粗略估一段文本占多少 token。纯函数，中文按字、英文按 4 字符折算。

    空串返回 0 —— 别返回 _PER_MESSAGE，那是"一条消息"的开销，不该算在裸文本头上。
    """
    if not text:
        return 0
    stripped = _CJK_RE.sub("", text)
    cjk = len(text) - len(stripped)
    ascii_chars = len(stripped)
    rough = cjk / _CJK_CHARS_PER_TOKEN + ascii_chars / _ASCII_CHARS_PER_TOKEN
    # ceil 而不是 int()：哪怕算出 0.4 个 token 也实实在在占一个位置。
    # 非空文本至少算 1 —— 返回 0 会让上层的预算比较得出"这条免费"的荒谬结论。
    return max(1, math.ceil(rough * _SAFETY_MARGIN))


def estimate_message_tokens(message: Message) -> int:
    """一条消息占多少 token —— 含工具调用 / name 的结构开销。

    role="tool" 的内容常常是几十 KB 的工具输出，是全序列里最容易爆掉的一块，
    所以这里必须连 arguments / output 一起算，只算 content 会严重低估。
    """
    total = _PER_MESSAGE + estimate_tokens(message.content)
    for call in message.tool_calls or []:
        total += _PER_TOOL_CALL + estimate_tokens(call.name) + estimate_tokens(call.arguments)
    if message.name:
        total += estimate_tokens(message.name)
    return total


def estimate_messages_tokens(messages: list[Message]) -> int:
    """整个消息序列的 token 估算值（不含 system —— 那部分单独算，见调用方）。"""
    return sum(estimate_message_tokens(m) for m in messages)


def estimate_tools_tokens(tools: list[dict] | None) -> int:
    """tools schema 占多少 token —— 它是按 wire 格式一次性算进窗口的。

    放着 25 个 MCP 工具的 schema 可不是小数，数 Harish 工作时把它漏掉，
    "预算看着还剩一半"其实早就贴上限了。

    序列化失败（不可 JSON 化的怪 dict）返回 0：这只是预算的扣减项，
    估不出来最多是少放一点，不该让对话失败。
    """
    if not tools:
        return 0
    try:
        raw = json.dumps(tools, ensure_ascii=False)
    except (TypeError, ValueError):
        return 0
    return estimate_tokens(raw)


@dataclass(frozen=True)
class Pressure:
    """一次"上下文压力"读数。

    做成对象而不是一个 bool，是因为要对用户说话：只说"要不要压"不够，
    用户需要知道**压的那一刻到底有多满**（"已到 83% 上限"远比"触发了压缩"可解释）。
    """

    used: int = 0        # 估算出来的实际占用
    budget: int = 0      # 预算上限（0 = 不限）
    trigger: int = 0     # 触发线 = budget * ratio
    ratio_pct: int = 0   # 触发线占预算的百分比

    @property
    def unlimited(self) -> bool:
        return self.budget <= 0

    @property
    def percent(self) -> int:
        """占用百分比（上限 999，防止除零和离谱数字撑爆格式串）。"""
        if self.budget <= 0:
            return 0
        return min(999, round(self.used * 100 / self.budget))

    @property
    def over_trigger(self) -> bool:
        """是否越过触发线。不限预算时恒为假 —— 没有上限就没有"接近上限"。"""
        if self.budget <= 0 or self.trigger <= 0:
            return False
        return self.used >= self.trigger


NO_PRESSURE = Pressure()


def measure(
    messages: list[Message],
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    budget: int,
    ratio_pct: int = 0,
) -> Pressure:
    """度量"这一轮将要发出去的请求"占了多少窗口。

    budget<=0（不限）或 ratio_pct<=0（关闭自动压缩）都直接返回 ``NO_PRESSURE`` ——
    没设上限的话，"接近上限"这句话本身就不成立。
    """
    if budget <= 0 or ratio_pct <= 0:
        return NO_PRESSURE
    used = (
        estimate_tokens(system or "")
        + estimate_messages_tokens(messages)
        + estimate_tools_tokens(tools)
    )
    return Pressure(
        used=used,
        budget=budget,
        # max(1, ...)：预算很小或触发线很低时整除会把触发线算成 0，
        # 而 trigger<=0 被当成"没触发线"处理 —— 那等于用一条算术副作用把
        # 自动压缩悄悄关了，这种事最该早点暴露而不是静默。
        trigger=max(1, budget * ratio_pct // 100),
        ratio_pct=ratio_pct,
    )


def align_cut(messages: list[Message], index: int) -> int:
    """把切点往前退到最近的**块起点**（见 :func:`split_blocks`）。

    为什么非对齐不可：切点要是落在一个工具往返的中间，切完之后剩下序列的第一条
    就是孤儿 ``role="tool"`` —— 端点直接 4xx。压缩会把切点持久化下来当窗口起点，
    所以这里的错不会当场暴露，而是在下一轮、下一次重启之后才炸，属于那种最难查的 bug
    （表面症状是"重启完 start typing 就报错"）。

    已经在块起点上（或越界）就原样返回。
    """
    if index <= 0:
        return index
    if index >= len(messages):
        return len(messages)
    blocks = split_blocks(messages)
    seen = 0
    for block in blocks:
        nxt = seen + len(block)
        if index < nxt:      # 切点落在这一块内部 → 退到块起点
            return seen
        if index == nxt:     # 正好在边界上
            return index
        seen = nxt
    return index


# ---- 自动压缩：到预算的百分之多少就动手 ----

# 默认触发线。为什么是 80：压一次是要**额外调一次模型**的，必须在还有余量时动身。
# 剩下的 20% 是给"本轮提问 + 工具往返 + 模型回答"留的周转空间 —— 真到 100% 就
# 只剩硬裁剪（把最老的历史整块丢掉）这一条路，那一步是不可逆的信息丢弃。
DEFAULT_COMPACT_RATIO = 80

# 触发之后要压到多低（= 触发线的几分之一）。没有这个滞后区间的话，压完刚好回到
# 触发线下方，下一轮加两条消息立刻又触发 —— 每轮压一次的代价远高于偶尔多占一点预算。
_COMPACT_TARGET_DIVISOR = 2


@dataclass(frozen=True)
class CompactPlan:
    """一次自动压缩的决策。做成对象而不是一个 bool，是因为压完要对用户交代。

    ``cut`` 是要压掉的消息条数（从最早开始数），``0`` = 不动手。
    触发那一刻的占用百分比也在里面（``pressure``）—— 事后再拼就拼不出来了。
    """

    pressure: Pressure = NO_PRESSURE
    cut: int = 0
    # system prompt 与 tools schema 的固定占用就已经在触发线之上：
    # 再怎么压消息也降不下来 —— 见 plan_compaction 里的解释。
    futile: bool = False

    @property
    def should_compact(self) -> bool:
        return self.cut > 0


def plan_compaction(
    messages: list[Message],
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    budget: int,
    ratio_pct: int = DEFAULT_COMPACT_RATIO,
) -> CompactPlan:
    """用量到了预算的 ``ratio_pct``% 就给出一个"该压了"的方案。

    与 :func:`trim_history` 的分工，两条是被同一个预算驱动的先后手：

    - **本函数是主动回收**：余量还够时先把老历史总结成摘要，**先留一份信息**再缩小
      窗口，原始对话一条不删（历史回放照旧，见 store.messages）；
    - ``trim_history`` 是最后防线：真正溢出那一步才动手，直接把老历史丢掉。

    三条与 ``trim_history`` 同源的约束（违反任何一条的代价都远超省下的那点预算）：
    1. 切点必须落在块边界上（见 :func:`align_cut`）—— 压缩窗口会被持久化成下一轮
       的起点，切错位置要到下一轮才炸，是最难查的一类 bug；
    2. **最后一块不压**：那里装着用户这一轮的提问，压掉等于没听见。
    3. 一次压到目标线（触发线的一半减去固定开销）而不是压一点点，避免轮轮都压；
       固定开销（system + tools）单独就已经越过触发线时认输（``futile``）——
       那时再压也降不下来，只会每轮白跑一次压缩。

    ``budget <= 0`` 或 ``ratio_pct <= 0`` 恒返回不动手：没设上限的话，
    "接近上限"这句话本身就不成立。
    """
    pressure = measure(
        messages, system=system, tools=tools, budget=budget, ratio_pct=ratio_pct
    )
    if not pressure.over_trigger or not messages:
        return CompactPlan(pressure=pressure)

    # system prompt 与 tools schema 是**每轮都付**的固定开销，压不掉。
    # 它们单独就已经越过触发线时，再压消息也降不到线下 —— 结果是每轮白跑一次
    # 压缩（多一次 LLM 调用），而预算一点没省。这种情况必须认出来就地停手，
    # 交给调用方去抱怨配置，而不是假装还在努力。
    overhead = pressure.used - estimate_messages_tokens(messages)
    if overhead >= pressure.trigger:
        return CompactPlan(pressure=pressure, futile=True)

    blocks = split_blocks(messages)
    # 整段就是本轮这一块时不能压：那等于把用户的提问扔进黑洞
    if len(blocks) < 2:
        return CompactPlan(pressure=pressure)

    costs = [sum(estimate_message_tokens(m) for m in block) for block in blocks]
    # 目标线要在**消息**这个尺度上说：固定开销先扣掉，剩下的才是消息能占的额度。
    target = max(0, pressure.trigger // _COMPACT_TARGET_DIVISOR - overhead)
    last = len(blocks) - 1
    used = sum(costs)
    cut_blocks = 0

    # 从最老往新压。**至少压一块**：既已越过触发线，压一块总比留着增长好；
    # 之后再按需继续，直到再压就要掉到目标线以下为止。
    for index in range(last):
        if cut_blocks > 0 and used - costs[index] < target:
            break
        used -= costs[index]
        cut_blocks += 1

    if cut_blocks == 0:
        return CompactPlan(pressure=pressure)

    raw_cut = sum(len(block) for block in blocks[:cut_blocks])
    # 算出来的切点本来就在块边界上，这一行是给"将来有人改坏了计数"买保险
    return CompactPlan(pressure=pressure, cut=align_cut(messages, raw_cut))


def describe_compact(pressure: Pressure, *, dropped: int, summarized: bool) -> str:
    """把一次自动压缩写成一句人话 —— 提示文案集中在此，免得两处说法漂移。"""
    head = (
        f"上下文已到预算的 {pressure.percent}%"
        f"（{pressure.used}/{pressure.budget} token）"
    )
    if not summarized:
        # 摘要没压出来就不许缩小窗口（见 kernel._auto_compact），
        # 但也不能完全不出声 —— 用户得知道余量已经不多了。
        return (
            f"{head}，自动压缩没能产出摘要，较早的对话先原样保留；"
            f"继续累积后会按预算省略较早的历史。"
        )
    return (
        f"{head}，已把较早的 {dropped} 条对话压缩成摘要。"
        f"原始记录仍在会话里，历史回放不受影响。"
    )


@dataclass(frozen=True)
class TrimReport:
    """裁剪结果回执 —— 用来告诉用户"我替你扔了什么"，不说就是黑箱。"""

    before: int = 0            # 裁剪前的消息条数
    after: int = 0             # 裁剪后的消息条数
    dropped: int = 0           # 被丢掉的消息条数
    kept_first_user: bool = False  # 保留窗口之外，额外抢救了首轮用户消息
    overflowing: bool = False  # 连最后一块都超预算（已无法再裁，只能硬发）

    @property
    def happened(self) -> bool:
        """是否真的动了刀 —— 没丢东西时不该打扰用户。"""
        return self.dropped > 0


#: 没发生裁剪时的回执，省得调用方到处 new 一个空对象
NO_TRIM = TrimReport()


def split_blocks(messages: list[Message]) -> list[list[Message]]:
    """把消息序列切成"不可分割的块"。

    为什么必须分块：`assistant(带 tool_calls)` 与紧跟其后的 `role="tool"` 结果是
    **一个原子块**。只留一半会产出孤儿 tool 消息 —— OpenAI 兼容端点见到没有前置
    tool_calls 的 role="tool" 是直接 4xx 的，比超预算还难排查。

    其余消息各自成块（一条 user / 一条 assistant 纯文本 = 一块）。
    """
    blocks: list[list[Message]] = []
    i = 0
    total = len(messages)
    while i < total:
        message = messages[i]
        if message.role == "assistant" and message.tool_calls:
            block = [message]
            j = i + 1
            while j < total and messages[j].role == "tool":
                block.append(messages[j])
                j += 1
            blocks.append(block)
            i = j
        else:
            blocks.append([message])
            i += 1
    return blocks


def trim_history(
    messages: list[Message],
    budget: int,
    *,
    keep_first_user: bool = True,
) -> tuple[list[Message], TrimReport]:
    """把消息序列裁到 ``budget`` token 以内（保尾弃头）。

    ``budget <= 0`` 表示"不限" —— 原样返回（`NO_TRIM`），保持与此前完全一致的行为。

    三条硬约束：
    1. **最后一块必留**：它装着本轮的用户输入，丢掉等于没问。
    2. **整块存亡**：见 :func:`split_blocks` —— 绝不切开工具往返。
    3. **首轮 user 尽量留**：它通常是整个会话的任务声明（"帮我把这个项目改成 X"），
       比中间几轮的寒暄更值得占预算，所以单独抢救一格。

    注意这里**不截断单条消息的内容**：一口气贴了一整个文件的用户消息要是被从中间
    剪断，意图就丢了，还不如让它整体超一点预算。单条工具输出另有一道
    ``tools.max_bytes`` 兜着，也不会长到离谱。
    """
    if budget <= 0 or not messages:
        return list(messages), NO_TRIM

    if estimate_messages_tokens(messages) <= budget:
        return list(messages), NO_TRIM

    blocks = split_blocks(messages)
    costs = [sum(estimate_message_tokens(m) for m in block) for block in blocks]

    # 从尾部倒着装：越新的内容越值钱。装不下就停，于是保留区间天然是
    # 一段**连续**的尾部 \[start, end\] —— 后面补首轮 user 时可以 O(1) 挤空间。
    last = len(blocks) - 1
    start = last
    used = costs[last]
    for index in range(last - 1, -1, -1):
        if used + costs[index] > budget:
            break
        used += costs[index]
        start = index

    # 首轮用户消息所在块（工具往返块里不会有 user，所以取到的一定是本轮的提问）
    first_block: int | None = None
    if keep_first_user:
        for index, block in enumerate(blocks):
            if any(m.role == "user" for m in block):
                first_block = index
                break

    kept_first = False
    if first_block is not None and first_block < start:
        used += costs[first_block]
        kept_first = True

    # 补了首轮之后可能又超了：从最老端往外挤，但最后一块永远不许挤掉
    while used > budget and start < last:
        used -= costs[start]
        start += 1

    flat: list[Message] = []
    if kept_first and first_block is not None:
        flat.extend(blocks[first_block])
    for block in blocks[start:]:
        flat.extend(block)

    report = TrimReport(
        before=len(messages),
        after=len(flat),
        dropped=len(messages) - len(flat),
        kept_first_user=kept_first,
        # 只剩最后一块还超预算 = 没得裁了，如实标记出来（调用方据此提示用户）
        overflowing=start >= last and used > budget,
    )
    return flat, report


def compose_system(base: str | None, sections: list[str]) -> str | None:
    """把若干段"运行时说明"拼到用户写的 system prompt 后面。

    为什么要这么个函数：系统提示现在有**多个来源** —— 用户手写的
    ``AGENTD_SYSTEM_PROMPT``、工作目录的浅层结构（见 tools.workspace_brief）、
    将来的跨会话记忆。各自往上拼字符串迟早有人写出两个换行 or 零个换行，
    统一在这里拼。

    空段会被跳过；全空返回 None（不传 system 给后端 —— 有的后端收到空 system
    消息会抱怨，干脆不给）。
    """
    parts = [p.strip() for p in [base or "", *sections]]
    kept = [p for p in parts if p]
    return "\n\n".join(kept) if kept else None


def sanitize_history(messages: list[Message]) -> list[Message]:
    """剔掉会招致端点 4xx 的畸形式：**没头或没尾的工具往返**。

    两种必须在这里兜住的情况：

    1. 上一轮在工具执行期间被叫停 —— 库里留下一条 ``assistant(带 tool_calls)``，
       却没有紧随的 ``role="tool"`` 响应。OpenAI 兼容端点的规定是"assistant 发出
       tool_calls 后，每个 id 都必须有一条 tool 消息回应"，缺一个就是 4xx。
    2. 反过来：``role="tool"`` 前面没有带 tool_calls 的 assistant（历史被手工
       改过、或旧版本留下的行）。

    两种都**整段丢弃**而不是缺啥补啥：半残的工具往返喂进去，模型会以为自己真
    的读过某个文件 —— 比它老老实实说"没有上下文"更糟。

    注意这里也顺手保证了 tool 结果的顺序与 tool_calls 一致（端点只认 id
    对应关系，但我们自己看历史时的可读性不是免费的）.
    """
    out: list[Message] = []
    pending: Message | None = None
    expected: list[str] = []
    answered: dict[str, Message] = {}

    def commit() -> None:
        """把攒着的这条工具往返落到输出里 —— 前提是它完整。"""
        nonlocal pending, answered
        if pending is not None and all(call_id in answered for call_id in expected):
            out.append(pending)
            out.extend(answered[call_id] for call_id in expected)
        pending = None
        answered = {}

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            commit()  # 上一条还没结算就又来了一条 ⇒ 上一轮没走到响应该有的地方
            pending = message
            expected = [tc.id for tc in message.tool_calls]
            answered = {}
            continue
        if message.role == "tool":
            call_id = message.tool_call_id or ""
            # 空 id 兜底放行：自己造的 id 一定有值，但万一某个后端不带，
            # 丢掉它比留着更可能让模型看不到工具输出。
            if pending is not None and (not call_id or call_id in expected):
                answered[call_id] = message
            # 孤儿 / id 对不上：留着就是一条 4xx，直接丢
            continue
        commit()
        out.append(message)
    commit()
    return out


def describe_trim(report: TrimReport, *, budget: int) -> str:
    """把回执写成一句人话 —— 给用户看的提示语集中在这里，免得两处文案漂移。"""
    tail = (
        f"（预算 {budget} token，实际仍在超窗，模型可能会截断输出）"
        if report.overflowing
        else f"（预算 {budget} token）"
    )
    rescued = "已保留首轮提问，" if report.kept_first_user else ""
    return (
        f"上下文超出预算，{rescued}省略了较早的 {report.dropped} 条历史消息{tail}。"
        f"如需完整上下文，请提高 AGENTD_MAX_CONTEXT_TOKENS（当前支持的最大窗口取决于所选模型）。"
    )
