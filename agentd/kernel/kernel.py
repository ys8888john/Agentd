"""内核入口：把会话、存储、模式编排缝在一起，对外只暴露 Event 流。

传输层（HTTP/SSE、ACP）只跟 handle() 打交道。
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import (
    Done,
    ErrorEvent,
    Event,
    Notice,
    ToolCallDone,
    ToolCallStart,
    MessageDone,
    new_run_id,
    new_session_id,
)
from .llm import LLM
from .context import (
    DEFAULT_COMPACT_RATIO,
    CompactPlan,
    Pressure,
    compose_system,
    describe_compact,
    plan_compaction,
    sanitize_history,
)
from .memory import MemoryFile, compact, render_memory_block
from .models import Message, ToolRecord
from .modes import AgentMode, Mode, ModeContext, SingleMode
from .store import InMemorySessionStore, SessionStore, UnknownSessionError
from .tools import TOOL_PROFILES, ApproveHandler, NativeToolbox, workspace_brief

class UnknownModeError(KeyError):
    """请求的模式未注册。"""


@dataclass
class AgentKernel:
    llm: LLM
    store: SessionStore = field(default_factory=InMemorySessionStore)
    system: str | None = None

    # 原生工具（read_file / glob / grep / write_file / edit / run_command）。
    # 取值见 tools.TOOL_PROFILES：native（默认）| read_only | off。
    native_tools: str = "native"
    tools_allow_outside: bool = False   # 允许原生工具碰 cwd 之外的路径（默认禁止）
    tools_timeout: float = 30.0         # run_command 默认超时
    tools_max_bytes: int = 65536        # 单个工具返回文本上限
    approval_policy: str = "native"     # 见 tools.needs_approval

    # ---- 跨会话记忆 ----
    # 累积这么多条消息就该压一次摘要（0 = 完全关闭记忆压缩）。
    # 选"条数"而不是"token 数"当触发条件：token 估算本身有 10% 的抖动，拿它当
    # 阈值会让"什么时候触发"变得难预测、难复现。
    summary_every: int = 20
    # 新会话往 system prompt 里塞多少条历史摘要
    summary_recall: int = 6
    # 事实层文件位置。给的是**临时目录注入口**：生产默认走 ~/.agentd/memory.md，
    # 单测必须能指到 tmp 去，否则一堆测试会把开发机上的真实记忆读进来（或写进去）。
    memory_file: str | Path | None = None

    # ---- 自动压缩：到上下文预算的百分之多少就动手 ----
    # 为什么要早动手：压一次本身要额外调一次模型，得在还有余量时做；
    # 真到 100% 就只剩"把老历史整块丢掉"这条不可逆的路了（见 context.trim_history）。
    compact_ratio: int = DEFAULT_COMPACT_RATIO
    # 上下文预算。**用 callable 而不是一个 int**：GUI 的 token 预算是按模型 profile
    # 热写的（本地 8b 与云端 GLM 差三倍以上），启动定死会让"切了模型预算还是旧值"。
    context_budget: Callable[[], int] = field(default_factory=lambda: lambda: 0)

    def __post_init__(self) -> None:
        self._modes: dict[str, Mode] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # 每个会话的额外配置（cwd / mcp_servers），由 create_session 记下
        self._session_opts: dict[str, dict[str, Any]] = {}
        # 每个会话"已经压到第几条"。只在进程内存里：重启后会重新数一遍行，
        # 宁可重压一次，也不要持久化一个可能与库内容不一致的下标。
        self._compacted: dict[str, int] = {}
        # 每个会话的"固定开销已超触发线"提醒过没有（见 _warn_about_overhead）
        self._overhead_warned: dict[str, bool] = {}
        self._memory = MemoryFile(self.memory_file)
        self.register(SingleMode())
        self.register(AgentMode())

    def _make_toolbox(
        self, cwd: str | None, additional_roots: list[Any] | None = None
    ) -> NativeToolbox | None:
        """按会话 cwd 造一个原生工具箱；profile=off 或构造失败都返回 None。

        每个 run 造一个（而不是内核级共享）：cwd 是会话级的，
        而且构造只是建个字典，成本可以忽略。
        """
        if self.native_tools not in TOOL_PROFILES:
            print(
                f"[agentd] 未知 AGENTD_TOOLS 取值 {self.native_tools!r}，按 native 处理",
                file=sys.stderr,
                flush=True,
            )
        profile = self.native_tools if self.native_tools in TOOL_PROFILES else "native"
        if not TOOL_PROFILES[profile]:
            return None
        try:
            return NativeToolbox(
                cwd=cwd,
                additional_roots=additional_roots or [],
                allow_outside=self.tools_allow_outside,
                profile=profile,
                max_bytes=self.tools_max_bytes,
                timeout=self.tools_timeout,
            )
        except OSError as exc:  # 边界处兜底：工具箱建不起来不该让整轮对话挂掉
            print(
                f"[agentd] 原生工具箱初始化失败，本会话禁用：{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

    def _system_prompt(self, opts: dict[str, Any], memory: str = "") -> str | None:
        """本轮真正要用的 system prompt = 用户写的 + 运行时说明。

        运行时说明目前有两类 source（将来的还可以继续往 sections 里加）：
        1. 工作目录长什么样（见 tools.workspace_brief）—— 没有它，模型开局不知道
           自己站在哪个目录、有哪些文件，第一轮常常浪费在探路上，或者直接猜错
           文件名；
        2. 跨会话记忆（事实 + 历史摘要，见 kernel.memory）—— 没有它，每次开新会话
           都像是第一次见面，"上次做到哪儿了"永远答不上来。

        刻意**不给没有工具的会话灌工作目录**：读不到文件的会话里，一份目录清单
        只会让模型以为自己能操作文件 —— 那是比"不知道"更糟的错误。
        记忆不受这条限制：认识用户跟能不能碰文件系统是两回事。

        memory 由调用方传进来而不是从 opts 里取 —— opts 是**会话级配置**
        （create_session 写一次），把每轮都变的记忆文本塞进去，等于让读取
        者分不清"这是配置还是临时产物"。
        """
        sections: list[str] = [memory]
        if self.native_tools in TOOL_PROFILES and TOOL_PROFILES[self.native_tools]:
            cwd = opts.get("cwd")
            if cwd:
                try:
                    sections.append(workspace_brief(Path(cwd)))
                except OSError:
                    pass  # 目录被删了 / 没权限读：环境说明掉了不该整轮失败
        return compose_system(self.system, sections)

    async def _maybe_compact(self, session_id: str, rows: list[Message]) -> None:
        """够久了就压一次对话成为长期记忆，并在 opts 里留下"已压到第几条"。

        放在**下一轮开头**而不是轮末（理由见 kernel.memory 的模块注释）：
        传输层拿到 Done 就收工了，跟在最后一个 yield 后面的代码不保证会跑。

        失败不抛：记忆是"有了更好"的能力，压不动就这一轮照常 —— 让对话挂掉去
        换一份记忆，这笔买卖不划算。
        """
        if self.summary_every <= 0:
            return
        done = self._compacted.get(session_id, 0)
        pending = [m for m in rows if m.role in ("user", "assistant", "tool")]
        if len(pending) - done < self.summary_every:
            return
        # 只压"还没压过"的那一段，避免每次重压全文 —— 那会让耗的 token 随会话长度平方增长
        fresh = pending[done:]
        try:
            result = await compact(self.llm, fresh, self._memory)
        except Exception as exc:  # noqa: BLE001 - 旁路能力，任何失败都到此为止
            # compact 内部已经兜过一轮，这里是双保险（比如 store 写入失败）
            print(f"[agentd] 记忆压缩未能完成：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return
        if result.summary:
            try:
                await self.store.save_summary(session_id, result.summary)
            except Exception as exc:  # noqa: BLE001
                print(f"[agentd] 摘要落库失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        # 压过就记账 —— 没压出内容也要记，否则每轮都会重试同一段
        self._compacted[session_id] = len(pending)

    @staticmethod
    def _context_pairs(
        rows: list[Message], cut: int
    ) -> list[tuple[int, Message]]:
        """窗口内的上下文消息，带上它们在 ``rows`` 里的**原始下标**。

        水位为什么记原始下标而不是"过滤后的序号"：``role="tool_record"`` 的行不进
        上下文，但会随着工具调用被 append 进来 —— 用过滤后的序号当水位，下一轮窗口
        会静默偏移几条，症状是"模型偶尔看不到上一轮的工具结果"。原始行只增不改，
        下标是稳的。
        """
        return [
            (index, m)
            for index, m in enumerate(rows[cut:], start=cut)
            if m.role != "tool_record"
        ]

    @staticmethod
    def _window(messages: list[Message], rows: list[Message], cut: int) -> list[Message]:
        """把**首轮用户消息**抢回到窗口里（如果它被压出去了的话）。

        与 ``trim_history`` 里的"抢救首轮 user"同源：那句通常写着任务本身是什么
        （"帮我把这个项目改成 X"），比摘要里被小模型重写过的一句转述可靠得多，
        而代价只是一条消息。
        """
        if cut <= 0 or not rows or rows[0].role != "user":
            return messages
        # cut>0 意味着第 0 行已经被挤出窗口，此时它绝不可能已经在 messages 里，
        # 补进去不会重复。
        return [rows[0], *messages]

    def _warn_about_overhead(self, session_id: str, pressure: Pressure) -> None:
        """固定开销压过触发线时，每个会话提醒一次。

        只提一次是因为它是**配置问题**，不是每一轮都要报的事；但完全不提又会让用户
        盯着"为什么历史突然被丢了几条"找一整天 —— stderr 里的一句话正好折中。
        """
        if self._overhead_warned.get(session_id):
            return
        self._overhead_warned[session_id] = True
        print(
            f"[agentd] 会话 {session_id}：system prompt 与 tools schema 已占 {pressure.used} token，"
            f"超过触发线 {pressure.trigger}（预算 {pressure.budget}）—— 再压缩历史也没用。"
            f"请调大 AGENTD_MAX_CONTEXT_TOKENS 或关掉一部分工具。",
            file=sys.stderr,
            flush=True,
        )

    async def _auto_compact(
        self,
        session_id: str,
        rows: list[Message],
        cut: int,
        pairs: list[tuple[int, Message]],
        *,
        plan: CompactPlan,
    ) -> tuple[int, str]:
        """把 ``plan`` 划出来的那一段压成摘要，返回 ``(新水位, 给用户的提示)``。

        **摘要没压出来就原地不动**：把还没留下备份的对话从上下文里抹掉，等于为了省
        地方删了文件才发现没备份 —— 宁可这一轮多花点 token（超了还有硬裁剪兜着），
        也不能静默丢上下文。

        压缩这里必须是"有账可查"的动作：成了要在 system prompt 里挂上自己那份摘要
        （见 ``_recall`` 的 ``own``），没成也要给用户一行提示 —— 悄悄跑和悄悄失败，
        都只会让人以为"它突然忘了"。
        """
        victim = [m for _index, m in pairs[: plan.cut]]
        failed = describe_compact(plan.pressure, dropped=0, summarized=False)
        try:
            result = await compact(self.llm, victim, self._memory)
        except Exception as exc:  # noqa: BLE001 - 旁路能力，任何失败都到此为止
            print(
                f"[agentd] 上下文压缩未能完成：{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return cut, failed
        if not result.summary:
            # 模型交白卷（小模型常见）：这次不动手，下一次触发时再试一次
            return cut, failed

        new_cut = pairs[plan.cut][0] if plan.cut < len(pairs) else len(rows)
        try:
            await self.store.save_compaction(session_id, result.summary, new_cut)
        except Exception as exc:  # noqa: BLE001 - 水位写不进去，那就别假装压过了
            print(
                f"[agentd] 压缩水位落库失败：{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return cut, failed

        # 这一段已经进过摘要了，别让"按条数触发"的压缩再为它付一次钱
        done = sum(1 for m in rows[:new_cut] if m.role in ("user", "assistant", "tool"))
        self._compacted[session_id] = max(self._compacted.get(session_id, 0), done)
        return new_cut, describe_compact(
            plan.pressure, dropped=len(victim), summarized=True
        )

    async def _compaction_of(self, session_id: str) -> tuple[str, int]:
        """读回本会话的压缩水位 ``(摘要, 已压掉的行数)``；读不动按"没压过"算。

        **没压过就把摘要也抹掉**：``save_summary`` 那条路（按消息条数触发的记忆整理）
        也往同一个字段里写内容，但它是"给将来的会话留个印象"，窗口一寸没动。
        这种情况下把摘要读回来塞进本会话的 system prompt，等于把上下文里原封不动
        存在的对话再复述一遍 —— 纯浪费预算，还占用了"更早的会话摘要"的位置。
        """
        try:
            summary, cut = await self.store.compaction(session_id)
        except Exception as exc:  # noqa: BLE001 - 摘要表不存在也要能聊天
            print(
                f"[agentd] 读压缩水位失败：{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return "", 0
        return (summary if cut > 0 else ""), cut

    async def _recall(self, session_id: str, own: str = "") -> str:
        """把两层记忆读出来拼成一段 —— 这就是"跨会话记忆"的全部可见部分。

        ``own`` 是本会话自己被压掉的那一段的摘要。平时一个会话不会读自己的摘要
        （那些内容本来就在上下文里），但窗口一旦往前移，移出去的部分就只剩这份
        摘要替它说话 —— 少了它，模型是真的"什么都不记得"。
        """
        if self.summary_every <= 0:
            return ""
        try:
            facts = await self._memory.read()
        except OSError as exc:
            print(f"[agentd] 读长期记忆失败：{exc}", file=sys.stderr, flush=True)
            facts = ""
        try:
            summaries = await self.store.recent_summaries(
                self.summary_recall, exclude=session_id
            )
        except Exception as exc:  # noqa: BLE001 - 摘要表不存在也要能聊天
            print(f"[agentd] 读历史摘要失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            summaries = []
        return render_memory_block(facts, summaries, own=own)

    def register(self, mode: Mode) -> None:
        if not mode.name:
            raise ValueError("Mode.name 不能为空")
        self._modes[mode.name] = mode

    def modes(self) -> list[str]:
        return sorted(self._modes)

    def _get_mode(self, name: str) -> Mode:
        try:
            return self._modes[name]
        except KeyError as exc:
            raise UnknownModeError(name) from exc

    async def create_session(
        self,
        *,
        cwd: str | None = None,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[Any] | None = None,
    ) -> str:
        session_id = new_session_id()
        await self.store.create(session_id)
        # 记住本会话的工作目录与 MCP server 配置，handle() 时交给模式使用
        self._session_opts[session_id] = {
            "cwd": cwd,
            "mcp_servers": list(mcp_servers or []),
            "additional_directories": list(additional_directories or []),
        }
        return session_id

    async def adopt_session(
        self,
        session_id: str,
        *,
        cwd: str | None = None,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[Any] | None = None,
    ) -> bool:
        """给**已存在**的会话补上会话级配置 —— ACP session/load 的恢复路径。

        客户端重启后拿着旧 sessionId 来 load：历史还在库里，但工具的工作目录、
        MCP 声明只存在于当初的 session/new 里，不重绑的话续聊的工具会在错误的
        cwd 下执行。setdefault 语义：本进程已经 attach 过（new 过）的会话不被
        覆盖；会话不存在返回 False，由传输层转成协议错误。
        """
        if not await self.store.exists(session_id):
            return False
        self._session_opts.setdefault(
            session_id,
            {
                "cwd": cwd,
                "mcp_servers": list(mcp_servers or []),
                "additional_directories": list(additional_directories or []),
            },
        )
        return True

    async def history(self, session_id: str) -> list[Message]:
        return await self.store.history(session_id)

    async def validate(self, session_id: str, mode: str) -> None:
        """提前校验参数（会话存在 + 模式已注册），失败抛 UnknownSessionError / UnknownModeError。

        理由：handle() 是异步生成器，代码要等到首次 __anext__ 才执行，届时响应头已发出，
        无法再改成错误状态，故传输层必须先调本方法。
        """
        if not await self.store.exists(session_id):
            raise UnknownSessionError(session_id)
        self._get_mode(mode)

    async def handle(
        self,
        session_id: str,
        user_input: str,
        *,
        mode: str = "single",
        approve: ApproveHandler | None = None,
        cancel: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        """执行一次对话，产出事件流。

        约定：末尾必发 Done；失败时先 ErrorEvent 再 Done(stop_reason="error")；
        同一 session 多次 handle 串行执行（history 有序，并行会写乱）。

        `approve` 是审批回调（工具要写文件/跑命令时用），由传输层注入；
        不传等于"无人可问"，一律放行。
        `cancel` 是可选的停止信号（asyncio.Event）：传输层收到 session/cancel 时
        把它 set 上（SDK 每帧一个 task，这个 handler 能与本循环并发执行），内核和
        模式在 chunk/工具边界查询、尽快收尾；本轮结束时 Done(stop_reason="cancelled")。
        None = 本轮不可取消。
        """

        if not await self.store.exists(session_id):
            raise UnknownSessionError(session_id)
        impl = self._get_mode(mode)

        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            run_id = new_run_id()

            # 用户消息先落库，这样传给模式的 history 里已经包含本轮输入
            await self.store.append(session_id, Message.user(user_input))
            rows = await self.store.history(session_id)

            # ---- 上下文窗口 ----
            # cut 是本会话"给模型的窗口"从第几行开始（此前的内容已被压成摘要）。
            # 被切掉的行一条不删 —— 它们仍在库里，UI 的历史回放照旧完整。
            #
            # 过滤与消毒的顺序有两件事：
            # 1. 工具记录（role="tool_record"）只服务客户端的历史回放（工具卡片），
            #    不是对话语境 —— 严禁混进喂给 LLM 的消息序列，否则 OpenAI 兼容端点
            #    会因为不认识的 role 直接 4xx；
            # 2. 被中途叫停的轮次会在库里留下"assistant 举手要调工具却没有工具响应"
            #    这种残缺片段，它是端点 4xx 的经典来源，整段丢掉。
            #    而 role="tool" 必须留下 —— 丢掉它，模型就不知道自己上轮读过什么。
            own_summary, cut = await self._compaction_of(session_id)
            pairs = self._context_pairs(rows, cut)
            history = self._window(sanitize_history([m for _i, m in pairs]), rows, cut)

            # 跨会话记忆：**先压再读**。顺序不能反 —— 压的过程中会把新抽出来的
            # 事实写进事实层，读完再压等于新事要等到下一轮才生效。
            # 两者内部都自己吞掉异常，这里不用再包一层 try。
            await self._maybe_compact(session_id, rows)
            memory = await self._recall(session_id, own=own_summary)

            opts = self._session_opts.get(session_id, {})
            toolbox = self._make_toolbox(
                opts.get("cwd"), opts.get("additional_directories") or []
            )
            # 工具 schema 也占窗口（放着二十几个 MCP 工具的 schema 不是小数），
            # 所以触发线要把它算进去。MCP 的工具要连上 server 才知道有哪些，
            # 而连接是每轮开一次的重量级动作 —— 这里只估原生工具，
            # 缺口由发请求前的硬裁剪（LiveLLM._trim）兜底，那道是必经之路。
            tools_schema = toolbox.tool_schema() if toolbox is not None else []
            system = self._system_prompt(opts, memory)

            notice = ""
            plan = plan_compaction(
                [m for _i, m in pairs],
                system=system,
                tools=tools_schema,
                budget=self.context_budget(),
                ratio_pct=self.compact_ratio,
            )
            if plan.futile:
                # 固定开销（system + tools）已经压过触发线。这种配置错得比较隐蔽
                # （症状是"压缩提示一条都没有，但历史在被硬裁"），所以哪怕每个会话
                # 只提示一次，也要在 stderr 里留句话。
                self._warn_about_overhead(session_id, plan.pressure)
            if plan.should_compact:
                new_cut, notice = await self._auto_compact(
                    session_id, rows, cut, pairs, plan=plan
                )
                if new_cut != cut:
                    # 窗口真的动了：上下文要按新水位重切一遍
                    cut = new_cut
                    pairs = self._context_pairs(rows, cut)
                    history = self._window(
                        sanitize_history([m for _i, m in pairs]), rows, cut
                    )
                    own_summary, _ = await self._compaction_of(session_id)
                    # 记忆要重读：刚才压出来的摘要此刻才对得上窗口移走的那一段
                    memory = await self._recall(session_id, own=own_summary)
                    system = self._system_prompt(opts, memory)

            ctx = ModeContext(
                session_id=session_id,
                run_id=run_id,
                llm=self.llm,
                history=history,
                system=system,
                mcp_servers=opts.get("mcp_servers", []),
                cwd=opts.get("cwd"),
                toolbox=toolbox,
                approve=approve,
                approval_policy=self.approval_policy,
                cancel=cancel,
            )

            if notice:
                yield Notice(session_id=session_id, run_id=run_id, text=notice)

            try:
                # Start/Done 在这里配对出 (title, kind)：Done 事件本身不带这些字段，
                # 内核本来就要顺着事件流走，查表比对契约加两个可选字段便宜
                tool_meta: dict[str, tuple[str, str]] = {}
                async for event in impl.run(ctx, user_input):
                    # 持久化由内核统一负责：模式只管产事件
                    if isinstance(event, Message):
                        # 模式夹带出来的"要进上下文的消息"： assistant 举手要调工具
                        # （带 tool_calls）与 role="tool" 的工具结果，成对出现。
                        # 落库但不外发 —— 传输层完全不需要知道这两件事存在，
                        # 它们下一轮由 store.history() 自己回到上下文里。
                        await self.store.append(session_id, event)
                        continue
                    if isinstance(event, MessageDone):
                        await self.store.append(session_id, Message.assistant(event.text))
                    elif isinstance(event, ToolCallStart):
                        tool_meta[event.call_id] = (event.title, event.kind)
                    elif isinstance(event, ToolCallDone):
                        title, kind = tool_meta.get(event.call_id, (event.call_id, "generic"))
                        # 每张工具卡片整体落一条 tool_record 行：续聊时客户端能
                        # 重放卡片，工具往返不再是"本轮内的临时状态"
                        await self.store.append(
                            session_id,
                            Message.from_tool_record(
                                ToolRecord(
                                    call_id=event.call_id,
                                    title=title,
                                    kind=kind,
                                    status=event.status,
                                    output=event.output,
                                )
                            ),
                        )
                    yield event
            except Exception as exc:  # noqa: BLE001 - 边界处统一转成事件
                yield ErrorEvent(
                    session_id=session_id, run_id=run_id, message=f"{type(exc).__name__}: {exc}"
                )
                yield Done(session_id=session_id, run_id=run_id, stop_reason="error")
                return

            # 被叫停的轮次必须把 cancelled 递到协议层（ACP 的 stop_reason 里有它），
            # 客户端才能把界面从"生成中"收成"已停止"而不是"正常说完了"。
            stop = "cancelled" if (cancel is not None and cancel.is_set()) else "end_turn"
            yield Done(session_id=session_id, run_id=run_id, stop_reason=stop)
