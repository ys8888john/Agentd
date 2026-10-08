# Agentd

ACP（[Agent Client Protocol](https://agentclientprotocol.com)）native 的 agent 内核。
**三个等价入口**共用同一个内核（建会话 / 续聊 / 流式对话 / 切模式 / 叫停 / 工具审批 / 历史），
差别只在"怎么和外面说话"：

| 入口 | 命令 | 协议 | 谁用 |
|---|---|---|---|
| ACP | `python -m agentd` | stdio / JSON-RPC | Zed、[ForgeAgent-GUI](../ForgeAgent-GUI) 之类的 ACP 客户端 |
| Gateway | `python -m agentd gateway` | HTTP + SSE | 浏览器 / curl / 任何能读 SSE 的东西 |
| CLI | `python -m agentd cli` | 终端交互 | 人在终端里直接聊天 |

```
ForgeAgent-GUI ──stdio/JSON-RPC──┐
curl / 浏览器  ──HTTP+SSE───────┤──> agentd ──HTTP──> Ollama / OpenAI 兼容端点
人（终端）      ──命令行────────┘
```

"功能一致"是**结构性**保证的，不是三份代码各写一遍：三入口的会话语义
（模式覆盖 / 取消信号 / "本会话总是允许"的审批记忆）都收在
`transports/runtime.py` 的 `SessionRuntime`，各入口只负责"怎么问用户"。

## 跑起来

```bash
pip install -e .

# ACP（默认，等 JSON-RPC 帧，单跑没输出是正常的）
python -m agentd

# Gateway：HTTP + SSE，只绑 127.0.0.1，端口随机；SSE 在 /v1/sessions/<id>/stream
python -m agentd gateway --port 8765

# CLI：终端里直接聊天（:help 看命令，:quit 退出）
python -m agentd cli
```

也装了三个 console script：`agentd` / `agentd-gateway` / `agentd-cli`。

### Gateway 的一轮对话长什么样

```bash
# 1) 建会话
curl -s localhost:8765/v1/sessions -H 'content-type: application/json' -d '{"cwd":"."}'
# → {"ok":true,"session_id":"sess_…","modes":["agent","single"]}

# 2) 订阅 SSE（一个会话一条流，多轮复用），另开一个终端发消息
curl -N localhost:8765/v1/sessions/sess_…/stream
curl -s localhost:8765/v1/sessions/sess_…/prompt \
  -H 'content-type: application/json' -d '{"text":"你好"}'

# SSE 里会收到 message_delta / thought_delta / tool_call_start / tool_call_done / done，
# 与 ACP 的 session/update 语义一一对应。

# 工具要审批时，流里会推一条 permission_request：
curl -s localhost:8765/v1/sessions/sess_…/permission \
  -H 'content-type: application/json' \
  -d '{"request_id":"perm_…","option_id":"allow_once"}'   # allow_once|allow_session|reject

# 其它：POST …/cancel 叫停、POST …/mode {"mode_id":"single"} 切模式、GET …/history 看历史
```

## 配置

优先级：**代码默认值 < 项目根的 `.env` < 真实环境变量**

`.env` 只补缺、不覆盖真实环境变量——否则 CI 里一份误提交的 `.env`
就能悄悄改掉整个后端行为。复制 `.env.example` 改名成 `.env` 即可。

| 变量 | 作用 | 默认 |
|---|---|---|
| `AGENTD_LLM_BACKEND` | `ollama` / `openai_compat` / `mimo` / `zhipu` / `fake` / `script` | `ollama` |
| `AGENTD_OLLAMA_HOST` | Ollama 地址 | `http://localhost:11434` |
| `AGENTD_OLLAMA_MODEL` | 模型名，**`auto` = 去 `/api/tags` 问本机有什么** | `auto` |
| `AGENTD_OLLAMA_PREFER` | auto 时的偏好顺序（子串匹配） | `qwen3.5,qwen3,glm,…` |
| `AGENTD_OLLAMA_THINK` | Ollama 原生 think 参数 | `false` |
| `AGENTD_OPENAI_BASE_URL` / `_MODEL` / `_API_KEY` | OpenAI 兼容端点三件套 | `http://localhost:11434/v1` / `qwen3` / `ollama` |
| `AGENTD_MIMO_API_KEY` / `AGENTD_MIMO_MODEL` | 小米 MiMo（OpenAI 兼容）：key（兜底读真实环境变量 `MIMO_API_KEY`）/ 模型 | 空（必填）/ `mimo-v2.5-pro` |
| `AGENTD_MIMO_BASE_URL` | MiMo 端点，Token Plan 要换成 tp- 专属地址 | `https://api.xiaomimimo.com/v1` |
| `AGENTD_ZHIPU_API_KEY` / `AGENTD_ZHIPU_MODEL` | 智谱 BigModel（OpenAI 兼容）：key（兜底读真实环境变量 `ZHIPU_API_KEY`）/ 模型 | 空（必填）/ `glm-4.5-air` |
| `AGENTD_ZHIPU_BASE_URL` | 智谱端点，注意路径是 `/api/paas/v4`（没有 `/v1`） | `https://open.bigmodel.cn/api/paas/v4` |
| `AGENTD_FAKE_REPLY` | `fake` 后端的固定回复 | 一句话 |
| `AGENTD_SCRIPT_JSON` | `script` 后端的回放脚本（内联 JSON，优先） | 空 |
| `AGENTD_SCRIPT_FILE` | `script` 后端的回放脚本（文件路径） | 空 |
| `AGENTD_SYSTEM_PROMPT` | 系统提示词 | 空 |
| `AGENTD_MAX_CONTEXT_TOKENS` | 上下文预算（历史消息上限，system/tools 另算）；`0`=不限 | `0` |
| `AGENTD_MAX_OUTPUT_TOKENS` | 单次回答的最大 token；`0`=交给后端默认 | `0` |
| `AGENTD_COMPACT_RATIO` | 占用到预算的百分之几就自动压缩；`0`=只保留溢出时的裁剪 | `80` |
| `AGENTD_SUMMARY_EVERY` | 多少条消息压一次摘要+抽一次事实；`0`=关闭跨会话记忆 | `20` |
| `AGENTD_SUMMARY_RECALL` | 新会话带几条历史摘要 | `6` |
| `AGENTD_MEMORY_FILE` | 长期事实文件（可直接手改的 Markdown） | `~/.agentd/memory.md` |
| `AGENTD_STORE` | `sqlite` / `memory` | `sqlite` |
| `AGENTD_DB_PATH` | SQLite 会话库位置 | `~/.agentd/sessions.db` |
| `AGENTD_TOOLS` | 原生工具范围：`native` / `read_only` / `off` | `native` |
| `AGENTD_TOOLS_ALLOW_OUTSIDE` | 允许原生工具访问 cwd 之外的路径 | `false` |
| `AGENTD_TOOLS_TIMEOUT` | `run_command` 默认超时（秒） | `30` |
| `AGENTD_TOOLS_APPROVE` | 审批策略：`native` / `all` / `none` | `native` |
| `AGENTD_SEARCH_ENDPOINT` | `web_search` 的搜索后端地址 | `https://cn.bing.com/search` |
| `AGENTD_SOGOU_ENDPOINT` | `web_search` 的搜狗兜底后端地址 | `https://www.sogou.com/web` |

### `script` 后端：不靠模型也能验工具链

本机的小模型不一定真的会吐 `tool_calls`，「MCP 通不通」不该赌在这上面。
`script` 后端按 JSON 脚本回放输出，每次 LLM 调用取一条，用完重复最后一条：

```json
[
  {"tool_calls": [{"name": "echo__echo", "arguments": {"text": "hi"}}]},
  {"text": "工具说：echo: hi"}
]
```

```bash
AGENTD_LLM_BACKEND=script \
AGENTD_SCRIPT_JSON='[{"tool_calls":[{"name":"echo__echo","arguments":{"text":"hi"}}]},{"text":"好了"}]' \
python -m agentd.server
```

GUI 侧的 `scripts/mcp_e2e.py` 就是靠它把整条链路（mcp.json → ACP → agentd → 真
stdio MCP server → 工具事件 → 界面 reducer）跑通的，不需要 Ollama。

### `mimo` 后端：小米 MiMo 开放平台

MiMo 提供 OpenAI 兼容协议（`https://api.xiaomimimo.com/v1`），所以后端就是
`OpenAICompatLLM` 加了一组预设，没有新协议代码：

```bash
AGENTD_LLM_BACKEND=mimo \
AGENTD_MIMO_API_KEY=sk-xxxx \
python -m agentd.server
```

两个实现决定：

- **模型默认写死 `mimo-v2.5-pro` 而不是 `auto`。** MiMo 的 `/v1/models` 会把
  `mimo-v2.5-asr` / `mimo-v2.5-tts` 这些非对话模型一并列出来，auto 按子串匹配
  挑错一个就是一次 404/400；云端模型的差异也不像本机 Ollama 那样"装什么用什么"，
  默认值就写给官方示例模型，要换显式改 `AGENTD_MIMO_MODEL`（设成 `auto` 仍可走列表探测）。
- **余额不足必须在第一次调用时就看得懂。** 实测 key 有效但没余额时，chat 返回
  HTTP 402 + `Insufficient account balance`，这条会被转成 `LLMError` 原样带出来
  （`test_mimo_402_balance_error_is_surfaced`），不会表现为"回复空白"。

真 key 的连通性验证（默认跳过，防止 CI 无 key 红）：

```bash
AGENTD_LIVE_MIMO=1 AGENTD_MIMO_API_KEY=sk-xxxx pytest tests/test_mimo_live.py -v
```

### `zhipu` 后端：智谱 BigModel 开放平台

与 MiMo 同一个思路：OpenAI 兼容端点（`https://open.bigmodel.cn/api/paas/v4`）
加一组预设，没有新协议代码：

```bash
AGENTD_LLM_BACKEND=zhipu \
AGENTD_ZHIPU_API_KEY=xxxx.yyyy \
python -m agentd.server
```

两个实现决定：

- **默认模型 `glm-4.5-air`。** GLM 系列里 4.6v 是视觉、search-* 是联网搜索档，
  `glm-4.6` 是强档；通用对话默认给 4.5-air，要切换显式改 `AGENTD_ZHIPU_MODEL`。
- **Key 是两段的 `id.secret`。** 网页上"xxxx...yyyy"是脱敏显示，不能只拼 `id` 段用
  —— 只用 id 段实测返回 401「令牌已过期或验证不正确」。secret 只有创建时能看，
  掉了只能重新创建。

真 key 的连通性验证同样默认跳过：

```bash
AGENTD_LIVE_ZHIPU=1 AGENTD_ZHIPU_API_KEY=xxxx.yyyy pytest tests/test_zhipu_live.py -v
```

### provider 热加载：改后端不用重启

provider / key / model / base_url 这一类「可能随时改」的配置，走的是**热加载**——
内核持有一个 `LiveLLM` 包装，每轮对话开始前按「当前」配置重建底层后端，于是切换
provider 在**下一轮对话**即生效，不用重启进程。

两种触发方式，另加一条 GUI 通道（三档优先级）：

1. **GUI 热配置文件**（ForgeAgent-GUI 切模型专用，**不重启 agentd**）：GUI 切换
   provider / 模型时，把选中的 profile 环境变量写进一个 JSON 文件，路径由启动时的
   `AGENTD_HOTENV` 环境变量指定（GUI 默认指向它自己管理的 `~/.agentd/gui/hotenv.json`；
   非 GUI 场景缺省回落 `~/.agentd/hotenv.json`）。`current_settings()` 每轮都读它，于是
   **运行中的 agentd 下一轮即用，子进程完全不用重启**——会话历史 / 工具循环 / 已加载的
   MCP 全都不用重建。这是前端「会话中随时切模型」的主通道。

2. **直接改 `.env`**：`current_settings()` 每轮都重新解析 `.env` 文件，改完下一轮自动生效。
   优先级是 `启动时的真实环境变量 > 重新解析的 .env > 代码默认`，所以进程外
   `export` 的环境变量依旧最优先、且不会被 `..env` 覆盖（保留了「CI 里误提交的 `.env`
   改不掉行为」这条不变式）。

3. **程序化切换**（未来的 admin 命令 / GUI 按钮可用）：

   ```python
   from agentd.boot import RUNTIME_CONFIG
   RUNTIME_CONFIG.set(backend="mimo", mimo_api_key="sk-...")   # 下一轮切到 MiMo
   RUNTIME_CONFIG.clear()                                       # 清回 .env 的值
   ```

完整优先级（逐档回落）：
`RUNTIME_CONFIG.overrides  >  GUI 热配置文件  >  启动时的真实环境变量  >  重新解析的 .env  >  代码默认`。

缺 key / 未知后端时，重建会在该轮抛出 `ValueError`，由内核 `handle()` 统一转成
`ErrorEvent`（不会让整轮对话挂掉）。注意：**store 路径、工具策略、审批策略这类结构性
配置仍是启动定死**，热加载它们会带来数据/安全风险，不在本次范围内。

## 模式：single / agent

| 模式 | 行为 |
|---|---|
| `single` | 调一次 LLM 就结束。无工具的老行为。 |
| `agent` | **默认**。工具循环：LLM → 若要调工具就执行 → 结果回灌 → 再调 LLM → 直到出纯文本（上限 12 步）。没有可用工具时与 `single` 等价。 |

这组模式已通过 ACP 的 session modes 对客户端声明（`session/new` 和
`session/load` 的响应里带 `modes.currentModeId / availableModes`），
`session/set_mode` 收到的 id 就是这里的 `single` / `agent`。
切换成功后 agentd 会广播标准的 `current_mode_update` 通知 —— 所有在看这个
会话的客户端同步到新模式，不依赖各自的轮询。

### 停止（中断）正在跑的一轮

客户端可以随时发 ACP 的 `session/cancel` **通知**把这一轮停下来：

- 传输层给每个进行中的 prompt 挂一个 `asyncio.Event`（SDK 每帧一个 task，
  cancel 与 prompt 并发执行），内核和模式在**每个 chunk / 工具边界**上查看信号；
- 流式输出最多推迟一个 chunk 生效；工具循环则是"没派发的批次不再派发"——
  模型已经举手、但还没开始执行的调用直接跳过，不会出现"点了停止还在写文件"；
- 审批同理：信号置位后，尚未等到答复的审批**一律按拒绝**处理；
- 被停的一轮以 `stopReason="cancelled"` 收尾，**已流出的文本照常落库**
  （一个字都没流出来时落一句"（已手动停止）"），续聊时上下文是完整的；
- 中断粒度是 chunk/工具边界，不是掐死正在执行的子进程——`run_command`
  有 `AGENTD_TOOLS_TIMEOUT` 超时兜底，MCP 调用同层。

## 工具：原生工具 + MCP

agent 模式下的工具来自两条路，对模型完全透明（合并成一个 `tools` 数组，
靠名字路由回各自的执行器）：

### 1. 原生工具（进程内，默认开启）

| 工具 | kind | 要不要审批 | 干什么 |
|---|---|---|---|
| `read_file` | `read` | 否 | 读文本文件，返回带行号的内容，支持 `offset` / `limit` |
| `glob` | `search` | 否 | 按通配符找文件（支持 `**`），按修改时间倒序 |
| `grep` | `search` | 否 | 正则搜内容，返回 `文件:行号: 内容`，自动跳过 `.git` / `node_modules` |
| `write_file` | `edit` | **是** | 整体写文件（覆盖），父目录自动创建 |
| `edit` | `edit` | **是** | 精确字符串替换；`old_string` 不唯一时报错，除非 `replace_all=true` |
| `run_command` | `execute` | **是** | 在工作目录跑 shell 命令，返回退出码 + stdout/stderr |
| `web_search` | `search` | 否 | Bing 搜网页（答非所问时自动换搜狗重试），回"标题 + 直链 + 摘要"，`count` 上限 10 |
| `web_fetch` | `fetch` | 否 | 抓一个 http(s) 页面，剥成纯文本（最多 2MB） |

`write_file` / `edit` 的输出带 unified diff（行首 `--- / +++ / @@ / - / +`，
上下文 2 行、超长截断），客户端据此在工具卡片里直接渲染改动前后对比。

约束与限额：路径必须落在会话 cwd 内，或客户端在 `session/new` /
`session/load` 里声明的 `additionalDirectories`（额外工作区根）里
（`..` 会被 `resolve` 展开后再判，`AGENTD_TOOLS_ALLOW_OUTSIDE=true` 才放开）；
单次返回文本上限 64KB；
`run_command` 默认 30 秒超时、非零退出码按失败上报。

**为什么不全走 MCP。** MCP 的 stdio 客户端内部是 anyio task group，
必须与 enter/exit 同 task，所以现在每轮 run 都要现开现关一次子进程
（见 `mcp.py` 的生命周期说明）。读文件、搜代码这类动作一次对话要用几十次，
全走 MCP 等于每轮重启一次 node/python。所以「高频、无状态、纯本地」的动作
放进程内做；MCP 留给「本来就是独立服务」的能力（fetch / git / 数据库 / 厂商托管）。

#### 联网工具（`web_search` / `web_fetch`）

`web_search` 拿一条 query 回"标题 + 直链 + 摘要"列表；`web_fetch` 把一页 HTML 折成
纯文本（去脚本样式 → 块级标签转换行 → 剥标签 → 解实体）。抓取上限 2MB，非文本
content-type 直接报错不猜 —— 防一个大页面或一个 PDF 把内存和上下文一起灌爆。

**搜索后端：Bing 为主 + 搜狗兜底，都是实测筛出来的（2026-09-14 / 2026-10-08），不是偷懒：**

| 后端 | 实测结果 |
|---|---|
| DuckDuckGo（`lite.` / `html.`） | 走隧道全是 502 —— 自己验不了的解析器等于埋雷，宁可不写（2026-10-08 复测仍连不上） |
| 公共 SearXNG 实例 | 同上，被挡 |
| Baidu | 2026-09-14 返回"百度安全验证"反爬页；2026-10-08 复测通了，但结果 URL 全是 `baidu.com/link` 跳转，反爬时好时坏，不做后端 |
| **Bing（`cn.bing.com`）** | **主后端**。中英文都能稳定拿到 10 条，且结果是目标站直链 |
| **搜狗（`www.sogou.com/web`）** | **兜底后端**。Bing 答非所问/挂了时顶上；结果带 `/link` 中转，`_resolve_sogou_links` 会解成真实地址 |

要加后端形状很固定：拉 HTML → 解析成 `{title, url, snippet}`，在 `_parse_bing`
旁边照写一个即可。地址可以用 `AGENTD_SEARCH_ENDPOINT` / `AGENTD_SOGOU_ENDPOINT`
换（见配置表）。

**Bing 的已知毛病与对策：**

- **带出行/票务意图的 query 会被"降级"（2026-10-08 实测）**：「成都飞北京机票价格」
  返回的其实是搜「成都」的结果（连"约 26,800 个结果"都等于「成都」的结果数），
  「北京到上海高铁票价格」同理只搜了「北京」。cookie 预热、`mkt`、`format=rss`、
  新 UA、完整浏览器头、引号强查全试过无效 —— 是 Bing 对非交互客户端的处理，
  修不了。**对策**：`_looks_degraded` 用 query 的内容指纹（中文 2-gram + 英数词）
  在 top 结果里做命中抽查，几乎命中不到就换搜狗重试；兜底也失败时把 Bing 结果
  附上"可能答非所问"的警告返回，让模型自己决定换说法重搜。
  指到自定义后端（`AGENTD_SEARCH_ENDPOINT`）时不做这个抽查 —— 假后端的
  离线 e2e 结果和 query 本来就零词面重叠，抽查必然误判，显式指定的后端也不该被二次猜疑。
- **Bing 查不到东西时不给"无结果"标记**，而是塞一批不相关结果（乱码 query 实测返回
  一屏"抖音"）。页面上也没有可靠的空结果标志（`b_no` 在有结果的页面上同样出现，
  是别的用途）。同上由相关性抽查 + 搜狗兜底来兜。
- **一页只有 10 条，`count` 也封顶在 10。** 实测传 `count=20` 仍只回 10 个
  `b_algo` 块，所以不做翻页 —— 要更多结果就打几个不同的 query。

**为什么这两个工具不弹审批。** 它们确实会出网，但有两条：一是 `read_file` 早就把本机
文件内容交给模型了，模型本可以把内容塞进 `web_search` 的 query 带出去 —— 在联网这
一步拦，拦不住真正的泄漏路径；二是只读动作一弹窗，用户三分钟就学会无脑点"允许"
（见下面"审批"一节的取舍）。真要防外泄得在模型出口做，不是在这里。

### 2. MCP 工具（客户端声明，agent 连）

MCP server 由**客户端**在 `session/new` 里声明（ACP 的设计就是"客户端声明、agent 连"），
内核每轮用 `McpHub` 现开现关地连上它们，把工具列成 `{server}__{tool}` 暴露给模型。
`agentd` 自己不读任何 MCP 配置文件 —— 那是前端的事。

MCP 协议里没有 ACP 的 kind 概念，所以只能从 `annotations` 反推：
`readOnlyHint=true` 当只读（图标是"读"，不弹审批），其余一律按 `execute` 呈现 ——
对没声明的东西保守一点。审批同理，只看工具自己声明的 `destructiveHint`，
不替它猜。

### 审批

会改变外部状态的动作在**真正执行前**停下来问一次。ACP 侧走
`session/request_permission`，弹三个选项：允许一次 / 本会话总是允许 / 拒绝。

- **只读动作（`read` / `search` / `fetch`）在任何策略下都不弹。** 每个 `ls` 都弹窗，
  用户三分钟就学会无脑点"允许"，审批本身也就废了。联网工具归在只读这一侧，理由见上面。
- 选过"本会话总是允许"的工具名会被记住（记忆在传输层，换客户端可以换策略）。
- 审批通道出错 / 客户端没实现该能力 / 用户关掉弹窗，**一律按拒绝处理**。
  审批这种拿不准的事必须往"拒绝"倒，反过来就是安全漏洞。
- 被拒后工具输出 `[错误] 用户拒绝执行 xxx`、`status=cancelled`，模型能看到这条
  反馈并绕路，整轮对话继续。

## 会话持久化

默认写 SQLite（`~/.agentd/sessions.db`），**进程退出后历史还在**。
想要"重启即丢"得显式设 `AGENTD_STORE=memory` —— 不持久化是例外，不是默认。

```bash
python scripts/sessions.py list            # 有哪些会话
python scripts/sessions.py show sess_xxx   # 看某个会话的完整历史
python scripts/sessions.py clear sess_xxx  # 清空某个会话（会话本身保留）
```

库结构只有两张表：

```
sessions(id, created_at)
messages(seq, session_id, role, content, name, payload, created_at)
```

`payload` 是整条 `Message` 的 JSON，**它是权威数据**；`role/content/name` 三列是冗余的，
存在的唯一理由是让人能用 `sqlite3` 直接看库、用 SQL 统计，而不是每次都写脚本解 JSON。
将来给 `Message` 加字段也不用迁移表。

除了 user / assistant 两种行，还有：

- `role="tool_record"`：每张完成的工具卡片一条
  （call_id/title/kind/status/output 全在 `payload.tool_record` 里，被拒绝的调用以
  `cancelled` 落库）。它是客户端历史回放用的 UI 记录 —— 内核加载 history 时会把
  这类行过滤掉，**永不进 LLM 上下文**；
- `role="tool"` 与「带 `tool_calls` 的 assistant」：模型与工具的一次完整往返。
  它们**也落库** —— 不落的话，下一轮模型就不知道自己上一步读过什么文件，
  "继续改"之类的追问必然答非所问；内核加载 history 时会先做一次
  `sanitize_history()`，把被中途叫停而残缺的往返（举手要调工具却没有结果，
  或反过来）整段丢掉 —— 那是 OpenAI 兼容端点 4xx 的经典来源。

第三张表是跨会话记忆的**摘要层**：

```
summaries(session_id, summary, updated_at)   -- 一个会话一段进展，反复压缩是覆盖
compactions(session_id, cut, updated_at)     -- 上下文窗口起点（第几条原始行之后才进上下文）
```

细节始终留在 `messages` 里，摘要只用来给上下文省空间；UI 回放仍走完整历史。
`cut` 只许往前 —— 往回退等于把刚压过的对话重新灌回上下文。

## 上下文：token 预算、裁剪与跨会话记忆

三件事共用一条链路（`store.history()` → `sanitize_history()` → 模式 → `LiveLLM`），
解决的是同一个症状的不同侧面：**"聊着聊着模型变笨 / 换会话像第一次见面"**。

- **token 估算**：先得知道占了多少；
- **自动压缩**（占 80% 就动手）：把较早的对话压成摘要，**留一份信息**再缩小窗口；
- **裁剪**（真溢出了才发生）：从最早的开始丢，此时也保证了
  「首轮提问 + 本轮提问」还在。

后两者的区别值得强调：压缩是"信息换位置"（原始行一条不删，UI 回放照旧完整），
裁剪是"信息换预算"（丢掉的就真的没了）。所以压缩必须排在前面。

### 自动压缩：到预算的 80% 就动手

`AGENTD_COMPACT_RATIO`（默认 80；`0` = 关掉，只剩溢出时的裁剪）。为什么是提前
这一截：

- 压一次要**额外调一次模型**，预算已经见底时做这件事等于把最后一点空间也花掉；
- 剩下那 20% 是留给"本轮提问 + 工具往返 + 模型回答"的周转空间。

一次压缩做三件事：把较早的一段交给 `memory.compact()` 压成摘要 + 事实、把窗口
起点往前推（`compactions` 表里的水位）、把这份摘要挂进 system prompt 替被移走的
部分说话。三条安全约束和一个反向安全阀：

- 切点必须落在工具往返的块边界上（水位会被持久化，切坏了要到下一轮才炸）；
- 本轮提问所在的那一块永远不压；
- 首轮 user 会从窗口外被救回来 —— 那里通常写着任务本身是什么；
- **摘要没压出来就原地不动**：宁可这一轮多占一点预算，也不能把还没留下备份的
  对话从上下文里抹掉。

触发与否只看一件事：实际占用占预算的百分比；而 `used` 的计算要把 **system prompt
和 tools schema** 一起算进去 —— 放着二十几个工具的 schema 不是小数，只数历史会在
"预算看着还剩一半"时早就贴上限。MCP 的工具要连上 server 才知道有哪些，而连接是每轮
开一次的重量级动作，这里只估原生工具；缺口由发请求前的裁剪兜底，那道必经。

压缩后会发一条 `[提示]` 说明占用百分比和压了多少条 —— 悄悄跑和悄悄失败都只会让人
以为"它突然忘了"。

一个反向安全阀：`system prompt + tools schema` 是每轮都要付的固定开销，压不掉。
如果它们单独就已经在触发线之上，再压缩历史也降不下来 —— 那样只会每轮白跑一次压缩
（多一次 LLM 调用）而窗口一点没变小。内核识别出来就停手，并在 stderr 里留一句
"请调大 `AGENTD_MAX_CONTEXT_TOKENS` 或关掉一部分工具"（每个会话一次）。

### token 估算：刻意不引 tiktoken

`kernel/context.py` 用字符比例估：CJK 按 1 字≈1 token、ASCII 按 4 字符≈1 token，
乘 1.1 安全余量后向上取整。理由：

- 各家词表差异在 20%~40%，谁也算不准本地小模型；
- 为一个只能"估得更准一点"的数引一个依赖不值当；
- 估不确定就**宁可高估**（×1.1）—— 估高了至多少放回几条历史，估低了会让
  请求真的超出窗口，症状是后端截断 + 模型答非所问。

### 裁剪：保尾弃头，永不切断工具往返

超预算时从最早的开始丢，但两条硬约束：

- 最后一块（本轮提问）**永远保留**；另外会额外抢救首轮 user —— 那里通常写着
  任务本身是什么；
- 「assistant(tool_calls) + 它引发的 tool 结果」是一个原子块。切开会让
  `role="tool"` 变成孤儿消息，端点直接 4xx。

实际上没有裁剪就**一声不出**（不会每轮刷一行系统提示让用户以为出了事）；
裁了才发一条 `[提示]` 说明丢了几条。

### 为什么不自动写 `num_ctx`

`max_output_tokens` 会翻译成 Ollama 的 `options.num_predict` / OpenAI 的
`max_tokens`，但 `num_ctx`（输入窗口）刻意不碰：只有使用者知道自己拉的模型是
8k 还是 40k，替他猜一个必然错 —— 而且设小了 Ollama 会**真的按这个值截断输入**，
症状跟"机器人变笨"一模一样，极难自查。要控输入量，就用 `max_context_tokens`
明确地裁：裁了几条、为什么裁，都能对用户讲清楚。

### 跨会话记忆：摘要层 + 事实层

```
        ┌──────────── 事实层 ~/.agentd/memory.md ────────────┐
        │  - 偏好简体中文                                     │
        │  - 本机 Ollama 模型是 qwen3.5:9b                    │   稳定、手可改
        └────────────────────────────────────────────────────┘
                     ↑ compact() 抽出来的"跨会话仍成立"的事实
                     │
一段久到够线  ────────┤
的对话 rows           │
                     ↓ compact() 压出来的"这段聊到哪了"
        ┌──────────── 摘要层 summaries 表 ───────────────────┐
        │  sess_a → 上次在给 memory.py 写测试                 │   按会话归档的进展
        │  sess_b → token 预算先用 8192 试                    │
        └────────────────────────────────────────────────────┘

        新会话的 system prompt = 事实（永远相关）+ 最近 N 条摘要（按需相关）
```

两层为什么要都有：只有摘要，十次之后摘要自己也有几万字，而且"用户是个什么样的
开发者"每次被重总结一遍、措辞次次不同；只有事实，对话过程全丢，"上次做到哪儿了"
照样答不上来。

三个不太显然的决定：

- **压缩发生在"下一轮开头"，不是轮末。** 传输层拿到 `Done` 就收工了，跟在最后一个
  `yield` 后面的代码不保证会跑到 —— 放在轮末等于"记忆看心情"，也不可单测。
- **只压上次没压过的那一段。** 全文重压会让压缩开销随会话长度平方增长，
  那正是"为了省上下文而反被上下文吃掉"。
- **压失败了静默但留日志。** 让一轮对话挂掉去换一份记忆，这笔买卖不划算；
  但也不能完全无声 —— "它为什么像第一次见我"必须有线素可查。

事实层刻意做成**可以直接打开改的 Markdown**："它记错了我的编辑器"这种事，
让用户去写 SQL 是不人道的。手动删改立即生效，写错也不会让对话失败。

### 三个实现决定

- **不用 aiosqlite。** 所有调用都经 `asyncio.to_thread` 丢进线程池，事件循环本来就不会
  被阻塞，为此多引一个依赖不划算。
- **单连接 + `threading.Lock`，而不是每线程一个连接。** `sqlite3` 连接默认
  `check_same_thread=True`，而 `to_thread` 每次可能落到不同线程。开
  `check_same_thread=False` 再自己串行化，写路径上根本不会出现 `SQLITE_BUSY`。
- **开 WAL + `busy_timeout=5s`。** GUI 是"一个窗口一个 agentd 进程"，多个进程会同时
  写同一个库。WAL 让读写不互相阻塞，`busy_timeout` 让并发写在 5 秒内排队等锁而不是
  立刻抛 `database is locked`。
- **`seq` 必须是 `INTEGER PRIMARY KEY AUTOINCREMENT`。** 普通的 `INTEGER PRIMARY KEY`
  会复用被删掉的 rowid，`clear()` 之后再 append，新消息的 seq 可能比老的小，
  `ORDER BY seq` 就把历史排乱了。

### 会话恢复（session/load）

客户端重启 / 断线重连后，可以带着 `sessionId` 调 ACP 的 `session/load`：
历史、工具卡片记录都在库里，agentd 只把会话级配置（请求参数里的 `cwd`、
`mcpServers`）重新绑上 —— 否则恢复出来的会话，工具会落在错误的目录。
会话不存在时返回协议错误，不会悄悄新建。`session/new` / `session/load`
的响应同时声明该会话的模式状态（见"模式"一节）。

### 为什么模型默认是 auto

默认值一度写死成 `qwen3`，但这台机器上装的是 `qwen3.5:9b-text`，Ollama 返回 404。
**404 的表现是回复一片空白、`stop_reason` 还是 `end_turn`** —— 界面干干净净，
一个错误字都没有，用户完全无从下手。

改成 auto 后：首次调用去 `/api/tags` 拉本机模型列表，按 `AGENTD_OLLAMA_PREFER`
挑一个，结果缓存住只查一次，选中的模型名打进 stderr 日志（前端 `Ctrl+L` 能看到）。

### WSL2 里跑 Ollama、Windows 上跑 agentd

`.wslconfig` 要开镜像网络，否则 Windows 侧连不上 WSL2 的端口：

```ini
[wsl2]
networkingMode=mirrored
firewall=false
```

改完 `wsl --shutdown` 生效。之后 Windows 侧 `http://localhost:11434` 直接可达。

## 已知坑

- **stdout 只能有 JSON-RPC 帧。** 所有日志必须走 stderr（`acp_stdio.py:_log`）。
  往 stdout 打一个字，客户端就解不出帧了。`tests/test_acp.py` 断言 stdout 每行可 JSON 解析。
- **`on_connect` 必须是同步 `def`。** SDK 是 `on_connect(self)` 直接调用、不 await
  （`acp/agent/connection.py:101-102`）。写成 `async def` 只会生成一个从未执行的协程，
  `self._conn` 永远是 `None`，直到 `prompt()` 里才炸 `'NoneType' has no attribute
  'session_update'` —— 报错点离病根十万八千里。
- **会话库打不开要报错，不能静默退回内存。** 静默降级会让"我明明聊过，重启后没了"
  变成一个查不出来的问题。`build_store()` 打不开库会抛 `RuntimeError`，并把
  `AGENTD_STORE=memory` 这条退路写进错误信息里让人自己选。
- **`scripts/chat.py` 会写真实会话库。** 它只把 `AGENTD_LLM_BACKEND` 设成 fake，
  存储仍是默认的 SQLite；想保持干净就给它设 `AGENTD_DB_PATH` 或 `AGENTD_STORE=memory`。
- **模式的最后一个事件必须是 `MessageDone` 而不是 `MessageDelta`。**
  流式已经把每个 chunk 发出去了，末尾再用 `MessageDelta` 发整份，客户端会收到两遍文本；
  而内核靠 `MessageDone` 把 assistant 回复落库，漏了它历史永远写不进去。
- **ACP 的 `mcpServers` 缺字段会被"静默清空"，不报错。** SDK 的 `McpServerStdio`
  把 `args` / `env` 声明成必填、`HttpMcpServer` 还要 `type`。少任何一个字段时
  pydantic 的 union 校验**不是抛错，而是把整份 mcpServers 折成 `[]`** ——
  现象是"前端配了 MCP，agentd 却说没接到 server"，而且两边日志都干干净净。
  写 mcp.json 的解析代码时宁可补空值（`args: []` / `env: []`）也不要省略。
- **ACP SDK 传来的 `mcpServers` 是 pydantic 模型，不是 dict。** 路由层已经用
  `NewSessionRequest.model_validate()` 校验过，`model_to_kwargs()` 取出来的是
  `McpServerStdio` 实例。`McpHub` 早先只认 dict，`isinstance(raw, dict)` 为假就
  把整条静默跳过 —— 现象是日志说"接入 1 个 server"但一个工具都列不出来。
  现在 `_as_config()` 兼容两种形态（`test_mcp_hub_accepts_acp_pydantic_model_config`）。
- **工具事件的 `kind` 必须是 ACP 的合法取值。** ACP 只认
  `read|edit|delete|move|search|execute|think|fetch|switch_mode|other`，
  内核多一个 `generic`（传输层译成 `other`）。取值对不上时的表现不是报错，而是
  **客户端静默卡在"运行中"**，两边日志都干干净净。所以 `contracts.ToolCallKind`
  写成 `Literal`（拼错当场 ValidationError），并且
  `test_acp_kind_covers_every_kernel_toolkind` 直接从 Literal 里取全部取值去比对 ——
  以后往 contracts 加新 kind，那个测试会立刻红。
- **审批问不到人时必须按拒绝处理。** `ctx.approve is None`（TUI / 单测）才是放行；
  一旦有回调但回调抛异常（客户端没实现 `session/request_permission`、
  请求超时、用户关掉弹窗），`AgentdAcpAgent._make_approver` 和
  `ModeContext.request_approval` 都会返回 False。这个方向的取舍不能反 ——
  反了就是"审批通道一坏，所有写操作自动放行"。
- **原生工具的路径必须 `resolve()` 之后再判越界。** 直接字符串比较
  `startswith(cwd)` 会放过 `../`，`Path.resolve()` 把 `..` 和符号链接都展开过，
  再 `relative_to(root)` 才是可靠的判据（`test_path_escape_outside_root_is_rejected`）。
- **搜索必须伪装成浏览器 UA。** 用 `httpx` 的默认 UA 会被 Bing 直接挡掉，而且
  **不报错** —— 它返回的是一个正常 HTML 页面，只是里面一个 `b_algo` 块都没有，
  现象是"搜索永远回'没能从结果页解析出条目'，看着像解析器坏了"。UA 在 `_WEB_HEADERS`。
- **`AGENTD_SEARCH_ENDPOINT` 是调用时读的，不是 import 时。** 所以指到别处立刻生效、
  不用重启进程（`_search_endpoint()` 每次读一次；写成模块级常量初值就做不到，
  单测 monkeypatch 也会失效）。跨仓库联网 e2e 就是靠它把后端指向本地假 Bing 的。
- **ACP 的 `ToolCallStatus` 里没有 `cancelled`。** 只有
  `pending|in_progress|completed|failed`。所以内核里"用户拒绝了这个工具"这个
  状态到了协议层被迫折成 `failed`（见 `_ACP_STATUS`）—— 于是"用户主动拒绝"和
  "工具真的炸了"在协议上长得一模一样。唯一的载体是输出的那行文案：
  `[错误] 用户拒绝执行 {tool}`。客户端靠这个字面量把它还原成 `cancelled`
  （ForgeAgent-GUI 的 `acp_client.DENY_MARK`）。两个仓库各自定义同一个字面量，
  改一处就要改另一处；改了不生效的表现是界面上"你点的拒绝"显示成红色"失败"。
- **审批回调抛异常 = 拒绝，不是放行。** `ModeContext.request_approval` 和
  `acp_stdio._make_approver` 都往"拒绝"倒。`ctx.approve is None`（TUI / 单测 /
  无人值守）才是放行。这个方向反了就是"审批通道一坏，所有写操作自动通过"。

## 测试

```bash
pip install -e ".[dev]"
pytest -q
```

取消（中断）的测试在 `tests/test_cancel.py`：single / agent 模式在流式中途、
工具批次之间收到停止信号的行为（工具不执行、正文留"已流出的部分"、
`cancelled` 收尾、历史照常落库），以及传输层 `cancel()` 与 `prompt()` 并发的
往返（带节奏钩子的假 LLM 确定性触发，不起子进程）。

`tests/test_session_load_and_diff.py` 锁 session/load 与 diff 输出：new/load
声明的模式状态、load 后相对路径工具命中恢复的 cwd、不存在会话报错，
以及 write_file/edit 输出里 diff 块的行首标记（前端着色的解析契约）。

MCP 工具循环的测试分三层：`AgentMode` 用假 Hub 测循环逻辑、`McpHub` 连真 stdio
echo server 测集成、`kernel.handle(mode="agent")` 测端到端。整条链路的真实验证
（含前端）在 ForgeAgent-GUI 的 `scripts/mcp_e2e.py`。

原生工具的测试在 `tests/test_native_tools.py`，分七块：

1. 六个**本地**工具各自的行为（真文件系统，全在 `tmp_path` 里）；
2. 路径边界（`../` 越界、`allow_outside`）与 profile（`read_only` 挡不挡得住写）；
3. `needs_approval` 判定矩阵 + `AgentMode` 的 kind 映射 / 审批通过 / 审批拒绝 /
   审批通道抛异常；
4. ACP 映射：`_ACP_KIND` 覆盖所有 kernel kind、`session/request_permission` 的
   三种应答（允许一次 / 本会话总是允许 / 取消）各自的翻译结果；
5. **联网工具**（第七块）：Bing 页面解析（含 `limit` 截断、空页 / 垃圾页）、
   跳转包装 URL 的解开（解不出、或解出来不像 URL 时退回原值）、HTML → 纯文本
   （脚本样式剥除、实体还原、连续空行压缩）、charset 判定、参数校验，
   以及"连不上必须回 `[错误]` 而不是抛异常"。

联网工具另有一套**真出网**的测试，默认跳过、要显式开：

```bash
AGENTD_LIVE_WEB=1 pytest tests/test_web_tools_live.py -v
```

它验的是离线套件验不了的那半件事：**Bing 现在的页面结构还认得出来吗**。
分开的原因很实际 —— 本机网络时通时不通（隧道会拦），把出网断言塞进默认套件
等于给 CI 埋一个随机红。断言刻意宽松到只认"有结果、链接是 http(s)"，
不去钉具体条数和具体站点，那样 Bing 一改版就得跟着修测试。

⚠️ 跑联网真测时**别摘 `HTTP_PROXY`** —— 本机出网正好是靠代理隧道走的
（下面跨仓库 e2e 反过来要摘，因为它只想连本机）。

再往上一层是 `tests/test_acp.py::test_native_tool_over_real_stdio`：起真子进程，
用 `script` 后端回放"模型要 glob" → 断言工具真执行了、`kind` 是 `search`、
stdout 上依然只有 JSON-RPC 帧。不依赖 Ollama。

**跨仓库的真实验证**在 ForgeAgent-GUI 的 `scripts/native_tools_e2e.py`，
两个场景都用真 `AcpClient` 拉起真 agentd、用 `script` 后端回放：

```bash
python scripts/native_tools_e2e.py                    # files：允许写入
python scripts/native_tools_e2e.py --deny             # files：拒绝写入
python scripts/native_tools_e2e.py --scenario web     # 联网工具（离线可跑）
```

- `files`：回放"读 + 搜（不该弹审批）→ 写（该弹审批）"，断言只弹了一次审批、
  `kind` 分别是 `read`/`search`/`edit`、允许时文件真落盘 / 拒绝时文件绝不存在。
- `web`：回放 `web_search`（`count=2`）→ `web_fetch`，断言两个都**没弹审批**、
  `kind` 是 `search`/`fetch`、`count` 真截断了、HTML 真被剥成了纯文本。

  **它不出网**：脚本自己起一个假的搜索/网页后端（真 socket、真 HTML），再用
  `AGENTD_SEARCH_ENDPOINT` 把 agentd 的搜索后端指过去，所以离线可复现。
  agentd 子进程会继承 `HTTP_PROXY`（本机把出网全导给本地代理），脚本顺带给它设了
  `NO_PROXY=127.0.0.1,localhost` —— 少了这一句，连本机假后端也会被代理成 502。

这一步是唯一能证明**两个仓库对审批帧的理解真的一致**的东西 ——
两边各自打自己造的帧，字段名对不上也测不出来。
