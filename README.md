# Agentd

ACP（[Agent Client Protocol](https://agentclientprotocol.com)）native 的 agent 内核。
对外就是一个 stdio 上的 JSON-RPC 服务，[ForgeAgent-GUI](../ForgeAgent-GUI) 之类的客户端
通过 stdio 把它拉起来对话。

```
ForgeAgent-GUI ──stdio/JSON-RPC──> agentd ──HTTP──> Ollama / OpenAI 兼容端点
```

## 跑起来

```bash
pip install -e .
python -m agentd.server      # 之后什么都不显示是正常的：它在等 JSON-RPC 帧
```

单独跑没意义（它是给人看不见的协议端点用的），配前端才有输出。

## 配置

优先级：**代码默认值 < 项目根的 `.env` < 真实环境变量**

`.env` 只补缺、不覆盖真实环境变量——否则 CI 里一份误提交的 `.env`
就能悄悄改掉整个后端行为。复制 `.env.example` 改名成 `.env` 即可。

| 变量 | 作用 | 默认 |
|---|---|---|
| `AGENTD_LLM_BACKEND` | `ollama` / `openai_compat` / `fake` | `ollama` |
| `AGENTD_OLLAMA_HOST` | Ollama 地址 | `http://localhost:11434` |
| `AGENTD_OLLAMA_MODEL` | 模型名，**`auto` = 去 `/api/tags` 问本机有什么** | `auto` |
| `AGENTD_OLLAMA_PREFER` | auto 时的偏好顺序（子串匹配） | `qwen3.5,qwen3,glm,…` |
| `AGENTD_OLLAMA_THINK` | Ollama 原生 think 参数 | `false` |
| `AGENTD_SYSTEM_PROMPT` | 系统提示词 | 空 |
| `AGENTD_STORE` | `sqlite` / `memory` | `sqlite` |
| `AGENTD_DB_PATH` | SQLite 会话库位置 | `~/.agentd/sessions.db` |

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

### 还有一件事没做

持久化只是**存下来**了，ACP 的 `new_session` 目前每次仍然造一个新会话，
所以"关掉 GUI 再打开，接着上次聊"还差一步：客户端要把 session_id 记下来并在下次
`new_session` 之后切回去。`list_sessions()` 已经备好了。

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

## 测试

```bash
pip install -e ".[dev]"
pytest -q
```
