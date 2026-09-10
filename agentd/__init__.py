"""agentd — Python agent 内核。

分层（依赖方向只能从上往下）：

    transports/  (HTTP+SSE, ACP stdio)   ← 只做 Event 的序列化
        ↓
    contracts.py (Event)                 ← 唯一的数据契约
        ↓
    kernel/      (会话 / 模式编排 / 记忆) ← 不认识任何传输层
"""

__version__ = "0.1.0"
