"""工具大结果外置化（BlobStore）—— 参考自 WorkBuddy（CodeBuddy CLI）的
ToolResultBlobService 设计（2026-10-08 从其 app.asar 实现中读到）：

    - 工具结果超过阈值（默认 50KB，CODEBUDDY_TOOL_RESULT_THRESHOLD_KB 可调）
      时，把**全文**写到会话目录下的文件里；
    - 模型与界面只拿到「预览 2048 字符 + 完整文件路径」，包在
      <persisted-output> 标签里；
    - 想看全文就用 read_file 打开那个路径。

为什么这么做（而不是把整块结果原样往前端塞）：
    1. **传输层安全**：真实事故（2026-10-08）里 agentd 把一篇几百 KB 的
       web_fetch 正文打成一行 JSON-RPC 帧，把 GUI 侧 asyncio 默认 64KB 的
       stdout 读入上限打爆，读帧任务死亡、GUI 误报「agent 进程已退出」。
       GUI 的 limit 已放大到 64MB 兜底（见 ForgeAgent-GUI acp_client.py），
       但那只是止血 —— 管道里的帧本来就**不该**这么大；
    2. **上下文友好**：模型大多数时候只需要开头一部分就知道下一步干什么，
       全文落盘后随时可以用 read_file 取回，上下文不必为一次性大输出买单；
    3. **界面友好**：工具卡片显示预览，用户要全文有路径可查。

本实现与 WorkBuddy 的差异：它的全文落在服务端会话目录、由 UI 托管预览；
 ours 是单机形态，直接落在 ~/.agentd/tool_results/<session>/ 下，路径对
模型和用户都可见。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# 阈值：超过这个字节数就外置。默认 50KB —— 与 WorkBuddy 的
# ToolResultBlobService 默认值对齐。AGENTD_TOOL_RESULT_THRESHOLD_KB 可调。
# ⚠️ 用**字节**而不是字符数：帧大小是字节决定的，而中文 1 字符 = 3 字节，
# 按字符数判断会把 64KB 级的中文页面漏放行 —— 恰恰是事故里那种帧。
_ENV_KB = os.getenv("AGENTD_TOOL_RESULT_THRESHOLD_KB")
THRESHOLD_BYTES = max(int(_ENV_KB or 50), 1) * 1024

# 预览长度：与 WorkBuddy 同为 2048 字符 —— 足够判断结果形状，又不至于把
# 上下文塞爆（模型要全文自己 read_file）。
PREVIEW_CHARS = 2048

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def externalize(output: str, session_id: str, call_id: str, *, home: Path | None = None) -> str:
    """超阈值时把全文落盘并返回 <persisted-output> 包装；小结果原样返回。

    包装格式刻意与 WorkBuddy 一致（<persisted-output> 标签）：模型见过这个
    形状，知道「要全文就按路径读文件」。
    """
    size_bytes = len(output.encode("utf-8"))
    if size_bytes <= THRESHOLD_BYTES:
        return output

    sid = _SAFE.sub("_", session_id)[:80] or "session"
    cid = _SAFE.sub("_", call_id)[:80] or "call"
    base = (home or (Path.home() / ".agentd")) / "tool_results" / sid
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{cid}.txt"
    path.write_text(output, encoding="utf-8")

    preview = output[:PREVIEW_CHARS]
    more = "\n..." if len(output) > PREVIEW_CHARS else ""
    kb = f"{size_bytes / 1024:.1f}KB"
    return "\n".join(
        [
            "<persisted-output>",
            f"输出过大（{kb}），完整内容已保存到：{path}",
            f"要全文就用 read_file 读上面的路径。",
            "",
            f"预览（前 {PREVIEW_CHARS} 字符）：",
            preview + more,
            "</persisted-output>",
        ]
    )
