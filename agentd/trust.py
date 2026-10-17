"""「完全信任」开关：落盘标记 + 审批策略生效判定。

用户在审批弹窗点「完全信任（永不再问）」之后，三件事发生：

1. 传输层（transports/acp_stdio）立刻把内核的 approval_policy 翻成 "none"
   —— **当前进程**后续所有会话、所有轮次（含计划模式的审批门）都不再问；
2. 本模块把标记写进 ``~/.agentd/trust_all.json``（带时间戳，人可读、可手改）；
3. 下次启动时 boot._build_settings 读到标记，等效
   ``AGENTD_TOOLS_APPROVE=none`` —— 重启不恢复提问。

优先级（effective_approval_policy）：

    **显式配置**（环境变量 / .env / GUI 热配置 / RUNTIME_CONFIG.overrides）
        > 信任标记 > 默认 native

显式配置赢，是因为设了 AGENTD_TOOLS_APPROVE 的人是在**用配置管理这个行为**，
点击记忆不该盖过它；没显式设（GUI 正常启动就是这样）时才让点击记忆生效。
注意启动快照 ``_INITIAL_ENV`` 也算法显式 —— 从 GUI/CLI 传进来的环境变量
同样是用户意图。

撤销：

- 当前进程：重启 agentd（标记还在，重启后依旧信任——所以要配合同步删文件）；
- 彻底恢复提问：删掉 ``~/.agentd/trust_all.json``（disable_trust()），
  下次启动即恢复；或显式设 ``AGENTD_TOOLS_APPROVE=native`` 并重启盖过。

标记文件损坏/非 dict/缺字段一律按"未开启"处理 —— 这种东西宁可多问一句，
不能因为一个坏文件就把闸门全拆了。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def trust_file() -> Path:
    """标记文件路径。**函数而非模块常量**：测试要能换 home 目录。

    默认 ``~/.agentd/trust_all.json``，与 session_aliases / hidden_sessions
    等同级 —— agentd 的用户态状态都住在这个目录下。
    """
    return Path.home() / ".agentd" / "trust_all.json"


def enable_trust() -> Path:
    """写标记（幂等；重复点就是刷新时间戳）。返回文件路径。"""
    path = trust_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": True,
        "since": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "完全信任：所有工具调用不再弹审批。删除本文件恢复提问；"
                "显式设置 AGENTD_TOOLS_APPROVE 可临时盖过。",
    }
    # 原子写：先写临时文件再替换，避免写到一半断电留下坏 JSON
    # （坏文件按"未开启"处理，但用户会莫名其妙丢一次信任）。
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def disable_trust() -> None:
    """删标记。文件本来不在也算成功（幂等）。"""
    try:
        trust_file().unlink()
    except FileNotFoundError:
        pass


def trust_enabled() -> bool:
    """标记是否生效。任何异常（不存在/坏 JSON/缺 enabled）都算未开启。"""
    try:
        data = json.loads(trust_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and bool(data.get("enabled"))


def effective_approval_policy(policy: str, *, explicit: bool) -> str:
    """把「点出来的完全信任」折进启动时的审批策略。

    - explicit=True：调用方明确拿到了配置值（环境变量/.env/热配置/启动快照），
      原样返回 —— 显式配置永远赢；
    - explicit=False：值是默认 native，此时若标记在，升级成 "none"。

    只认 "none" 一种升级目标：POLICIES 里另外两档（native/all）都比默认更严，
    "完全信任"语义上就是 none，不存在"信任到 all"这种中间态。
    """
    if explicit:
        return policy
    if trust_enabled():
        return "none"
    return policy
