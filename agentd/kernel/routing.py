"""模式自动路由：规划类请求从默认的 agent 提升到 plan。

为什么放在内核而不是传输层：三个入口（ACP/GUI、Gateway、CLI）都要回答
"这轮该用哪个模式"，路由规则只应有一份 —— 挂在内核 ``handle()`` 的入口，
三个入口自动一致；放传输层就得抄三遍，漂移是迟早的事。

三条边界，刻意保守：

1. **只提升，不覆盖**：用户显式选过 single / plan 时不插手（那是明确意图）。
   只有停在默认 ``agent`` 时，才按请求内容判断要不要走计划流程。
2. **判错的方向往"多一个审批框"倒**：误命中时用户点一次拒绝就回到老行为
   （"（计划未执行）"），代价是一次规划调用 + 一个弹窗；漏命中则是这个功能
   等于没有。所以执行向的词表收得比直觉略宽。
3. **咨询类话题一律不提升**（2026-10-13 加，用户实测反馈）：旅行/旅游/行程/
   攻略这类请求要的是一份**完整的答案文档**，不是"分步执行"的任务序列。
   实测"帮我规划一个7天的新疆旅行"被提升进 plan 后：先弹一次计划审批，再把
   一个回答拆成 7 轮 agent 循环逐步吐（93 秒、20+ 次工具调用），历史里还留下
   7 条"【计划 i/7】…"内部消息 —— 用户原话："这个问题不应该识别为一个 plan
   任务"。词表再准也分不清"给我一份行程"和"替我把这件事做了"，所以直接按
   话题豁免：命中咨询话题时，模型在 agent 模式里搜索+汇总，才是要的行为。

关掉：``AGENTD_PLAN_AUTOROUTE=0``（调用时读，改完下一句就生效，不用重启）。
"""

from __future__ import annotations

import os
import re

# 规划意图词表（执行向）：请求要"把一件事分步做出来"时才命中。
# 中文请求为主（GUI 用户群），英文常见说法一并收着。
# 刻意不收"行程 / 攻略 / itinerary"—— 它们是咨询名词（一篇攻略本身就是答案），
# 已挪进 _CONSULT_RX；也不收裸名词 roadmap（"看看 roadmap.md" 是读文件，
# 2026-10-13 实测这个词把 test_additional_roots 的 agent 用例绊翻过）。
_PLAN_INTENT_RX = re.compile(
    r"规划|计划|安排|策划|拆解|分解"
    r"|make\s+a\s+plan|plan\s+out",
    re.IGNORECASE,
)

# 咨询/问答类话题：这里的"规划"要的是一份**答案/建议文档**（行程、学习计划、
# 食谱…），不是上机执行的任务序列。命中即豁免自动路由 —— 与 _PLAN_INTENT_RX
# 的先后顺序刻意为之：先判豁免，再判意图（"帮我规划旅行"两个表都中，豁免赢）。
# 话题清单按"答案即交付物"的域维护：出行、学习、健康饮食、生活建议。
_CONSULT_RX = re.compile(
    # 出行旅游（用户实测的重灾区）
    r"旅行|旅游|游玩|自驾|露营|徒步|骑行|登山|景点|景区|海岛|行程|攻略|游记"
    r"|机票|酒店|民宿|签证|退税|遛娃|亲子游|蜜月|跟团"
    # 学习进修
    r"|学习计划|学习规划|备考|复习计划|课程表|书单"
    # 健康与饮食
    r"|健身计划|减肥计划|增肌|食谱|菜谱|菜单|穿搭"
    # 生活建议
    r"|礼物|起名|装修"
    # 英文常见说法
    r"|itinerary|travel\s+plan|trip\s+plan|road\s+trip|recipe",
    re.IGNORECASE,
)

DEFAULT_MODE = "agent"


def _plain_text(user_input: str | list[dict]) -> str:
    """多模态输入（内容块列表）里抠出纯文本；拿不出就返回空串（不路由）。"""
    if isinstance(user_input, str):
        return user_input
    parts = [
        str(b.get("text") or "")
        for b in user_input
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return " ".join(parts)


def autoroute_enabled() -> bool:
    """总开关。调用时读环境变量（不是 import 时）—— 改完下一句就生效。"""
    return os.environ.get("AGENTD_PLAN_AUTOROUTE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def looks_like_planning(user_input: str | list[dict]) -> bool:
    """这段请求是在要求"先规划后执行"式的任务吗（纯函数，单测直接打）。

    咨询类话题（_CONSULT_RX）优先返回 False：那些请求要的是完整答复，
    拆成计划逐步执行反而更糟（见模块 docstring 的第 3 条边界）。
    """
    text = _plain_text(user_input)
    if not text:
        return False
    if _CONSULT_RX.search(text):
        return False
    return bool(_PLAN_INTENT_RX.search(text))


def route_mode(user_input: str | list[dict], mode: str) -> str:
    """决定这轮实际用的模式：默认 agent + 规划意图 → plan，其余原样。"""
    if mode != DEFAULT_MODE:
        return mode  # 用户显式选过，不插手
    if not autoroute_enabled():
        return mode
    return "plan" if looks_like_planning(user_input) else mode
