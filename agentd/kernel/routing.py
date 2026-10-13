"""模式解析：把外部传进来的模式标识归一化成内核认识的模式名。

**这里刻意不做自动路由** —— 2026-10-13 去掉了「请求文本里出现『规划』就自动进
计划模式」的逻辑，理由见下。模式由用户在界面上显式选（Agent / Plan / Ask），
内核照做。

为什么去掉自动路由（用户实测，2026-10-13）：

关键词表分不清「给我一份行程」和「替我把这件事做了」。实测「帮我规划一个 7 天
的新疆旅行」被自动提升进 plan 后：先弹一次计划审批，再把一个本该一气呵成的回答
拆成 7 轮 agent 循环逐步吐（93 秒、20+ 次工具调用），历史里还留下 7 条
「【计划 i/7】…」内部消息 —— 用户原话：「这个问题不应该识别为一个 plan 任务」。

中间修过一版：加咨询类话题豁免（旅行 / 食谱 / 学习计划…）。但豁免名单是打地鼠，
名单外的误判照样发生；而「要不要先列计划再分步执行」这件事，用户点一下就知道，
模型猜十次也有错的时候。判断错了的代价也不对称 —— 自动路由猜对只是省一次点击，
猜错则整轮行为跑偏。所以连豁免一起去掉，交回显式选择。

为什么这个模块还留着：三个入口（ACP/GUI、Gateway、CLI）都要把外部传进来的模式
标识（None / 空串 / 别名 / 大小写混杂）收成一个内核认识的 id，规则只应有一份。
挂在内核 ``handle()`` 入口，三个入口自动一致；放传输层就得抄三遍，漂移是迟早的事。

未知取值**原样返回**，由内核的 ``_get_mode`` 抛 UnknownModeError —— 静默回退成
agent 会让「客户端选了个不存在的模式」这类错误消失得无声无息。
"""

from __future__ import annotations

DEFAULT_MODE = "agent"

# 对外（GUI 顶栏三档）暴露的模式，顺序即界面上的排列顺序：
# agent 全量执行 / plan 先列计划再分步执行 / ask 只读不动手。
#
# single（一次 LLM、不调工具）**不在这份清单里**：它是调试用的底层档位，
# 内核照旧支持（CLI `--mode single` 仍在），但用户没有"选它"的理由，
# 界面上不出现 —— 三档是给人的，底层档位是对机器的。
UI_MODES: tuple[str, ...] = ("agent", "plan", "ask")

# 别名 → 规范模式名。收这些是因为客户端 / 配置文件里真的出现过这些写法；
# 查表前先统一小写并去掉首尾空格。
_MODE_ALIASES: dict[str, str] = {
    "": DEFAULT_MODE,
    "auto": DEFAULT_MODE,
    "default": DEFAULT_MODE,
    "chat": DEFAULT_MODE,
    "ask": "ask",
    "readonly": "ask",
    "read_only": "ask",
    "planning": "plan",
    "once": "single",
}


def normalize_mode(mode: str | None) -> str:
    """把外部传来的模式标识收成内核认识的名字（纯函数，单测直接打）。

    认不出来的返回原样，让内核去抛 ``UnknownModeError`` —— 宁可报错，
    也不要悄悄换成 agent 跑一轮。
    """
    raw = (mode or "").strip().lower()
    if not raw:
        return DEFAULT_MODE
    return _MODE_ALIASES.get(raw, raw)
