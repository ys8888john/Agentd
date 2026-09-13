"""模式编排：单次对话、提示链、路由、并行、反思、工具调用。

每种模式就是一个 AsyncIterator[Event] 的生成器，不关心谁在消费它。
"""

from .base import Mode, ModeContext
from .single import SingleMode

__all__ = ["Mode", "ModeContext", "SingleMode"]