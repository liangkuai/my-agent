"""
钩子系统 —— 在 agent 对话循环的关键节点插入自定义逻辑。

架构分为两层：
1. 注册层：HOOKS 字典 + register_hook() 维护事件 → 回调的映射
2. 触发层：trigger_hooks() 按注册顺序依次执行回调，首个非 None 返回值即短路

钩子按返回值语义分为两类：
- 拦截型（permission_hook）：返回非 None 字符串 → 阻止工具执行，字符串作为拒绝原因回填给模型
- 观察型（log / summary / context_inject / large_output）：始终返回 None，仅记录或打印，不影响主流程

四个事件的具体触发时机见 app.py 中的 agent_loop，本模块只定义钩子逻辑本身。
"""

from typing import Any, Callable

from constant import WORKDIR
from permission import check_permission


# ── 事件注册表 ──────────────────────────────────────────────────────
# 每个事件名对应一个回调列表，触发时按注册顺序依次执行。
# 执行顺序影响行为：观察型钩子应排在拦截型钩子之前，否则被拦截的调用不会留下日志。

HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register_hook(event: str, callback: Callable[..., str | None]) -> None:
    """将回调注册到指定事件，后续 trigger_hooks 会按注册顺序调用。"""
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """按顺序执行 event 下的所有回调。

    返回语义：
    - None                  → 全部放行，流程继续
    - 非 None 字符串         → 第一个拦截结果，调用方应以此中止后续操作

    短路规则：任一回调返回非 None 时立即停止遍历、不再执行后续回调。
    这保证了拦截型钩子（如 permission_hook）生效后，排在它后面的回调
    不会再对本次调用进行处理。

    注意：不同事件的 *args 参数不同（见各钩子签名），调用方需保证参数匹配。
    """
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # 返回值 ≠ None → hook 拦截，短路
            return result
    return None


# ══════════════════════════════════════════════════════════════════════
# 钩子回调定义
#
# 每个回调的签名由对应事件的触发方约定：
#   UserPromptSubmit  → (query: str)
#   PreToolUse        → (block)              block 为 SDK ContentBlock 对象
#   PostToolUse       → (block, output: str)
#   Stop              → (messages: list)     完整的对话历史
#
# 拦截型回调返回 str（拒绝原因），观察型回调始终返回 None。
# ══════════════════════════════════════════════════════════════════════


def context_inject_hook(query: str) -> str | None:
    """UserPromptSubmit —— 每次用户输入后打印当前工作目录。

    仅作信息提示，不拦截。
    """
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None


def permission_hook(block: Any) -> str | None:
    """PreToolUse —— 在工具执行前走权限管道。

    委托给 permission.check_permission()，后者分三层：
    拒绝列表 → 规则检查 + 用户确认 → 放行。
    返回非 None 时 agent_loop 会跳过工具执行并回填拒绝原因给模型。
    """
    return check_permission(block)


def log_hook(block: Any) -> str | None:
    """PreToolUse —— 记录每个工具调用请求的简要日志。

    展示工具名和前 2 个参数值（截断到 60 字符），便于追踪模型行为。
    始终返回 None，不干预流程。

    注意：该钩子排在 permission_hook 之前，确保被拦截的调用也能留下日志。
    """
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None


def large_output_hook(block: Any, output: str) -> str | None:
    """PostToolUse —— 工具执行后检测超大输出（>100k 字符）并告警。

    超大输出可能撑爆模型上下文窗口，黄色告警提醒用户关注。
    始终返回 None，不拦截（输出已产生，告警只是提示）。
    """
    if len(output) > 100000:
        print(
            f"\033[33m[HOOK] ⚠ Large output from {block.name}: {len(output)} chars\033[0m"
        )
    return None


def summary_hook(messages: list) -> None:
    """Stop —— 对话循环结束后统计本次会话中执行过的工具调用总数。

    遍历完整对话历史的 tool_result 块计数，不包含被拦截未执行的调用。
    注意：messages 跨多轮查询累积（见 app.py 中 history_messages 的生命周期），
    因此统计的是整个 session 的累计值，而非当前单次查询。

    统计逻辑：
    - 外层遍历每条消息 m
    - 中层展开 m["content"]（仅当它是列表时；assistant 消息的 content 是列表，
      user 文本消息的 content 是字符串）
    - 内层过滤 type == "tool_result" 的字典块
    """
    tool_count = sum(
        1
        for m in messages
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    )
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None


# ── 注册钩子 ────────────────────────────────────────────────────────
# 注册顺序即执行顺序：
#   PreToolUse: 先记日志 (log_hook)，再做权限检查 (permission_hook)
#               这样被权限拦截的调用也会留下日志，方便排查。
#   PostToolUse: 仅检测超大输出，不重做权限检查（权限已在 PreToolUse 处理）。

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", log_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)
