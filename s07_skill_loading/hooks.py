"""
钩子系统 —— 在 agent 对话循环的关键节点插入自定义逻辑。

架构两层：
1. 注册层：HOOKS 字典 + register_hook()，事件 → 回调列表
2. 触发层：trigger_hooks()，按注册顺序执行，首个非 None 即短路

两类钩子：
- 拦截型：返回非 None 字符串 → 阻止执行，字符串作为拒绝原因回填
- 观察型：始终返回 None，仅记录或打印

四个事件（触发位置见 app.py agent_loop）：
- UserPromptSubmit  — 用户每次输入后
- PreToolUse        — 每个 tool_use 块执行前
- PostToolUse       — 每个 tool_use 块执行后
- Stop              — 模型停止请求工具、循环即将退出时

扩展：在下方定义回调，在模块末尾 register_hook 即可。
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
    """将回调注册到指定事件，后续 trigger_hooks 会按注册顺序调用。

    event 必须是 HOOKS 字典中已存在的事件名，否则会 KeyError——
    这是有意为之：不支持的事件名应立即暴露，避免回调静默不执行。
    """
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """按顺序执行 event 下的所有回调。

    返回语义：
    - None                  → 全部放行，流程继续
    - 非 None 字符串         → 第一个拦截结果，调用方应以此中止后续操作

    不同事件的 *args 参数不同，由各事件的触发位置决定：
    - UserPromptSubmit  → args = (query: str,)
    - PreToolUse        → args = (block,)
    - PostToolUse       → args = (block, output: str)
    - Stop              → args = (messages: list,)
    """
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # 返回值 ≠ None → hook 拦截，短路
            return result
    return None


# ══════════════════════════════════════════════════════════════════════
# 回调定义（签名由各事件的触发位置决定）
#
# 规则：拦截型返回 str（拒绝原因），观察型始终返回 None。
# 所有回调通过 side effect 发挥作用，返回值仅表示"是否拦截"。
# ══════════════════════════════════════════════════════════════════════


def context_inject_hook(query: str) -> str | None:
    """UserPromptSubmit —— 每次用户输入后打印当前工作目录。

    仅作信息提示，不拦截。可用于扩展：注入额外的上下文到对话历史中
    （例如在用户消息前追加当前 git 分支、最近文件变更等环境信息）。
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

    超大输出可能撑爆模型上下文窗口（单条 tool_result 过大导致后续几轮
    的 token 预算被占满），黄色告警提醒用户关注。
    始终返回 None，不拦截（输出已产生，告警只是提示）。
    """
    if len(output) > 100000:
        print(
            f"\033[33m[HOOK] ⚠ Large output from {block.name}: {len(output)} chars\033[0m"
        )
    return None


def summary_hook(messages: list) -> None:
    """Stop —— 对话循环结束后统计本次会话中执行过的工具调用总数。

    遍历完整对话历史的 tool_result 块计数，不包含被 PreToolUse 拦截未执行的调用。
    注意：messages 跨多轮查询累积（见 app.py 中 history_messages 的生命周期），
    因此统计的是整个 session 的累计值，而非当前单次查询。
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
