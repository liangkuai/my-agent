"""
权限管道 —— 在工具执行前对模型请求的每个 tool_use 块进行安全检查。

管道分三层：
1. 拒绝列表（硬阻止）—— 直接拦截，无交互机会
2. 规则检查（软阻止）—— 命中后弹出交互式确认
3. 放行           —— 以上均未命中，正常执行

工具层（tool.py）不再做权限判断，安全逻辑统一收敛到本模块。
"""

from typing import Any

from constant import WORKDIR


# ── 拒绝列表 ────────────────────────────────────────────────────────
# 命中即直接拒绝，模型看到 "Permission denied." 后可以调整策略。
#
# 注意：这里使用子串匹配，并非可靠的安全边界——多余空格、`;\nrm`、编码变体、
# 大小写变形等均可绕过。仅作为 CLI demo 的第一道粗筛，生产环境必须使用沙箱
# 或容器隔离来执行不可信命令。

DENY_LIST = [
    "rm -rf /",
    "sudo",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=",
    "> /dev/sda",
]


def check_deny_list(command: str) -> str | None:
    """命中则返回拒绝原因字符串，否则返回 None。"""
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# ── 规则检查 ────────────────────────────────────────────────────────
# 每条规则包含：
#   tools    — 适用的工具名列表
#   check    — 判断函数，接收工具参数字典，返回 True 表示拦截
#   message  — 拦截时展示给用户的警告信息
#
# 同样使用子串匹配，局限性同上。

PERMISSION_RULES = [
    {
        "tools": ["write_file", "edit_file"],
        "check": lambda args: (
            # 路径不存在时默认值 "" → WORKDIR / "" == WORKDIR，不触发越界。
            # 畸形的无 path 调用会在后续 safe_path 中以 ValueError 兜底。
            not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR)
        ),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: any(
            kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]
        ),
        "message": "Potentially destructive command",
    },
]


def check_rules(tool_name: str, args: dict) -> str | None:
    """遍历规则列表，首次命中即返回警告信息；均未命中返回 None。"""
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# ── 用户确认 ────────────────────────────────────────────────────────

def ask_user(tool_name: str, args: dict, reason: str) -> str:
    """弹交互式确认提示，返回 "allow" 或 "deny"。

    注意：input() 同步阻塞当前线程。CLI 单线程场景可接受；
    用于多线程／异步环境时需改为非阻塞确认机制（如 asyncio.Queue + callback）。
    """
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# ── 管道入口 ────────────────────────────────────────────────────────

def check_permission(block: Any) -> str | None:
    """权限管道主入口。

    参数 block 为 Anthropic SDK 返回的 tool_use 内容块，需要具备
    .name（str）和 .input（dict）两个属性。

    返回 None 表示放行，非 None 字符串表示拒绝原因（由 hooks 层直接作为
    tool_result 回填给模型，模型可据此调整策略）。
    """
    # 第一层：拒绝列表 —— bash 专属，直接拒绝
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\033[31m⛔ {reason}\033[0m")
            print()
            return "Permission denied by deny list"

    # 第二层：规则检查 —— 命中后进入用户交互确认
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        print()
        if decision == "deny":
            return "Permission denied by user"

    # 第三层：放行
    return None
