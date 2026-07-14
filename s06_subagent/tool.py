"""
Agent 可用的工具函数注册表。

本模块是 s06 的工具层，定义了主 agent 和子 agent 共用的全部工具实现：
shell 执行、文件读写、编辑、glob 搜索、todo 管理，以及子 agent 的 spawn
与独立的工具回填循环。

核心设计原则：
1. 不抛异常 → 所有工具函数的错误以 "Error: ..." 字符串返回给模型，
   让模型自行解读和重试，避免单次工具失败拖垮整个 agent 循环。
2. 路径安全 → safe_path() 是所有文件类工具的统一入口，通过 resolve
   展开符号链接和 `..`，再由 is_relative_to 拦掉逃逸路径。
3. 代码复用，声明分离 → 主/子 agent 共享同一组 handler 实现
   （run_bash / run_read / run_write / run_edit / run_glob），
   但工具定义分开——主 agent 额外拥有 todo_write 和 task。
"""

import subprocess
import json
import ast
from pathlib import Path

from constant import WORKDIR, MODEL, SUB_SYSTEM
from llm_client import client
from hooks import trigger_hooks


def run_bash(command: str) -> str:
    """执行 shell 命令，合并 stdout+stderr 并截断为字符串返回。

    所有工具函数的统一约定：不向上抛异常，错误以 "Error: ..." 字符串
    返回给模型，让模型自行解读和重试，避免单次工具失败拖垮整个 agent 循环。

    安全措施：
    - shell=True 让命令经由系统 shell 解释，支持管道、通配符等
    - cwd 锁定在 WORKDIR，配合 permission.py 的拒绝列表做双层防护
    - 120 秒超时保护，防止死循环或等待输入卡死 agent 循环
    - 输出截断到 50000 字符，避免超长输出撑爆模型上下文
    - 空输出返回占位符 "(no output)"，防止模型把空字符串误读为调用失败

    Returns:
        命令的合并输出（截断后），或 "Error: ..." 错误描述。
    """
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def safe_path(p: str) -> Path:
    """把相对路径解析为绝对路径，并确保落在 WORKDIR 内，防止目录穿越。

    安全机制：
    1. 先把 WORKDIR 也 resolve() 一次——is_relative_to 在 Python 3.9~3.11
       是纯字符串前缀比较，只有两边都是规范化的真实路径时结果才可靠。
    2. (workdir / p).resolve() 展开所有 `..` 和符号链接，让 ../etc/passwd
       这类攻击路径暴露真实位置。
    3. is_relative_to 作为最后一道门槛，拦掉所有逃逸路径。

    Raises:
        ValueError: 路径指向 WORKDIR 之外时抛出，由调用方的 try/except 捕获
                   并转为 "Error: ..." 字符串返回给模型。
    """
    workdir = Path(WORKDIR).resolve()
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容，可选截断到前 limit 行。

    Args:
        path: 相对于 WORKDIR 的文件路径，经 safe_path() 校验。
        limit: 若指定且小于实际行数，只返回前 limit 行，并在末尾追加
               "... (N more lines)" 提示——防止模型把截断误当成文件的真实结尾。

    Returns:
        文件内容字符串，或 "Error: ..." 错误描述。
    """
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入内容到文件，自动创建缺失的父目录。

    自动补建父目录让模型写新文件时不必先 mkdir——减少工具调用轮次。

    Returns:
        "Wrote N bytes to <path>" 确认消息，或 "Error: ..." 错误描述。
    """
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的一段文本，仅替换首次出现。

    设计决策：
    - 若 old_text 不存在 → 报错而非静默跳过，防止模型以为编辑成功。
    - 若 old_text 多次出现 → 只替换第一处（str.replace count=1），
      避免单次调用意外波及多处同名文本。模型可多次调用逐处修改。

    Returns:
        "Edited <path>" 确认消息，或 "Error: ..." 错误描述。
    """
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """在 WORKDIR 下按 glob 模式搜索文件，返回匹配的相对路径列表。

    搜索范围通过 root_dir 限定在 WORKDIR 内；但 glob 结果仍可能经由 `..`
    或符号链接指向外部，因此逐条用 is_relative_to 做二次校验。

    glob 模块仅本函数使用，采用内联 import 避免污染模块顶层命名空间。

    Returns:
        换行分隔的匹配路径列表，无匹配时返回 "(no matches)"。
    """
    import glob as g

    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
#  任务管理
#
#  CURRENT_TODOS 是 agent_loop 生命周期内唯一的可变会话状态：
#  - run_todo_write 负责校验并更新
#  - agent_loop 在每次 todo_write 调用后重置 rounds_since_todo 计数器
#  - 所有模块通过 `from tool import CURRENT_TODOS` 共享同一引用
# =============================================================================
CURRENT_TODOS: list[dict] = []


def run_todo_write(todos: list[dict] | str) -> str:
    """接收模型传来的任务列表，校验并更新全局状态，打印彩色面板到终端。

    严格先校验后更新：校验失败则错误信息直接返回给模型让它自行修正，
    校验通过才写入 CURRENT_TODOS，避免脏数据污染全局状态。

    Args:
        todos: 模型传入的 list[dict] 或 JSON/Python 字面量字符串，
               _normalize_todos 会统一处理两种格式。

    Returns:
        确认消息或错误描述，直接作为 tool_result 回填给模型。
    """
    global CURRENT_TODOS
    normalized_todos, error = _normalize_todos(todos)
    if error:
        return error
    # error 为 None ⇒ 解构出的 normalized 必为 list；但类型检查器无法自动
    # 缩窄联合元组的双向依赖，显式断言帮助缩窄。
    assert normalized_todos is not None
    CURRENT_TODOS = normalized_todos
    # 用 ANSI 转义码渲染彩色面板：黄色标题、青色进行中箭头、绿色勾
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {
            "pending": " ",
            "in_progress": "\033[36m▸\033[0m",
            "completed": "\033[32m✓\033[0m",
        }[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    print()
    return f"Updated {len(CURRENT_TODOS)} tasks"


def _normalize_todos(todos: list[dict] | str) -> tuple[list, None] | tuple[None, str]:
    """将模型输入规范化成合法的 todo 列表，返回 (结果, 错误) 二选一的元组。

    容错策略：模型可能传入已解析的 list[dict]，也可能传入 JSON 字符串或
    Python 风格的单引号字面量。json.loads 先试，失败后用 ast.literal_eval
    兜底（后者能处理单引号、None/True/False 等 Python 原生写法），兼顾 LLM
    输出的各种格式漂移。

    注意：返回类型标注为 list 而非 list[dict]——json.loads / ast.literal_eval
    返回 Any，类型检查器无法证明元素为 dict，但下方的逐元素 isinstance 校验
    在运行时保证了这一点。

    Returns:
        (todos, None)  — 校验通过，todos 为 list[dict]
        (None, error)  — 校验失败，error 为可直接返回给模型的错误描述
    """
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None


def spawn_subagent(description: str) -> str:
    """启动子 agent 独立完成一项复杂子任务，只返回最终结论。

    子 agent 拥有独立的工具集 SUB_TOOLS（bash/read/write/edit/glob，不含
    todo_write 和 task），在全新上下文中运行——只保留 description 作为首条
    user 消息，不继承主 agent 的对话历史。

    主 agent 调用 task 工具后只需等待最终结果，不必参与子 agent 的每一步推理，
    因为子 agent 通过 client 自驱动工具调用循环，而非交由上层 agent_loop 调度。

    循环流程（最多 30 轮）：
    1. 发送 description → 模型返回 reply（可能含 tool_use）
    2. 逐一执行工具（PreToolUse / PostToolUse hooks 照常触发）
    3. 将 tool_result 列表回填 → 下一轮
    4. stop_reason != "tool_use" → 循环结束，提取文本结果
    5. 超限后向前查找最后一条 assistant 消息中的文本；仍无则返回提示

    Args:
        description: 描述子任务的纯文本，作为子 agent 的初始 user 消息。

    Returns:
        子 agent 的最终文本回复；若 30 轮后无文本输出则返回提示信息。
    """
    print(f"\n\033[35m[Subagent spawned]\033[0m")

    # 子 agent 使用全新上下文，不继承主 agent 的对话历史。
    messages = [{"role": "user", "content": description}]
    for _ in range(30):
        response = client.messages.create(
            model=MODEL,
            system=SUB_SYSTEM,
            messages=messages,
            tools=SUB_TOOLS,
            max_tokens=8000,
        )
        # 将 assistant 回复追加到消息历史。
        messages.append({"role": "assistant", "content": response.content})

        # 非 tool_use 的 stop_reason（end_turn / stop_sequence / max_tokens）
        # 表示模型已经给出最终回复，循环结束。
        if response.stop_reason != "tool_use":
            break

        # 遍历 assistant 回复中的 tool_use 块，逐一执行并收集结果。
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # PreToolUse hook：允许外部拦截（如权限控制），返回非空则跳过执行。
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(blocked),
                        }
                    )
                    continue

                # 从 SUB_TOOL_HANDLERS 查找 handler 并执行。
                handler = SUB_TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"

                # PostToolUse hook：审计、日志等后置处理。
                trigger_hooks("PostToolUse", block, output)
                print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )

        # 将本轮的 tool_result 列表作为一条 user 消息追加。
        messages.append({"role": "user", "content": results})

    # 从最后一条消息中提取文本结果。
    result = extract_text(messages[-1]["content"])
    if not result:
        # 最后一条可能仍是 tool_use 消息（30 轮耗尽），向前查找最后一条
        # assistant 消息中的文本内容。
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."

    print(f"\033[35m[Subagent done]\033[0m")
    print()
    return result


def extract_text(content) -> str:
    """从 API 返回的 content 中提取纯文本。

    Anthropic Messages API 的 content 字段是 ContentBlock 列表，可能混合
    text / tool_use / tool_result 三种类型。本函数过滤出 type == "text" 的块
    并拼接，用于从模型回复中提取最终的自然语言文本。

    Args:
        content: API 返回的 content（list[ContentBlock]），或已是纯字符串。

    Returns:
        拼接后的纯文本字符串。
    """
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text"
    )


# =============================================================================
#  工具定义：声明每个工具的名称、描述和参数 schema（Anthropic Tool Use 格式）。
#  TOOLS        → 提供给主 agent，包含全部 7 个工具。
#  SUB_TOOLS    → 提供给子 agent，仅含 5 个基础文件/shell 工具（不含 todo_write 和 task）。
#  TOOL_HANDLERS / SUB_TOOL_HANDLERS → 把工具名映射到对应的 Python 函数。
# =============================================================================

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "todo_write",
        "description": "Create and manage a task list for your current coding session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
    {
        "name": "task",
        "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
        "input_schema": {
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"],
        },
    },
]


# 主 agent 的工具处理函数映射：工具名 → 实现函数。
TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
    "todo_write": run_todo_write,
    "task": spawn_subagent,
}


# 子 agent 的工具定义：不含 todo_write（子 agent 不需要自己的 todo）和 task（防止无限嵌套）。
SUB_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]


# 子 agent 的工具处理函数映射：与主 agent 共享同一组 handler 实现，
# 但只映射子 agent 实际拥有的 5 个基础工具。
SUB_TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}
