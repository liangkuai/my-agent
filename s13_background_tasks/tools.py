"""
Agent 可用的工具函数注册表。

本模块定义了主 agent 和子 agent 共用的全部工具：shell 执行、文件读写、
编辑、glob 搜索、todo 管理、skill 加载、对话压缩，以及子 agent 的 spawn 与工具回填循环。

核心设计原则：
1. 所有工具函数都不抛异常——错误以 "Error: ..." 字符串返回给模型，
   让模型自行消化和重试，避免单次工具失败拖垮整个 agent 循环。
2. 路径安全：safe_path() 是所有文件类工具的统一入口，通过 resolve + is_relative_to
   双重校验防止目录穿越。
3. 主/子 agent 共享 handler（run_bash / run_read / run_write / run_edit / run_glob），
   但工具声明分开——主 agent 额外拥有 todo_write、task、load_skill、compact。
"""

import subprocess
import json
import ast
from pathlib import Path

from constant import WORKDIR, MODEL, SUB_SYSTEM
from llm_client import client
import skills
import hooks
import tasks


def run_bash(command: str, run_in_background: bool = False) -> str:
    """执行模型请求的 shell 命令，返回 stdout+stderr 合并后的字符串。

    设计要点：
    - 通过 shell=True 交由系统 shell 解释，支持管道、通配符等复杂语法。
    - 超时 120s：防止死循环或等待输入的命令无限挂起。
    - 输出截断到 50000 字符：避免超长输出撑爆上下文窗口。
    - 空输出返回 "(no output)"：防止模型把空字符串误读成「调用失败」。
    - 所有异常都转为 "Error: ..." 字符串返回，不向上抛出——
      让模型能读到错误并自行决定下一步，而不是让整个 agent 循环崩溃。
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
    """将模型给的相对路径解析为 WORKDIR 下的安全绝对路径。

    通过 resolve() + is_relative_to() 双重校验防止目录穿越攻击：
    1. resolve() 展开 `..` 和符号链接，暴露真实位置。
    2. is_relative_to() 确保结果仍在 WORKDIR 子树内。

    两次 resolve（WORKDIR 和拼接结果各一次）是必须的——
    只有两边都经过 resolve() 规范化后，is_relative_to 的结果才可靠。
    如果路径越界则抛出 ValueError，由调用方（各 run_*）捕获并转为 "Error: ..." 返回给模型。

    Raises:
        ValueError: 路径逃逸 WORKDIR 时抛出。
    """
    workdir = Path(WORKDIR).resolve()
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容，可选截断到前 limit 行。

    与 run_bash 同理：所有文件类工具都把异常转成 "Error: ..." 字符串返回给模型，
    而不是向上抛出——让模型读到错误并自行重试，避免单次工具失败拖垮整个循环。
    """
    try:
        lines = safe_path(path).read_text().splitlines()
        # 只读前 limit 行时，补一行 "... (N more lines)" 提示，免得模型把截断
        # 误当成文件的真实结尾。
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入内容到文件，自动创建缺失的父目录。"""
    try:
        file_path = safe_path(path)
        # 自动补建缺失的父目录，这样模型写新文件时不必先手动创建目录。
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的一段文本（仅替换首次出现）。

    若 old_text 在文件中不存在则报错，避免静默地什么都没改、却让模型以为成功了。
    若 old_text 多次出现，每次调用只替换一处，更可控也避免误伤。
    """
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        # count=1：只替换首次出现，单次调用不会意外波及多处同名文本。
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """在 WORKDIR 下按 glob 模式搜索文件，返回匹配的相对路径列表。

    搜索范围限定在 WORKDIR 内；结果会二次校验确保没有路径逃逸。
    """
    import glob as g

    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            # 二次校验：glob 结果可能经由 `..` 或符号链接指向 WORKDIR 之外，
            # 这里和 safe_path 同样用 is_relative_to 过滤掉越界路径。
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# === 任务管理 ===
# CURRENT_TODOS 是进程生命周期内唯一的可变会话状态。
# run_todo_write 负责更新，agent_loop 在每次 todo_write 调用后重置提醒计数器。
CURRENT_TODOS: list[dict] = []


def run_todo_write(todos: list[dict] | str) -> str:
    """接收模型传来的任务列表，校验、记录状态，并打印彩色任务面板到终端。

    严格顺序：先校验 → 失败则把错误信息返回给模型让它自行修正；
    校验通过后再更新全局状态，避免写进脏数据。
    """
    global CURRENT_TODOS
    normalized_todos, error = _normalize_todos(todos)
    if error:
        return error
    # error 为 None ⇒ 解构出的 normalized 必为 list，但类型检查器无法自动
    # 缩窄联合元组的双向依赖，这里显式断言帮助缩窄。
    assert normalized_todos is not None
    CURRENT_TODOS = normalized_todos
    # 用 ANSI 转义码渲染彩色任务面板：黄色标题、青色进行中箭头、绿色勾
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


# 返回类型用 list 而非 list[dict]：json.loads / ast.literal_eval 返回 Any，
# 类型检查器无法证明元素为 dict——但下方的逐元素 isinstance 校验在运行时保证了这一点。
def _normalize_todos(todos: list[dict] | str) -> tuple[list, None] | tuple[None, str]:
    """将模型输入规范化成合法的 todo 列表，返回 (结果, 错误) 二选一的元组。

    模型可能传入已解析的 list[dict]，也可能传入 JSON 字符串（甚至 Python
    风格的单引号字面量）。json.loads 先试，失败后用 ast.literal_eval 兜底，
    兼顾 LLM 输出的各种格式漂移。

    Returns:
        (todos, None)  — 校验通过，todos 为 list[dict]
        (None, error)  — 校验失败，error 为可直接返回给模型的错误描述
    """
    # 模型偶尔输出未解析的 JSON/Python 字面量字符串，先尝试反序列化
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                # ast.literal_eval 比 json.loads 更宽松，能处理单引号、
                # None/True/False 等 Python 原生写法
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    # 类型 + 结构校验，逐层确保数据形状符合预期
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

    子 agent 在一个清新的上下文中运行（只保留 description 作为首条 user 消息），
    拥有独立的工具集 SUB_TOOLS（bash/read/write/edit/glob，不含 todo_write 和 task）。
    通过 client 直接驱动工具调用循环，而不是让上层 agent_loop 统一调度——
    这样主 agent 调用 task 工具后只需等待最终结果，不必参与子 agent 的每一步推理。

    循环机制：
    1. 将 description 作为首条 user 消息发送给模型。
    2. 若模型返回 tool_use，逐一执行工具并将结果以 tool_result 回填。
    3. 工具执行前后触发 PreToolUse / PostToolUse hooks，支持拦截和审计。
    4. 若 stop_reason 不再是 tool_use（即模型给出最终文本回复），循环结束。
    5. 最多 30 轮；超限后从最近一条 assistant 消息中提取文本结果返回。

    Args:
        description: 描述子任务的纯文本，作为子 agent 的初始 user 消息。

    Returns:
        子 agent 的最终文本回复；若 30 轮后无文本输出则返回提示信息。
    """
    print("\n\033[35m[Subagent spawned]\033[0m")

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
                blocked = hooks.trigger_hooks("PreToolUse", block)
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
                hooks.trigger_hooks("PostToolUse", block, output)
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
    print("\033[35m[Subagent done]\033[0m")
    print()
    return result


def extract_text(content) -> str:
    """从 API 返回的 content 中提取纯文本。

    API 的 content 字段是 ContentBlock 列表（可能混合 text / tool_use / tool_result），
    本函数过滤出 type == "text" 的块并拼接，用于获取模型最终的文本回复。
    若 content 本身已是字符串则原样返回。
    """
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text"
    )


def load_skill(name: str) -> str:
    """按名称从 SKILL_REGISTRY 中查找并返回技能的完整 SKILL.md 原文。

    注册表由 skills._scan_skills() 在模块导入时一次性填充，本函数只做 O(1) 字典查找。
    未找到时返回错误提示字符串而非抛异常，与其余工具函数的约定保持一致。
    """
    skill = skills.get_skill(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]


def run_create_task(
    subject: str, description: str = "", blockedBy: list[str] | None = None
) -> str:
    """创建新任务并持久化到 .tasks/{id}.json，向终端打印蓝色创建提示。

    参数直接转发至 tasks.create_task()，由后者生成唯一 ID、写入磁盘。
    blockedBy 声明前置依赖——所列任务全部完成后本任务才能被认领。

    Args:
        subject: 任务标题（必填）。
        description: 详细描述（可选）。
        blockedBy: 前置依赖任务 ID 列表（可选）。

    Returns:
        创建确认字符串，如 "Created task_xxx: 标题 (blockedBy: t1, t2)"。
    """
    task = tasks.create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """列出所有任务，以 Unicode 图标区分状态，空列表时返回友好提示。

    每条任务显示：图标 ID 标题 [状态] [owner] (依赖)。图标含义：
    ○ = pending  ● = in_progress  ✓ = completed

    Returns:
        多行字符串；无任务时返回 "No tasks. Use create_task to add some."。
    """
    all_tasks = tasks.list_tasks()
    if not all_tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in all_tasks:
        icon = {"pending": "○", "in_progress": "●", "completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} [{t.status}]{owner}{deps}")
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    """获取单条任务的 JSON 格式详情。

    tasks.get_task() 内部将 Task 对象序列化为缩进 JSON 字符串，
    本函数仅额外捕获 FileNotFoundError，将异常转为 "Error: ..." 字符串。

    Args:
        task_id: 任务唯一标识（如 "task_1699999999_0001"）。

    Returns:
        格式化 JSON 字符串，或 "Error: Task {id} not found"。
    """
    try:
        return tasks.get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    """认领 pending 任务，将其状态切换为 in_progress，owner 固定为 "agent"。

    底层 tasks.claim_task() 会校验：
    1. 任务当前状态必须为 "pending"
    2. 所有 blockedBy 依赖必须已满足（依赖任务状态为 completed）

    任一条件不满足时返回具体原因字符串，让模型自行调整。

    Args:
        task_id: 要认领的任务 ID。

    Returns:
        成功确认消息或失败原因字符串。
    """
    return tasks.claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    """完成任务（in_progress → completed），自动报告被解除阻塞的下游任务。

    底层 tasks.complete_task() 会：
    1. 校验任务状态为 "in_progress"
    2. 切换状态为 "completed" 并保存
    3. 扫描所有 pending 任务，找出因本任务完成而 can_start 的下游任务
    4. 返回确认消息时附带解除阻塞的下游任务列表

    Args:
        task_id: 要完成的任务 ID。

    Returns:
        成功确认消息（含解除阻塞的下游任务列表）或失败原因字符串。
    """
    return tasks.complete_task(task_id)


# =============================================================================
#  工具定义：声明每个工具的名称、描述和参数 schema（Anthropic Tool Use 格式）。
#  TOOLS        → 提供给主 agent，包含全部 14 个工具。
#  SUB_TOOLS    → 提供给子 agent，仅含 5 个基础文件/shell 工具（不含 todo_write、task 等）。
#  TOOL_HANDLERS / SUB_TOOL_HANDLERS → 把工具名映射到对应的 Python 函数。
# =============================================================================

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "run_in_background": {"type": "boolean"},
            },
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
    {
        "name": "load_skill",
        "description": "Load the full content of a skill by name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "compact",
        "description": "Summarize earlier conversation to free context space.",
        "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}},
    },
    {
        "name": "create_task",
        "description": "Create a new task with optional blockedBy dependencies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "blockedBy": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["subject"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List all tasks with status, owner, and dependencies.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_task",
        "description": "Get full details of a specific task by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "claim_task",
        "description": "Claim a pending task. Sets owner, changes status to in_progress.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "complete_task",
        "description": "Complete an in-progress task. Reports unblocked downstream tasks.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
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
    "load_skill": load_skill,
    "create_task": run_create_task,
    "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task,
    "complete_task": run_complete_task,
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


def list_tool_name() -> list:
    """返回主 agent 所有可用工具的名称列表。

    从 TOOLS 列表（面向 API 的工具定义，即模型实际可调用的工具集合）中
    提取 name 字段，而非从 TOOL_HANDLERS 字典。因为 compact 等元操作工具
    在 app.agent_loop 中特殊拦截处理，不经过 TOOL_HANDLERS 分发，但模型
    仍然可以调用它们——system prompt 中声明的工具列表应与 API 声明一致。

    供 context.update_session_context() 调用，填入 context["enabled_tools"]。
    """
    return [t["name"] for t in TOOLS]


def use_tool(name: str, input: dict) -> str:
    """主 agent 工具调度的统一入口：按名查找 handler 并执行。

    从 TOOL_HANDLERS 字典中查找对应函数，将 input ** 解包传入。
    未注册的工具名返回 "Unknown: {name}" 而非抛异常，让模型自行纠正。

    Args:
        name: 工具名（"bash"、"read_file" 等）。
        input: 模型传入的参数字典（如 {"command": "ls"}）。

    Returns:
        工具输出的字符串；未注册时返回 "Unknown: {name}"。
    """
    handler = TOOL_HANDLERS.get(name)
    return handler(**input) if handler else f"Unknown: {name}"
