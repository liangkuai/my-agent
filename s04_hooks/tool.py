"""
工具定义与执行层 —— 向 LLM 暴露的五个工具（bash / read_file / write_file / edit_file / glob）。

本模块包含两部分：
1. 五个工具的执行函数（run_*），负责实际的文件/命令操作，统一返回字符串给模型
2. TOOLS（Anthropic tool-use schema）和 TOOL_HANDLERS（名称→函数映射表），
   供 app.py 中的 agent_loop 使用

安全约定：
- 所有文件操作都经过 safe_path() 做目录穿越检测
- 所有异常都转为 "Error: ..." 字符串返回，不向上抛，让模型自行解读和重试
- 权限判断不在此模块处理，统一由 permission.py 的管道在 PreToolUse 钩子中完成
"""

import subprocess
from pathlib import Path

from constant import WORKDIR


def run_bash(command: str) -> str:
    """执行模型请求的 shell 命令，返回 stdout+stderr 合并后的字符串。

    设计决策：
    - 无论成功或失败都返回字符串、不向上抛异常，让模型看到错误后自行调整
    - 输出截断到 50000 字符，防止超长结果撑爆上下文窗口
    - 120 秒超时保护，避免死循环或等待输入的命令永久阻塞
    - 空输出返回 "(no output)" 占位，避免模型将空字符串误解为调用失败
    """

    try:
        r = subprocess.run(
            command,  # 要执行的命令
            shell=True,  # 通过系统 shell 解释命令（支持管道、通配符等）
            cwd=WORKDIR,  # 指定这条命令在哪个目录下执行
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,  # 以字符串（而非 bytes）返回输出
            encoding="utf-8",
            errors="replace",
            timeout=120,  # 超时保护：命令最多跑 120 秒，超时则抛 TimeoutExpired
        )
        # 合并标准输出与标准错误：模型同样需要看到报错内容才能判断成败。
        out = (r.stdout + r.stderr).strip()
        # 截断到 50000 字，避免超长输出撑爆上下文；空输出回一个占位符，
        # 以免模型把空字符串误读成「调用失败」。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        # 命令超时（如死循环、等待输入），返回提示而非让异常冒泡。
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        # 进程无法启动等系统级错误（例如 shell 不存在、资源不足）。
        return f"Error: {e}"


def safe_path(p: str) -> Path:
    """将模型给的相对路径解析为绝对路径，确保不逃出 WORKDIR。

    两层防御：
    1. resolve() 展开 .. 和符号链接，暴露真实路径
    2. is_relative_to() 校验最终路径在 WORKDIR 子树内

    调用前对 WORKDIR 再次 resolve()，消除「调用方碰巧传了已解析的目录」
    这一隐含前提，确保 is_relative_to 在两边都是规范路径时执行比较。
    """
    workdir = Path(WORKDIR).resolve()
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容，支持行数截断。

    limit 参数作用：只返回前 limit 行，并在末尾追加 "... (N more lines)" 提示，
    避免模型将截断误读为文件的真实结尾。
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
    """写入文件，自动创建缺失的父目录。

    这样模型创建新文件时无需先手动 mkdir，减少一次工具往返。
    """
    try:
        file_path = safe_path(path)
        # 自动补建缺失的父目录，这样模型写新文件时不必先手动创建目录。
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的文本（仅替换首次出现）。

    设计决策：
    - 找不到 old_text 时返回错误而非静默跳过，避免模型误以为修改成功
    - replace(..., 1) 只替换第一处，防止同名片段被误伤
    - 若需替换多处，模型应多次调用 edit_file
    """
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        # 找不到待替换文本就报错，避免静默地什么都没改、却让模型以为成功了。
        if old_text not in text:
            return f"Error: text not found in {path}"
        # replace 的第三个参数 1 表示只替换第一处：若 old_text 在文件中多次出现，
        # 一次只改一处更可控，也避免误伤其他同名片段。
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """在 WORKDIR 下匹配 glob 模式，返回匹配路径（一行一个）。

    对结果做二次 is_relative_to 校验，过滤经由 .. 或符号链接逃逸的路径。
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


# ── 工具 Schema 注册表 ──────────────────────────────────────────────────
# Anthropic tool-use API 要求的 JSON Schema 格式。
# 每个定义包含 name、description 和 input_schema：
#   name          — 工具唯一标识，需与 TOOL_HANDLERS 的键严格一致
#   description   — 模型据此判断何时调用该工具，应描述「做什么」而非「怎么做」
#   input_schema  — JSON Schema，模型按此结构填充参数

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
]


# ── 工具名称 → 执行函数映射 ────────────────────────────────────────────
# agent_loop 在处理 tool_use 块时，通过 block.name 在此字典中查找对应的
# 执行函数。键名必须与 TOOLS 列表中的 name 字段严格一致。

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}
