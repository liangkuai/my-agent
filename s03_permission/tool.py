"""
工具层 —— 定义 AI 代理可调用的所有工具及其处理函数。

本模块包含:
- 5 个工具处理函数: run_bash / run_read / run_write / run_edit / run_glob
- safe_path: 路径安全校验, 防止目录穿越攻击
- TOOLS: 向 Anthropic API 注册的工具定义列表
- TOOL_HANDLERS: 工具名 -> 处理函数的映射

设计原则:
- 所有处理函数统一返回字符串, 无论成功还是失败都不向上抛异常,
  错误信息以 "Error: ..." 格式返回, 让模型自行判断并重试。
- safe_path 是安全边界, 所有涉及文件路径的操作必须先经过它校验。
"""

import subprocess
from pathlib import Path

from constant import WORKDIR


def run_bash(command: str) -> str:
    """执行模型请求的 shell 命令并返回结果字符串。

    通过 subprocess.run 在 WORKDIR 下以 shell 模式运行命令,
    捕获 stdout/stderr 合并后截断返回。超时或系统错误均转为
    字符串返回, 不向上抛异常。

    Args:
        command: 要执行的 shell 命令字符串 (支持管道、通配符等)。

    Returns:
        命令输出 (截断到 50000 字符) 或错误信息。
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
    """将相对路径解析为绝对路径, 并确保不逃逸出 WORKDIR。

    工作原理:
    1. 对 WORKDIR 和输入路径分别 resolve(), 展开 .. 和符号链接,
       像 ../../etc/passwd 这类越权路径会在此暴露真实位置
    2. 用 is_relative_to 验证最终路径仍在工作目录内
    3. 越界则抛出 ValueError, 调用方应捕获并转为 "Error: ..." 返回

    这是所有文件操作的安全关口 —— 无论调用方是谁, 要想接触文件系统
    就必须先过这一道检查。

    Args:
        p: 相对于 WORKDIR 的路径字符串。

    Returns:
        规范化后的绝对 Path 对象。

    Raises:
        ValueError: 路径逃逸出 WORKDIR 时抛出。
    """
    # 同样把 WORKDIR resolve 一次: is_relative_to 是纯字符串前缀比较,
    # 只有当两边都是规范化的真实路径时结果才可靠。这样本函数就不再依赖
    # 调用方「恰好传进来一个已 resolve 的目录」这一隐含前提。
    workdir = Path(WORKDIR).resolve()
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容, 可选限制行数。

    先经 safe_path 校验路径, 然后按行读取。若指定 limit 且文件行数
    超出, 会在末尾追加 "... (N more lines)" 提示, 避免模型将截断
    误读为文件结尾。

    Args:
        path: 相对于 WORKDIR 的文件路径。
        limit: 最大读取行数, None 表示读取全部。

    Returns:
        文件内容字符串, 或 "Error: ..." 错误信息。
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
    """将内容写入文件, 自动创建缺失的父目录。

    Args:
        path: 相对于 WORKDIR 的目标文件路径。
        content: 要写入的文本内容。

    Returns:
        成功时返回 "Wrote N bytes to <path>", 失败返回 "Error: ..."。
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
    """精确替换文件中的文本, 仅替换首次出现。

    先读取文件, 查找 old_text; 若未找到则返回错误 (避免静默失败)。
    使用 str.replace(..., 1) 只替换第一处, 防止误伤同名片段。

    Args:
        path: 相对于 WORKDIR 的目标文件路径。
        old_text: 待替换的原始文本, 必须与文件中完全匹配。
        new_text: 替换后的新文本。

    Returns:
        成功返回 "Edited <path>", 失败返回 "Error: ..."。
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
    """按 glob 模式搜索文件, 返回匹配路径列表。

    在 WORKDIR 下执行 glob 搜索, 并对每个匹配结果做二次校验,
    防止经由 .. 或符号链接逃逸出工作目录的路径被返回。

    Args:
        pattern: glob 模式字符串 (如 "**/*.py")。

    Returns:
        换行分隔的匹配路径列表, 无匹配时返回 "(no matches)"。
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


# ── 工具注册表 ────────────────────────────────────────────────────────
# TOOLS 是 Anthropic API 要求的工具定义列表, 每个工具需声明 name,
# description 和 input_schema。模型根据这些定义决定何时调用哪个工具。
#
# TOOL_HANDLERS 将工具名映射到对应的处理函数, agent_loop 中通过
# TOOL_HANDLERS.get(block.name) 查找并执行。

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


TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}
