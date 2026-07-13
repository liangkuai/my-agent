"""s02: Tool Use — 工具定义与处理函数

本模块定义 Agent 可用的 5 个工具（bash / read_file / write_file / edit_file / glob）
及其处理函数。所有 handler 遵循统一约定：
- 第一个参数固定为 workdir: Path，其余由模型传入
- 任何错误都转成 "Error: ..." 字符串返回，不向上抛异常，让模型自行判读是否重试

目录穿越防护由 safe_path() 统一负责，run_bash 额外内置命令黑名单。
"""

import subprocess
from pathlib import Path


def run_bash(workdir: Path, command: str) -> str:
    # 执行 shell 命令，将 stdout/stderr 或异常信息作为字符串返回给模型。
    # 无论成功还是失败都不抛异常——让模型读到结果后自行判读下一步，而非让整个循环中断。

    # 仅作演示的极简防护：用子串黑名单挡掉几条最常见的破坏性命令。
    # 注意这远不是真正的安全边界——诸如 `rm -rf ~`、多空格变体、`dd`、fork 炸弹等
    # 都能轻易绕过。生产环境必须改用沙箱／容器隔离来执行不可信命令。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(
            command,  # 要执行的命令
            shell=True,  # 通过系统 shell 解释命令（支持管道、通配符等）
            cwd=workdir,  # 指定这条命令在哪个目录下执行
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,  # 以字符串（而非 bytes）返回输出
            timeout=120,  # 超时保护：命令最多跑 120 秒，超时则抛 TimeoutExpired
        )
        # 合并标准输出与标准错误：模型同样需要看到报错内容才能判断成败。
        out = (r.stdout + r.stderr).strip()
        # 截断到 50000 个字符，避免超长输出撑爆上下文；空输出回一个占位符，
        # 以免模型把空字符串误读成「调用失败」。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        # 命令超时（如死循环、等待输入），返回提示而非让异常冒泡。
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        # 进程无法启动等系统级错误（例如 shell 不存在、资源不足）。
        return f"Error: {e}"


def safe_path(workdir: Path, p: str) -> Path:
    # 把模型给的相对路径解析成绝对路径，并确保它仍落在工作目录内，防止目录穿越。
    # resolve() 会展开 `..` 和符号链接，所以像 `../../etc/passwd` 这类越权路径
    # 在这一步会暴露真实位置；随后用 is_relative_to 拦掉所有逃出 workdir 的路径。

    # 先把 workdir 也 resolve 一次：is_relative_to 基于路径语义而非纯字符串做归属
    # 判断，路径中含 `..` 或符号链接会干扰结果。两边都 resolve 才能保证可靠，
    # 这样本函数就不再依赖调用方「恰好传入已 resolve 的目录」这一隐含前提。
    workdir = Path(workdir).resolve()
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(workdir: Path, path: str, limit: int | None = None) -> str:
    # 与 run_bash 同理：所有文件类工具都把异常转成 "Error: ..." 字符串返回给模型，
    # 而不是向上抛出——让模型读到错误并自行重试，避免单次工具失败拖垮整个循环。
    try:
        lines = safe_path(workdir, path).read_text().splitlines()
        # 只读前 limit 行时，补一行 "... (N more lines)" 提示，免得模型把截断
        # 误当成文件的真实结尾。
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(workdir: Path, path: str, content: str) -> str:
    try:
        file_path = safe_path(workdir, path)
        # 自动补建缺失的父目录，这样模型写新文件时不必先手动创建目录。
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(workdir: Path, path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(workdir, path)
        text = file_path.read_text()
        # 找不到待替换文本时报错，避免静默无操作却让模型以为修改已生效。
        if old_text not in text:
            return f"Error: text not found in {path}"
        # replace 的第三个参数 1 表示只替换第一处——若 old_text 在文件中多次出现，
        # 逐次修改更可控，也避免一次调用误伤文件中其他相同文本。
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(workdir: Path, pattern: str) -> str:
    # 在 workdir 下执行 glob 模式匹配（支持 *、? 等通配符），返回匹配结果列表。
    # 注意：未传 recursive=True，** 递归匹配不会生效。
    # 通过 is_relative_to 二次校验过滤掉通过 .. 或符号链接逃逸到 workdir 之外的路径。
    import glob as g

    try:
        results = []
        for match in g.glob(pattern, root_dir=workdir):
            # 二次校验：glob 结果可能经由 `..` 或符号链接指向 workdir 之外，
            # 这里和 safe_path 同样用 is_relative_to 过滤掉越界路径。
            if (workdir / match).resolve().is_relative_to(workdir):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# Anthropic tool-use 格式的工具定义列表，传给 client.messages.create 的 tools 参数。
# 每个工具需声明 name、description（帮助模型判断何时调用）和 input_schema（JSON Schema 格式）。
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


# tool name → handler 函数的映射表，agent_loop 根据模型返回的 block.name 查找对应处理函数。
# handler 签名为 (workdir: Path, **kwargs) -> str，kwargs 由模型在 tool_use 块中提供。
TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}
