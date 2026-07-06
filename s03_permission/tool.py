import subprocess
from pathlib import Path

from constant import WORKDIR


def run_bash(command: str) -> str:
    # 执行模型请求的 shell 命令，把结果（成功输出或错误信息）作为字符串返回。
    # 返回值会原样回填给模型，因此无论成功还是失败都返回字符串、不向上抛异常，
    # 让模型能读到错误并自行决定下一步，而不是让整个 REPL 崩溃。

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
    # 把模型给的相对路径解析成绝对路径，并确保它仍落在工作目录内，防止目录穿越。
    # resolve() 会展开 `..` 和符号链接，所以像 `../../etc/passwd` 这类越权路径
    # 在这一步会暴露真实位置；随后用 is_relative_to 拦掉所有逃出 WORKDIR 的路径。

    # 先把 WORKDIR 也 resolve 一次：is_relative_to 是纯字符串前缀比较，只有当两边
    # 都是规范化的真实路径时结果才可靠。这样本函数作为安全边界就不再依赖调用方
    # 「恰好传进来一个已 resolve 的目录」这一隐含前提。
    workdir = Path(WORKDIR).resolve()
    path = (workdir / p).resolve()
    if not path.is_relative_to(workdir):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: int | None = None) -> str:
    # 与 run_bash 同理：所有文件类工具都把异常转成 "Error: ..." 字符串返回给模型，
    # 而不是向上抛出——让模型读到错误并自行重试，避免单次工具失败拖垮整个循环。
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
    try:
        file_path = safe_path(path)
        # 自动补建缺失的父目录，这样模型写新文件时不必先手动创建目录。
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
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
