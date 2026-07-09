import os
from pathlib import Path

import config


# 初始化静态变量

# resolve() 得到规范化的绝对路径：safe_path 内部的 is_relative_to 越界判断
# 依赖两边都是真实路径，这里先把根目录定死，工具层就有了可靠的安全边界。
WORKDIR = Path.cwd().resolve()

MODEL = os.getenv("MODEL_ID", "")

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "For complex sub-problems, use the task tool to spawn a subagent."
)

SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)
