"""
共享常量 —— 所有模块引用同一份配置，避免循环导入。

当前只包含工作目录 WORKDIR，工具层用它作为安全边界（safe_path 的
is_relative_to 检查），hooks 层用它展示上下文信息。
"""

from pathlib import Path

# 所有文件操作的安全根目录。
# resolve() 展开符号链接和相对路径，保证后续 is_relative_to 比较的可靠性。
WORKDIR = Path.cwd().resolve()
