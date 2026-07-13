"""
全局常量 —— 存放整个应用共享的静态配置。

当前仅包含:
- WORKDIR: 工作目录根路径, 作为所有文件操作的安全边界。
  tool.py 中的 safe_path() 与 permission.py 中的 check_rules()
  均依赖此值来判断路径越界。
"""

from pathlib import Path


# 以当前进程的工作目录为基准, resolve() 消除符号链接和相对路径,
# 确保后续所有 is_relative_to 检查基于真实路径。
WORKDIR = Path.cwd().resolve()
