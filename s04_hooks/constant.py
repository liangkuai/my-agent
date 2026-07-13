"""
全局常量 —— 跨模块共享的不可变配置。

WORKDIR 被以下模块共同引用：
  tool.py       — 工具执行的工作目录与安全边界
  hooks.py      — UserPromptSubmit 钩子中展示当前工作目录
  permission.py — 文件写入越界检测的参照根目录
"""

from pathlib import Path


# 工作目录 —— agent 所有文件操作的安全根目录。
# resolve() 将相对路径展开为绝对路径，消除符号链接和 `..`，
# 确保 is_relative_to 越界判断拿到的是真实路径而非字面字符串。
WORKDIR = Path.cwd().resolve()
