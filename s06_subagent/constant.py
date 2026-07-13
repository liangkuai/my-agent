"""
全局常量 —— 工作目录、模型名和系统提示词。

本模块在 config.py 之后被导入，依赖 `load_dotenv()` 已将 .env 注入 os.environ。

设计原则：
- WORKDIR 是整个应用的「安全根」—— 所有文件操作必须落在该目录内。
- SYSTEM 与 SUB_SYSTEM 是主/子 agent 各自的系统提示词，分别定义了它们的
  角色边界和可用能力。
"""

import os
from pathlib import Path

import config  # noqa: F401 — 保证 load_dotenv 在本模块之前执行

# ── 工作目录 ──────────────────────────────────────────────────────────
# resolve() 得到规范化的绝对路径：safe_path 内部的 is_relative_to 越界判断
# 依赖两边都是真实路径，这里先把根目录定死，工具层就有了可靠的安全边界。
# 后续所有路径操作都在这个目录的子树内完成。
WORKDIR = Path.cwd().resolve()

# ── 模型标识 ──────────────────────────────────────────────────────────
# 从环境变量 MODEL_ID 读取；空字符串表示未配置，llm_client 会回退到 SDK 默认值。
MODEL = os.getenv("MODEL_ID", "")

# ── 系统提示词 ────────────────────────────────────────────────────────
# 主 agent 提示词：定位为 coding agent，引导其对复杂子问题使用 task 工具。
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "For complex sub-problems, use the task tool to spawn a subagent."
)

# 子 agent 提示词：定位为一次性执行者，完成指定子任务后返回摘要，严禁再委托。
# "Do not delegate further" 防止子 agent 递归 spawn 孙子 agent，避免无限嵌套。
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)
