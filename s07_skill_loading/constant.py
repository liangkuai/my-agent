"""
全局常量 —— 进程启动时一次性计算，后续只读。

所有模块（app/tool/hooks/skills/permission）都从本模块导入常量，
避免重复计算路径或在多处散落魔法字符串。
"""

import os
from pathlib import Path

import config  # 确保 load_dotenv 先于 os.getenv 执行

# ── 路径常量 ──────────────────────────────────────────────────────────
# WORKDIR 是整个 agent 的安全边界根目录。
# resolve() 消除符号链接和相对路径：safe_path() 的 is_relative_to 越界判断
# 依赖两边都是真实路径——这里先规范化，工具层就有了可靠的安全锚点。
WORKDIR = Path.cwd().resolve()

# SKILLS_DIR 是技能文件树的根目录，每个子目录为一个技能包。
# 技能扫描逻辑见 skills._scan_skills()。
SKILLS_DIR = WORKDIR / "skills"

# ── 模型配置 ──────────────────────────────────────────────────────────
# 从环境变量 MODEL_ID 读取模型标识符。若 .env 中未设置此项，
# 传给 API 的 model 参数为空字符串，调用时报错——检查 .env 是否包含 MODEL_ID=xxx。
MODEL = os.getenv("MODEL_ID", "")

# ── 子 agent 系统提示 ─────────────────────────────────────────────────
# 子 agent（由 tool.spawn_subagent 启动）使用独立的系统提示。
# "Do not delegate further." 是关键约束：防止子 agent 递归 spawn 孙 agent，
# 导致调用栈无限膨胀——子 agent 的工具集中不含 task 工具，加上提示强调，
# 双保险避免嵌套。
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)
