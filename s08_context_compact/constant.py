"""
s08 Context Compact —— 全局常量与配置。

本模块在导入时计算所有路径常量，其余模块直接引用，避免散落魔术数字。
按类别分为三组：路径、LLM 配置、压缩阈值。
"""

import os
from pathlib import Path

import config


# =============================================================================
# 路径常量 —— 所有路径基于 WORKDIR 派生，确保工具层的安全边界一致
# =============================================================================

# resolve() 得到规范化的绝对路径：safe_path 内部的 is_relative_to 越界判断
# 要求两边都是真实路径，这里预先 resolve WORKDIR，工具层即可基于此构建可靠的安全边界。
WORKDIR = Path.cwd().resolve()

# skill 定义文件存放目录，list_skills / load_skill 从此处扫描
SKILLS_DIR = WORKDIR / "skills"

# 超长 tool_result 经 persist_large_output 持久化后写入此目录
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"

# compact_history / reactive_compact 将原始对话以 JSONL 格式写入此目录
TRANSCRIPT_DIR = WORKDIR / ".transcripts"


# =============================================================================
# LLM 配置
# =============================================================================

# 模型 ID，通过环境变量 MODEL_ID 注入，支持运行时切换不同模型
MODEL = os.getenv("MODEL_ID", "")

# 子 agent 的 system prompt（Workflow / Agent 工具分派子任务时使用）。
# 与 app.build_system() 构建的主 agent prompt 不同：子 agent 只需完成单一任务
# 并返回摘要，不需要 skill 目录等额外上下文。
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# =============================================================================
# 压缩阈值 —— 控制各压缩策略的触发条件和行为参数
# =============================================================================

# persist_large_output 的触发阈值（字符数）。
# 超过此长度的 tool_result 会被持久化到磁盘，上下文内仅保留路径 + 前 2000 字符预览。
# 30000 字符约 7500 token，对于大多数文件读取操作来说足够看到完整内容。
PERSIST_THRESHOLD = 30000

# micro_compact 保留的最近 tool_result 数量。
# 最近 3 条工具结果通常与当前任务直接相关（如刚读的文件、刚执行的命令），
# 更早的结果大概率已被模型消化。
KEEP_RECENT = 3

# 上下文大小上限（字符数），触发 compact_history 硬重启的阈值。
# agent_loop 每轮执行三层压缩后，若 estimate_size 仍超过此值则全量 LLM 摘要替代。
# 50000 字符约 12500 token，为大多数模型上下文窗口的保守下界。
CONTEXT_LIMIT = 50000
