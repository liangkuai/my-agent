"""
s11 Error Recovery —— 全局常量与配置。

本模块在导入时计算所有路径常量，其余模块直接引用，避免散落魔术数字。
按类别分组：路径、LLM 配置、压缩阈值、记忆、错误恢复。
"""

import os
from pathlib import Path

import config


# =============================================================================
# 路径常量 —— 所有路径基于 WORKDIR 派生，确保工具层的安全边界一致
# =============================================================================

# resolve() 得到规范化的绝对路径：safe_path 内部的 is_relative_to 越界判断
# 依赖两边都是真实路径，这里先把根目录定死，工具层就有了可靠的安全边界。
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

# 备用模型 ID，由 FALLBACK_MODEL_ID 注入。
# 连续 529 过载超阈值时，with_retry() 自动切换到此模型继续重试，
# 避免长时间卡在过载的主模型上。未配置时仅重试不切换。
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")

# 单次 LLM 请求的默认 max_tokens。8000 对大多数单轮回答和中等规模工具
# 调用结果足够，同时控制 token 消耗在可预期范围内。
DEFAULT_MAX_TOKENS = 8000

# max_tokens 用尽后的提升值。app.agent_loop 在首次遇到 max_tokens
# stop_reason 时自动将 max_tokens 提升到此值，给模型更多空间完成输出。
ESCALATED_MAX_TOKENS = 64000

# 子 agent 的 system prompt（Workflow / Agent 工具分派子任务时使用）。
# 与 system_prompt.get_system_prompt() 构建的主 agent prompt 不同：子 agent 只需完成单一任务
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


# 记忆持久化目录。每条记忆一个 .md 文件（YAML frontmatter + Markdown 正文），
# 由 memory 模块统一管理读写与索引。
MEMORY_DIR = WORKDIR / ".memory"

# 记忆索引文件，位于 MEMORY_DIR 下。由 _rebuild_index() 自动维护，
# 内容为 Markdown 链接列表（如 "- [name](file.md) — description"）。
# 由 memory.read_memory_index() 读取，经 context → system_prompt 注入 system prompt。
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"

# 记忆合并触发阈值。当记忆文件数 >= 此值时，consolidate_memories()
# 调用 LLM 合并重复、删除过时条目，保持记忆库精简。
# 设为 10：积累约 10 条记忆后开始定期整理，避免过早合并。
CONSOLIDATE_THRESHOLD = 10


# =============================================================================
# 错误恢复配置 —— 控制 API 调用失败时的重试、续写和降级行为
# =============================================================================

# 单次查询中 max_tokens 续写的最大次数。
# 模型输出达 max_tokens 上限时，app.agent_loop 追加 CONTINUATION_PROMPT 让模型继续，
# 达到此次数后停止续写，避免无限追加。
MAX_RECOVERY_RETRIES = 3

# 追加给模型的续写提示，要求直接继续、不要道歉或复述。
# 放在 tool_result 之后作为新的 user 消息注入。
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)

# 单次 API 调用的最大重试次数。
# with_retry() 对 429/529 错误最多重试此轮数，超过后抛出 RuntimeError。
# 设为 10：足够覆盖数分钟的临时限流 / 过载恢复窗口。
MAX_RETRIES = 10

# 指数退避的基延迟（毫秒）。
# 第 n 次重试的等待时间 = min(500 * 2^n, 32000)ms + 随机抖动。
# 500ms 起始足够快、上限 32s 不至于等太久。
BASE_DELAY_MS = 500

# 连续 529 过载错误的降级阈值。
# 达到此次数后 with_retry() 切换到 FALLBACK_MODEL（若已配置），
# 避免长时间卡在过载的主模型上。
MAX_CONSECUTIVE_529 = 3


TASKS_DIR = WORKDIR / ".tasks"
