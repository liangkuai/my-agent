"""
System prompt 组装模块 —— 将静态模板片段与运行时上下文拼接，生成发给 LLM 的系统提示。

设计要点：
1. 带缓存的组装：将 context 序列化为规范化 JSON 串，与上次快照逐字比对；
   仅在 context 变化时才重建 prompt，否则直接返回缓存。
2. 动态段落：tools / workspace / memories 三段内容来自 context 字典（由
   context.update_context() 填充），而非硬编码——工具列表从 TOOLS 定义
   自动提取（含 compact 等特殊拦截工具），工作目录和记忆索引也反映运行时实际状态。
3. 按需注入：memories 为空时不追加记忆段落，避免空字符串占用 prompt 空间。
4. 显式拼接顺序：assemble_system_prompt 按 identity → tools → workspace →
   (可选) memories 的固定顺序拼接，新增段落需同时修改本函数和
   context.update_context()，不是自动发现——这样更可控，且一眼就能看清段落顺序。

对外接口：
  get_system_prompt(context) → str
    agent_loop 每轮调用，返回系统提示字符串（缓存命中时跳过重建）。
"""

import json

import constant


# ── 静态模板 ───────────────────────────────────────────────────────────
# 仅保留 identity 为静态片段（不随运行时变化）。tools / workspace / memories
# 三段内容由 assemble_system_prompt 从 context 字典中动态取值并格式化。

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {constant.WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    """将静态模板与 context 中的运行时信息拼接为完整的 system prompt 字符串。

    拼接顺序固定：identity → tools → workspace → (可选) memories。

    context 由 context.update_context() 构建，包含三个字段：
    - enabled_tools: list[str] — 已注册工具名，由 tools.list_tool_name() 提供
    - workspace: str — 当前工作目录的绝对路径
    - memories: str — MEMORY.md 索引内容（无记忆时为空字符串 ""）
    """
    sections = []

    # 静态段落：身份声明，每次必有
    sections.append(PROMPT_SECTIONS["identity"])

    # 动态段落：工具列表、工作目录 —— 从 context 取值，反映运行时实际状态
    tools_list = context.get("enabled_tools", [])
    sections.append(f"Available tools: {', '.join(tools_list)}.")
    sections.append(f"Working directory: {context.get('workspace', str(constant.WORKDIR))}")

    # 可选段落：记忆索引 —— 仅在记忆库非空时注入
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")

    # 双换行分隔：让模型清楚地感知段落边界，降低混淆风险
    return "\n\n".join(sections)


# ── 缓存 ───────────────────────────────────────────────────────────────
# context 字典中 enabled_tools 来自 tools.TOOLS 列表（静态工具定义），
# workspace 来自 WORKDIR（常量），仅 memories 在记忆索引文件更新时变化。
# 因此绝大多数 agent_loop 迭代中 context 不变，缓存命中率极高。
# 两个模块级变量分别保存上一次的 context 快照和生成的 prompt 结果。
#
# 缓存失效条件：MEMORY_INDEX 文件内容变化（提取或合并记忆后索引重建）。

_last_context_key = None   # 上一次 context 序列化后的规范化 JSON 字符串
_last_prompt = None        # 上一次生成的 system prompt 完整文本


def get_system_prompt(context: dict) -> str:
    """获取或重建 system prompt（带缓存）。

    每次调用时先对 context 做规范化 JSON 序列化，与上次快照比较：
    - 相同 → 缓存命中，直接返回上次结果（打印灰色提示）
    - 不同 → 重建 prompt，更新缓存（打印绿色提示列出加载的段落）

    sort_keys=True 消除字典 key 遍历顺序差异导致的假阴性；
    default=str 处理 Path、datetime 等非 JSON 原生类型。
    """
    global _last_context_key, _last_prompt

    # 规范化 context 为稳定字符串：排序 key + 非标量降级为 str()
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)

    if key == _last_context_key and _last_prompt:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt

    # 缓存未命中 → 重建
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)

    # 打印已加载的段落列表，方便调试时确认 prompt 结构
    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt
