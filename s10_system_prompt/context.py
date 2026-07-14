"""
上下文压缩模块 —— 在对话历史过长时，通过裁剪、压缩、持久化、摘要等手段控制 token 消耗。

每条消息以 role: user/assistant 交替排列，assistant 消息的 content 为内容块列表
（text / tool_use），user 消息可能包含 tool_result 块。压缩的核心挑战是在不破坏
tool_use ↔ tool_result 配对的前提下减少上下文体积。

压缩策略由轻到重，app.agent_loop 每轮依次执行：
  1. tool_result_budget → 单轮 tool_result 总大小超出 max_bytes 时，贪心持久化最大的结果到磁盘
  2. snip_compact       → 消息数超出 max_messages 时，裁剪中间旧消息保留头尾（保持配对完整）
  3. micro_compact      → 保留最近 KEEP_RECENT 条 tool_result，其余长内容替换为占位文本
  4. compact_history    → 全部历史写入转录文件，用 LLM 摘要完全替代上下文（"硬重启"）
  5. reactive_compact   → 保留尾部约 5 条消息，其余用 LLM 摘要替代（API 报 prompt_too_long 时触发）

工具函数：
  estimate_size / collect_tool_results / persist_large_output / write_transcript / summarize_history
"""

import time
import json
from typing import Any
from pathlib import Path

import constant
from llm_client import client
import tools


def estimate_size(msgs: list):
    """粗略估算消息列表的字符串长度，用于判断是否接近上下文窗口上限。

    直接对列表做 str() 而非 json.dumps()：速度更快，且对大小比较来说精度足够。
    实际 token 数与此成正比，不需要精确计数。
    """
    return len(str(msgs))


def _block_type(block: Any):
    """获取 content 块（dict 或对象）的 type 字段，兼容两种表示形式。

    消息历史中的 content 元素可能是 dict（序列化后）或 SDK 返回的富对象
    （如 ToolUseBlock），这里统一取 type 字段以消除调用方的心智负担。
    """
    return (
        block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
    )


def _message_has_tool_use(msg: dict) -> bool:
    """判断一条 assistant 消息中是否包含 tool_use 块。"""
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(block) == "tool_use" for block in content)


def _is_tool_result_message(msg: dict) -> bool:
    """判断一条 user 消息的 content 中是否包含至少一个 tool_result 块。"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def snip_compact(messages: list, max_messages: int = 50) -> list:
    """裁剪压缩：保留前 3 条和最近的 N-3 条消息，中间部分用占位消息替代。

    为什么要保留前 3 条：通常包含 system prompt 的等效信息（身份声明、规则等），
    模型需要这些上下文才能正确理解后续对话。

    边界处理 —— 避免切断 tool_use ↔ tool_result 配对：
    - 头部：若裁剪边界前一条是 tool_use，则向后吞入后续的 tool_result 直到完整。
    - 尾部：若裁剪边界恰好在 tool_result 上且前一条是其 tool_use，则向前扩展以保留配对。
      否则模型会因为 tool_result 找不到对应的 tool_use 而产生困惑。

    Returns:
        裁剪后的消息列表（原地修改了中间部分，但返回新列表）。
    """
    if len(messages) <= max_messages:
        return messages
    keep_head, keep_tail = 3, max_messages - 3
    head_end, tail_start = keep_head, len(messages) - keep_tail

    # 头部边界修正：确保不把 tool_use 和它的 tool_result 拆开
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1

    # 尾部边界修正：确保不把 tool_result 和它的 tool_use 拆开
    if (
        tail_start > 0
        and tail_start < len(messages)
        and _is_tool_result_message(messages[tail_start])
        and _message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1

    # 如果修正后头尾重叠或无空间裁剪，原样返回
    if head_end >= tail_start:
        return messages
    snipped = tail_start - head_end
    return (
        messages[:head_end]
        + [{"role": "user", "content": f"[snipped {snipped} messages]"}]
        + messages[tail_start:]
    )


def collect_tool_results(messages: list) -> list:
    """收集消息列表中所有 tool_result 块的位置信息。

    Returns:
        list[tuple[int, int, dict]]: 每个元素为 (消息索引, 块索引, tool_result 块字典)。
    """
    blocks = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))
    return blocks


def micro_compact(messages: list) -> list:
    """微压缩：保留最近 KEEP_RECENT 条 tool_result，其余长内容用占位文本替换。

    目的：减少早期工具调用结果占用的上下文空间，同时保留最近结果的可用性。
    最近的结果往往与当前任务相关（模型刚读了某个文件、刚执行了某个命令），
    而早期结果通常已被模型消化并体现在后续行为中。

    120 字符阈值：短结果（如 "文件已写入"）本身占用很小，替换为占位文本反而
    更占空间，所以只压缩超过该阈值的结果。
    """
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= constant.KEEP_RECENT:
        return messages
    # 对最近 N 条之外的 tool_result，若内容较长则替换为占位文本
    for _, _, block in tool_results[:-constant.KEEP_RECENT]:
        if len(block.get("content", "")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


def persist_large_output(tool_use_id: str, output: str) -> str:
    """将超长 tool_result 写入磁盘文件，返回指向该文件的摘要占位文本。

    超过 PERSIST_THRESHOLD（默认 30000 字符）的输出不再在上下文中完整保留，
    改为写入 TOOL_RESULTS_DIR 下的 {tool_use_id}.txt。

    幂等性：if not path.exists() 防止重复写入同一结果（同一 tool_use_id 可能
    被 budget 控制多次尝试持久化）。

    返回的占位文本包含文件路径 + 前 2000 字符预览，模型看到此文本后可选择
    用读文件工具获取完整内容。
    """
    if len(output) <= constant.PERSIST_THRESHOLD:
        return output
    constant.TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = constant.TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists():
        path.write_text(output)
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"


def tool_result_budget(messages: list, max_bytes: int = 200_000) -> list:
    """对单轮对话中的 tool_result 做预算控制：总大小超出限制时，贪心地持久化最大的结果。

    为什么只检查最后一条消息：一轮 agent_loop 可能同时执行多个工具（如并行读取
    多个文件），所有 tool_result 打包在同一条 user 消息中回填。如果这条消息的
    tool_result 总量过大，即使消息数未超限也可能撑爆上下文窗口。

    贪心策略：按内容大小降序排列，优先持久化最大的结果以最快速缩减总量。
    只持久化超过 PERSIST_THRESHOLD 的结果 —— 小结果持久化不划算（文件 IO +
    占位文本可能比原内容还大）。
    """
    last = messages[-1] if messages else None
    if (
        not last
        or last.get("role") != "user"
        or not isinstance(last.get("content"), list)
    ):
        return messages

    blocks = [
        (i, b)
        for i, b in enumerate(last["content"])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes:
        return messages

    # 按内容大小降序排列，优先压缩最大的结果以最快速缩减总量
    ranked = sorted(
        blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True
    )
    for _, block in ranked:
        if total <= max_bytes:
            break
        content = str(block.get("content", ""))
        if len(content) <= constant.PERSIST_THRESHOLD:
            continue
        tid = block.get("tool_use_id", "unknown")
        block["content"] = persist_large_output(tid, content)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)

    return messages


def write_transcript(messages: list) -> Path:
    """将完整消息历史以 JSONL 格式写入转录文件，用于事后回溯或调试。

    使用 json.dumps(..., default=str) 处理 SDK 返回的富对象（它们可能
    包含不可直接序列化的类型），降级为 str() 而非抛出异常。
    文件名带时间戳以便按时间排序。
    """
    constant.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = constant.TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    """调用 LLM 将对话历史总结为一段紧凑的摘要文本。

    截取前 80000 字符作为输入（约 20000 token，为大多数模型的合理输入窗口），
    指示 LLM 保留五个关键维度：当前目标、关键发现/决策、已读/已修改的文件、
    待完成工作、用户约束。

    max_tokens=2000：摘要输出足够紧凑，且不会显著增加后续轮次的上下文负担。
    """
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
        "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n"
        + conversation
    )
    response = client.messages.create(
        model=constant.MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000
    )
    return (
        "\n".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        or "(empty summary)"
    )


def compact_history(messages: list) -> list:
    """全量压缩（硬重启）：将全部历史写入转录文件，用 LLM 摘要完全替代上下文。

    这是最重的压缩手段。返回的新消息列表中只有一条 user 消息，相当于
    告诉模型"之前的对话已经总结完毕，请基于摘要继续工作"。

    触发条件：三层常规压缩（tool_result_budget → snip → micro）后
    估算大小仍超过 CONTEXT_LIMIT。
    """
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


def reactive_compact(messages: list) -> list:
    """响应式压缩：保留最近约 5 条消息的原始内容，其余用 LLM 摘要替代。

    触发场景：API 返回 prompt_too_long 错误时调用（app.agent_loop 的 except 分支）。

    与 compact_history 的关键区别：不丢弃全部历史，而是保留尾部最近交互的
    完整性。这样模型可以看到"刚才在做什么"——包括可能正在进行中的
    tool_use ↔ tool_result 配对，从而在压缩后继续当前任务而非从头开始。

    尾部边界修正：若保留的起始位置恰好切断了一对 tool_use/tool_result，
    向前扩展以保持配对完整。
    """
    transcript = write_transcript(messages)
    tail_start = max(0, len(messages) - 5)
    # 尾部边界修正：避免切断 tool_use ↔ tool_result 配对
    if (
        tail_start > 0
        and tail_start < len(messages)
        and _is_tool_result_message(messages[tail_start])
        and _message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1
    summary = summarize_history(messages[:tail_start])
    return [
        {"role": "user", "content": f"[Reactive compact]\n\n{summary}"},
        *messages[tail_start:],
    ]


def update_context(context: dict, messages: list) -> dict:
    memories = ""
    if constant.MEMORY_INDEX.exists():
        content = constant.MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": tools.list_tool_name(),
        "workspace": str(constant.WORKDIR),
        "memories": memories,
    }
