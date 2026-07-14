"""
s08 Context Compact —— 带上下文压缩的编码 Agent REPL 应用。

架构概览：
  main()        → REPL 循环，读取用户输入，调用 agent_loop，打印模型回复
  agent_loop()  → 核心循环：压缩 → 调用 LLM → 执行工具 → 回填结果，直到模型停止请求工具
  build_system()→ 构建 system prompt，注入可用 skill 列表

上下文压缩管线（每轮 agent_loop 开始时依次执行）：
  tool_result_budget → 单轮 tool_result 总大小超限时，持久化最大的结果到磁盘
  snip_compact       → 消息数超限时，裁剪中间旧消息，保留头尾
  micro_compact      → 压缩早期 tool_result 内容为占位文本
  若仍超 CONTEXT_LIMIT → compact_history（LLM 摘要完全替代历史）
  若 API 报 prompt_too_long → reactive_compact（保留尾部 + LLM 摘要）
"""

# macOS 内置的 libedit（readline 的替代品）在处理多字节字符（如中文）
# 时存在退格删除和光标定位问题。以下四行 bind 配置将 libedit 切换到
# 8-bit 模式，使其把中文字节当作原始数据处理，从而修复输入体验。
# 如果环境中没有 readline（如 Windows），静默跳过。
try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

import config
from constant import WORKDIR, MODEL, CONTEXT_LIMIT
from llm_client import client
import skills
import tools
import hooks
import context


# 距离上次 todo_write 调用已经过的 agent_loop 轮数。
# 每轮 tool_use 循环 +1，todo_write 调用时清零。
# 连续 3 轮未更新任务列表时向模型注入提醒，防止模型沉浸在一连串
# 工具调用中忘记向用户同步进度。
rounds_since_todo = 0


def build_system() -> str:
    """构建 system prompt：声明 agent 身份，注入当前可用的 skill 目录。"""
    catalog = skills.list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )


# system prompt 在模块加载时一次性构建，后续所有 API 调用复用同一份
SYSTEM = build_system()

# prompt_too_long 时的最大重试次数：首次报错 → reactive_compact → 重试，
# 若仍报错则直接抛出，避免无限循环
MAX_REACTIVE_RETRIES = 1


def agent_loop(messages: list) -> None:
    """反复「调用模型 → 执行工具 → 回填结果」，直到模型不再请求工具为止。

    每轮检查是否需要注入 todo 提醒；退出前触发 Stop hook，允许外部注入
    追问消息并继续循环。
    """
    global rounds_since_todo
    reactive_retries = 0
    while True:
        # === 三层压缩管线：每轮先压缩再调用模型 ===
        # 第一层：单轮 tool_result 预算控制 → 持久化超长结果到磁盘
        messages[:] = context.tool_result_budget(messages)
        # 第二层：消息数裁剪 → 保留头尾，中间用占位消息替代
        messages[:] = context.snip_compact(messages)
        # 第三层：早期 tool_result 内容压缩 → 替换为占位文本
        messages[:] = context.micro_compact(messages)

        # 三层压缩后仍超限 → 全量 LLM 摘要替代
        if context.estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = context.compact_history(messages)

        # 连续 N 轮未调用 todo_write 时注入提醒。仅在模型已创建过任务列表
        # 的前提下才提醒，从未创建就不打扰。
        if rounds_since_todo >= 3 and messages and tools.CURRENT_TODOS:
            messages.append(
                {"role": "user", "content": "<reminder>Update your todos.</reminder>"}
            )
            rounds_since_todo = 0

        try:
            response = client.messages.create(
                model=MODEL,
                system=SYSTEM,
                messages=messages,
                tools=tools.TOOLS,
                max_tokens=8000,
            )
            reactive_retries = 0
        except Exception as e:
            # prompt_too_long → 上下文太长，API 拒绝请求。
            # 用 reactive_compact 保留尾部最近消息并压缩其余，然后重试。
            # 最多重试 MAX_REACTIVE_RETRIES 次，避免死循环。
            if (
                "prompt_too_long" in str(e).lower()
                or "too many tokens" in str(e).lower()
            ) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                messages[:] = context.reactive_compact(messages)
                reactive_retries += 1
                continue
            raise

        # 把模型本轮回复（可能含文本和 tool_use 块）原样追加进历史
        messages.append({"role": "assistant", "content": response.content})

        # 模型没有请求工具，说明本轮任务已给出最终回答，结束循环
        if response.stop_reason != "tool_use":
            # Stop hook 在循环退出前触发，返回非 None 时可注入一条 user 消息
            # 并 continue 继续循环，让模型基于该消息再产生回复。
            # 典型场景：自动追问（"需要我继续吗？"）、会话摘要注入等。
            force = hooks.trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        # 仅在 tool_use 轮次递增计数器。文本回复轮次（模型给出最终答案）
        # 不需要提醒——提醒只针对模型陷入一连串工具调用却忘记更新任务列表的场景。
        rounds_since_todo += 1

        # === 工具执行阶段 ===
        # 遍历模型返回的所有 tool_use 块，逐一执行并收集结果
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # hooks 工具执行前
            blocked = hooks.trigger_hooks("PreToolUse", block)
            if blocked:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(blocked),
                    }
                )
                continue

            if block.name == "compact":
                # compact 工具是特殊的元操作：直接压缩当前历史，然后 break 跳出
                # 本轮的工具循环，下一轮 agent_loop 迭代将用压缩后的历史继续
                messages[:] = context.compact_history(messages)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "[Compacted. Conversation history has been summarized.]",
                    }
                )
                messages.append({"role": "user", "content": results})
                break

            output = tools.use_tool(block.name, block.input)

            # PostToolUse hook：工具执行后触发，可用于日志记录、结果后处理等
            hooks.trigger_hooks("PostToolUse", block, output)

            # 模型主动调用了 todo_write，重置提醒计数器，避免重复注入
            if block.name == "todo_write":
                rounds_since_todo = 0

            # tool_use_id 必须与请求一一对应，模型据此匹配结果
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )
        else:
            # for...else：未被 break 中断 → 所有 tool_use 都执行完毕
            # 将收集到的 tool_result 以 user 角色回填，进入下一轮循环
            messages.append({"role": "user", "content": results})


def main() -> None:
    """REPL 入口：循环读取用户输入，交给 agent_loop 处理，打印模型最终回复。"""
    print("s08: Context Compact")
    print("输入问题，回车发送。输入 q 退出。\n")

    history_messages = []

    while True:
        try:
            query = input("\033[36ms08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        # 通知 hooks 用户已提交输入
        hooks.trigger_hooks("UserPromptSubmit", query)

        history_messages.append({"role": "user", "content": query})
        agent_loop(history_messages)

        # agent_loop 结束后，末尾必为 assistant 消息。
        # content 为内容块列表时只挑文本块展示（工具调用块由 run_todo_write 等函数
        # 内置的 print 在终端直接渲染）。
        response_content = history_messages[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()


if __name__ == "__main__":
    main()
