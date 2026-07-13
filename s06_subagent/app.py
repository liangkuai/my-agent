"""
主入口 —— REPL 交互循环与 agent 工具调用循环。

运行方式：`python app.py`，启动后在终端与 agent 对话，输入 q 退出。

架构分层（自顶向下）：
1. main()        → REPL：读用户输入 → 调 agent_loop → 打印回复
2. agent_loop()  → 核心循环：调模型 → 执行工具 → 回填结果，直到模型不再请求工具
3. tool.py       → 工具实现（bash / read / write / edit / glob / todo / task）
4. hooks.py      → 钩子系统（日志 / 权限 / 超大输出告警 / 会话统计）
5. permission.py → 权限管道（拒绝列表 → 规则检查 → 用户交互确认）

导入顺序保证（config → constant → llm_client → tool / hooks → app）：
config.py 在最顶部被导入时执行 load_dotenv，后续模块才能读到环境变量。
"""

try:
    import readline

    # macOS 的 libedit 在处理中文等多字节字符输入时存在退格只能删半个字符
    # 的问题。下面四条 bind 指令修复该行为：关闭终端特殊字符绑定、开启 8-bit
    # 输入输出、关闭 meta 位转换——让 readline 正确按 UTF-8 字节序列处理。
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    # Windows 或未安装 readline 的环境下静默跳过，不影响功能。
    pass

import config  # noqa: F401 — 保证 load_dotenv 最先执行
from constant import MODEL, SYSTEM
from llm_client import client
from tool import TOOLS, TOOL_HANDLERS, CURRENT_TODOS
from hooks import trigger_hooks


# ── todo 提醒计数器 ───────────────────────────────────────────────────
# 自上次 todo_write 调用以来已完成的工具调用 round 数。
# agent_loop 中每轮 tool_use 后 +1；todo_write 调用时清零。
# 达到阈值（3 轮）时向模型注入提醒消息，防止模型长时间沉浸在一连串
# bash/read/write 调用中却忘记更新任务列表状态。
rounds_since_todo = 0


def agent_loop(messages: list) -> None:
    """反复「调用模型 → 执行工具 → 回填结果」，直到模型不再请求工具为止。

    这是 s06 的核心循环，每轮做什么：
    1. 检查是否需要向模型注入 todo 提醒（连续 3 轮 tool_use 未调 todo_write）。
    2. 调用 Anthropic Messages API，传入当前对话历史和工具定义。
    3. 把模型本轮回复（文本 + tool_use 块）原样追加到 messages。
    4. 若 stop_reason 不是 tool_use → 模型已给出最终回答 → 触发 Stop hook
       → 若 hook 注入追问消息则 continue 继续，否则 return 退出。
    5. 若 stop_reason 是 tool_use → 逐一执行工具：
       a. PreToolUse hook（日志 + 权限拦截）
       b. 从 TOOL_HANDLERS 查找 handler 并执行
       c. PostToolUse hook（超大输出告警）
       d. 若执行了 todo_write 则重置提醒计数器
    6. 将 tool_result 列表作为 user 消息回填 → 回到步骤 1。

    退出条件：
    - 模型返回 stop_reason != "tool_use"
    - 且 Stop hook 未注入新的用户消息

    Args:
        messages: 完整的对话历史列表（含 system 之外的所有 role），
                  本函数会原地修改该列表。
    """
    global rounds_since_todo
    while True:
        # 连续 N 轮未调用 todo_write 时注入提醒。仅在模型已创建过任务列表
        # 的前提下才提醒，从未创建就不打扰。
        if rounds_since_todo >= 3 and messages and CURRENT_TODOS:
            messages.append(
                {"role": "user", "content": "<reminder>Update your todos.</reminder>"}
            )
            rounds_since_todo = 0

        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        # 把模型本轮回复（可能含文本和 tool_use 块）原样追加进历史
        messages.append({"role": "assistant", "content": response.content})

        # 模型没有请求工具，说明本轮任务已给出最终回答，结束循环
        if response.stop_reason != "tool_use":
            # Stop hook 在循环退出前触发，返回非 None 时可注入一条 user 消息
            # 并 continue 继续循环，让模型基于该消息再产生回复。
            # 典型场景：自动追问（"需要我继续吗？"）、会话摘要注入等。
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        # 仅在 tool_use 轮次递增计数器。文本回复轮次（模型给出最终答案）
        # 不需要提醒——提醒只针对模型陷入一连串工具调用却忘记更新任务列表的场景。
        rounds_since_todo += 1

        # 执行模型请求的每个工具调用，收集结果
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # hooks 工具执行前
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(blocked),
                    }
                )
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"

            # hooks 工具执行后
            trigger_hooks("PostToolUse", block, output)

            # 模型主动调用了 todo_write，重置提醒计数器，避免重复注入
            if block.name == "todo_write":
                rounds_since_todo = 0

            # tool_use_id 必须与请求一一对应，模型据此匹配结果
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )

        # 工具结果以 user 角色回填，进入下一轮让模型据此继续
        messages.append({"role": "user", "content": results})


def main() -> None:
    """REPL 入口：循环读取用户输入，交给 agent_loop 处理，打印模型最终回复。

    运行流程：
    1. 打印欢迎信息。
    2. 维护 history_messages 列表，跨多轮查询累积（模型能记住之前的对话）。
    3. 每次用户输入后：
       a. 触发 UserPromptSubmit hook（打印工作目录等上下文）
       b. 追加 user 消息到 history_messages
       c. 调用 agent_loop（模型推理 + 工具执行闭环）
       d. agent_loop 返回后，打印模型的最终文本回复
    4. 输入 q / exit / quit 或 EOF / Ctrl+C 退出。

    注意：history_messages 的生命周期跨越整个 REPL session，因此 hooks
    中的 summary_hook 统计的是 session 累计工具调用数，而非单次查询。
    """
    print("s06: Subagent")
    print("输入问题，回车发送。输入 q 退出。\n")

    # 跨查询共享的对话历史；agent_loop 在原列表上追加，不回传新列表。
    history_messages = []

    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 空输入和退出指令均退出 REPL
        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        # 通知 hooks 用户已提交输入（当前仅 context_inject_hook 打印工作目录）
        trigger_hooks("UserPromptSubmit", query)

        history_messages.append({"role": "user", "content": query})
        agent_loop(history_messages)

        # agent_loop 结束后，末尾必为 assistant 消息。
        # content 为内容块列表时只挑文本块展示——工具调用块（如 todo 面板）
        # 已由 run_todo_write 内置的 print 在终端直接渲染，这里不再重复输出。
        response_content = history_messages[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()


if __name__ == "__main__":
    main()
