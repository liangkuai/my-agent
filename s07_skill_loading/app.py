"""
s07: Skill Loading —— 带技能系统的 CLI coding agent。

本模块是程序的入口，负责：
1. 组装系统提示（含技能目录）并启动 REPL 交互循环
2. 驱动 agent_loop：发送消息 → 执行工具 → 回填结果，直到模型给出最终回复
3. 管理 todo 提醒节奏，避免模型长时间不更新任务列表

架构关系：
    app.py          ← 主循环 + 消息管理（本文件）
    tool.py         ← 工具定义与实现
    hooks.py        ← 事件钩子（权限、日志、注入）
    skills.py       ← 技能扫描与注册
    permission.py   ← 权限管道（拒绝列表 → 规则 → 确认）
    constant.py     ← 全局常量
    llm_client.py   ← LLM client 初始化
    config.py       ← .env 加载
"""

try:
    import readline

    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    # Windows 或未安装 readline 的环境：不影响功能，仅失去行编辑能力
    pass

import config  # 必须在所有 os.getenv 调用前加载
from constant import WORKDIR, MODEL
from skills import list_skills
from llm_client import client
from tool import TOOLS, TOOL_HANDLERS, CURRENT_TODOS
from hooks import trigger_hooks


# 自上次 todo_write 调用以来经过的 agent_loop 轮数。
# agent_loop 每轮（tool_use）递增 1；todo_write 调用时清零。达到阈值 3 时
# 向模型注入一条 user 提醒，防止模型沉浸在一连串工具调用中忘记更新任务列表。
# 只在模型已创建过 todo（CURRENT_TODOS 非空）时才提醒——从未创建就不打扰。
rounds_since_todo = 0


def build_system() -> str:
    """组装系统提示：注入工作目录上下文和已安装的技能目录。

    list_skills() 从内存注册表读取，新增技能文件后需重新调用本函数才能
    反映到系统提示中——当前实现里 SYSTEM 在模块导入时一次性固化，
    要动态更新需改为每次 agent_loop 调用时重新构建。
    """
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )


# 模块导入时固化系统提示。
SYSTEM = build_system()


def agent_loop(messages: list) -> None:
    """反复「调用模型 → 执行工具 → 回填结果」，直到模型不再请求工具为止。

    每轮检查是否需要注入 todo 提醒；退出前触发 Stop hook，允许外部注入
    追问消息并继续循环（例如：自动追问"需要我继续吗？"）。
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

        # 把模型本轮回复（可能含 text + tool_use 混合块）原样追加进历史。
        # 注意：SDK 的 response.content 是 ContentBlock 对象列表而非 list[dict]，
        # 但 Anthropic API 的 messages 字段接受对象和字典两种形式。
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

            # PreToolUse hook：权限检查、日志记录（见 hooks.py）。
            # 若 hook 返回非 None 字符串，跳过真实执行，将字符串作为
            # 拒绝原因回填给模型。
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

            # PostToolUse hook：超大输出告警、审计日志等（见 hooks.py）。
            # 始终不拦截——此时工具已执行完毕，hook 只做观察。
            trigger_hooks("PostToolUse", block, output)

            # 模型主动调用了 todo_write，重置提醒计数器，避免重复注入
            if block.name == "todo_write":
                rounds_since_todo = 0

            # tool_use_id 必须与请求一一对应，模型据此匹配结果
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )

        # 工具结果以 user 角色回填，进入下一轮让模型据此继续推导
        messages.append({"role": "user", "content": results})


def main() -> None:
    """REPL 入口：循环读取用户输入，交给 agent_loop 处理，打印模型最终回复。

    交互约定：
    - 空行 / q / exit / quit → 退出
    - EOF / Ctrl+C              → 退出
    - 其他文本                   → 作为 user 消息发给模型

    history_messages 在整个 session 期间持续累积，不会被清空——
    后续每轮查询都能看到之前所有的工具调用和回复，实现上下文连续。
    """
    print("s07: Skill Loading")
    print("输入问题，回车发送。输入 q 退出。\n")

    history_messages = []

    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        # 通知 hooks 用户已提交输入
        trigger_hooks("UserPromptSubmit", query)

        history_messages.append({"role": "user", "content": query})
        agent_loop(history_messages)

        # agent_loop 结束后，末尾必为 assistant 消息。
        # content 为 ContentBlock 列表时只挑 text 块展示：
        # tool_use 块由 run_todo_write 等 handler 中内置的 print 直接渲染，
        # 不需要再打印它们的文本表示。
        response_content = history_messages[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()


if __name__ == "__main__":
    main()
