import os
from dotenv import load_dotenv
from anthropic import Anthropic

try:
    import readline

    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

from constant import WORKDIR
from tool import TOOLS, TOOL_HANDLERS
from hooks import trigger_hooks


# 加载配置
load_dotenv()

# 初始化静态变量
# resolve() 得到规范化的绝对路径：safe_path 内部的 is_relative_to 越界判断
# 依赖两边都是真实路径，这里先把根目录定死，工具层就有了可靠的安全边界。
MODEL = os.getenv("MODEL_ID", "")
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)

# 初始化 LLM Client
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


# 自上次 todo_write 调用以来已完成的 round 数。
# agent_loop 每轮 +1；todo_write 调用时清零。达到阈值时向模型注入提醒，
# 防止模型长时间不更新任务列表（例如沉浸在一连串 bash/read 调用中）。
rounds_since_todo = 0


def agent_loop(messages: list) -> None:
    """反复「调用模型 → 执行工具 → 回填结果」，直到模型不再请求工具为止。

    每轮检查是否需要注入 todo 提醒；退出前触发 Stop hook，允许外部注入
    追问消息并继续循环。
    """
    global rounds_since_todo
    while True:
        # 连续 N 轮未调用 todo_write 时，注入一条系统提醒催促模型更新任务列表
        if rounds_since_todo >= 3 and messages:
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

        # 每轮 rounds_since_todo += 1（追踪未调用 todo_write 的连续轮次）
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
            # print(f"\033[33m$ {block.name}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            # print(output[:200])
            # print()
            # hooks 工具执行后
            trigger_hooks("PostToolUse", block, output)  # s05: post hook

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
    """REPL 入口：循环读取用户输入，交给 agent_loop 处理，打印模型最终回复。"""
    print("s05: Todo Write")
    print("输入问题，回车发送。输入 q 退出。\n")

    history_messages = []

    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        # 通知 hooks 用户已提交输入（s05: 会话记录等）
        trigger_hooks("UserPromptSubmit", query)

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
