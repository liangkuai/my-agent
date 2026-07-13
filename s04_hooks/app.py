"""
s04 Hooks —— 带钩子系统的 AI 编码 Agent REPL。

在 agent_loop 的四个关键节点插入钩子，实现可扩展的拦截与观察：
  UserPromptSubmit → 用户输入提交后、进入 LLM 前
  PreToolUse       → 工具调用执行前（可拦截，返回非 None 即阻止执行）
  PostToolUse      → 工具调用执行后（纯观察，不干预流程）
  Stop             → 对话循环结束、向用户展示回复前

钩子由 hooks.py 统一注册和触发，权限检查由 permission.py 的三层管道实现。
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

try:
    import readline

    # macOS 默认使用 libedit（而非 GNU readline），在处理中文等多字节字符
    # 输入时会出现退格删除异常（一次退格只删半个字符导致乱码）。
    # 以下四行关闭 libedit 的 tty 特殊字符绑定并开启 meta 模式，
    # 让 readline 以字节流方式处理输入，从而修复中文退格问题。
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    # 非 macOS 或无 readline 的环境静默跳过，不影响核心功能
    pass

from constant import WORKDIR
from tool import TOOLS, TOOL_HANDLERS
from hooks import trigger_hooks


# ── 初始化配置 ─────────────────────────────────────────────────────────

load_dotenv()

# MODEL 从 .env 读取，缺省时 Anthropic SDK 会使用默认模型
MODEL = os.getenv("MODEL_ID", "")
# 系统提示词告知模型其角色与工作目录，模型可据此判断文件操作的上下文
SYSTEM = f"You are a coding agent at {WORKDIR}. All destructive operations require user approval."

# 初始化 LLM Client，base_url 从环境变量注入（支持自定义 API 端点）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


def agent_loop(messages: list) -> None:
    """反复「调用模型 → 执行工具 → 回填结果」，直到模型不再请求工具为止。

    每次循环：
    1. 调用 LLM，获取回复（可能包含 tool_use 块）
    2. 若模型不再请求工具（stop_reason != "tool_use"），触发 Stop 钩子后返回
    3. 否则依次处理每个 tool_use 块：PreToolUse → 执行 → PostToolUse
    4. 将工具结果以 user 角色回填，进入下一轮
    """
    while True:
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

        # 执行模型请求的每个工具调用，收集结果
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # PreToolUse 钩子：权限检查、日志记录；返回非 None 即拦截该次调用
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

            # PostToolUse 钩子：超大输出告警等事后观察，不干预流程
            trigger_hooks("PostToolUse", block, output)

            # tool_use_id 必须与请求一一对应，模型据此匹配结果
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )

        # 工具结果以 user 角色回填，进入下一轮让模型据此继续
        messages.append({"role": "user", "content": results})


def main() -> None:
    """REPL 主循环 —— 接收用户输入，驱动 agent_loop，展示回复。

    流程：
    1. 读取用户输入 → 触发 UserPromptSubmit 钩子
    2. 将输入追加到 history_messages（跨轮累积，保留完整会话上下文）
    3. 调用 agent_loop() 进入「调用模型 → 执行工具 → 回填结果」循环
    4. agent_loop 返回后，提取最后一条 assistant 消息中的文本块展示
    5. 回到步骤 1，直到用户输入退出指令或 EOF
    """
    print("s04: Hooks")
    print("输入问题，回车发送。输入 q 退出。\n")

    # 跨轮累积的完整对话历史，每次 agent_loop 在此基础上追加新的问答对
    history_messages = []

    while True:
        # 获取用户输入，Ctrl+D / Ctrl+C 优雅退出
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 空输入或退出指令 → 结束 REPL
        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        # UserPromptSubmit 钩子：在用户输入提交后、进入 LLM 前触发
        # 当前实现为 context_inject_hook，打印工作目录提示
        trigger_hooks("UserPromptSubmit", query)

        # 将用户输入追加到对话历史，然后驱动 agent_loop
        history_messages.append({"role": "user", "content": query})
        agent_loop(history_messages)

        # agent_loop 结束后，末尾必为 assistant 消息。
        # content 可能为纯文本字符串，也可能为内容块列表（含 tool_use 块），
        # 这里只挑出 text 类型的块展示给用户。
        response_content = history_messages[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()


if __name__ == "__main__":
    main()
