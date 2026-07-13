"""s02: Tool Use — Agent Loop 与工具调用

本模块实现带工具调用能力的 AI Agent 交互循环：
1. agent_loop — 反复「调用模型 → 执行工具 → 回填结果」直到模型给出最终回答
2. main   — 命令行交互入口，管理多轮对话历史与用户输入输出

核心流程：
  用户输入 → 拼入 messages → agent_loop() 驱动多轮 tool_use 往返 → 打印最终回答
"""

import os
from pathlib import Path
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

# 工具定义与 name → handler 映射，agent_loop 据此查找并执行对应的工具处理函数
from tool import TOOLS, TOOL_HANDLERS


# 加载 .env 配置文件（本模块依赖其中的 ANTHROPIC_BASE_URL、MODEL_ID 等）
load_dotenv()

# 初始化全局配置
# resolve() 得到规范化的绝对路径——safe_path 内的 is_relative_to 越界判断
# 要求两边都是真实路径，这里先定死根目录，工具层就有可靠的安全边界。
WORKDIR = Path.cwd().resolve()
MODEL = os.getenv("MODEL_ID", "")
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

# 初始化 LLM Client
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


def agent_loop(messages: list) -> None:
    """反复「调用模型 → 执行工具 → 回填结果」直到模型给出最终回答。

    每一轮循环：
    1. 调用 Anthropic API，传入当前 messages 与可用工具定义
    2. 将模型返回的 assistant 消息原样追加进历史（含 text 与 tool_use 块）
    3. 若 stop_reason 不是 "tool_use"，说明模型已完成任务，退出循环
    4. 否则遍历 tool_use 块，调用对应 handler 执行，收集 tool_result
    5. 将 tool_result 以 user 角色回填历史，进入下一轮
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
            return

        # 遍历模型返回的 tool_use 块，逐一执行并收集结果
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.name}\033[0m")
                handler = TOOL_HANDLERS.get(block.name)
                output = (
                    handler(WORKDIR, **block.input)
                    if handler
                    else f"Unknown: {block.name}"
                )
                print(output[:200])
                print()
                # tool_use_id 必须与对应的 tool_use 块一一匹配，模型据此关联结果
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )

        # 工具结果以 user 角色回填，进入下一轮让模型据此继续
        messages.append({"role": "user", "content": results})


def main() -> None:
    """命令行交互入口，管理多轮对话历史与用户输入输出。

    - 每轮对话将用户输入追加到 history_messages，交给 agent_loop 驱动工具往返
    - agent_loop 结束后，从历史末尾取出模型的文本回复展示给用户
    - 输入 q / exit / quit 或 Ctrl-C / Ctrl-D 退出
    """
    print("s02: Tool Use")
    print("输入问题，回车发送。输入 q 退出。\n")

    # 多轮对话历史：跨 agent_loop 调用累积，模型能记住之前的所有交互
    history_messages = []

    while True:
        # 获取输入
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 校验输入：空行或退出关键词则结束程序
        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        # 追加用户输入到对话历史，交给 agent_loop 驱动工具往返
        history_messages.append({"role": "user", "content": query})
        agent_loop(history_messages)

        # agent_loop 结束后，最新一条消息必为 assistant 回复
        # content 可能是纯字符串或内容块列表（含 text / tool_use），
        # 这里只挑出文本块展示给用户；工具调用过程已在 agent_loop 中打印
        response_content = history_messages[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()


if __name__ == "__main__":
    main()
