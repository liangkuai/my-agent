"""
s03_permission —— 带权限管道的 AI 编程代理 CLI。

本模块是整个应用的入口，负责：
1. 配置 readline（修复 macOS libedit 中文输入问题）
2. 加载环境变量并初始化 Anthropic 客户端
3. 实现 agent_loop：模型 ↔ 工具调用的主循环
4. 提供交互式 REPL（main 函数），用户输入问题后交由 agent_loop 处理

权限检查在 agent_loop 中通过 check_permission 管道完成（见 permission.py），
工具定义和处理函数来自 tool.py，工作目录来自 constant.py。
"""

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
from permission import check_permission


# 加载 .env 中的配置（MODEL_ID、ANTHROPIC_BASE_URL 等）
load_dotenv()

# 运行时常量：MODEL 和 SYSTEM 会在每次 agent_loop 中传给 API
MODEL = os.getenv("MODEL_ID", "")
SYSTEM = f"You are a coding agent at {WORKDIR}. All destructive operations require user approval."

# 初始化 Anthropic 客户端，BASE_URL 支持自定义代理 / 兼容 API
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


def agent_loop(messages: list) -> None:
    """运行「调用模型 → 执行工具 → 回填结果」循环, 直到模型给出最终回答。

    流程:
    1. 调用模型, 传入当前消息历史和可用工具列表
    2. 将模型回复 (文本 + tool_use 块) 追加到消息历史
    3. 若模型未请求工具 (stop_reason != "tool_use"), 循环结束
    4. 否则遍历每个 tool_use 块:
       a. 通过 check_permission 管道进行权限检查
       b. 被拒绝则回填 "Permission denied."
       c. 放行则调用对应的 TOOL_HANDLERS 执行
    5. 将工具结果以 user 角色回填, 回到步骤 1

    每次用户输入可能触发多轮工具调用, 本函数会一直循环直到模型
    不再请求工具为止。

    Args:
        messages: 消息历史列表, 每项为 {"role": ..., "content": ...},
                  本函数会在原地修改该列表 (追加 assistant 和 user 消息)。
    """
    # 反复调用模型 → 执行工具 → 回填结果, 直到模型不再请求工具为止
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

        # 执行模型请求的每个工具调用，收集结果
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"\033[33m$ {block.name}\033[0m")

            # 执行前运行权限管道
            if not check_permission(block):
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Permission denied.",
                    }
                )
                print()
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(output[:200])
            print()
            # tool_use_id 必须与请求一一对应，模型据此匹配结果
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )

        # 工具结果以 user 角色回填，进入下一轮让模型据此继续
        messages.append({"role": "user", "content": results})


def main() -> None:
    """启动交互式 REPL 主循环。

    用户在提示符下输入问题, 回车后交由 agent_loop 处理;
    输入 q / exit / quit 或空行退出程序。
    agent_loop 结束后打印模型的文本回复 (工具调用块已在循环内展示)。
    """
    print("s03: Permission")
    print("输入问题，回车发送。输入 q 退出。\n")

    history_messages = []

    while True:
        # 获取输入
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 校验输入
        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        # 追加输入
        history_messages.append({"role": "user", "content": query})
        agent_loop(history_messages)

        # 打印 LLM 最近一次输出（agent_loop 结束后，末尾必为 assistant 消息）
        # content 为内容块列表时，只挑出文本块展示给用户（工具调用块已在循环中打印）
        response_content = history_messages[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()


if __name__ == "__main__":
    main()
