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
SYSTEM = f"You are a coding agent at {WORKDIR}. All destructive operations require user approval."

# 初始化 LLM Client
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


def agent_loop(messages: list) -> None:
    # 反复“调用模型 → 执行工具 → 回填结果”，直到模型不再请求工具为止
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
            # hooks 一轮对话循环即将退出时
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
            trigger_hooks("PostToolUse", block, output)  # s04: post hook

            # tool_use_id 必须与请求一一对应，模型据此匹配结果
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )

        # 工具结果以 user 角色回填，进入下一轮让模型据此继续
        messages.append({"role": "user", "content": results})


def main() -> None:
    print("s04: Hooks")
    print("输入问题，回车发送。输入 q 退出。\n")

    history_messages = []

    while True:
        # 获取输入
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # hooks 用户输入提交后、进入 LLM 前
        trigger_hooks("UserPromptSubmit", query)

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
