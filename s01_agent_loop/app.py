import os
import subprocess
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


# 加载配置
load_dotenv()

# 初始化静态变量
MODEL = os.environ["MODEL_ID"]
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]

# 初始化 LLM Client
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


def run_bash(command: str) -> str:
    # 执行模型请求的 shell 命令，把结果（成功输出或错误信息）作为字符串返回。
    # 返回值会原样回填给模型，因此无论成功还是失败都返回字符串、不向上抛异常，
    # 让模型能读到错误并自行决定下一步，而不是让整个 REPL 崩溃。

    # 仅作演示的极简防护：用子串黑名单挡掉几条最常见的破坏性命令。
    # 注意这远不是真正的安全边界——诸如 `rm -rf ~`、多空格变体、`dd`、fork 炸弹等
    # 都能轻易绕过。生产环境必须改用沙箱／容器隔离来执行不可信命令。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(
            command,  # 要执行的命令
            shell=True,  # 通过系统 shell 解释命令（支持管道、通配符等）
            cwd=os.getcwd(),  # 指定这条命令在哪个目录下执行
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,  # 以字符串（而非 bytes）返回输出
            timeout=120,  # 超时保护：命令最多跑 120 秒，超时则抛 TimeoutExpired
        )
        # 合并标准输出与标准错误：模型同样需要看到报错内容才能判断成败。
        out = (r.stdout + r.stderr).strip()
        # 截断到 50000 字，避免超长输出撑爆上下文；空输出回一个占位符，
        # 以免模型把空字符串误读成「调用失败」。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        # 命令超时（如死循环、等待输入），返回提示而非让异常冒泡。
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        # 进程无法启动等系统级错误（例如 shell 不存在、资源不足）。
        return f"Error: {e}"


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
            return

        # 执行模型请求的每个工具调用，收集结果
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200])
                print()
                # tool_use_id 必须与请求一一对应，模型据此匹配结果
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )

        # 工具结果以 user 角色回填，进入下一轮让模型据此继续
        messages.append({"role": "user", "content": results})


def main() -> None:
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history_messages = []

    while True:
        # 获取输入
        try:
            query = input("\033[36ms01 >> \033[0m")
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
