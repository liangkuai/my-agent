"""
s01: Agent Loop —— 一个最简的 AI 编码助手 REPL。

核心流程：
1. 用户在终端输入自然语言任务
2. 将任务发送给 Claude（带 bash 工具定义）
3. Claude 返回文本回复或 tool_use（bash 命令）
4. 本程序在本地执行 bash 命令，将结果回填给模型
5. 重复步骤 3-4 直到模型给出最终文本回复，展示给用户，等待下一个输入

这是一个教学性质的实现，展示了 agentic loop 的基本原理。
"""

import os
import subprocess
from dotenv import load_dotenv
from anthropic import Anthropic

# —— readline 配置（仅用于改善 macOS 终端中文输入体验） ——
try:
    import readline

    # macOS 自带的 libedit 在处理中文输入时有退格异常（吃掉两个字符等），
    # 以下四行关闭 libedit 的 tty 特殊字符绑定，并开启 8-bit 元字符处理，
    # 从而修复中文输入下的退格行为。
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    # Windows / 非标准环境可能没有 readline，忽略即可
    pass


# =============================================================================
# 配置与常量
# =============================================================================

# 从 .env 文件加载环境变量（MODEL_ID、ANTHROPIC_BASE_URL 等）
load_dotenv()

# 模型 ID，例如 "claude-sonnet-4-5-20251001"
MODEL = os.environ["MODEL_ID"]

# 系统提示词：
# - 告诉模型「你是一个编码助手，当前工作目录是什么」
# - "Act, don't explain" 要求模型直接行动（调工具），而非长篇解释
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# 工具定义列表（目前仅暴露一个 bash 工具）
# 符合 Anthropic tool-use 协议：name + description + JSON Schema 格式的 input_schema
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command. Returns stdout, stderr, or an error message.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                }
            },
            "required": ["command"],
        },
    }
]

# Anthropic 客户端实例
# base_url 可通过 ANTHROPIC_BASE_URL 环境变量指向代理或兼容 API
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


# =============================================================================
# 工具执行
# =============================================================================


def run_bash(command: str) -> str:
    """执行模型请求的 shell 命令，以字符串形式返回结果。

    无论命令成功还是失败，本函数都以字符串返回（不抛异常），
    这样模型可以读取错误信息并自行决定下一步，
    而不会因为一个异常就让整个 REPL 崩溃。

    Args:
        command: 要执行的 shell 命令字符串。

    Returns:
        - 成功时：截断后的 stdout + stderr（最多 50000 字符）；
        - 空输出时：占位字符串 "(no output)"；
        - 超时 / 执行失败时：以 "Error:" 开头的错误描述。
    """
    # 极简安全防护：子串匹配黑名单。
    # ⚠️ 仅供演示，绝非安全边界 —— `rm -rf ~`、多空格、`dd`、fork 炸弹等均可绕过。
    # 生产环境务必使用沙箱 / 容器来隔离不可信命令。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        r = subprocess.run(
            command,
            shell=True,  # 通过 shell 执行（支持管道、通配符）
            cwd=os.getcwd(),  # 在当前工作目录下执行
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,  # 输出为字符串而非 bytes
            timeout=120,  # 超时保护：120 秒
        )
        # 合并 stdout 与 stderr —— 错误信息同样有助于模型判断。
        out = (r.stdout + r.stderr).strip()
        # 截断过长输出，避免撑爆上下文窗口；空输出时给占位符，
        # 防止模型把空串误解为「调用失败」。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        # 命令超时（死循环、等待输入等），返回可读的提示
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        # 进程无法启动等系统级错误（shell 不存在、磁盘满、资源不足等）
        return f"Error: {e}"


# =============================================================================
# Agent 循环
# =============================================================================


def agent_loop(messages: list) -> None:
    """运行一次 agentic loop：反复「调用模型 → 执行工具 → 回填结果」。

    循环终止条件：模型的 stop_reason 不再是 "tool_use"，
    这意味着模型已经给出了最终文本回复（end_turn）或达到 token 上限。

    Args:
        messages: 对话历史，list[dict]，每个 dict 包含 role 和 content：
            - role="user"     → content 可以是 str（纯文本）或 list[ContentBlock]（回填工具结果时）
            - role="assistant"→ content 固定为 list[ContentBlock]（TextBlock / ToolUseBlock）
            本函数会原地修改该列表，将 assistant 回复和 user tool_result 追加进去。
    """
    while True:
        # 调用 Anthropic Messages API，返回 Message 对象，主要字段：
        #   response.id          — 本次 API 调用的唯一标识（如 "msg_01AbCd..."）
        #   response.model       — 实际使用的模型 ID
        #   response.role        — 固定为 "assistant"
        #   response.stop_reason — 停止原因："end_turn"（正常结束）、"tool_use"（请求调工具）、
        #                          "max_tokens"（达到上限）、"stop_sequence"（命中自定义停止序列）
        #   response.content     — 一堆 block 组成的列表 list[ContentBlock]，可以理解为模型的
        #                         「一轮回复」被拆成了多个独立的片段，按顺序排列：
        #
        #       ┌─ TextBlock ──────────────────────────────────────
        #       │  type = "text"
        #       │  text = "让我先看看目录结构..."
        #       │  说明: 模型直接输出的文本片段，拿来展示给用户就行
        #       └──────────────────────────────────────────────────
        #
        #       ┌─ ToolUseBlock ───────────────────────────────────
        #       │  type  = "tool_use"
        #       │  id    = "toolu_01AbCdEfGh..."    ← 用于回填结果时一一对应
        #       │  name  = "bash"                    ← 工具名，与 TOOLS 定义一致
        #       │  input = {"command": "ls -la"}     ← 工具参数，结构由 input_schema 决定
        #       │  说明: 模型请求执行工具，需要程序实际运行命令并回填 tool_result
        #       └──────────────────────────────────────────────────
        #
        #       多种 block 可以混在同一轮 response.content 里，比如：
        #       [TextBlock("我先查一下"), ToolUseBlock("ls"), TextBlock("目录如上")]
        #       遍历时通过 block.type 判断是哪一种即可
        #   response.usage       — token 用量: {input_tokens, output_tokens}
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,  # 单轮回复的最大 token 数
        )

        # 将模型本轮回复追加到历史（content 为 ContentBlock 列表）
        messages.append({"role": "assistant", "content": response.content})

        # 模型没有请求工具 → 任务已给出最终回答，退出循环
        if response.stop_reason != "tool_use":
            return

        # 遍历回复中的每个内容块，执行所有 tool_use
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 黄色打印执行的命令
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                # 仅预览前 200 字符，完整内容已通过工具结果传给模型
                print(output[:200])
                print()
                # tool_use_id 必须与请求一一对应，模型据此匹配结果
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )

        # 以 user 角色将所有工具结果回填到对话历史。
        # user 消息的 content 可以是：
        #   1. str                      — 纯文本输入
        #   2. list[ContentBlock]       — 内容块列表，每项可能是：
        #       TextBlock               : type="text",         text="..."
        #       ToolResultBlock         : type="tool_result",  tool_use_id="...", content="..."
        #       ImageBlock              : type="image",        source={...}
        #       DocumentBlock           : type="document",     source={...}
        # 这里因为要回传工具执行结果，所以 content 用列表，每项是 tool_result。
        messages.append({"role": "user", "content": results})


# =============================================================================
# 主入口
# =============================================================================


def main() -> None:
    """启动 REPL：循环读取用户输入，交给 agent_loop 处理，打印最终回复。"""
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    # 对话历史（跨多轮交互累积，实现上下文记忆）
    history_messages = []

    while True:
        # 读取用户输入（蓝色提示符）
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C → 优雅退出
            break

        # 空输入或退出指令 → 结束 REPL
        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        # 用户消息以字符串形式追加（纯文本输入）
        history_messages.append({"role": "user", "content": query})
        agent_loop(history_messages)

        # agent_loop 结束后，末尾必为 assistant 消息。
        # content 固定是 list[ContentBlock]（纯文本也是 [TextBlock]），
        # 只提取 text 块展示（tool_use 块已在循环中打印）。
        response_content = history_messages[-1]["content"]
        if isinstance(response_content, list):  # 防御性判断，正常情况恒为 True
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()


if __name__ == "__main__":
    main()
