"""
s14 Cron Scheduler —— 带定时任务调度、上下文压缩与工具调用循环的编码 Agent REPL 应用。

本模块是 s14 章节的入口，在 s13 的 Agent REPL 基础上新增 cron 定时任务调度能力：
模型可通过 schedule_cron / list_crons / cancel_cron 工具管理定时任务，
后台线程每秒轮询匹配的 cron 表达式，到期的 prompt 自动注入对话。

架构概览：
  main()        → REPL 循环，读取用户输入，调用 agent_loop，打印模型回复
  agent_loop()  → 核心循环：消费 cron_queue → 压缩 → 调用 LLM → 执行工具 → 回填结果
  queue_processor_loop() → 后台线程，非阻塞轮询 cron_queue 并派发到 agent_loop

上下文压缩管线（每轮 agent_loop 开始时依次执行）：
  tool_result_budget → 单轮 tool_result 总大小超限时，持久化最大的结果到磁盘
  snip_compact       → 消息数超限时，裁剪中间旧消息，保留头尾
  micro_compact      → 压缩早期 tool_result 内容为占位文本
  若仍超 CONTEXT_LIMIT → compact_history（LLM 摘要完全替代历史）
  若 API 报 prompt_too_long → reactive_compact（保留尾部 + LLM 摘要）

后台任务系统：
  耗时 shell 命令（install/build/test 等）可派发到 daemon 线程异步执行，
  完成后以 <task_notification> 片段注入下一轮对话。
"""

try:
    import readline

    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

import threading
import time
from typing import Any

import config
import constant
from llm_client import client
import system_prompt
import tools
import hooks
import context
import memory
import recovery
import tasks
import jobs


# 自上次 todo_write 调用以来已完成的 round 数。
# agent_loop 每轮 +1；todo_write 调用时清零。达到阈值时向模型注入提醒，
# 防止模型长时间不更新任务列表（例如沉浸在一连串 bash/read 调用中）。
rounds_since_todo = 0


# === 后台任务系统 ===
# 将耗时 shell 命令（install/build/test 等）派发到后台线程执行，主 agent
# 循环不阻塞。任务完成后以 <task_notification> XML 片段注入下一轮对话。
#
# 数据流：
#   模型调用 bash(run_in_background=True) → tasks.should_run_background 判定为 true
#   → start_background_task 启动 daemon 线程 → 立即返回占位 tool_result
#   → 后续轮次 collect_background_results 收集完成的通知 → 注入对话
#
# 三个模块级变量 + 一把锁：
#   _bg_counter        — 自增计数器，生成唯一 bg_id（bg_0001, bg_0002, ...）
#   background_tasks   — bg_id → {tool_use_id, command, status} 任务元数据
#   background_results — bg_id → output 字符串，worker 线程写入的输出内容
#   background_lock    — 保护上述两个字典的 threading.Lock

_bg_counter = 0
background_tasks: dict[str, dict] = {}
background_results: dict[str, str] = {}
background_lock = threading.Lock()


def start_background_task(block: Any) -> str:
    """将耗时工具调用（如 install/build/test）派发到后台线程异步执行。

    主循环不等待后台任务完成——立即返回占位 tool_result 告知模型"已派发"，
    任务完成后由 collect_background_results() 将结果注入下一轮对话。

    1. 分配全局唯一 bg_id（格式 bg_0001）
    2. 持锁注册任务状态为 "running"
    3. 启动 daemon 线程执行 worker()，结果写入 background_results
    4. 立即返回 bg_id，主循环继续

    Args:
        block: 模型请求的 tool_use 内容块，需具备 .id、.name、.input 属性。

    Returns:
        后台任务 ID 字符串（如 "bg_0001"）。
    """
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        """后台执行体：调用工具，将结果写入共享字典。

        用 try/except 包裹 tools.use_tool()，防止工具调用抛出未预期的异常
        （如 TypeError、OSError 等非 "Error: ..." 字符串所能覆盖的错误）导致
        worker 静默死亡。异常时以 "Error: ..." 字符串写入结果，确保
        collect_background_results 总能收到完成通知，不会永久泄漏任务条目。
        """
        try:
            result = tools.use_tool(block.name, block.input)
        except Exception as e:
            result = f"Error: background task failed — {e}"
        # 持锁写入两个共享字典，确保与 collect_background_results 的 pop 操作互斥
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running",
        }
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(f"  \033[33m[background] dispatched {bg_id}: {cmd[:40]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    """收集已完成后台任务的结果，生成 <task_notification> XML 片段。

    在 agent_loop 工具执行阶段末尾调用，若无已完成任务则返回空列表。

    1. 持锁扫描 background_tasks，收集所有 status == "completed" 的任务 ID
    2. 逐任务持锁 pop 元数据与结果（pop 后从字典移除，防止重复注入）
    3. 截取前 200 字符作为摘要，包装为 <task_notification> XML
    4. 打印绿色终端提示

    线程安全策略：分两阶段操作，将快路径（字典读写）与慢路径（字符串
    格式化和终端 I/O）分离——持锁只做 dict 扫描/pop，释放锁后再构建 XML
    和打印，避免 worker 线程因等待终端输出而被阻塞。

    Returns:
        <task_notification> XML 字符串列表，以 text 块追加到 user 消息中。
    """
    with background_lock:
        ready_ids = [
            bid
            for bid, task in background_tasks.items()
            if task["status"] == "completed"
        ]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>"
        )
        print(
            f"  \033[32m[background done] {bg_id}: "
            f"{task['command'][:40]} ({len(output)} chars)\033[0m"
        )
    return notifications


def agent_loop(messages: list, session_context: dict) -> dict:
    """反复「调用模型 → 执行工具 → 回填结果」，直到模型不再请求工具为止。

    每轮检查是否需要注入 todo 提醒；退出前触发 Stop hook，允许外部注入
    追问消息并继续循环。
    """
    global rounds_since_todo

    memories_content = memory.load_memories(messages)
    # 只将记忆注入到本轮对话的首条 user 消息之前（即最新一条纯文本 user 消息）。
    # 如果最新一条 user 消息的 content 是 list（含 tool_result 块），说明
    # 上一轮模型刚执行完工具，本轮尚未加入新 user 消息——此时 memory_turn 为 None，
    # 记忆将在下一轮用户真正提交文本输入时才注入。
    memory_turn = (
        len(messages) - 1
        if messages and isinstance(messages[-1].get("content"), str)
        else None
    )

    state = recovery.RecoveryState()
    max_tokens = constant.DEFAULT_MAX_TOKENS
    system = system_prompt.get_system_prompt(session_context)

    while True:
        fired = jobs.consume_cron_queue()
        for job in fired:
            messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
            print(f"  \033[35m[inject cron] {job.prompt[:50]}\033[0m")

        # 在压缩前拍一份消息历史快照（统一转换为纯 dict 形式）。
        # 压缩管线（tool_result_budget / snip / micro / compact_history）会
        # 修改 messages 中的 block 内容或裁切消息，而 extract_memories 需要
        # 压缩前的完整对话才能提取准确的记忆——所以提前保存。
        pre_compress = [
            m
            if isinstance(m, dict)
            else {"role": m.get("role", ""), "content": str(m.get("content", ""))}
            for m in messages
        ]

        # === 三层压缩管线：每轮先压缩再调用模型 ===
        # 第一层：单轮 tool_result 预算控制 → 持久化超长结果到磁盘
        messages[:] = context.tool_result_budget(messages)
        # 第二层：消息数裁剪 → 保留头尾，中间用占位消息替代
        messages[:] = context.snip_compact(messages)
        # 第三层：早期 tool_result 内容压缩 → 替换为占位文本
        messages[:] = context.micro_compact(messages)

        # 三层压缩后仍超限 → 全量 LLM 摘要替代
        if context.estimate_size(messages) > constant.CONTEXT_LIMIT:
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
            request_messages = messages
            if (
                memories_content
                and memory_turn is not None
                and memory_turn < len(messages)
            ):
                request_messages = messages.copy()
                request_messages[memory_turn] = {
                    **messages[memory_turn],
                    "content": memories_content
                    + "\n\n"
                    + messages[memory_turn]["content"],
                }
            response = recovery.with_retry(
                lambda mt=max_tokens, mdl=state.current_model: client.messages.create(
                    model=mdl,
                    system=system,
                    messages=request_messages,
                    tools=tools.TOOLS,
                    max_tokens=mt,
                ),
                state,
            )
            state.has_attempted_reactive_compact = False
        except Exception as e:
            # prompt_too_long → 上下文太长，API 拒绝请求。
            # 用 reactive_compact 保留尾部最近消息并压缩其余，然后重试。
            # 仅尝试一次 reactive_compact，通过 has_attempted_reactive_compact 标志防止重复压缩。
            if recovery.is_prompt_too_long_error(e):
                if not state.has_attempted_reactive_compact:
                    print("[reactive compact]")
                    messages[:] = context.reactive_compact(messages)
                    state.has_attempted_reactive_compact = True
                    continue
                print("  \033[31m[unrecoverable] still too long after compact\033[0m")
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "[Error] Context too large, cannot continue.",
                            }
                        ],
                    }
                )
                return session_context

            name = type(e).__name__
            print(f"  \033[31m[unrecoverable] {name}: {str(e)[:100]}\033[0m")
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"[Error] {name}: {str(e)[:200]}"}
                    ],
                }
            )
            return session_context

        if response.stop_reason == "max_tokens":
            if not state.has_escalated:
                max_tokens = constant.ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(
                    f"  \033[33m[max_tokens] escalating"
                    f" {constant.DEFAULT_MAX_TOKENS} -> {constant.ESCALATED_MAX_TOKENS}\033[0m"
                )
                continue
            messages.append({"role": "assistant", "content": response.content})
            if state.recovery_count < constant.MAX_RECOVERY_RETRIES:
                messages.append(
                    {"role": "user", "content": constant.CONTINUATION_PROMPT}
                )
                state.recovery_count += 1
                print(
                    f"  \033[33m[max_tokens] continuation"
                    f" {state.recovery_count}/{constant.MAX_RECOVERY_RETRIES}\033[0m"
                )
                continue
            print("  \033[31m[max_tokens] recovery limit reached\033[0m")
            return session_context

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
            memory.extract_memories(pre_compress)
            memory.consolidate_memories()
            return session_context

        # 仅在 tool_use 轮次递增计数器。文本回复轮次（模型给出最终答案）
        # 不需要提醒——提醒只针对模型陷入一连串工具调用却忘记更新任务列表的场景。
        rounds_since_todo += 1

        # === 工具执行阶段 ===
        # 遍历模型返回的所有 tool_use 块，逐一执行并收集结果
        results = []
        for block in response.content:
            if block.type != "tool_use":
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

                session_context = context.update_session_context(
                    session_context, messages
                )
                system = system_prompt.get_system_prompt(session_context)
                break

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

            # === 后台任务判断：should_run_background ===
            # 两个条件任一满足即派发到后台：
            # 1. run_in_background=True：模型显式声明（如 bash 工具的参数）
            # 2. is_slow_operation 命中：命令包含 install/build/test 等耗时关键词
            if tasks.should_run_background(block.name, block.input):
                bg_id = start_background_task(block)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"[Background task {bg_id} started] "
                        f"Command: {block.input.get('command', '')}. "
                        f"Result will be available when complete.",
                    }
                )
            else:
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
            user_content = list(results)
            # === 后台任务结果注入 ===
            # 工具执行阶段的最后一步：收集已完成后台任务的结果，
            # 以 text 块形式追加到 user 消息中。模型在下一轮迭代中
            # 将这些 <task_notification> 当作系统通知来解析。
            bg_notifications = collect_background_results()
            if bg_notifications:
                for notif in bg_notifications:
                    user_content.append({"type": "text", "text": notif})
                print(
                    f"  \033[32m[inject] {len(bg_notifications)} background "
                    f"notification(s)\033[0m"
                )

            # for...else：未被 break 中断 → 所有 tool_use 都执行完毕
            # 将收集到的 tool_result 以 user 角色回填，进入下一轮循环
            messages.append({"role": "user", "content": user_content})

            session_context = context.update_session_context(session_context, messages)
            system = system_prompt.get_system_prompt(session_context)


def print_latest_assistant_text(history_messages: list) -> None:
    """打印对话历史中最后一条 assistant 消息的文本内容。

    兼容两种 content 格式：纯字符串（旧版或简化场景）和 ContentBlock 列表
    （SDK 返回的富对象，需从中过滤 type=="text" 的块）。
    无消息或最后一条非 assistant 角色时静默返回。
    """
    if not history_messages:
        return
    msg = history_messages[-1]
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return
    content = msg.get("content", "")
    if isinstance(content, str):
        print(content)
        return
    for block in content:
        if getattr(block, "type", None) == "text":
            print(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""))


def run_agent_turn_locked(
    history_messages: list, session_context: dict, user_query: str | None = None
) -> None:
    if user_query is not None:
        # 通知 hooks 用户已提交输入（s14: 会话记录等）
        hooks.trigger_hooks("UserPromptSubmit", user_query)

        history_messages.append({"role": "user", "content": user_query})

    session_context = agent_loop(history_messages, session_context)
    session_context = context.update_session_context(session_context, history_messages)
    print_latest_assistant_text(history_messages)
    print()


def queue_processor_loop(history_messages: list, session_context: dict) -> None:
    """后台线程：非阻塞轮询 cron_queue，有任务时获取 agent_lock 并派发。

    每 200ms 检查一次，空队列或无空闲锁时立即重试。双重检查 has_cron_queue
    （获取锁前后各一次）避免在等锁期间队列被消费后空跑一轮 agent_loop。
    """
    while True:
        time.sleep(0.2)
        if not jobs.has_cron_queue():
            continue
        if not agent_lock.acquire(blocking=False):
            continue
        try:
            if not jobs.has_cron_queue():
                continue
            print("\n  \033[35m[queue processor] delivering scheduled work\033[0m")
            run_agent_turn_locked(history_messages, session_context)
        finally:
            agent_lock.release()


agent_lock = threading.Lock()


def main() -> None:
    """REPL 入口：循环读取用户输入，交给 agent_loop 处理，打印模型最终回复。"""
    print("s14: Cron Scheduler")
    print("输入问题，回车发送。输入 q 退出。\n")

    history_messages = []
    session_context = context.update_session_context({}, history_messages)

    threading.Thread(
        target=queue_processor_loop,
        args=(history_messages, session_context),
        daemon=True,
    ).start()

    while True:
        try:
            query = input("\033[36ms14 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", "quit", ""):
            break

        with agent_lock:
            run_agent_turn_locked(history_messages, session_context, query)

        # 通知 hooks 用户已提交输入（s14: 会话记录等）
        # hooks.trigger_hooks("UserPromptSubmit", query)

        # turn_start = len(history_messages)

        # history_messages.append({"role": "user", "content": query})
        # agent_loop(history_messages, session_context)
        # session_context = context.update_session_context(
        #     session_context, history_messages
        # )

        # agent_loop 结束后，末尾必为 assistant 消息。
        # content 为内容块列表时只挑文本块展示（工具调用块由 run_todo_write 等函数
        # 内置的 print 在终端直接渲染）。
        # for msg in history_messages[turn_start:]:
        #     if msg.get("role") != "assistant":
        #         continue
        #     for block in msg["content"]:
        #         if getattr(block, "type", None) == "text":
        #             print(block.text)
        #         elif isinstance(block, dict) and block.get("type") == "text":
        #             print(block.get("text", ""))
        # print()


if __name__ == "__main__":
    main()
