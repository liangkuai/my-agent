"""
s12 任务管理系统 —— 文件级别的任务持久化与依赖追踪。

以 JSON 文件为存储后端，提供任务的完整生命周期管理
（创建、列出、认领、完成）和基于 blockedBy 的依赖关系追踪。

每条任务对应 .tasks/ 下一个 {task_id}.json 文件，
ID 格式为 task_{timestamp}_{random}，单机环境下保证唯一。

设计要点：
1. 纯文件存储 —— 无需数据库，可直接 ls/cat 查看调试
2. 依赖追踪 —— blockedBy 约束任务启动顺序，完成时自动报告解除阻塞的下游任务
3. 状态机 —— pending → in_progress → completed，转换点有强制校验
4. 容错约定 —— 面向 agent 的函数将错误以字符串形式返回而非抛异常，
   让 agent 自行消化并重试（load_task 例外，它是内部构造器，异常由调用方处理）
"""

import time
import random
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import constant


@dataclass
class Task:
    """一条任务记录，对应 .tasks/ 下一个 JSON 文件。

    Attributes:
        id: 唯一标识，格式 task_{timestamp}_{random}
        subject: 任务标题（简短摘要）
        description: 任务详细描述
        status: 当前状态，取值为 pending | in_progress | completed
        owner: 认领该任务的 agent 名称，未认领时为 None
        blockedBy: 前置依赖任务 ID 列表，全部完成后本任务才能认领
    """
    id: str
    subject: str
    description: str
    status: str  # pending | in_progress | completed
    owner: str | None  # Agent name (multi-agent scenarios)
    blockedBy: list[str]  # Dependency task IDs


def _task_path(task_id: str) -> Path:
    """返回任务 JSON 文件的完整路径。"""
    return constant.TASKS_DIR / f"{task_id}.json"


def save_task(task: Task):
    """将 Task 对象序列化为 JSON 并写入磁盘（覆盖已有文件）。

    自动确保 TASKS_DIR 目录存在，因此调用方无需手动创建。
    """
    constant.TASKS_DIR.mkdir(parents=True, exist_ok=True)
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    """从磁盘读取并反序列化任务 JSON 文件为 Task 对象。

    Raises:
        FileNotFoundError: 任务文件不存在时抛出，由调用方决定如何处理。
    """
    return Task(**json.loads(_task_path(task_id).read_text()))


def create_task(
    subject: str, description: str = "", blockedBy: list[str] | None = None
) -> Task:
    """创建一个新的 pending 状态任务，写入磁盘后返回 Task 对象。

    Args:
        subject: 任务标题（必填）
        description: 任务详细描述（可选）
        blockedBy: 前置依赖任务 ID 列表，这些任务全部完成后本任务才能认领

    Returns:
        新建的 Task 对象，status 固定为 "pending"。
    """
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject,
        description=description,
        status="pending",
        owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def list_tasks() -> list[Task]:
    """列出所有任务，按文件名排序（等效于按创建时间排序）。

    Returns:
        Task 列表，无任务时返回空列表 []。
    """
    return [
        Task(**json.loads(p.read_text()))
        for p in sorted(constant.TASKS_DIR.glob("task_*.json"))
    ]


def get_task(task_id: str) -> str:
    """获取单条任务的 JSON 格式详情字符串。

    与 load_task() 的区别：本函数返回格式化的 JSON 字符串（适合直接展示），
    而 load_task() 返回 Task 对象（供内部逻辑使用）。
    """
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """检查任务的所有前置依赖是否均已满足（文件存在且状态为 completed）。

    两条规则缺一不可：
    1. 依赖任务文件必须在磁盘上存在（被删除的依赖视为不满足）
    2. 依赖任务状态必须为 "completed"

    Returns:
        True 表示没有阻塞项，任务可以认领。
    """
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领一个 pending 任务，将其状态切换为 in_progress。

    认领前会做两项校验：
    1. 任务当前状态必须为 "pending"，否则返回错误提示
    2. 所有 blockedBy 依赖必须已满足，否则返回阻塞列表

    Args:
        task_id: 要认领的任务 ID
        owner: 认领者名称，默认为 "agent"

    Returns:
        成功时返回确认消息；失败时返回错误原因（如状态不对、被阻塞）。
    """
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id):
        deps = [
            d
            for d in task.blockedBy
            if not _task_path(d).exists() or load_task(d).status != "completed"
        ]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress (owner: {owner})\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    """完成一个 in_progress 任务，将其状态切换为 completed。

    完成后会扫描所有 pending 任务，找出因本任务完成而解除阻塞的下游任务，
    将其列在返回消息中并打印彩色提示。

    校验：任务当前状态必须为 "in_progress"，否则返回错误提示。

    Returns:
        成功时返回确认消息（含解除阻塞的下游任务列表）；失败时返回错误原因。
    """
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    unblocked = [
        t.subject
        for t in list_tasks()
        if t.status == "pending" and t.blockedBy and can_start(t.id)
    ]
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")
    return msg
