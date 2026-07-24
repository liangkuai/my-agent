"""
s14 Cron Scheduler —— cron 定时任务的注册、验证、触发与持久化。

本模块维护一个后台调度线程，每秒轮询已注册任务，匹配到期的 cron 表达式
后注入 cron_queue，由 app.agent_loop 消费并转为对话消息。

支持五种 cron 字段格式（*、*/N、N、列表、范围），日/周字段按标准 crontab
OR 语义执行。durable 任务写入 .scheduled_tasks.json，下次启动自动恢复。

线程安全：cron_scheduler_loop（后台）与 agent_loop（主线程）通过 cron_lock
互斥访问所有共享状态。
"""

import time
import threading
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime

import constant


@dataclass
class CronJob:
    """一条 cron 定时任务记录。

    Attributes:
        id: 唯一标识，格式 cron_{6位随机数}（如 cron_004227）。
        cron: 标准五段式 cron 表达式（"分 时 日 月 周"），如 "0 9 * * *" = 每天9:00。
        prompt: 触发时以 "[Scheduled] {prompt}" 格式注入对话的消息文本。
        recurring: True=周期性重复触发；False=一次性任务，首次触发后自动移除。
        durable: True=写入磁盘持久化（跨会话保留）；False=仅内存（会话结束消失）。
    """

    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool


# === 模块级共享状态（受 cron_lock 保护） ===

# 已注册的定时任务注册表，job_id → CronJob
scheduled_jobs: dict[str, CronJob] = {}

# 已触发但尚未被 agent_loop 消费的任务队列（FIFO）
cron_queue: list[CronJob] = []

# 读写 scheduled_jobs / cron_queue / _last_fired 的全局锁
cron_lock = threading.Lock()

# 记录每个 job 上次触发的分钟标识（"YYYY-MM-DD HH:MM"），
# 同一分钟内只触发一次，防止调度循环的每秒轮询导致重复入队
_last_fired: dict[str, str] = {}


# =============================================================================
# Cron 表达式匹配 —— 支持 */step、逗号列表、范围、通配符五种格式
# =============================================================================


def _cron_field_matches(field: str, value: int) -> bool:
    """判断单个 cron 字段是否匹配给定的时间值。

    按优先级尝试五种格式：
      *        → 匹配所有值
      */N      → 每隔 N 个单位匹配一次（如 */5 = 每5分钟）
      N,M,...  → 逗号列表，任一命中即为匹配
      A-B      → 闭区间范围
      N        → 精确匹配

    Args:
        field: cron 字段字符串。
        value: 当前时间的对应整数值（如 minute=30）。
    """
    if field == "*":
        return True
    # */N 步进：每 N 个单位匹配一次
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    # 逗号列表：递归检查每个子项（任一匹配即为 True）
    if "," in field:
        return any(_cron_field_matches(f.strip(), value) for f in field.split(","))
    # 范围 A-B：value 落在 [A, B] 闭区间内
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    # 精确匹配：value 必须等于字段值
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """判断给定的 cron 五段表达式是否匹配指定时间。

    日/周字段遵循标准 crontab OR 语义：
      dom=* 且 dow=*        → 每天都匹配
      dom=* 且 dow=具体值   → 仅按周匹配
      dom=具体值 且 dow=*   → 仅按日匹配
      dom=具体值 且 dow=具体值 → 任一匹配（OR）

    注意：Python weekday() 返回 0=Monday，cron dow 字段 0=Sunday，
    因此需要 (weekday() + 1) % 7 做映射。

    Args:
        cron_expr: 五段式 cron 字符串（如 "30 9 * * 1-5" = 工作日9:30）。
        dt: 要检查的 datetime 对象。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields

    # Python 的 weekday() 返回 0=Monday，而 cron 的 dow 字段 0=Sunday。
    # 将 Python weekday 值 +1 后 mod 7 映射到 cron 的星期编号。
    dow_val = (dt.weekday() + 1) % 7

    # 先检查分、时、月 —— 这三个字段不匹配则直接短路
    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    if not (m and h and month_ok):
        return False

    # 日/周字段的 OR 逻辑（与标准 crontab 一致）
    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True  # 每天都匹配
    if dom_unconstrained:
        return dow_ok  # 仅按周匹配
    if dow_unconstrained:
        return dom_ok  # 仅按日匹配
    return dom_ok or dow_ok  # 任一匹配（OR）


# =============================================================================
# Cron 表达式校验 —— schedule 前确保表达式合法，避免磁盘残留无效任务
# =============================================================================


def validate_cron(cron_expr: str) -> str | None:
    """校验 cron 表达式格式是否合法。

    对五个字段逐一做范围检查，确保每个字段的值在其合法范围内：
    minute [0-59], hour [0-23], day-of-month [1-31], month [1-12], day-of-week [0-6]。

    Args:
        cron_expr: 五段式 cron 字符串。

    Returns:
        None 表示校验通过；非 None 字符串描述第一个错误（格式 "字段名: 错误原因"）。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"

    # 五个字段的合法范围与可读名称，按位置一一对应
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]

    for i, (field, (lo, hi), name) in enumerate(zip(fields, bounds, names)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """递归校验单个 cron 字段的合法性与范围。

    支持五种格式：*、*/N、逗号列表、范围 A-B、精确数字。
    对逗号列表递归拆解后逐项校验；范围需验证边界在 [lo, hi] 内且 start <= end。

    Args:
        field: 单个字段的字符串表示。
        lo: 该字段的合法下界（含）。
        hi: 该字段的合法上界（含）。

    Returns:
        None 表示校验通过；非 None 字符串描述错误原因。
    """
    if field == "*":
        return None

    # */N 步进：N 必须是正整数
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None

    # 逗号列表：递归校验每个子项
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None

    # 范围 A-B：两端必须是数字、边界在合法范围内、起始不大于结束
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"Invalid range: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None

    # 精确数字：必须在合法范围内
    if not field.isdigit():
        return f"Invalid field: {field}"
    val = int(field)
    if val < lo or val > hi:
        return f"Value {val} out of bounds [{lo}-{hi}]"
    return None


# =============================================================================
# 任务生命周期 —— schedule 注册、cancel 取消
# =============================================================================


def schedule_job(
    cron: str, prompt: str, recurring: bool = True, durable: bool = True
) -> CronJob | str:
    """注册一个新的 cron 定时任务。

    先校验 cron 表达式合法性，校验通过后生成唯一 ID、写入内存注册表，
    可选（durable=True）持久化到磁盘。返回 CronJob 对象供调用方展示。

    由 tools.run_schedule_cron() 封装为 Agent 可调用工具。

    Args:
        cron: 五段式 cron 表达式（如 "0 9 * * 1-5" = 工作日9:00）。
        prompt: 触发时注入对话的消息文本，格式为 "[Scheduled] {prompt}"。
        recurring: True=周期性重复；False=一次性（触发后自动移除）。
        durable: True=持久化到 .scheduled_tasks.json，跨会话保留。

    Returns:
        成功时返回 CronJob 对象；校验失败时返回错误原因字符串。
    """
    err = validate_cron(cron)
    if err:
        return err

    # 6 位随机数作为 ID 后缀，碰撞概率可忽略（1/1,000,000 per job）
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron,
        prompt=prompt,
        recurring=recurring,
        durable=durable,
    )
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()
    print(f"  \033[35m[cron register] {job.id} '{cron}' → {prompt[:40]}\033[0m")
    return job


def cancel_job(job_id: str) -> str:
    """按 ID 取消（删除）一个已注册的 cron 任务。

    从 scheduled_jobs 字典中移除；若任务为 durable 则同步更新磁盘文件。

    Args:
        job_id: 任务唯一标识（如 "cron_004227"）。

    Returns:
        成功确认消息或 "Job {id} not found"。
    """
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()
    print(f"  \033[31m[cron cancel] {job_id}\033[0m")
    return f"Cancelled {job_id}"


# =============================================================================
# 调度循环 —— 后台 daemon 线程入口，每秒轮询匹配
# =============================================================================


def cron_scheduler_loop() -> None:
    """定时任务的主调度循环（在后台 daemon 线程中运行）。

    每秒醒来一次，遍历所有已注册的 scheduled_jobs：
    1. 调用 cron_matches() 判断当前时间是否命中 cron 表达式
    2. 命中后检查 _last_fired：同一分钟内已触发过的 job 跳过（防重复）
    3. 通过检查的 job 追加到 cron_queue，等待 agent_loop 消费
    4. 非 recurring 的 job 首次触发后从注册表中移除
    5. 每次遍历用 try/except 包裹，单个 job 的异常不影响其他 job 的调度

    线程安全：所有对 scheduled_jobs / cron_queue / _last_fired 的读写
    都在 cron_lock 保护下进行。
    """
    while True:
        time.sleep(1)
        now = datetime.now()
        # 精确到分钟的标识串，用于去重——同一分钟内同一 job 最多触发一次
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        with cron_lock:
            # 用 list() 快照遍历，避免在遍历过程中因 pop 修改字典导致 RuntimeError
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        # 防重复：同一分钟内已触发过则跳过
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)
                            _last_fired[job.id] = minute_marker
                            print(
                                f"  \033[35m[cron fire] {job.id} → "
                                f"{job.prompt[:40]}\033[0m"
                            )
                        # 一次性任务：触发后立即从注册表移除
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:
                    print(f"  \033[31m[cron error] {job.id}: {e}\033[0m")


# =============================================================================
# 队列消费 —— app.agent_loop 每轮开始时取出待注入的触发任务
# =============================================================================


def has_cron_queue() -> bool:
    """检查 cron_queue 中是否有待消费的触发任务。

    供 app.queue_processor_loop() 轮询使用，非空时才尝试获取 agent_lock。
    空队列直接跳过可避免无意义的锁竞争。
    """
    with cron_lock:
        return bool(cron_queue)


def consume_cron_queue() -> list[CronJob]:
    """取出 cron_queue 中所有待消费的触发任务并清空队列。

    每个 agent_loop 迭代开始时由 app.agent_loop 调用，将取出的 job 以
    "[Scheduled] {prompt}" 格式的 user 消息注入对话历史。

    Returns:
        CronJob 列表（可能为空），按入队顺序排列。
    """
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


# =============================================================================
# 持久化 —— durable 任务写入 .scheduled_tasks.json，跨会话保留
# =============================================================================


def save_durable_jobs() -> None:
    """将 scheduled_jobs 中所有 durable=True 的任务序列化写入磁盘。

    写入路径：constant.DURABLE_PATH（.scheduled_tasks.json）。
    调度循环中每次修改 durable 任务后立即调用，确保磁盘与内存一致。

    注意：仅持久化 durable=True 的 job，session-only 的 job 不写入，
    因此会话结束后自动消失。
    """
    durable = [asdict(j) for j in scheduled_jobs.values() if j.durable]
    constant.DURABLE_PATH.write_text(json.dumps(durable, indent=2))


def load_durable_jobs() -> None:
    """从磁盘恢复持久化的 cron 任务（模块导入时自动调用一次）。

    读取 .scheduled_tasks.json，逐条反序列化 → 校验 cron 表达式 →
    注册到 scheduled_jobs。校验失败的条目会打印警告并跳过，不阻塞启动。

    容错设计：文件不存在（首次运行）或 JSON 解析异常时静默返回，
    确保新安装环境下应用正常启动。
    """
    if not constant.DURABLE_PATH.exists():
        return
    try:
        jobs = json.loads(constant.DURABLE_PATH.read_text())
        for j in jobs:
            job = CronJob(**j)
            err = validate_cron(job.cron)
            if err:
                print(f"  \033[31m[cron] skipping invalid job {job.id}: {err}\033[0m")
                continue
            scheduled_jobs[job.id] = job
        # 仅统计成功加载的 job（校验失败的被过滤掉了）
        valid = [j for j in jobs if j["id"] in scheduled_jobs]
        if valid:
            print(f"  \033[35m[cron] loaded {len(valid)} durable job(s)\033[0m")
    except Exception:
        pass


# === 模块初始化：恢复持久化任务 + 启动调度线程 ===
# 这两行在模块首次 import 时执行（app.py import jobs），
# 确保调度器在 REPL 启动前就已开始运行。

load_durable_jobs()
# daemon=True：主线程退出时调度线程自动终止，不会阻止进程退出
threading.Thread(target=cron_scheduler_loop, daemon=True).start()
print("  \033[35m[cron] scheduler thread started\033[0m")
