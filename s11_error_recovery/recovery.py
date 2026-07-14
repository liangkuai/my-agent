"""
错误恢复模块 —— LLM API 调用的重试、降级与错误分类。

三层容错，由轻到重：
1. 指数退避重试 —— 429（限流）/ 529（过载）自动重试，带随机抖动避免惊群
2. 模型降级      —— 连续 529 超阈值时自动切换到 FALLBACK_MODEL
3. 错误分类      —— 识别 prompt_too_long，供 app.agent_loop 触发响应式压缩

with_retry() 是所有 LLM 调用的统一容错入口，调用方只需传入可调用对象和
RecoveryState 即可获得完整的重试与降级保护。

RecoveryState 在 agent_loop 内以局部变量创建，生命周期与单次用户查询绑定——
每次新查询重置状态，防止上一次的降级/升级决策污染下一次。
"""

import random
import time
from collections.abc import Callable

import constant


class RecoveryState:
    """单次 agent_loop 调用期间的恢复状态追踪器。

    每个字段在 __init__ 中初始化为默认值，新查询创建新实例保证状态隔离。

    Attributes:
        has_escalated: 是否已将 max_tokens 从 DEFAULT 提升到 ESCALATED。
            由 app.agent_loop 在 max_tokens stop_reason 时置 True。
        recovery_count: 连续 max_tokens 触发的续写次数。
            每次追加 CONTINUATION_PROMPT 后 +1，达 MAX_RECOVERY_RETRIES 后停止。
        consecutive_529: 连续 529 过载次数。
            每次 529 错误 +1，成功调用后清零。达阈值时切换模型。
        has_attempted_reactive_compact: 是否已在本轮尝试过响应式压缩。
            prompt_too_long 时置 True，防止重复压缩。
        current_model: 当前正在使用的模型 ID。
            初始为 constant.MODEL，连续 529 超阈值后可能切换为 constant.FALLBACK_MODEL。
    """
    def __init__(self):
        self.has_escalated = False
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.has_attempted_reactive_compact = False
        self.current_model = constant.MODEL


def retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """计算第 attempt 次重试的等待秒数。

    指数退避公式：delay = min(BASE_DELAY_MS × 2^attempt, 32000) / 1000
    在此基值上叠加 25% 以内的均匀随机抖动，防止多个客户端同时重试（惊群效应）。
    若调用方传入 retry_after（来自 API 的 Retry-After 响应头），则直接使用该值。

    Args:
        attempt: 从 0 开始的重试序号。
        retry_after: API 建议的重试等待秒数，为 None 时由公式计算。

    Returns:
        等待秒数（float）。
    """
    if retry_after:
        return retry_after
    base = min(constant.BASE_DELAY_MS * (2 ** attempt), 32000) / 1000
    jitter = random.uniform(0, base * 0.25)
    return base + jitter


def with_retry(fn: Callable, state: RecoveryState):
    """带智能重试的 LLM API 调用包装器，错误恢复模块的核心入口。

    所有需要容错的 LLM 调用（主/子 agent）都应通过此函数封装。

    三类错误的处理策略：

    1. 429 Rate Limit
       - 按指数退避等待后重试
       - 最多重试 MAX_RETRIES 次

    2. 529 Overloaded
       - 按指数退避等待后重试，累计 consecutive_529 计数
       - 计数达 MAX_CONSECUTIVE_529 时：
         · 已配置 FALLBACK_MODEL → 切换模型、清零计数、继续
         · 未配置 FALLBACK_MODEL → 清零计数、用原模型继续（等待恢复）
       - 任意一次成功调用后计数器归零

    3. 其他异常（含 prompt_too_long）
       - 直接向上抛出，不做重试
       - prompt_too_long 重试无意义：不压缩上下文每次都会因同样原因失败

    Args:
        fn: 零参数可调用对象，内部引用当前的 model 和 max_tokens。
            典型形式：lambda mt=tokens, mdl=state.current_model: client.messages.create(...)
        state: 当前查询的 RecoveryState，追踪降级/续写状态。

    Returns:
        fn() 的成功返回值。

    Raises:
        非 429/529 异常：直接向上抛出。
        RuntimeError: MAX_RETRIES 次重试全部失败。
    """
    for attempt in range(constant.MAX_RETRIES):
        try:
            result = fn()
            # 成功调用后重置 529 计数器——模型已恢复正常
            state.consecutive_529 = 0
            return result
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()

            # ── 429 Rate Limit ──
            # 纯速率限制，等待后重试即可。
            # 匹配：异常类名含 "ratelimit" 或消息体含 "429"。
            if "ratelimit" in name.lower() or "429" in msg:
                delay = retry_delay(attempt)
                print(
                    f"  \033[33m[429 rate limit] retry {attempt + 1}/{constant.MAX_RETRIES},"
                    f" wait {delay:.1f}s\033[0m"
                )
                time.sleep(delay)
                continue

            # ── 529 Overloaded ──
            # 服务端过载错误，可能持续较长时间。连续失败时尝试降级到备用模型。
            # 匹配：异常类名或消息体含 "overloaded"，或消息体含 "529"。
            if "overloaded" in name.lower() or "529" in msg or "overloaded" in msg:
                state.consecutive_529 += 1
                if state.consecutive_529 >= constant.MAX_CONSECUTIVE_529:
                    if constant.FALLBACK_MODEL:
                        # 有备用模型 → 切换并重置计数器
                        state.current_model = constant.FALLBACK_MODEL
                        state.consecutive_529 = 0
                        print(
                            f"  \033[31m[529 x{constant.MAX_CONSECUTIVE_529}]"
                            f" switching to {constant.FALLBACK_MODEL}\033[0m"
                        )
                    else:
                        # 无备用模型 → 仅清零计数器，继续重试等原模型恢复
                        state.consecutive_529 = 0
                        print(
                            f"  \033[31m[529 x{constant.MAX_CONSECUTIVE_529}]"
                            f" no FALLBACK_MODEL_ID configured, continuing retry\033[0m"
                        )
                delay = retry_delay(attempt)
                print(
                    f"  \033[33m[529 overloaded] retry {attempt + 1}/{constant.MAX_RETRIES},"
                    f" wait {delay:.1f}s\033[0m"
                )
                time.sleep(delay)
                continue

            # ── 其他异常（含 prompt_too_long）──
            # 不做重试，向上抛出由调用方处理。
            # prompt_too_long 重试无意义：不压缩上下文则每次都失败。
            raise

    # 所有重试次数耗尽仍未成功
    raise RuntimeError(f"Max retries ({constant.MAX_RETRIES}) exceeded")


def is_prompt_too_long_error(e: Exception) -> bool:
    """判断异常是否为上下文过长（prompt_too_long）错误。

    将异常消息转为小写后，匹配以下任一模式即返回 True：
    - "prompt" 与 "long" 同时出现（宽松匹配，覆盖各类 API 的变体措辞）
    - "prompt_is_too_long"（Anthropic API 标准错误码）
    - "context_length_exceeded"（部分兼容 API 的错误码）
    - "max_context_window"（部分自部署模型的错误码）

    Args:
        e: LLM API 调用抛出的异常。

    Returns:
        True 表示可触发 reactive_compact 压缩后重试。
    """
    msg = str(e).lower()
    return (
        ("prompt" in msg and "long" in msg)
        or "prompt_is_too_long" in msg
        or "context_length_exceeded" in msg
        or "max_context_window" in msg
    )
