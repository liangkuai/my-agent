"""
LLM 客户端初始化 —— 创建全局唯一的 Anthropic SDK client 实例。

app.py 和 tool.py（子 agent）共用同一个 client 对象，共享底层 HTTP 连接池。
Anthropic SDK 的 client 是线程安全的，多轮顺序调用的单线程 agent 循环不需要加锁。

base_url 从 ANTHROPIC_BASE_URL 环境变量读取：
- 未设置时为 None → SDK 连接 api.anthropic.com（官方 API）
- 设置后 → 可以指向代理网关或兼容 API（如 OpenRouter、本地模型服务）
"""

import os
from anthropic import Anthropic

import config  # 确保 load_dotenv 在 client 创建前已执行

# 全局唯一的 LLM client，整个进程生命周期内复用。
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
