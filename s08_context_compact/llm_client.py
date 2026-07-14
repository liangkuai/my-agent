"""
LLM 客户端初始化模块。

创建全局唯一的 Anthropic SDK 客户端实例，供 app.py、context.py、tools.py
等模块直接 import 使用。全局复用同一个 client 实例可以避免重复建立连接，
同时 SDK 内部线程池和连接池也能被所有调用方共享。

base_url 通过 ANTHROPIC_BASE_URL 环境变量注入，支持指向 Anthropic API
兼容的代理网关（如 LiteLLM 等），便于在本地调试或企业内网环境中
通过统一网关访问模型。未设置时 SDK 默认连接 api.anthropic.com。
API Key 由 SDK 自动从 ANTHROPIC_API_KEY 环境变量读取，无需显式传入。
"""

import os
from anthropic import Anthropic

import config  # noqa: F401  — 触发 load_dotenv，确保环境变量在 client 构造前已加载

# 全局 LLM 客户端实例。base_url 仅在非官方端点时设置：
# - 默认（未设置环境变量）→ SDK 使用 api.anthropic.com
# - 设置为代理地址    → 所有请求经代理转发
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
