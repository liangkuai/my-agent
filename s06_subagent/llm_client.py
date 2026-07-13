"""
LLM 客户端 —— 封装 Anthropic SDK 的初始化，向外暴露唯一的 `client` 实例。

本模块是项目中唯一的 SDK 导入点：其他模块（tool.py、app.py）不直接依赖
anthropic 包，而是通过 `from llm_client import client` 获取已配置的客户端。
这样当需要切换模型提供商或调整客户端参数时，只需修改本文件。

配置来源：
- ANTHROPIC_BASE_URL：API 端点地址，未设置则使用 Anthropic 默认值。
  常见用法是指向本地代理（如 http://localhost:8080/v1）或兼容网关。
- ANTHROPIC_API_KEY：由 SDK 自动从环境变量读取，无需显式传入。
"""

import os
from anthropic import Anthropic

import config  # noqa: F401 — 保证 load_dotenv 在本模块之前执行

# 全局共享的 Anthropic 客户端实例。
# base_url 指向 API 端点：留空用官方地址，设值则指向本地代理或兼容网关。
# 所有模型调用（主 agent 循环和子 agent spawn）共用同一个 client，
# SDK 内部管理连接池，无需多次创建。
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
