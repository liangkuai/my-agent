"""
LLM 客户端模块 —— 初始化并暴露全局共享的 Anthropic SDK 客户端实例。

所有需要调用 Claude API 的模块（app、context、tools、memory）通过
`from llm_client import client` 共享同一实例，避免重复初始化。

ANTHROPIC_BASE_URL 为空时 SDK 默认连接 api.anthropic.com；
设置后可指向自定义代理或兼容 API 端点。
"""

import os
from anthropic import Anthropic

import config

# 全局共享的 Anthropic 客户端。base_url 由环境变量注入，
# 支持指向自定义代理或兼容服务端点。
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
