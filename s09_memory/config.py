"""
应用配置入口 —— 在模块导入链的最前端加载 .env 环境变量。

constant.py import config，因此所有 import constant 的模块在首次加载时
都会先触发 dotenv 注入。这确保了 MODEL、WORKDIR 等依赖环境变量的常量
在定义时就拿到正确的值。

override=True：.env 中的值覆盖系统环境变量已存在的同名变量，
本地开发时 .env 可完全控制配置，无需手动 unset。
"""

from dotenv import load_dotenv

load_dotenv(override=True)

