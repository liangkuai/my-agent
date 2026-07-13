"""
全局配置加载 —— 在模块导入时一次性把 .env 注入 os.environ。

本模块是整个项目的配置入口，必须在所有依赖环境变量的模块（llm_client、constant 等）
之前被导入。Python 的模块缓存机制保证 load_dotenv 只执行一次，后续 import config
不会重复加载。
"""

from dotenv import load_dotenv

# override=True：.env 中的值覆盖已有的同名环境变量。
# 这在 IDE 内嵌终端等场景下很重要——系统可能预设了不同的环境变量，
# 而项目级 .env 应该优先。
load_dotenv(override=True)

