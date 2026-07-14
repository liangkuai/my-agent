"""
应用配置加载模块。

负责在项目启动时从 .env 文件加载环境变量，供 llm_client、constant 等
模块通过 os.getenv() 读取。所有敏感配置（API Key、Base URL、Model ID 等）
均通过环境变量注入，不写入代码中，避免凭据泄露。
"""

from dotenv import load_dotenv

# override=True：.env 文件中的值会覆盖已存在的同名环境变量。
# 这样即使 shell 环境中已设置了旧值，也能保证项目根目录 .env 的配置优先生效，
# 便于在开发/测试/CI 等不同环境下切换配置。
load_dotenv(override=True)
