"""
配置初始化 —— 在项目导入链的最前端加载环境变量。

本模块是整个 s06_subagent 包中第一个被导入的模块（所有其他模块的 `import config`
都会触发它），因此承担「最早注入环境变量」的职责。

导入顺序保证：
1. config.py        → load_dotenv 把 .env 注入 os.environ
2. constant.py      → 从 os.environ 读取 MODEL_ID 等变量
3. llm_client.py    → 用环境变量初始化 Anthropic 客户端
4. tool.py / app.py → 消费上面的常量和客户端

`override=True`：.env 中的值会覆盖系统环境变量中已有的同名变量，确保本地
.env 文件作为单一事实来源，避免被 shell 环境中残留的旧变量干扰。
"""

from dotenv import load_dotenv

load_dotenv(override=True)
