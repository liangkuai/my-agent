import os
from anthropic import Anthropic

import config

# 初始化 LLM Client
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
