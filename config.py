import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

# 模型配置
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Pro"
IMAGE_MODEL_NAME = "Qwen/Qwen-Image"
BASE_URL = "https://api.siliconflow.cn/v1"
API_KEY = os.getenv("SILICONFLOW_API_KEY")
TEMPERATURE = 0.3

# 记忆配置
MAX_HISTORY_LENGTH = 20  # 超过20条触发压缩
KEEP_RECENT = 10         # 压缩时保留最近10条