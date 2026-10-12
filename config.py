"""
config.py - 项目配置

所有敏感配置（API key、网关地址）都从环境变量读取，不写进代码。
首次使用请复制 .env.example 为 .env 并填入自己的配置。

注意：.env 已在 .gitignore 中，不会被提交到仓库。
"""

import os


def _load_dotenv(path: str):
    """极简 .env 加载器（不引入额外依赖）。

    规则：每行 KEY=VALUE，忽略空行和 # 开头的注释。
    已存在的环境变量优先，不会被 .env 覆盖。
    """
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


# 项目根目录的 .env（与 config.py 同级）
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


# ---- API 配置 ----
# 请在 .env 或系统环境变量中设置，不要硬编码进代码（避免密钥泄露）
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# ---- 模型配置 ----
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ---- 数据配置 ----
# 模拟对话条目数
DATA_SIZE = int(os.getenv("DATA_SIZE", "10000"))

# ---- 检索配置 ----
TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# ---- 向量存储配置 ----
# 两种引擎：numpy（暴力检索）和 faiss（ANN 近似最近邻）
VECTOR_ENGINE = os.getenv("VECTOR_ENGINE", "numpy")  # numpy | faiss
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1536"))  # 必须与 EMBEDDING_MODEL 输出维度一致（text-embedding-3-small=1536，bge-m3=1024）

# ---- 生成配置 ----
TEMPERATURE = 0.7
MAX_TOKENS = 4096