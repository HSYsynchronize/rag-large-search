"""
embeddings.py - Embedding 客户端

复用旧项目，封装 OpenAI 兼容 API。
"""

from openai import OpenAI
import config


class EmbeddingClient:
    def __init__(self):
        if not config.API_KEY:
            raise RuntimeError(
                "未找到 OPENAI_API_KEY。\n"
                "请复制 .env.example 为 .env，并填入你的 API key 与网关地址，"
                "或直接设置同名环境变量。"
            )
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.API_BASE)
        self.model = config.EMBEDDING_MODEL

    def embed(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码，减少 API 调用次数"""
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]
