"""
vector_store.py - 向量存储（NumPy 暴力检索版）

和旧项目的区别：
- 支持元数据（timestamp、topic、role），检索时可过滤
- 批量操作优化
- 检索时返回耗时，方便和 FAISS 对比

这是基线版本，用来感受 NumPy 暴力检索在 1000 条下的性能瓶颈。
"""

import json
import os
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
import config


@dataclass
class ConversationEntry:
    """一条对话记忆（带元数据）"""
    id: str
    text: str
    embedding: list[float]
    role: str          # "user" | "assistant"
    topic: str          # 主题分类
    timestamp: str      # ISO 时间


class NumPyVectorStore:
    """NumPy 暴力检索：每次检索遍历全部向量"""

    def __init__(self, path: str = None):
        self.path = path or os.path.join(
            os.path.dirname(__file__), "..", "data", "vector_store.json"
        )
        # embedding 单独存 .npy 二进制文件，避免 JSON 几百 MB 读写慢
        self._emb_path = self.path.replace(".json", "_embeddings.npy")
        self.entries: list[ConversationEntry] = []
        self._matrix: Optional[np.ndarray] = None
        self.load()

    def add_batch(self, entries: list[ConversationEntry]):
        """批量添加"""
        self.entries.extend(entries)
        self._matrix = None
        self.persist()

    def _build_matrix(self) -> np.ndarray:
        if not self.entries:
            return np.array([])
        vectors = [e.embedding for e in self.entries]
        self._matrix = np.array(vectors, dtype=np.float32)
        return self._matrix

    def search(self, query_vec: list[float], top_k: int = 5,
               threshold: float = None,
               topic_filter: str = None,
               exclude_text: str = None) -> list[tuple[ConversationEntry, float, float]]:
        """
        检索最相似的 top_k 条。

        返回: [(entry, similarity_score, search_time_ms), ...]
        search_time_ms 是本次检索耗时，用于和 FAISS 对比
        """
        if not self.entries:
            return [], 0.0

        t_start = time.perf_counter()

        matrix = self._build_matrix()
        query = np.array(query_vec, dtype=np.float32)

        # 余弦相似度
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query)
        if query_norm == 0 or np.any(norms == 0):
            return [], 0.0

        similarities = (matrix / norms) @ (query / query_norm)

        # 元数据过滤：在排序前先过滤，减少排序量
        valid_mask = np.ones(len(self.entries), dtype=bool)
        for i, entry in enumerate(self.entries):
            if topic_filter and entry.topic != topic_filter:
                valid_mask[i] = False
            if exclude_text and entry.text == exclude_text:
                valid_mask[i] = False

        # 过滤后的索引排序
        valid_indices = np.where(valid_mask)[0]
        valid_scores = similarities[valid_indices]

        fetch_k = top_k + 5 if exclude_text else top_k
        top_local = np.argsort(valid_scores)[::-1][:fetch_k]

        results = []
        for local_idx in top_local:
            global_idx = valid_indices[local_idx]
            score = float(similarities[global_idx])
            entry = self.entries[global_idx]

            if threshold and score < threshold:
                continue
            if score > 0:
                results.append((entry, score))

            if len(results) >= top_k:
                break

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000

        return results, elapsed_ms

    def count(self) -> int:
        return len(self.entries)

    def clear(self):
        self.entries = []
        self._matrix = None
        if os.path.exists(self.path):
            os.remove(self.path)
        if os.path.exists(self._emb_path):
            os.remove(self._emb_path)

    def persist(self):
        """持久化：元数据存 JSON，embedding 存 .npy（二进制，紧凑百倍）"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # 元数据存 JSON（不含 embedding，文件小）
        meta_data = []
        for e in self.entries:
            d = asdict(e)
            d.pop("embedding", None)
            meta_data.append(d)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False)

        # embedding 存 .npy（2 万条 × 1024 维 = 80MB，JSON 要 400MB+）
        if self.entries:
            embeddings = np.array([e.embedding for e in self.entries], dtype=np.float32)
            np.save(self._emb_path, embeddings)

    def load(self):
        """加载：先读元数据 JSON，再读 embedding .npy，组合成 entries"""
        # 元数据
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except (json.JSONDecodeError, Exception):
                meta_data = []
        else:
            meta_data = []

        # embedding
        embeddings = np.array([], dtype=np.float32)
        if os.path.exists(self._emb_path):
            try:
                embeddings = np.load(self._emb_path)
            except Exception:
                embeddings = np.array([], dtype=np.float32)

        # 组装
        self.entries = []
        for i, d in enumerate(meta_data):
            if i < len(embeddings):
                d["embedding"] = embeddings[i].tolist()
            else:
                d["embedding"] = []
            self.entries.append(ConversationEntry(**d))
        self._matrix = None
