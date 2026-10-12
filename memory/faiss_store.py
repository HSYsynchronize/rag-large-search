"""
faiss_store.py - FAISS 向量存储

用 FAISS 的 IVF 索引替代 NumPy 暴力检索。
核心区别：FAISS 用 ANN（近似最近邻）算法，不全量遍历，大数据量下快得多。

索引类型选择：
- IndexFlatIP: 精确检索（和 NumPy 一样慢，作为对照基线）
- IndexIVFFlat: 近似检索（先聚类再搜索，牺牲少量精度换大幅提速）
"""

import json
import os
import time
import numpy as np
import faiss
from dataclasses import asdict
from typing import Optional
import config
from .vector_store import ConversationEntry


class FaissVectorStore:
    """FAISS 向量存储：支持精确检索和近似检索两种模式"""

    def __init__(self, path: str = None, mode: str = "flat"):
        """
        mode:
          "flat" → IndexFlatIP，精确检索（和 NumPy 一样的全量遍历，作为对照）
          "ivf"  → IndexIVFFlat，近似检索（ANN，大数据量下快）
        """
        self.path = path or os.path.join(
            os.path.dirname(__file__), "..", "data", "faiss_store.json"
        )
        self.mode = mode
        self.entries: list[ConversationEntry] = []
        self._index: Optional[faiss.Index] = None
        self._dim = config.VECTOR_DIM
        self.load()

    def add_batch(self, entries: list[ConversationEntry]):
        """批量添加：构建 FAISS 索引"""
        self.entries.extend(entries)

        # 准备向量矩阵，L2 归一化后再入 FAISS
        # 为什么：FAISS 用内积（IP）模拟余弦相似度，前提是向量已归一化
        # NumPy 那边手动做了 matrix/norms，FAISS 这边也要做，否则聚类质量差
        vectors = np.array([e.embedding for e in self.entries], dtype=np.float32)
        faiss.normalize_L2(vectors)

        if self.mode == "ivf":
            # IVF 索引：先聚类，搜索时只查最近的 nprobe 个聚类
            # IVF 参数选择：
            # nlist（聚类数）：数据量越大越多，但太多则每个聚类没几个向量
            #   经验值：n/100，上限 1024
            # nprobe（搜索聚类数）：越小越快但越不精确
            #   2000 条时 nlist//2 搜 50% 还能接受
            #   大规模时必须降到 nlist//8（12.5%），否则 IVF 和 Flat 没区别
            nlist = int(min(max(len(self.entries) // 100, 4), 1024))
            nlist = max(nlist, 1)
            quantizer = faiss.IndexFlatIP(self._dim)
            # 关键：必须显式指定 METRIC_INNER_PRODUCT，否则 IVFFlat 默认用 L2 距离
            # L2 距离越小越相似，和余弦相似度（越大越相似）方向相反，会导致阈值过滤把正确结果全部干掉
            self._index = faiss.IndexIVFFlat(quantizer, self._dim, nlist,
                                             faiss.METRIC_INNER_PRODUCT)
            self._index.train(vectors)
            self._index.add(vectors)
            # 搜 12.5% 的聚类：大规模下这才是 IVF 的意义
            self._index.nprobe = max(nlist // 8, 4)
        else:
            # Flat 索引：精确检索，全量遍历
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(vectors)

        self.persist()

    def search(self, query_vec: list[float], top_k: int = 5,
               threshold: float = None,
               topic_filter: str = None,
               exclude_text: str = None) -> list[tuple[ConversationEntry, float, float]]:
        """
        检索。返回: [(entry, score, search_time_ms), ...]

        注意：FAISS 的 topic 过滤是先检索再过滤（post-filter），
        不是 NumPy 那种先过滤再排序。因为 FAISS 索引不感知元数据。
        所以过滤后返回的条目数可能少于 top_k。
        """
        if not self.entries or self._index is None:
            return [], 0.0

        t_start = time.perf_counter()

        query = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(query)  # query 也要归一化

        # FAISS 检索：返回距离和索引
        # 过滤时多取一些，保证过滤后还有足够数量
        fetch_k = top_k
        if topic_filter or exclude_text:
            fetch_k = top_k * 5  # 多取 5 倍用于过滤

        scores, indices = self._index.search(query, fetch_k)

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000

        # 调试：打印原始返回的前 5 个分数
        if os.getenv("FAISS_DEBUG"):
            raw_scores = [f"{float(scores[0][i]):.4f}" for i in range(min(5, len(indices[0])))]
            print(f"  [DEBUG] raw top5 scores: {raw_scores}")

        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            score = float(scores[0][i])

            if idx < 0:  # FAISS 返回 -1 表示没有更多结果
                continue
            if score <= 0:
                continue

            entry = self.entries[idx]

            # post-filter
            if topic_filter and entry.topic != topic_filter:
                continue
            if exclude_text and entry.text == exclude_text:
                continue
            if threshold and score < threshold:
                continue

            results.append((entry, score))
            if len(results) >= top_k:
                break

        return results, elapsed_ms

    def count(self) -> int:
        return len(self.entries)

    def clear(self):
        self.entries = []
        self._index = None
        if os.path.exists(self.path):
            os.remove(self.path)

    def persist(self):
        """持久化：元数据存 JSON，FAISS 索引存 .index 文件"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # 存元数据（不含 embedding，太大）
        data = []
        for e in self.entries:
            d = asdict(e)
            d.pop("embedding", None)  # embedding 存在 FAISS 索引里
            data.append(d)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        # 存 FAISS 索引
        if self._index is not None:
            index_path = self.path.replace(".json", ".index")
            faiss.write_index(self._index, index_path)

    def load(self):
        """加载元数据，索引需要重新构建（因为 embedding 不存 JSON）"""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 注意：这里加载的条目没有 embedding
                # 需要调用 rebuild_index() 重建索引
                self.entries = []
                for d in data:
                    d["embedding"] = []  # 占位
                    self.entries.append(ConversationEntry(**d))
            except (json.JSONDecodeError, Exception):
                self.entries = []
        self._index = None
