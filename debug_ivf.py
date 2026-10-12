"""
debug_ivf.py - 诊断 FAISS IVF 为什么返回空

直接打印 IVF 检索的原始分数和索引，不做任何过滤。
"""

import sys
import os
import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(__file__))

from memory.embeddings import EmbeddingClient
from memory.vector_store import NumPyVectorStore
import config

def main():
    embedder = EmbeddingClient()
    numpy_store = NumPyVectorStore()

    if numpy_store.count() == 0:
        print("向量库为空")
        return

    # 从 NumPy store 拿向量
    vectors = np.array([e.embedding for e in numpy_store.entries], dtype=np.float32)
    print(f"向量数: {len(vectors)}, 维度: {vectors.shape[1]}")
    print(f"向量模长范围: {np.linalg.norm(vectors, axis=1).min():.4f} ~ {np.linalg.norm(vectors, axis=1).max():.4f}")

    # 归一化
    faiss.normalize_L2(vectors)

    # 构建 IVF
    nlist = 20
    quantizer = faiss.IndexFlatIP(config.VECTOR_DIM)
    index = faiss.IndexIVFFlat(quantizer, config.VECTOR_DIM, nlist)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = 10  # 搜一半聚类

    print(f"IVF: nlist={nlist}, nprobe={index.nprobe}, 向量数={index.ntotal}")

    # 测试 3 个查询
    test_queries = [
        "C++ 智能指针怎么用",
        "项目需要实现分布式锁",
        "周末去爬山了感觉很累",
    ]

    for query in test_queries:
        query_vec = np.array([embedder.embed(query)], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = index.search(query_vec, 10)

        print(f"\n查询: {query}")
        print(f"  原始 top-10 分数: {[f'{float(s):.4f}' for s in scores[0]]}")
        print(f"  原始 top-10 索引: {[int(i) for i in indices[0]]}")

        # 对比 NumPy
        query_np = np.array(embedder.embed(query), dtype=np.float32)
        matrix = vectors  # 已归一化
        sims = matrix @ query_np  # 内积 = 余弦相似度（已归一化）
        top_np = np.argsort(sims)[::-1][:5]
        print(f"  NumPy top-5: {[(int(i), f'{float(sims[i]):.4f}') for i in top_np]}")
        print(f"  NumPy top-1 文本: {numpy_store.entries[top_np[0]].text[:50]}")


if __name__ == "__main__":
    main()
