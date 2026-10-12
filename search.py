"""
search.py - 检索测试 + 性能基准

对 NumPy 暴力检索做性能测试：
- 多个查询的检索耗时
- 元数据过滤效果
- 和不同 top_k 的性能对比

运行方式:
  python search.py
"""

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.dirname(__file__))

from memory.embeddings import EmbeddingClient
from memory.vector_store import NumPyVectorStore
import config

TEST_QUERIES = [
    "C++ 智能指针怎么用",
    "RAII 管理资源生命周期",
    "项目需要实现分布式锁",
    "最近在学操作系统，虚拟内存怎么理解",
    "GPT-5 发布了有什么新功能",
    "周末去爬山了感觉很累",
    "内存泄漏怎么排查",
    "微服务还是单体架构怎么选",
    "TCP 三次握手的核心原理",
    "Rust 进 Linux 内核了",
]


def run_search():
    embedder = EmbeddingClient()
    store = NumPyVectorStore()

    if store.count() == 0:
        print("向量库为空，请先运行 python indexer.py")
        return

    print(f"向量库条目数: {store.count()}")
    print(f"检索引擎: NumPy 暴力检索")
    print(f"Top-K: {config.TOP_K}")
    print()

    # 1. 基础检索 + 耗时
    print("=" * 50)
    print("  1. 基础检索")
    print("=" * 50)

    latencies = []
    for query in TEST_QUERIES:
        query_vec = embedder.embed(query)
        results, elapsed_ms = store.search(
            query_vec, top_k=config.TOP_K,
            threshold=config.SIMILARITY_THRESHOLD,
        )
        latencies.append(elapsed_ms)

        top1_text = results[0][0].text[:50] if results else "(无结果)"
        top1_score = f"{results[0][1]:.4f}" if results else "N/A"
        top1_topic = results[0][0].topic if results else "N/A"
        print(f"  [{elapsed_ms:.2f}ms] {query[:20]}... → top1={top1_score} [{top1_topic}] {top1_text}")

    print(f"\n  平均耗时: {statistics.mean(latencies):.2f}ms")
    print(f"  最大耗时: {max(latencies):.2f}ms")
    print(f"  最小耗时: {min(latencies):.2f}ms")

    # 2. 元数据过滤效果
    print()
    print("=" * 50)
    print("  2. 元数据过滤（topic=cpp_programming）")
    print("=" * 50)

    query = "智能指针内存泄漏怎么排查"
    query_vec = embedder.embed(query)

    # 不过滤
    results_no_filter, time_no_filter = store.search(
        query_vec, top_k=config.TOP_K, threshold=config.SIMILARITY_THRESHOLD)
    print(f"  无过滤: [{time_no_filter:.2f}ms] {len(results_no_filter)} 条结果")
    for entry, score in results_no_filter:
        print(f"    [{score:.4f}] ({entry.topic}) {entry.text[:50]}")

    # 过滤
    results_filtered, time_filtered = store.search(
        query_vec, top_k=config.TOP_K, threshold=config.SIMILARITY_THRESHOLD,
        topic_filter="cpp_programming")
    print(f"\n  过滤后: [{time_filtered:.2f}ms] {len(results_filtered)} 条结果")
    for entry, score in results_filtered:
        print(f"    [{score:.4f}] ({entry.topic}) {entry.text[:50]}")

    # 3. 不同 top_k 性能对比
    print()
    print("=" * 50)
    print("  3. Top-K 性能对比")
    print("=" * 50)

    query_vec = embedder.embed("C++ 内存管理方案")
    for k in [1, 5, 10, 20, 50]:
        results, elapsed = store.search(query_vec, top_k=k, threshold=config.SIMILARITY_THRESHOLD)
        print(f"  top_k={k:3d}: {elapsed:.2f}ms, 返回 {len(results)} 条")

    print()
    print("=" * 50)
    print("  测试结束")
    print("=" * 50)


if __name__ == "__main__":
    run_search()
