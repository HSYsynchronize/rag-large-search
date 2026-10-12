"""
benchmark.py - NumPy vs FAISS 性能对比

同样的数据，同样的查询，对比三种引擎的耗时和结果质量：
  A) NumPy 暴力检索（基线）
  B) FAISS Flat（精确检索，对照）
  C) FAISS IVF（近似检索，ANN）

两部分验证：
  1. 手写 10 条查询 → 性能对比 + 逐条一致性检查
  2. 从数据集随机抽 100 条 → 批量一致性验证（只看 top-1 是否和 NumPy 一致）

运行方式:
  python benchmark.py
"""

import sys
import os
import time
import json
import statistics
import random

sys.path.insert(0, os.path.dirname(__file__))

from memory.embeddings import EmbeddingClient
from memory.vector_store import NumPyVectorStore
from memory.faiss_store import FaissVectorStore
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

# 批量验证的查询数
BATCH_VERIFY_COUNT = 100


def benchmark_engine(name, store, embedder, queries):
    """对一个引擎跑全部查询，返回结果和耗时"""
    print(f"\n  [{name}]")
    print(f"  {'─' * 46}")

    latencies = []
    results_all = []  # 存每条查询的 top-1，用于和别的引擎对比

    for query in queries:
        query_vec = embedder.embed(query)
        results, elapsed_ms = store.search(
            query_vec, top_k=config.TOP_K,
            threshold=config.SIMILARITY_THRESHOLD,
        )
        latencies.append(elapsed_ms)

        top1 = results[0] if results else None
        top1_text = top1[0].text[:40] if top1 else "N/A"
        top1_score = f"{top1[1]:.4f}" if top1 else "N/A"
        results_all.append((top1_text, top1_score))

        print(f"    [{elapsed_ms:.2f}ms] {query[:18]}... → {top1_score} {top1_text}")

    print(f"\n  平均: {statistics.mean(latencies):.2f}ms | "
          f"最大: {max(latencies):.2f}ms | "
          f"最小: {min(latencies):.2f}ms")

    return results_all


def run_benchmark():
    embedder = EmbeddingClient()

    # 加载三种引擎
    # NumPy 版（从已存的 JSON 加载）
    numpy_store = NumPyVectorStore()

    # FAISS Flat 版（需要重新构建索引，因为上次存的是 NumPy 格式）
    print("构建 FAISS Flat 索引...")
    faiss_flat_store = FaissVectorStore(mode="flat")
    faiss_flat_store.clear()
    # 从 NumPy store 复制数据
    faiss_flat_store.add_batch(numpy_store.entries)
    print(f"  FAISS Flat: {faiss_flat_store.count()} 条")

    # FAISS IVF 版
    print("构建 FAISS IVF 索引...")
    t_build = time.perf_counter()
    faiss_ivf_store = FaissVectorStore(mode="ivf")
    faiss_ivf_store.clear()
    faiss_ivf_store.add_batch(numpy_store.entries)
    ivf_build_time = time.perf_counter() - t_build
    print(f"  FAISS IVF: {faiss_ivf_store.count()} 条 (构建耗时 {ivf_build_time:.2f}s)")
    # 打印 IVF 参数
    if faiss_ivf_store._index is not None:
        ivf_index = faiss_ivf_store._index
        nlist = ivf_index.nlist
        nprobe = ivf_index.nprobe
        scan_ratio = nprobe / nlist * 100 if nlist > 0 else 0
        print(f"  IVF 参数: nlist={nlist}, nprobe={nprobe} (扫描 {scan_ratio:.1f}% 聚类)")

    print(f"\n向量库条目数: {numpy_store.count()}")
    print(f"Top-K: {config.TOP_K}")

    # 三路对比
    print("\n" + "=" * 50)
    print("  性能对比：NumPy vs FAISS Flat vs FAISS IVF")
    print("=" * 50)

    numpy_results = benchmark_engine("NumPy 暴力检索", numpy_store, embedder, TEST_QUERIES)
    flat_results = benchmark_engine("FAISS Flat (精确)", faiss_flat_store, embedder, TEST_QUERIES)
    ivf_results = benchmark_engine("FAISS IVF (近似)", faiss_ivf_store, embedder, TEST_QUERIES)

    # 结果质量对比：FAISS IVF 的 top-1 和 NumPy 的 top-1 是否一致
    print("\n" + "=" * 50)
    print("  结果一致性：FAISS IVF vs NumPy")
    print("=" * 50)

    match_count = 0
    for i, query in enumerate(TEST_QUERIES):
        numpy_text, numpy_score = numpy_results[i]
        ivf_text, ivf_score = ivf_results[i]
        match = numpy_text == ivf_text
        if match:
            match_count += 1
        status = "一致" if match else "不同"
        print(f"  [{status}] {query[:20]}...")
        if not match:
            print(f"    NumPy: {numpy_score} {numpy_text}")
            print(f"    IVF:   {ivf_score} {ivf_text}")

    print(f"\n  一致率: {match_count}/{len(TEST_QUERIES)}")

    # ---- 批量验证：从数据集随机抽 100 条做查询 ----
    print("\n" + "=" * 50)
    print(f"  批量一致性验证：随机 {BATCH_VERIFY_COUNT} 条查询")
    print("=" * 50)

    # 从 conversations.json 随机抽取 user 消息作为查询
    data_path = os.path.join(os.path.dirname(__file__), "data", "conversations.json")
    with open(data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    random.seed(123)
    sampled = random.sample(conversations, min(BATCH_VERIFY_COUNT, len(conversations)))
    batch_queries = [c["user"] for c in sampled]

    # 用 NumPy 和 IVF 分别检索，比较 top-5 的 entry.id 重叠率
    # 不打印每条（100 条太多），只统计结果
    batch_match_top1 = 0
    batch_overlap_sum = 0  # top-5 重叠数累加
    batch_mismatches = []  # 记录 top-1 不一致的，最多打印 5 条
    batch_numpy_latencies = []
    batch_ivf_latencies = []
    batch_score_diffs = []  # top-1 分数差异

    for query in batch_queries:
        query_vec = embedder.embed(query)

        numpy_res, numpy_ms = numpy_store.search(query_vec, top_k=5)
        ivf_res, ivf_ms = faiss_ivf_store.search(query_vec, top_k=5)
        batch_numpy_latencies.append(numpy_ms)
        batch_ivf_latencies.append(ivf_ms)

        numpy_ids = set(e[0].id for e in numpy_res)
        ivf_ids = set(e[0].id for e in ivf_res)

        # top-1 一致性
        numpy_top1_id = numpy_res[0][0].id if numpy_res else None
        ivf_top1_id = ivf_res[0][0].id if ivf_res else None
        if numpy_top1_id == ivf_top1_id:
            batch_match_top1 += 1

        # top-5 重叠率
        overlap = len(numpy_ids & ivf_ids)
        batch_overlap_sum += overlap

        # top-1 分数差异
        numpy_top1_score = numpy_res[0][1] if numpy_res else 0
        ivf_top1_score = ivf_res[0][1] if ivf_res else 0
        batch_score_diffs.append(abs(numpy_top1_score - ivf_top1_score))

        # 记录不一致的
        if numpy_top1_id != ivf_top1_id and len(batch_mismatches) < 5:
            batch_mismatches.append({
                "query": query[:40],
                "numpy_id": numpy_top1_id,
                "numpy_score": f"{numpy_top1_score:.4f}",
                "ivf_id": ivf_top1_id,
                "ivf_score": f"{ivf_top1_score:.4f}",
                "overlap": f"{overlap}/5",
            })

    n = len(batch_queries)
    avg_overlap = batch_overlap_sum / n
    avg_score_diff = statistics.mean(batch_score_diffs)
    max_score_diff = max(batch_score_diffs)

    print(f"  样本数: {n}")
    print(f"  top-1 一致: {batch_match_top1}/{n} ({batch_match_top1/n*100:.1f}%)")
    print(f"  top-5 平均重叠: {avg_overlap:.1f}/5 ({avg_overlap/5*100:.1f}%)")
    print(f"  top-1 分数差异: 平均 {avg_score_diff:.4f}, 最大 {max_score_diff:.4f}")
    print(f"\n  耗时对比 (批量):")
    print(f"    NumPy 平均: {statistics.mean(batch_numpy_latencies):.2f}ms")
    print(f"    IVF   平均: {statistics.mean(batch_ivf_latencies):.2f}ms")

    if batch_mismatches:
        print(f"\n  top-1 不一致的查询 (最多显示 5 条):")
        for m in batch_mismatches:
            print(f"    [{m['query']}...]")
            print(f"      NumPy: {m['numpy_score']} {m['numpy_id']}")
            print(f"      IVF:   {m['ivf_score']} {m['ivf_id']}")
            print(f"      top-5 重叠: {m['overlap']}")
    else:
        print(f"\n  全部一致，IVF 在本数据集上 100% 召回")

    print("\n" + "=" * 50)
    print("  总结")
    print("=" * 50)
    print("  NumPy:  精确，但 O(n) 遍历，数据量翻倍耗时翻倍")
    print("  Flat:   精确，结果和 NumPy 一样，FAISS 底层优化略快")
    print("  IVF:    近似，可能漏掉少量结果，但大数据量下快得多")
    print("  规模建议: <1万条用 Flat 就行，>1万条用 IVF")


if __name__ == "__main__":
    run_benchmark()
