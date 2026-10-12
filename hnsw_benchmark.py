"""
hnsw_benchmark.py - IVF 与 HNSW 对比实验

回答一个问题：同样是近似检索，IVF 和 HNSW 该怎么选？

实验设计：
- 不调 API：直接复用已生成的 1024 维向量（data/vector_store_embeddings.npy），
  用数据集内随机抽样作为查询——完全可复现，且不产生任何 API 费用
- 精度基准：以 FAISS Flat（精确检索）的结果为 ground truth
- 对比四个维度：构建耗时 / 索引体积 / 检索延迟 / 召回率
- 额外做参数扫描：HNSW 的 efSearch、IVF 的 nprobe，看各自的精度-速度曲线

运行方式:
  python hnsw_benchmark.py
"""

import os
import sys
import json
import time
import statistics

import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NPY_PATH = os.path.join(DATA_DIR, "vector_store_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "vector_store.json")
RESULT_PATH = os.path.join(DATA_DIR, "hnsw_benchmark_result.json")

# ---- 实验参数 ----
N_QUERIES = 200   # 随机查询数
TOP_K = 10        # 召回率评估的 k
SEED = 42         # 固定种子，保证可复现
REPEATS = 5       # 每条查询重复次数，取中位数以抑制系统调度噪声

# HNSW 参数
HNSW_M = 32                  # 邻居连接数（内存与质量的主要旋钮）
HNSW_EF_CONSTRUCTION = 200   # 建图搜索宽度
HNSW_EF_SEARCH = 64          # 查询搜索宽度（默认值）

# IVF 参数
IVF_NLIST_RATIO = 100        # nlist = n / 100
IVF_NPROBE_RATIO = 8         # nprobe = nlist / 8（与本项目既有配置一致）


def load_vectors():
    """加载已生成的向量与元数据（不调 API）"""
    vectors = np.load(NPY_PATH).astype(np.float32)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # 归一化：内积等价余弦相似度的前提
    faiss.normalize_L2(vectors)
    return vectors, meta


def index_size_mb(index) -> float:
    """索引序列化后的体积（MB），作为存储/内存开销的量化指标"""
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), f"_size_{id(index)}.idx")
    try:
        faiss.write_index(index, tmp)
        size = os.path.getsize(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return size / 1024 / 1024


def build_flat(vectors):
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, time.perf_counter() - t0


def build_ivf(vectors):
    n = len(vectors)
    nlist = max(min(n // IVF_NLIST_RATIO, 1024), 1)
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    # 必须显式指定 METRIC_INNER_PRODUCT，否则 IVFFlat 默认 L2 距离（本项目的关键踩坑）
    index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist,
                               faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = max(nlist // IVF_NPROBE_RATIO, 1)
    return index, time.perf_counter() - t0


def build_hnsw(vectors, m=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION,
               ef_search=HNSW_EF_SEARCH):
    t0 = time.perf_counter()
    # HNSW 无需 train，直接建图；同样必须显式指定内积度量
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    index.hnsw.efSearch = ef_search
    return index, time.perf_counter() - t0


def run_queries(index, queries, k, warmup=5):
    """逐条查询并计时，返回 (结果索引矩阵, 延迟列表ms)

    每条查询重复 REPEATS 次并取中位数：单次计时会被系统调度/缓存抖动干扰，
    中位数能反映稳定性能，让参数扫描的曲线单调可比。
    """
    for i in range(min(warmup, len(queries))):
        index.search(queries[i:i + 1], k)

    latencies = []
    all_ids = []
    for i in range(len(queries)):
        reps = []
        ids = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            _, ids = index.search(queries[i:i + 1], k)
            reps.append((time.perf_counter() - t0) * 1000)
        latencies.append(statistics.median(reps))
        all_ids.append(ids[0])
    return np.array(all_ids), latencies


def recall_at_k(ground_truth, got, k):
    """召回率：ANN 结果与精确结果的 top-k 重合比例"""
    hits = 0
    for gt_row, got_row in zip(ground_truth, got):
        hits += len(set(gt_row[:k].tolist()) & set(got_row[:k].tolist()))
    return hits / (len(ground_truth) * k)


def latency_stats(latencies):
    s = sorted(latencies)
    return {
        "mean": statistics.mean(s),
        "p50": s[len(s) // 2],
        "p95": s[int(len(s) * 0.95)],
    }


def main():
    print("=" * 62)
    print("  IVF vs HNSW 对比实验")
    print("=" * 62)

    vectors, meta = load_vectors()
    n, dim = vectors.shape
    print(f"\n向量库: {n} 条 × {dim} 维")
    print(f"查询数: {N_QUERIES} 条（数据集内随机抽样，seed={SEED}）")
    print(f"top_k:  {TOP_K}  |  每条查询重复 {REPEATS} 次取中位数")

    # 随机抽样作为查询（固定种子保证可复现）
    rng = np.random.RandomState(SEED)
    q_idx = rng.choice(n, size=min(N_QUERIES, n), replace=False)
    queries = vectors[q_idx].copy()

    # ---- 构建四种索引 ----
    print("\n" + "-" * 62)
    print("  构建索引")
    print("-" * 62)

    flat, t_flat = build_flat(vectors)
    print(f"  Flat : 构建 {t_flat:.2f}s")

    ivf, t_ivf = build_ivf(vectors)
    print(f"  IVF  : 构建 {t_ivf:.2f}s (nlist={ivf.nlist}, nprobe={ivf.nprobe}, "
          f"扫描 {ivf.nprobe / ivf.nlist * 100:.1f}% 聚类)")

    hnsw, t_hnsw = build_hnsw(vectors)
    print(f"  HNSW : 构建 {t_hnsw:.2f}s (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}, "
          f"efSearch={HNSW_EF_SEARCH})")

    # 索引体积
    size_flat = index_size_mb(flat)
    size_ivf = index_size_mb(ivf)
    size_hnsw = index_size_mb(hnsw)
    print(f"\n  索引体积: Flat {size_flat:.1f}MB | IVF {size_ivf:.1f}MB | "
          f"HNSW {size_hnsw:.1f}MB")

    # ---- 精度基准：Flat 的结果视为正确答案 ----
    gt_ids, _ = run_queries(flat, queries, TOP_K)
    print(f"\n  精度基准已建立（Flat 精确结果）")

    # ---- 三引擎对比（各自默认参数）----
    print("\n" + "=" * 62)
    print("  性能与精度对比（默认参数）")
    print("=" * 62)

    results = []
    for name, index, build_time, size_mb in [
        ("Flat (精确)", flat, t_flat, size_flat),
        ("IVF (近似)", ivf, t_ivf, size_ivf),
        ("HNSW (近似)", hnsw, t_hnsw, size_hnsw),
    ]:
        ids, lats = run_queries(index, queries, TOP_K)
        st = latency_stats(lats)
        rec = recall_at_k(gt_ids, ids, TOP_K)
        results.append({
            "engine": name,
            "build_s": round(build_time, 2),
            "size_mb": round(size_mb, 2),
            "mean_ms": round(st["mean"], 3),
            "p50_ms": round(st["p50"], 3),
            "p95_ms": round(st["p95"], 3),
            "recall@10": round(rec, 4),
        })
        print(f"\n  [{name}]")
        print(f"    构建 {build_time:.2f}s | 体积 {size_mb:.1f}MB")
        print(f"    延迟 平均 {st['mean']:.3f}ms  P50 {st['p50']:.3f}ms  P95 {st['p95']:.3f}ms")
        print(f"    召回@{TOP_K} = {rec * 100:.2f}%")

    # ---- HNSW efSearch 扫描：精度-速度旋钮 ----
    print("\n" + "=" * 62)
    print("  HNSW 参数扫描：efSearch 对精度/速度的影响")
    print("=" * 62)
    print(f"  {'efSearch':>9} | {'平均延迟':>9} | {'召回@10':>8}")
    print(f"  {'-' * 9}-+-{'-' * 9}-+-{'-' * 8}")

    hnsw_sweep = []
    for ef in [16, 32, 64, 128, 256]:
        hnsw.hnsw.efSearch = ef
        ids, lats = run_queries(hnsw, queries, TOP_K, warmup=3)
        st = latency_stats(lats)
        rec = recall_at_k(gt_ids, ids, TOP_K)
        hnsw_sweep.append({"efSearch": ef,
                           "mean_ms": round(st["mean"], 3),
                           "recall@10": round(rec, 4)})
        print(f"  {ef:>9} | {st['mean']:>7.3f}ms | {rec * 100:>7.2f}%")

    # ---- IVF nprobe 扫描（对照组）----
    print("\n" + "=" * 62)
    print("  IVF 参数扫描：nprobe 对精度/速度的影响")
    print("=" * 62)
    print(f"  {'nprobe':>9} | {'扫描比例':>8} | {'平均延迟':>9} | {'召回@10':>8}")
    print(f"  {'-' * 9}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 8}")

    ivf_sweep = []
    for nprobe in [1, 2, 4, 8, 16, 32, 64]:
        if nprobe > ivf.nlist:
            continue
        ivf.nprobe = nprobe
        ids, lats = run_queries(ivf, queries, TOP_K, warmup=3)
        st = latency_stats(lats)
        rec = recall_at_k(gt_ids, ids, TOP_K)
        ivf_sweep.append({"nprobe": nprobe,
                          "scan_ratio": round(nprobe / ivf.nlist, 4),
                          "mean_ms": round(st["mean"], 3),
                          "recall@10": round(rec, 4)})
        print(f"  {nprobe:>9} | {nprobe / ivf.nlist * 100:>7.1f}% | "
              f"{st['mean']:>7.3f}ms | {rec * 100:>7.2f}%")

    # ---- HNSW M 扫描：图的质量上限（回答"精度天花板在哪"）----
    # efSearch 只能让搜索更充分，但若图本身连边不够，召回率就是封顶的。
    # M 决定图的连通度，因此是召回率上限的真正来源。
    print("\n" + "=" * 62)
    print("  HNSW 参数扫描：M（图连通度）对精度/体积的影响")
    print("=" * 62)
    print(f"  {'M':>4} | {'构建耗时':>9} | {'索引体积':>9} | {'召回@10':>8}")
    print(f"  {'-' * 4}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}")

    hnsw_m_sweep = []
    for m in [8, 16, 32, 64]:
        idx_m, t_m = build_hnsw(vectors, m=m, ef_search=128)
        size_m = index_size_mb(idx_m)
        ids, _ = run_queries(idx_m, queries, TOP_K, warmup=3)
        rec = recall_at_k(gt_ids, ids, TOP_K)
        hnsw_m_sweep.append({"M": m,
                             "build_s": round(t_m, 2),
                             "size_mb": round(size_m, 2),
                             "recall@10": round(rec, 4)})
        print(f"  {m:>4} | {t_m:>7.2f}s | {size_m:>7.1f}MB | {rec * 100:>7.2f}%")

    # ---- 结论 ----
    r_ivf = next(r for r in results if r["engine"].startswith("IVF"))
    r_hnsw = next(r for r in results if r["engine"].startswith("HNSW"))
    print("\n" + "=" * 62)
    print("  结论")
    print("=" * 62)
    speed_gain = r_ivf["mean_ms"] / r_hnsw["mean_ms"] if r_hnsw["mean_ms"] else 0
    print(f"  默认参数下：")
    print(f"    HNSW 延迟 {r_hnsw['mean_ms']:.3f}ms vs IVF {r_ivf['mean_ms']:.3f}ms "
          f"→ HNSW 快 {speed_gain:.1f}x")
    print(f"    HNSW 召回 {r_hnsw['recall@10'] * 100:.2f}% vs IVF {r_ivf['recall@10'] * 100:.2f}%")
    print(f"    HNSW 体积 {r_hnsw['size_mb']:.1f}MB vs IVF {r_ivf['size_mb']:.1f}MB "
          f"→ HNSW 多占 {r_hnsw['size_mb'] / r_ivf['size_mb']:.2f}x")
    print(f"    构建耗时 HNSW {r_hnsw['build_s']:.2f}s vs IVF {r_ivf['build_s']:.2f}s")

    # 落盘结果，供报告引用
    payload = {
        "dataset": {"n": int(n), "dim": int(dim)},
        "n_queries": int(len(queries)),
        "top_k": TOP_K,
        "seed": SEED,
        "repeats": REPEATS,
        "hnsw_params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION,
                        "efSearch": HNSW_EF_SEARCH},
        "ivf_params": {"nlist": int(ivf.nlist), "nprobe": int(ivf.nprobe)},
        "results": results,
        "hnsw_efSearch_sweep": hnsw_sweep,
        "hnsw_M_sweep": hnsw_m_sweep,
        "ivf_nprobe_sweep": ivf_sweep,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {RESULT_PATH}")


if __name__ == "__main__":
    main()