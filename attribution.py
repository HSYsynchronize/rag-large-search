"""
attribution.py - 召回损失的逐一归因

用途：不做"召回率 98.8% 是不是假象"这种二选一的判断，而是把每一个未命中位
     单独拿出来，判定它属于哪一类：

   A. exact_duplicate  未命中项在库里存在文本完全相同的副本，且该副本被 ANN 找到了
                       → 语义等价，纯属"选了另一个副本"，不是损失
   B. near_duplicate   未命中项虽无完全相同文本，但 ANN 找到的结果与它余弦相似度
                       高于阈值（近似改写句）
                       → 语义近似，可争议，但仍不是"漏掉了"
   C. boundary         未命中项的分数 <= ANN 实际返回结果的最低分
                       → 它本就在 top-k 的排序边界上，ANN 返回的集合按分数并不更差
   D. unexplained      以上都不成立 → 这才是真正的召回损失

产出：
  - 每类的计数与占比
  - 阈值敏感性曲线（A+B 的占比如何随阈值变化）
  - 分引擎结果（HNSW / IVF 各跑一遍，说明这不是某个引擎的问题）

设计要点：查询取自库内（与 hnsw_benchmark.py 一致），不调 API，固定种子可复现。

运行方式:
  python attribution.py                        # 默认 2000 条查询
  python attribution.py --queries 500 --k 10
"""

import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NPY_PATH = os.path.join(DATA_DIR, "vector_store_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "vector_store.json")
RESULT_PATH = os.path.join(DATA_DIR, "attribution_result.json")

# 判定"近似重复"的余弦相似度阈值（向量已归一化，内积即余弦）
TIE_THRESHOLD = 0.999
# 阈值敏感性扫描
THRESHOLD_SWEEP = [0.9999, 0.999, 0.995, 0.99, 0.98, 0.95, 0.90]

# 与项目既有实验保持一致的 IVF / HNSW 参数
IVF_NLIST_RATIO = 100
IVF_NPROBE_RATIO = 8
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def load_data():
    vectors = np.load(NPY_PATH).astype(np.float32)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    faiss.normalize_L2(vectors)
    return vectors, meta


def build_indexes(vectors):
    n, dim = vectors.shape
    indexes = {}

    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)
    indexes["Flat (精确, 基准)"] = flat

    nlist = max(min(n // IVF_NLIST_RATIO, 1024), 1)
    quantizer = faiss.IndexFlatIP(dim)
    ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)
    ivf.add(vectors)
    ivf.nprobe = max(nlist // IVF_NPROBE_RATIO, 1)
    indexes[f"IVF (nlist={nlist}, nprobe={ivf.nprobe})"] = ivf

    hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    hnsw.add(vectors)
    hnsw.hnsw.efSearch = HNSW_EF_SEARCH
    indexes[f"HNSW (M={HNSW_M}, efSearch={HNSW_EF_SEARCH})"] = hnsw

    return indexes


def attribute(vectors, query_ids, gt_ids, ann_ids, k, dup_text_group, thresholds):
    """把每个未命中位分类。

    query_ids:      查询向量在库中的 id（查询取自库内，本就是库里的条目）
    dup_text_group: dict[id] -> set(ids)，文本完全相同的条目组
    返回：分类计数 + 阈值敏感性 + 未命中明细
    """
    n_q = len(gt_ids)
    counts = Counter()
    per_threshold = {t: Counter() for t in thresholds}
    missed_total = 0
    details = []

    for i in range(n_q):
        qv = vectors[query_ids[i]]          # 查询向量本身
        gt = [int(x) for x in gt_ids[i][:k] if x >= 0]
        ann = [int(x) for x in ann_ids[i][:k] if x >= 0]
        gt_set, ann_set = set(gt), set(ann)
        missed = gt_set - ann_set
        if not missed:
            continue

        # ANN 实际返回结果中的最低分（用查询向量打分，未命中项也用同一把尺子）
        ann_scores = [float(qv @ vectors[a]) for a in ann]
        ann_min_score = min(ann_scores) if ann_scores else float("-inf")

        for m in missed:
            missed_total += 1
            mv = vectors[m]
            # 未命中项在 ANN 返回集合中最接近的替代者，及其相似度
            best_sim, best_sub = -1.0, None
            for a in ann:
                s = float(mv @ vectors[a])
                if s > best_sim:
                    best_sim, best_sub = s, a

            # A：库里存在与未命中项文本完全相同的条目，且 ANN 已经找到了那个副本
            exact_group = dup_text_group.get(m)
            is_exact_dup = bool(exact_group and best_sub in exact_group)

            if is_exact_dup:
                counts["A_exact_duplicate"] += 1
            elif best_sim >= TIE_THRESHOLD:
                counts["B_near_duplicate"] += 1
            else:
                # C：未命中项自身的分数并不高于 ANN 返回集合的最低分
                #    → 它本就在 top-k 的排序边界上，ANN 的集合按分数并不更差
                self_score = float(qv @ mv)
                if self_score <= ann_min_score + 1e-6:
                    counts["C_boundary"] += 1
                else:
                    counts["D_unexplained"] += 1

            for t in thresholds:
                per_threshold[t]["explained" if best_sim >= t else "unexplained"] += 1

            details.append({
                "query_idx": i,
                "missed_id": int(m),
                "best_substitute": int(best_sub) if best_sub is not None else None,
                "best_sim": round(best_sim, 6),
                "is_exact_dup": bool(is_exact_dup),
            })

    return counts, per_threshold, missed_total, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("  召回损失逐一归因")
    print("=" * 70)

    vectors, meta = load_data()
    n, dim = vectors.shape
    print(f"\n向量库: {n} 条 × {dim} 维")

    # 查询取自库内（与 hnsw_benchmark.py 一致），不调 API
    rng = np.random.RandomState(args.seed)
    q_idx = rng.choice(n, size=min(args.queries, n), replace=False)
    queries = vectors[q_idx].copy()
    print(f"查询数: {len(queries)} 条（库内随机抽样，seed={args.seed}）")
    print(f"top_k : {args.k}  → 命中位总数 {len(queries) * args.k}")

    # 文本完全相同的条目组
    text_to_ids = {}
    for i, m in enumerate(meta):
        text_to_ids.setdefault(m["text"], set()).add(i)
    dup_text_group = {i: grp for grp in text_to_ids.values() if len(grp) > 1 for i in grp}
    n_dup_items = sum(len(g) for g in text_to_ids.values() if len(g) > 1)
    print(f"文本重复: {n - len(text_to_ids)} 条重复条目（唯一文本 {len(text_to_ids)}）")

    indexes = build_indexes(vectors)
    # 注意：faiss 的 search 返回 (distances, labels)，别把距离当 ID 用
    _, gt_ids = indexes["Flat (精确, 基准)"].search(queries, args.k)

    summary = {"n": int(n), "dim": int(dim), "n_queries": int(len(queries)),
               "k": args.k, "seed": args.seed, "dup_items": int(n - len(text_to_ids)),
               "engines": {}}

    for name, index in indexes.items():
        if name.startswith("Flat"):
            continue
        print("\n" + "=" * 70)
        print(f"  {name}")
        print("=" * 70)

        _, ann_ids = index.search(queries, args.k)

        hits = sum(len(set(int(x) for x in gt_ids[i][:args.k]) &
                       set(int(x) for x in ann_ids[i][:args.k]))
                   for i in range(len(queries)))
        id_recall = hits / (len(queries) * args.k)

        counts, per_threshold, missed_total, details = attribute(
            vectors, q_idx, gt_ids, ann_ids, args.k, dup_text_group, THRESHOLD_SWEEP)

        print(f"\n  ID 层面召回@{args.k} = {id_recall * 100:.2f}%"
              f"  （未命中 {missed_total} 位 / 命中位 {len(queries) * args.k}）")

        print(f"\n  【归因分解】")
        labels = {
            "A_exact_duplicate": "A 文本完全重复的等价副本",
            "B_near_duplicate": "B 语义近似（改写句）",
            "C_boundary": "C 排序边界（分数不高于 ANN 最低分）",
            "D_unexplained": "D 未解释（真实损失）",
        }
        for key in ("A_exact_duplicate", "B_near_duplicate", "C_boundary", "D_unexplained"):
            c = counts.get(key, 0)
            rate = c / missed_total * 100 if missed_total else 0
            print(f"    {labels[key]:<32} {c:>5} 位  {rate:>6.1f}%")

        print(f"\n  【阈值敏感性】有多少未命中位可被『找到了等价替代』解释")
        print(f"    {'相似度阈值':>10} | {'可解释':>7} | {'未解释':>7} | {'可解释占比':>10}")
        print(f"    {'-' * 10}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 10}")
        sweep = []
        for t in THRESHOLD_SWEEP:
            e = per_threshold[t]["explained"]
            u = per_threshold[t]["unexplained"]
            tot = e + u
            rate = e / tot * 100 if tot else 0
            print(f"    {t:>10.4f} | {e:>7} | {u:>7} | {rate:>9.1f}%")
            sweep.append({"threshold": t, "explained": e,
                          "unexplained": u, "explained_rate": round(rate / 100, 4)})

        summary["engines"][name] = {
            "id_recall@k": round(id_recall, 4),
            "missed_positions": missed_total,
            "counts": dict(counts),
            "threshold_sweep": sweep,
            # 只留最有诊断价值的少量明细，避免结果文件过大
            "unexplained_samples": [d for d in details
                                    if not d["is_exact_dup"]
                                    and d["best_sim"] < TIE_THRESHOLD][:20],
        }

    print("\n" + "=" * 70)
    print("  怎么读这个结果")
    print("=" * 70)
    print("  1. 若 A+B 占未命中的绝大多数 → 所谓『召回损失』主要是评测指标造成的")
    print("     假象，实际检索质量接近无损。")
    print("  2. 若 D_unexplained 还有可观数量 → 存在真实的召回损失，需要调参数或")
    print("     考虑是近似检索本身的固有代价。")
    print("  3. 阈值敏感性表告诉你这个结论有多稳健：若在 0.99~0.9999 之间可解释占比")
    print("     都稳定在高位，说明结论不依赖阈值的选择。")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {RESULT_PATH}")


if __name__ == "__main__":
    main()