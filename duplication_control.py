"""
duplication_control.py - 重复率的控制变量实验

问题：attribution.py 在真实数据集上发现，HNSW 的召回损失 100% 来自"并列副本"，
      IVF 则还有一部分真实损失。但这是**观察**，不是**因果证明**。

本实验做控制变量：人为构造不同内容重复率的数据集，保持其它条件不变，观察
ID 层面召回率如何随重复率变化。

  - 若 ID 召回随重复率单调下降，而去掉"并列副本假象"后的真实召回基本不变
    → 证明"ID 一致率"这个指标本身会随数据重复率系统性失真
  - r = 0 的那一行就是**零重复对照组**

这同时回答了 README「下一步」的第 2、3 条。

运行方式:
  python duplication_control.py
  python duplication_control.py --base 8000 --queries 1500 --rates 0,0.1,0.25,0.5
"""

import os
import sys
import json
import argparse

import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribution import attribute, THRESHOLD_SWEEP  # noqa: E402  复用归因逻辑

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NPY_PATH = os.path.join(DATA_DIR, "vector_store_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "vector_store.json")
RESULT_PATH = os.path.join(DATA_DIR, "duplication_control_result.json")

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
IVF_NLIST_RATIO = 100
IVF_NPROBE_RATIO = 8


def build_synthetic_library(base_vecs, base_texts, dup_rate, rng):
    """按指定重复率构造向量库。

    重复条目 = 从基础库里随机挑一条，把它的向量和文本原样复制一份。
    这模拟"模板回复被反复入库"的情形 —— 内容完全相同，向量完全相同。
    """
    n_base = len(base_vecs)
    if dup_rate <= 0:
        dup_count = 0
    else:
        # dup_count / (n_base + dup_count) = dup_rate
        dup_count = int(round(n_base * dup_rate / (1 - dup_rate)))

    if dup_count == 0:
        vectors = base_vecs.copy()
        texts = list(base_texts)
    else:
        pick = rng.choice(n_base, size=dup_count, replace=True)
        vectors = np.vstack([base_vecs, base_vecs[pick]]).astype(np.float32)
        texts = list(base_texts) + [base_texts[i] for i in pick]

    vectors = vectors.copy()
    faiss.normalize_L2(vectors)

    # 文本分组：用于判定"未命中项是否存在完全相同的副本"
    text_to_ids = {}
    for i, t in enumerate(texts):
        text_to_ids.setdefault(t, set()).add(i)
    dup_group = {i: grp for grp in text_to_ids.values() if len(grp) > 1 for i in grp}

    return vectors, texts, dup_group, dup_count


def evaluate(vectors, dup_group, n_queries, k, seed):
    """在给定库上跑 Flat / IVF / HNSW，返回 ID 召回与归因分解。"""
    n, dim = vectors.shape
    rng = np.random.RandomState(seed)
    q_idx = rng.choice(n, size=min(n_queries, n), replace=False)
    queries = vectors[q_idx].copy()

    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)

    nlist = max(min(n // IVF_NLIST_RATIO, 1024), 1)
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist,
                             faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)
    ivf.add(vectors)
    ivf.nprobe = max(nlist // IVF_NPROBE_RATIO, 1)

    hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    hnsw.add(vectors)
    hnsw.hnsw.efSearch = HNSW_EF_SEARCH

    _, gt_ids = flat.search(queries, k)

    out = {}
    for name, index in [("IVF", ivf), ("HNSW", hnsw)]:
        _, ann_ids = index.search(queries, k)
        hits = sum(len(set(int(x) for x in gt_ids[i][:k]) &
                       set(int(x) for x in ann_ids[i][:k]))
                   for i in range(len(queries)))
        id_recall = hits / (len(queries) * k)

        counts, _, missed, _ = attribute(
            vectors, q_idx, gt_ids, ann_ids, k, dup_group, THRESHOLD_SWEEP)

        total_pos = len(queries) * k
        n_a = counts.get("A_exact_duplicate", 0)
        n_d = counts.get("D_unexplained", 0)
        # 真实召回：把"并列副本假象"从损失里剔除
        real_recall = 1 - n_d / total_pos
        # 假象造成的损失占全部损失的比例
        artifact_share = n_a / missed if missed else 0.0

        out[name] = {
            "id_recall@k": round(id_recall, 4),
            "real_recall@k": round(real_recall, 4),
            "missed_positions": missed,
            "A_exact_duplicate": n_a,
            "D_unexplained": n_d,
            "artifact_share_of_loss": round(artifact_share, 4),
        }
    return out


def ivf_real_loss_sweep(base_vecs, base_texts, n_queries, k, seed, nprobe_list):
    """在零重复库里扫 IVF 的 nprobe：真实损失降到 0 需要多大 nprobe？

    零重复库排除了并列分数假象，因此这里的「未解释损失」就是 ANN 的固有近似误差。
    """
    lib_vecs, _, dup_group, _ = build_synthetic_library(
        base_vecs, base_texts, 0.0, np.random.RandomState(seed))
    n, dim = lib_vecs.shape
    rng = np.random.RandomState(seed)
    q_idx = rng.choice(n, size=min(n_queries, n), replace=False)
    queries = lib_vecs[q_idx].copy()

    flat = faiss.IndexFlatIP(dim)
    flat.add(lib_vecs)
    _, gt_ids = flat.search(queries, k)
    total_pos = len(queries) * k
    nlist = max(min(n // IVF_NLIST_RATIO, 1024), 1)

    print("\n" + "=" * 78)
    print("  IVF 的固有近似误差：零重复库上扫 nprobe")
    print("=" * 78)
    print(f"  库 {n} 条（零重复），{len(queries)} 查询 × top_{k} = {total_pos} 命中位")
    print(f"  {'nprobe':>7} | {'扫描':>7} | {'ID召回':>8} | {'真实召回':>9} | {'真实损失':>8}")
    print(f"  {'-' * 7}-+-{'-' * 7}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 8}")

    rows = []
    for npb in nprobe_list:
        if npb > nlist:
            continue
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist,
                                 faiss.METRIC_INNER_PRODUCT)
        ivf.train(lib_vecs)
        ivf.add(lib_vecs)
        ivf.nprobe = npb
        _, ann_ids = ivf.search(queries, k)
        hits = sum(len(set(int(x) for x in gt_ids[i][:k]) &
                       set(int(x) for x in ann_ids[i][:k]))
                   for i in range(len(queries)))
        id_recall = hits / total_pos
        counts, _, _, _ = attribute(
            lib_vecs, q_idx, gt_ids, ann_ids, k, dup_group, THRESHOLD_SWEEP)
        n_d = counts.get("D_unexplained", 0)
        real_recall = 1 - n_d / total_pos
        print(f"  {npb:>7} | {npb / nlist * 100:>6.1f}% | {id_recall * 100:>7.2f}% "
              f"| {real_recall * 100:>8.2f}% | {(1 - real_recall) * 100:>7.2f}%")
        rows.append({"nprobe": npb, "scan_ratio": round(npb / nlist, 4),
                     "id_recall@k": round(id_recall, 4),
                     "real_recall@k": round(real_recall, 4)})

    print("\n  读法：零重复库里没有并列副本，未解释损失就是 ANN 的固有近似误差。")
    print("        它随 nprobe 单调降到 0 —— 即 IVF 的损失是可以通过参数消除的真损失，")
    print("        与 HNSW 那种『纯评测假象』性质不同。")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=8000,
                    help="基础（唯一）条目数，各重复率共用同一基础库以保证可比")
    ap.add_argument("--queries", type=int, default=1500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rates", type=str, default="0,0.1,0.25,0.5")
    ap.add_argument("--ivf-sweep", action="store_true",
                    help="额外在零重复库上扫 IVF 的 nprobe（量化其固有近似误差）")
    ap.add_argument("--nprobe-list", type=str, default="20,40,64,100,166")
    args = ap.parse_args()

    rates = [float(x) for x in args.rates.split(",") if x.strip()]

    print("=" * 78)
    print("  重复率的控制变量实验：ID 召回率会随数据重复率失真吗？")
    print("=" * 78)

    vectors, meta = np.load(NPY_PATH).astype(np.float32), json.load(
        open(META_PATH, "r", encoding="utf-8"))
    faiss.normalize_L2(vectors)

    # 只取"文本唯一"的条目作为基础库 —— 保证 r=0 时库里真的没有重复
    from collections import Counter
    cnt = Counter(m["text"] for m in meta)
    unique_ids = np.array([i for i, m in enumerate(meta) if cnt[m["text"]] == 1])
    print(f"\n真实数据: {len(meta)} 条，其中文本唯一的条目 {len(unique_ids)} 条")

    rng = np.random.RandomState(args.seed)
    take = rng.choice(len(unique_ids), size=min(args.base, len(unique_ids)),
                      replace=False)
    base_ids = unique_ids[take]
    base_vecs = vectors[base_ids].copy()
    base_texts = [meta[i]["text"] for i in base_ids]
    print(f"基础库: {len(base_ids)} 条（全部文本唯一）")
    print(f"查询数: {args.queries} 条/组  |  top_k: {args.k}  |  seed: {args.seed}")

    rows = []
    for r in rates:
        # 用固定子种子，避免不同重复率之间共用同一批随机抽出的重复项
        sub_rng = np.random.RandomState(args.seed + int(r * 1000))
        lib_vecs, lib_texts, dup_group, dup_count = build_synthetic_library(
            base_vecs, base_texts, r, sub_rng)
        total = len(lib_vecs)
        actual_rate = dup_count / total if total else 0.0

        print("\n" + "-" * 78)
        print(f"  重复率 {r:.0%}  实际 {actual_rate:.1%}"
              f"  （库 {total} 条 = {len(base_ids)} 唯一 + {dup_count} 重复副本）")
        print("-" * 78)

        res = evaluate(lib_vecs, dup_group, args.queries, args.k, args.seed)
        row = {"target_rate": r, "actual_rate": round(actual_rate, 4),
               "library_size": total, "dup_count": dup_count}
        for eng in ("IVF", "HNSW"):
            e = res[eng]
            row[eng] = e
            print(f"    {eng:<5} ID 召回 {e['id_recall@k'] * 100:>6.2f}%"
                  f"  |  真实召回 {e['real_recall@k'] * 100:>6.2f}%"
                  f"  |  未命中 {e['missed_positions']:>3} 位"
                  f"  |  其中假象 {e['A_exact_duplicate']:>3} 位"
                  f"  |  未解释 {e['D_unexplained']:>3} 位")
        rows.append(row)

    # ---- 汇总表 ----
    print("\n" + "=" * 78)
    print("  汇总：ID 召回 vs 真实召回（HNSW, top_%d）" % args.k)
    print("=" * 78)
    print(f"  {'重复率':>7} | {'ID 召回':>9} | {'真实召回':>10} | {'损失中假象占比':>14}")
    print(f"  {'-' * 7}-+-{'-' * 9}-+-{'-' * 10}-+-{'-' * 14}")
    for row in rows:
        h = row["HNSW"]
        print(f"  {row['actual_rate']:>6.1%} | {h['id_recall@k'] * 100:>8.2f}% "
              f"| {h['real_recall@k'] * 100:>9.2f}% | "
              f"{h['artifact_share_of_loss'] * 100:>13.1f}%")

    print("\n  怎么读：")
    print("  - 『ID 召回』随重复率下降 → 该指标本身受数据重复率干扰，不是检索变差了")
    print("  - 『真实召回』基本持平 → 检索质量实际没有下降")
    print("  - 重复率 0% 那一行即零重复对照组：此时不存在并列副本，假象无法产生")

    ivf_rows = []
    if args.ivf_sweep:
        npl = [int(x) for x in args.nprobe_list.split(",") if x.strip()]
        ivf_rows = ivf_real_loss_sweep(base_vecs, base_texts, args.queries,
                                       args.k, args.seed, npl)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"base_size": len(base_ids), "n_queries": args.queries,
                   "k": args.k, "seed": args.seed, "rows": rows,
                   "ivf_nprobe_sweep_zero_dup": ivf_rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {RESULT_PATH}")


if __name__ == "__main__":
    main()