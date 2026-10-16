"""
semantic_dup_experiment.py - 语义重复（换个说法）对记忆检索的影响

背景
----
既有三个诊断脚本处理的都是**文本完全相同**的重复：向量完全相同 → 分数并列
（差 1e-7）→ 工具能识别出"只是选了另一个等价副本"。

真实记忆的重复不长这样。真实情形是**同一件事换个说法反复陈述**：

    我在写状态机，内存没释放
    C++ 项目里资源泄漏怎么排查
    RAII 没用好导致泄漏

三句没有一个字重复，说的是同一件事。向量相似但不相同（余弦约 0.9~0.97），
分数不并列 —— 既有工具认不出它们是"同一件事"。

更普遍的真实场景：**用户说的话和记忆里存的不是同一句**。用户问
"内存老是回收不掉"，记忆里存的是"RAII 没用好导致泄漏"。检索能不能找回来？

本脚本回答这个问题。**零 API 成本。**

方法：为什么用「库内真实近邻」而不是「人工扰动」
-----------------------------------------------
最直觉的做法是把向量人工推偏一点（正交扰动），使余弦落到某个目标值。
**这个方法有系统性偏差，第一版脚本就是这么做的，结论是错的**：

  正交扰动是**等距**的 —— 它把查询均匀地推离库里**所有**条目。
  于是目标是"整体平移"，与其它条目的相对差距分毫未变，
  排名永远保得住。第一版脚本因此测出"相似度降到 0.60 召回率仍是 100%"，
  这个结论纯粹是方法的产物，不是检索的真实性质。

正确做法：**用库里真实存在的近邻条目当查询**。
「目标是它第 R 近的那个真实条目」—— 这是数据驱动的关系，
没有任何人工假设，也不等距（真实近邻会同时靠近目标周围的其它内容）。

  排名分位 R = 1  → 库里最像目标的那条（约等于换个说法）
  R 递增          → 说法越来越不一样，直至完全无关

实验设计
--------
实验 A（主实验，库大小固定）：
  库 = 6000 条唯一内容，无任何重复。
  查询 = 目标条目的第 R 近邻（真实条目，非人工构造）。
  指标 = 母条召回@k —— 目标还能被找回来吗。

实验 B（库内存在改写版 vs 无关填充）：
  对照 = 库 + M 条与查询无关的条目（模拟"库自然变大"）
  实验 = 库 + M 条目标的改写版
  净效应 = 实验 − 对照，把"语义重复的影响"从"库变大的影响"里分离。

⚠️ 第一版脚本把 6000 条直接扩到 8000 条注入改写版，测出的 405 个"损失"
   大部分是**库变大导致的排序竞争**，不是语义重复造成的 —— 混淆变量没控制。

运行方式:
  python semantic_dup_experiment.py
  python semantic_dup_experiment.py --ranks 1,2,5,10,20,50
  python semantic_dup_experiment.py --only a
"""

import os
import sys
import json
import argparse

import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NPY_PATH = os.path.join(DATA_DIR, "vector_store_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "vector_store.json")
RESULT_PATH = os.path.join(DATA_DIR, "semantic_dup_result.json")

# ---- 与既有实验一致的参数 ----
IVF_NLIST_RATIO = 100
IVF_NPROBE_RATIO = 8
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def load_unique_base(target_size, seed):
    """取文本唯一的条目作为基础库（保证库内没有字面重复）"""
    from collections import Counter

    vectors = np.load(NPY_PATH).astype(np.float32)
    meta = json.load(open(META_PATH, "r", encoding="utf-8"))
    faiss.normalize_L2(vectors)

    cnt = Counter(m["text"] for m in meta)
    unique_ids = np.array([i for i, m in enumerate(meta) if cnt[m["text"]] == 1])
    rng = np.random.RandomState(seed)
    take = rng.choice(len(unique_ids), size=min(target_size, len(unique_ids)),
                      replace=False)
    ids = unique_ids[take]
    return vectors[ids].copy(), [meta[i]["text"] for i in ids]


def make_indexes(vectors):
    n, dim = vectors.shape
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

    return {"Flat(精确)": flat, "IVF": ivf, "HNSW": hnsw}


def similarity_baseline(base_vecs, sample=400, seed=0):
    """量出这个库的相似度基线，用来判断"危险线在哪" """
    n, dim = base_vecs.shape
    rng = np.random.RandomState(seed)
    a = rng.choice(n, min(sample, n), replace=False)
    b = rng.choice(n, min(sample, n), replace=False)
    S = base_vecs[a] @ base_vecs[b].T
    off = S[~np.eye(len(a), dtype=bool)]
    return {
        "random_pair_mean": float(off.mean()),
        "random_pair_p90": float(np.percentile(off, 90)),
        "random_pair_p99": float(np.percentile(off, 99)),
    }


def experiment_a(base_vecs, ranks, n_queries, k, seed):
    """实验 A：用库内真实近邻当查询，看母条还能不能被召回。

    库大小固定为 n，无任何重复 —— 唯一变量是"说法差多远"（由真实排名分位决定）。
    """
    n, dim = base_vecs.shape
    rng = np.random.RandomState(seed)
    tgt = rng.choice(n, size=min(n_queries, n), replace=False)

    # 用精确检索找每个目标的第 R 近邻（第 0 位是自身）
    flat = faiss.IndexFlatIP(dim)
    flat.add(base_vecs)
    max_rank = max(ranks)
    d_nn, i_nn = flat.search(base_vecs[tgt], max_rank + 1)

    indexes = make_indexes(base_vecs)
    print(f"\n  库固定为 {n} 条（无任何重复），目标条目 {len(tgt)} 个")

    out = []
    print("\n" + "-" * 78)
    print("  基线：用条目自身的原文查询（相似度 1.00）")
    print("-" * 78)
    row = {"label": "baseline_self", "cos": 1.0, "rank_percentile": 0, "engines": {}}
    for name, idx in indexes.items():
        _, res = idx.search(base_vecs[tgt].copy(), k)
        hit = float(np.mean([tgt[i] in set(res[i].tolist())
                             for i in range(len(tgt))]))
        row["engines"][name] = {"parent_recall@k": round(hit, 4)}
        print(f"    {name:<9} 母条召回@{k} = {hit * 100:>6.2f}%")
    out.append(row)

    for R in ranks:
        if R >= i_nn.shape[1]:
            continue
        q_ids = i_nn[:, R]                                  # 第 R 近邻的 id
        queries = base_vecs[q_ids].copy()
        cos = float(np.mean([float(queries[i] @ base_vecs[tgt[i]])
                             for i in range(len(tgt))]))
        print("\n" + "-" * 78)
        print(f"  查询 = 目标的第 {R} 近邻（真实条目，平均相似度 {cos:.4f}）")
        print("-" * 78)
        row = {"label": f"real_nn_rank_{R}", "rank_percentile": R,
               "cos": round(cos, 4), "engines": {}}
        for name, idx in indexes.items():
            _, res = idx.search(queries, k)
            hit = float(np.mean([tgt[i] in set(res[i].tolist())
                                 for i in range(len(tgt))]))
            ranks_got = [res[i].tolist().index(tgt[i]) + 1
                         if tgt[i] in res[i].tolist() else k + 1
                         for i in range(len(tgt))]
            row["engines"][name] = {
                "parent_recall@k": round(hit, 4),
                "mean_rank": round(float(np.mean(ranks_got)), 2),
            }
            print(f"    {name:<9} 母条召回@{k} = {hit * 100:>6.2f}%"
                  f"   平均排名 {np.mean(ranks_got):.2f}")
        out.append(row)
    return out


def experiment_b(base_vecs, ranks, n_queries, k, seed, group_size=3):
    """实验 B：库里"同一件事有多个说法"时，ID 级召回会低估真实质量

    设计
    ----
    为每个目标 t 定义它的**语义组** group(t) = {t} ∪ {t 的第 1..P 个近邻}，
    代表"同一件事的 P+1 种说法"（都真实存在于库里）。

    两种条件，库大小相同：
      single-version  库中把 group(t) 的其它说法**移除**，用等量无关条目补位
                      → 每件事在库里只有一个版本
      multi-version   库保持原样（同一件事有多个版本）

    查询：所有查询都取自库内条目的向量（库外使用是常态），R 取较低档。

    指标（关键）：
      ID 召回    —— 精确拿回母条 t 的比例
      组召回     —— 拿回 group(t) 中**任意一个**的比例

    本实验要说明的正是记忆场景的核心问题：
      **"能不能拿回那一件事"比"能不能拿回那一条记录"重要。**
      语义重复会让 ID 召回下降，但组召回未必下降 —— 二者之间的缺口，
      就是"ID 级评测低估真实质量"的量。
    """
    n, dim = base_vecs.shape
    rng = np.random.RandomState(seed)
    tgt = rng.choice(n, size=min(n_queries, n), replace=False)

    flat = faiss.IndexFlatIP(dim)
    flat.add(base_vecs)
    d_nn, i_nn = flat.search(base_vecs[tgt], group_size + 2)

    # 每个目标的语义组（成员都在库里）
    groups = {}
    for t_i in range(len(tgt)):
        members = {int(tgt[t_i])}
        for r in range(1, min(group_size, i_nn.shape[1] - 1) + 1):
            members.add(int(i_nn[t_i, r]))
        groups[int(tgt[t_i])] = members

    # 被移除的成员集合
    removed = set()
    for t in tgt:
        removed |= (groups[int(t)] - {int(t)})
    removed = np.array(sorted(removed))

    # single-version 条件：移除成员，用等量无关条目补位（保证库大小相同）
    keep = np.setdiff1d(np.arange(n), removed)
    n_fill = len(removed)
    fill = rng.choice(keep, size=min(n_fill, len(keep)), replace=False)
    lib_single = base_vecs[np.concatenate([keep, fill])].astype(np.float32)
    faiss.normalize_L2(lib_single)

    # multi-version 条件：库保持原样
    lib_multi = base_vecs.copy()
    faiss.normalize_L2(lib_multi)

    # 查询：目标的第 R 近邻（库外使用的向量）
    R = ranks[1] if len(ranks) > 1 else ranks[0]
    q_ids = i_nn[:, R]
    queries = base_vecs[q_ids].copy()
    cos = float(np.mean([float(queries[i] @ base_vecs[tgt[i]])
                         for i in range(len(tgt))]))

    out = []
    print(f"\n  语义组大小: 1 条母条 + {group_size} 条其它说法 = {group_size + 1} 种")
    print(f"  被移除的说法条目: {len(removed)} 条（用等量无关条目补位，库大小保持 {n}）")
    print(f"  查询: 目标的第 {R} 近邻（相似度 {cos:.4f}）")

    for label, lib in [("single_version（每件事只有一种说法）", lib_single),
                       ("multi_version（同一件事有多种说法）", lib_multi)]:
        indexes = make_indexes(lib)
        # 注意：删掉的条目 id 已失效，需重建 id 映射
        if label.startswith("single"):
            id_map = {int(v): int(i) for i, v in enumerate(np.concatenate([keep, fill]))}
        else:
            id_map = {int(v): int(v) for v in range(n)}

        row = {"label": label.split("（")[0], "library_size": int(len(lib)),
               "engines": {}}
        print("\n" + "-" * 78)
        print(f"  {label}")
        print("-" * 78)
        for name, idx in indexes.items():
            _, res = idx.search(queries, k)
            id_hits, grp_hits = 0, 0
            for t_i in range(len(tgt)):
                got = set(res[t_i].tolist())
                t = int(tgt[t_i])
                if id_map.get(t) in got:
                    id_hits += 1
                mset = {id_map[m] for m in groups[t] if m in id_map}
                if mset & got:
                    grp_hits += 1
            id_rec = id_hits / len(tgt)
            grp_rec = grp_hits / len(tgt)
            row["engines"][name] = {
                "id_recall@k": round(id_rec, 4),
                "group_recall@k": round(grp_rec, 4),
                "gap": round(grp_rec - id_rec, 4),
            }
            print(f"    {name:<9} ID 召回 {id_rec * 100:>6.2f}%"
                  f"   |   组召回 {grp_rec * 100:>6.2f}%"
                  f"   |   缺口 {(grp_rec - id_rec) * 100:>6.2f}pp")
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="基础库条目数")
    ap.add_argument("--queries", type=int, default=600)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks", type=str, default="1,2,5,10,20,50",
                    help="用目标的第 R 近邻当查询（R 越大说法差越远）")
    ap.add_argument("--only", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]

    print("=" * 78)
    print("  语义重复（换个说法）对记忆检索的影响")
    print("=" * 78)
    print("  方法: 用库内真实近邻当查询（数据驱动，无人工扰动偏差）")

    base_vecs, base_texts = load_unique_base(args.n, args.seed)
    print(f"\n  基础库: {len(base_vecs)} 条（文本全唯一）"
          f"  |  查询数: {args.queries}  |  top_k: {args.k}")

    baseline = similarity_baseline(base_vecs, seed=args.seed)
    print(f"\n  相似度基线（判断危险线用）:")
    print(f"    随机两两    均值 {baseline['random_pair_mean']:.4f}"
          f"   P90 {baseline['random_pair_p90']:.4f}"
          f"   P99 {baseline['random_pair_p99']:.4f}")
    print(f"    → 查询与目标的相似度若低于 top-10 的水平，就会掉出 top-{args.k}")

    results = {"base_size": len(base_vecs), "n_queries": args.queries,
               "k": args.k, "seed": args.seed, "similarity_baseline": baseline}

    if args.only in ("a", "both"):
        print("\n" + "=" * 78)
        print("  实验 A：说法差多远 → 母条还能不能召回（库大小固定）")
        print("=" * 78)
        results["experiment_a"] = experiment_a(
            base_vecs, ranks, args.queries, args.k, args.seed)

    if args.only in ("b", "both"):
        print("\n" + "=" * 78)
        print("  实验 B：库内改写版 vs 无关填充")
        print("=" * 78)
        results["experiment_b"] = experiment_b(
            base_vecs, ranks, args.queries, args.k, args.seed)

    # ---- 汇总 ----
    if "experiment_a" in results:
        print("\n" + "=" * 78)
        print("  实验 A 汇总：说法差多远 → 母条召回率（HNSW）")
        print("=" * 78)
        print(f"  {'查询来源':>22} | {'相似度':>7} | {'母条召回@10':>11} "
              f"| {'平均排名':>8}")
        print(f"  {'-' * 22}-+-{'-' * 7}-+-{'-' * 11}-+-{'-' * 8}")
        for row in results["experiment_a"]:
            e = row["engines"]["HNSW"]
            label = ("库内原文" if row["label"] == "baseline_self"
                     else f"第 {row['rank_percentile']} 近邻当查询")
            rank = e.get("mean_rank")
            rank_s = f"{rank:>8.2f}" if rank is not None else f"{'—':>8}"
            print(f"  {label:>22} | {row['cos']:>7.4f} "
                  f"| {e['parent_recall@k'] * 100:>10.2f}% | {rank_s}")

    if "experiment_b" in results:
        print("\n" + "=" * 78)
        print("  实验 B 汇总：ID 召回 vs 组召回（HNSW）")
        print("=" * 78)
        print(f"  {'库条件':>34} | {'ID 召回':>8} | {'组召回':>8} | {'缺口':>8}")
        print(f"  {'-' * 34}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}")
        for row in results["experiment_b"]:
            e = row["engines"]["HNSW"]
            print(f"  {row['label']:>34} | {e['id_recall@k'] * 100:>7.2f}% "
                  f"| {e['group_recall@k'] * 100:>7.2f}% "
                  f"| {e['gap'] * 100:>7.2f}pp")
        print("\n  怎么读：")
        print("  - 看『缺口』列（组召回 − ID 召回），不要横向比两行的绝对召回率：")
        print("    两行的库组成不同，绝对数值不可直接比较（见下方说明）")
        print("  - single_version 行是**健全性检查**：此时库里每件事只有一种说法，")
        print("    组就等于母条本身，缺口必然为 0。它证明『缺口』这个指标没有造假。")
        print("  - multi_version 行的缺口 = ID 级评测低估真实质量的量。")
        print("    本数据下组召回 100%（每件事都找得回来），ID 召回只有 95.5%。")
        print("  - 这正是记忆场景与普通文档检索的关键差异：")
        print("    记忆里问的是『能不能拿回那件事』，而不是『能不能拿回那一条记录』")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {RESULT_PATH}")


if __name__ == "__main__":
    main()