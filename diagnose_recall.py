"""
diagnose_recall.py - 召回率差异诊断工具

用途：当近似检索（IVF / HNSW）的召回率低于精确检索（Flat）时，
判断这是"真的漏了"还是"并列分数导致的 ID 不确定性"。

背景（本项目的真实发现）：
最初测 HNSW 时召回率 98.8%，一开始怀疑是图质量不足。于是做了两组对照：
  - efSearch 从 16 扫到 256 → 召回率纹丝不动，排除"搜索不充分"
  - M 从 8 扫到 64          → 召回率纹丝不动，排除"图连通度不足"
两次都无效，说明瓶颈不在索引参数，而在数据本身。

真正的根因：数据集中存在"文本重复 → 向量几乎相同 → 分数完全并列"的条目。
近似索引与精确索引各自选中了并列副本中的不同一个，ID 因此不同，
但两者给出的分数完全相同（差异量级 1e-7，属浮点误差）。
这不是精度损失，是评测指标的假象。

运行方式:
  python diagnose_recall.py
"""

import os
import sys
import json
from collections import Counter

import numpy as np
import faiss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NPY_PATH = os.path.join(DATA_DIR, "vector_store_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "vector_store.json")

N_QUERIES = 200
TOP_K = 10
SEED = 42
NEAR_DUP_THRESHOLD = 0.999   # 余弦相似度高于此值视为"近似重复"


def main():
    print("=" * 62)
    print("  召回率差异诊断：并列分数分析")
    print("=" * 62)

    vectors = np.load(NPY_PATH).astype(np.float32)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    faiss.normalize_L2(vectors)
    n, dim = vectors.shape

    # ---- 1. 文本层面的重复 ----
    print("\n[1] 文本重复情况")
    texts = [m["text"] for m in meta]
    counter = Counter(texts)
    dup_items = n - len(counter)
    dup_kinds = sum(1 for k in counter.values() if k > 1)
    print(f"  总条目 {n}，唯一文本 {len(counter)}，重复条目 {dup_items} 条 "
          f"({dup_items / n * 100:.1f}%)")
    print(f"  重复文本种类: {dup_kinds}")
    if dup_kinds:
        top_text, top_cnt = max(counter.items(), key=lambda x: x[1])
        print(f"  重复最多出现 {top_cnt} 次: {top_text[:60]}")

    roles = Counter(m["role"] for m in meta)
    print(f"  按角色: user {roles.get('user', 0)} / assistant {roles.get('assistant', 0)}")
    # 分别看两侧的重复，定位重复来自哪一侧
    for role in ("user", "assistant"):
        sub = [m["text"] for m in meta if m["role"] == role]
        u = len(set(sub))
        print(f"    {role}: {len(sub)} 条，唯一 {u} → 重复 {len(sub) - u} 条")

    # ---- 2. 向量层面的近似重复 ----
    print("\n[2] 向量近似重复（余弦相似度 > %.3f）" % NEAR_DUP_THRESHOLD)
    rng = np.random.RandomState(0)
    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)
    sample = rng.choice(n, min(300, n), replace=False)
    _, near_ids = flat.search(vectors[sample], 5)
    pairs = 0
    for row, base in zip(near_ids, sample):
        # 第 0 位是自身，从第 1 位开始
        scores = [float(vectors[base] @ vectors[j]) for j in row[1:]]
        pairs += sum(1 for s in scores if s > NEAR_DUP_THRESHOLD)
    print(f"  {len(sample)} 条抽样中发现 {pairs} 个近似重复对（相似度>{NEAR_DUP_THRESHOLD}）")

    # ---- 3. 并列分数验证 ----
    print("\n[3] 并列分数验证：近似检索的召回损失是否为假象")
    q_idx = rng.choice(n, min(N_QUERIES, n), replace=False)
    queries = vectors[q_idx].copy()

    d_flat, i_flat = flat.search(queries, TOP_K)

    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(vectors)
    hnsw.hnsw.efSearch = 64
    d_hnsw, i_hnsw = hnsw.search(queries, TOP_K)

    n_q = len(queries)
    id_recall = sum(len(set(i_flat[i][:TOP_K]) & set(i_hnsw[i][:TOP_K]))
                    for i in range(n_q)) / (n_q * TOP_K)

    # 分数差异：若只是"选了并列副本"，分数应几乎相同
    score_diff_top1 = float(np.abs(d_flat[:, 0] - d_hnsw[:, 0]).mean())
    score_diff_tail = float(np.abs(d_flat[:, -1] - d_hnsw[:, -1]).mean())

    # 分数集合一致但 ID 不同的查询数 —— 这是"假象"的直接证据
    same_scores_diff_ids = 0
    for i in range(n_q):
        if not np.allclose(np.sort(d_flat[i])[::-1], np.sort(d_hnsw[i])[::-1], atol=1e-5):
            continue
        if set(i_flat[i]) != set(i_hnsw[i]):
            same_scores_diff_ids += 1

    print(f"  基于 ID 的召回@{TOP_K}        = {id_recall * 100:.2f}%")
    print(f"  top-1  分数平均差异          = {score_diff_top1:.8f}")
    print(f"  top-{TOP_K} 末端分数平均差异     = {score_diff_tail:.8f}")
    print(f"  分数集合相同但条目不同的查询  = {same_scores_diff_ids}/{n_q}")

    # ---- 结论 ----
    print("\n" + "=" * 62)
    print("  结论")
    print("=" * 62)
    is_tie_artifact = (score_diff_top1 < 1e-5 and same_scores_diff_ids > 0)
    if is_tie_artifact:
        print(f"  召回率损失是【假象】：分数差异仅 {score_diff_top1:.1e}（浮点误差量级），")
        print(f"  且 {same_scores_diff_ids}/{n_q} 条查询的分数集合与精确检索完全一致，")
        print(f"  只是从并列副本中选中了不同条目。实际检索质量无损。")
        print(f"\n  → 根因是数据重复（{dup_items} 条重复文本），而非索引精度不足。")
        print(f"  → 修复方式：提升数据唯一性（不只 user 侧，assistant 侧也要去重）。")
    else:
        print(f"  召回率损失可能是【真实】的：分数差异 {score_diff_top1:.6f} 不可忽略。")
        print(f"  → 建议提高 efSearch / nprobe，或增大 M。")

    out = {
        "total": int(n),
        "unique_text": len(counter),
        "dup_items": dup_items,
        "dup_kinds": dup_kinds,
        "id_recall@10": round(id_recall, 4),
        "score_diff_top1": score_diff_top1,
        "score_diff_tail": score_diff_tail,
        "same_scores_diff_ids": same_scores_diff_ids,
        "n_queries": int(n_q),
        "verdict": "tie_artifact" if is_tie_artifact else "real_loss",
    }
    path = os.path.join(DATA_DIR, "recall_diagnosis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  诊断结果已保存: {path}")


if __name__ == "__main__":
    main()