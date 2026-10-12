"""
indexer.py - 批量入库脚本

读取 conversations.json → 批量 Embedding → 存入向量库

为什么需要批量：1000 条对话 = 2000 条记忆（user+assistant），
逐条调 API 要 2000 次请求，批量调只需要 ~200 次（每次 10 条）。

运行方式:
  python indexer.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from memory.embeddings import EmbeddingClient
from memory.vector_store import NumPyVectorStore, ConversationEntry
import config

BATCH_SIZE = 64  # 每次调 API 编码多少条文本（bge-m3 支持大批量）


def index_conversations():
    # 读取对话数据
    data_path = os.path.join(os.path.dirname(__file__), "data", "conversations.json")
    with open(data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"读取 {len(conversations)} 条对话")

    embedder = EmbeddingClient()
    store = NumPyVectorStore()
    store.clear()

    # 收集所有文本（user + assistant 分别入库）
    all_texts = []
    all_meta = []
    for conv in conversations:
        all_texts.append(conv["user"])
        all_meta.append({"role": "user", "topic": conv["topic"],
                         "timestamp": conv["timestamp"], "index": conv["index"]})
        all_texts.append(conv["assistant"])
        all_meta.append({"role": "assistant", "topic": conv["topic"],
                         "timestamp": conv["timestamp"], "index": conv["index"]})

    total = len(all_texts)
    print(f"待入库条目数: {total}（{len(conversations)} 轮对话 × 2）")
    print(f"批量大小: {BATCH_SIZE} 条/次，预计 {total // BATCH_SIZE} 次 API 调用")
    print()

    # 批量编码
    all_embeddings = []
    t_start = time.perf_counter()

    for i in range(0, total, BATCH_SIZE):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        batch_embs = embedder.embed_batch(batch_texts)
        all_embeddings.extend(batch_embs)

        done = min(i + BATCH_SIZE, total)
        if done % 500 == 0 or done == total:
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  进度: {done}/{total} ({rate:.1f} 条/秒, 剩余 {eta:.0f}s)")

    embed_time = time.perf_counter() - t_start
    print(f"\nEmbedding 完成: {total} 条, 耗时 {embed_time:.1f}s ({total/embed_time:.1f} 条/秒)")

    # 构建条目并存入
    entries = []
    for i in range(total):
        meta = all_meta[i]
        entry = ConversationEntry(
            id=f"{meta['role']}_{meta['index']}_{i}",
            text=all_texts[i],
            embedding=all_embeddings[i],
            role=meta["role"],
            topic=meta["topic"],
            timestamp=meta["timestamp"],
        )
        entries.append(entry)

    store.add_batch(entries)

    print(f"入库完成: {store.count()} 条")
    print(f"存储路径: {store.path}")

    # 统计
    topic_counts = {}
    for e in entries:
        topic_counts[e.topic] = topic_counts.get(e.topic, 0) + 1
    print("\n入库主题分布:")
    for topic, cnt in sorted(topic_counts.items()):
        print(f"  {topic}: {cnt}")


if __name__ == "__main__":
    index_conversations()
