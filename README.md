# RAG 大规模对话检索系统

把 1 万条模拟对话（2 万条记忆条目）存进向量库，用三种检索引擎做对比实验，验证 **IVF 近似检索在大规模数据下的真实优势**。

核心结论：**16K 向量下，FAISS IVF 比 NumPy 暴力检索快 740 倍，比 FAISS Flat 快 16 倍，且精度 100% 不丢。**

---

## 这个项目在做什么

一个 RAG 记忆系统的规模化验证：当对话记录从 10 条涨到 1 万条，暴力检索还能不能用？

答案是：NumPy 暴力检索在 16K 向量时已经要 543ms，10 万条预计 3.3 秒，不可接受。而 IVF 近似检索把线性复杂度降到亚线性，16K 时只要 0.73ms。

---

## 实验结果

### 性能对比（去重后数据）

| 检索引擎 | 2K 向量 | 16K 向量 | 增长倍数 | 说明 |
|---------|---------|---------|---------|------|
| NumPy 暴力检索 | 70ms | 543ms | 7.8x | O(n) 线性增长，10 万条预计 3.3s |
| FAISS Flat（精确） | 0.76ms | 11.5ms | 15x | 也线性，但 C++ 常数小 |
| **FAISS IVF（近似）** | 0.69ms | **0.73ms** | **1.06x** | 亚线性，10 万条预计 5-8ms |

### 精度验证（100 条随机查询）

| 指标 | 结果 |
|------|------|
| top-1 一致率 | 100/100 (100%) |
| top-5 重叠率 | 5.0/5 (100%) |
| top-1 分数差异 | 0.0000 |

**关键工程判断**：数据量 < 1 万条时 Flat 就够用（IVF 优势不明显）；> 1 万条时 IVF 的意义才显现——它不是"稍微快一点"，而是把复杂度从线性变成亚线性。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 需要 `faiss-cpu`。Windows / macOS / Linux 均可。

### 2. 配置 API

复制 `.env.example` 为 `.env`，填入你的 API 配置：

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=sk-你的密钥
OPENAI_API_BASE=https://api.openai.com/v1   # 或你自己的 OpenAI 兼容网关
EMBEDDING_MODEL=text-embedding-3-small
VECTOR_DIM=1536                             # 必须与 embedding 模型输出维度一致
```

> `.env` 已在 `.gitignore` 中，不会进入仓库。

### 3. 三步跑通

```bash
python data_generator.py   # 1. 生成 1 万条模拟对话（固定随机种子，可复现）
python indexer.py          # 2. 批量 Embedding 入库（64 条/批）
python benchmark.py        # 3. 三引擎性能对比 + 100 条随机一致性验证
```

单引擎检索测试：

```bash
python search.py           # NumPy 检索 + 元数据过滤 + top-k 对比
```

---

## 项目结构

```
rag-large-search/
├── config.py              # 配置（全部从环境变量 / .env 读取）
├── data_generator.py      # 模拟对话生成器（5 主题 × 1 万条，生成时去重）
├── indexer.py             # 批量入库（64 条/批 + .npy 二进制存储）
├── search.py              # 检索测试（基础检索 + 元数据过滤 + top_k 对比）
├── benchmark.py           # 三引擎性能对比（NumPy vs Flat vs IVF）
├── debug_ivf.py           # IVF 诊断工具（打印原始分数，定位 bug 用）
├── memory/
│   ├── embeddings.py      # Embedding 客户端（OpenAI 兼容 API）
│   ├── vector_store.py    # NumPy 向量存储（元数据 JSON + embedding .npy）
│   └── faiss_store.py     # FAISS 向量存储（Flat + IVF）
└── data/                  # 生成的数据（已 gitignore，由脚本重建）
```

---

## 踩过的坑

这份实验最值钱的不是性能数字，是下面这些坑——每一个都花了实际排查时间。

| 坑 | 现象 | 根因 | 教训 |
|----|------|------|------|
| **IVF 返回空结果** | 10 个查询 9 个空 | IVFFlat 默认用 L2 距离（越小越相似），代码却把分数当余弦相似度用（越大越相似），阈值过滤把正确结果全干掉了 | **FAISS 的 metric 不跟 quantizer 走**，IVF 必须显式指定 `METRIC_INNER_PRODUCT` |
| **IVF 参数优化不足** | 2K 时 IVF 和 Flat 差不多 | `nprobe=nlist//2` 扫了 50% 聚类，省不了多少 | 大规模必须降 nprobe 比例（本项目降到 12.5%），否则 IVF 没意义 |
| **数据重复率 94%** | 100 条随机验证 top-1 一致率只有 6% | 变量池太小，1 万条只有 582 条唯一；IVF 和 NumPy 找到的是分数相同的不同副本 | 生成时去重 + 扩大变量池，去重后一致率 100% |
| **验证样本太少** | 10 条手写查询 10/10，以为没问题 | 手写查询是 paraphrase，不在库内，没暴露重复问题 | 验证必须包含**库内随机抽样**，且数据需去重 |
| **JSON 文件过大** | 2 万条全 JSON 预计 400MB+ | 每个浮点数以 18 字符文本存储 | embedding 改存 `.npy` 二进制，省 80% 空间 |
| **json.dump 参数写反** | 存储时报 TypeError | 第一个参数应该是数据不是文件对象 | 低级错误，但程序崩了重来要重新调 API（浪费钱） |
| **JSON 文件损坏** | 加载时报 JSONDecodeError | 上次写入失败留了坏文件 | load 加 try-except 容错 |

---

## 设计要点

- **批量 Embedding**：64 条/批调 API，2 万条只需 312 次调用（逐条要 2 万次），耗时 191 秒
- **两级存储**：元数据存 JSON（可读、可过滤），embedding 存 `.npy`（紧凑、读写快）
- **元数据过滤**：每条记忆带 `topic` 和 `timestamp`，支持按主题检索
- **IVF 参数**：`nlist = n/100`（上限 1024），`nprobe = nlist/8`（扫 12.5% 聚类）
- **可复现**：数据生成固定随机种子（`random.seed(42)`），验证抽样也固定种子

---

## 环境

- Python 3.10+
- 依赖：`openai`、`numpy`、`faiss-cpu`
- 实验数据：8317 条唯一对话，2 万条记忆条目，1024/1536 维向量