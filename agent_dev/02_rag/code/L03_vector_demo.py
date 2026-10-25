"""L03_vector_demo.py —— 向量检索机制演示（阶段 2 · 第 3 课配套代码）

两部分（均离线可跑，仅依赖 numpy）：
  1. 迷你"真"embedding：用字符 bigram 词频向量给中文短文本编码，
     演示"语义（字形共现）邻近 → 向量邻近 → 检索命中"的完整机制
  2. 随机向量规模化：1000 个固定随机种子向量当文档库，演示
     L2 归一化 → 矩阵点积 → top-k 的标准检索流程（与真实向量库接口同构）

真实 embedding 模型把第 1 部分的"字形共现"升级为"语义编码"，机制完全相同：
换掉 mini_embed 函数、其余代码不动，就是可接入 checkpoint 的检索内核。
"""

import time

try:
    import numpy as np
except ImportError:  # 依赖未装时给出可操作的提示，而不是抛裸栈
    raise SystemExit("缺少 numpy 依赖：请先运行 uv add numpy（或 pip install numpy）")

# ---------- 第 1 部分：字符 bigram 迷你 embedding ----------
DOCS = [
    "退款流程：提交申请后 3-5 个工作日原路退回",      # 0
    "退货需要联系客服获取退货地址",                    # 1
    "会员等级分为银卡金卡与钻石卡三档",                # 2
    "发票将在订单完成后 24 小时内开具",                # 3
    "退款失败可联系财务复核",                          # 4
]
QUERIES = ["怎么退货", "退款要多久", "钻石会员有什么权益"]


def bigram_vec(text: str, vocab: dict[str, int], dim: int) -> np.ndarray:
    """把文本编码成 bigram 词频向量（土法 embedding：可解释、离线、无语义）。"""
    vec = np.zeros(dim)
    cleaned = text.replace(" ", "")
    for i in range(len(cleaned) - 1):
        gram = cleaned[i:i + 2]            # 相邻两字成一个 bigram
        if gram in vocab:
            vec[vocab[gram]] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec  # L2 归一化：余弦退化为点积


def build_vocab(texts: list[str]) -> dict[str, int]:
    """收集全部 bigram 建词表（真实 embedding 模型的词表是训练学出来的）。"""
    vocab: dict[str, int] = {}
    for t in texts:
        cleaned = t.replace(" ", "")
        for i in range(len(cleaned) - 1):
            vocab.setdefault(cleaned[i:i + 2], len(vocab))
    return vocab


def top_k(query: np.ndarray, doc_vecs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """标准 top-k 检索：矩阵点积 + argsort。与 FAISS/Chroma 的检索接口同构。"""
    sims = doc_vecs @ query                # 已归一化，点积即余弦相似度
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def demo_mini_embedding() -> None:
    print("=" * 24, "第 1 部分：mini embedding 检索", "=" * 24)
    vocab = build_vocab(DOCS + QUERIES)
    dim = len(vocab)
    doc_vecs = np.stack([bigram_vec(d, vocab, dim) for d in DOCS])
    for q in QUERIES:
        qv = bigram_vec(q, vocab, dim)
        hits = top_k(qv, doc_vecs, k=3)
        print(f"\n查询: {q}")
        for rank, (idx, sim) in enumerate(hits, 1):
            print(f"  #{rank} (cos={sim:.3f}) {DOCS[idx]}")
    print("\n注意：bigram 只看字形共现（退款/退货共享「款退/货需」类片段），")
    print("真实 embedding 模型把它升级成语义编码——检索机制一模一样。")


# ---------- 第 2 部分：随机向量规模化演示 ----------
def demo_random_scale() -> None:
    print("\n", "=" * 24, "第 2 部分：随机向量 top-k（规模化演示）", "=" * 24, sep="")
    rng = np.random.default_rng(42)        # 固定种子，可复现
    n, dim = 1000, 128
    doc_vecs = rng.normal(size=(n, dim))
    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)  # 归一化
    query = rng.normal(size=dim)
    query /= np.linalg.norm(query)

    t0 = time.perf_counter()
    hits = top_k(query, doc_vecs, k=5)
    ms = (time.perf_counter() - t0) * 1000
    print(f"库规模 {n}×{dim}，暴力扫一次 top-5 耗时 {ms:.2f} ms")
    for rank, (idx, sim) in enumerate(hits, 1):
        print(f"  #{rank} doc#{idx} cos={sim:.3f}")

    # max_distance 过滤：top-k 只保证"最相关"，不保证"真相关"
    threshold = 0.3
    kept = [(i, s) for i, s in hits if s >= threshold]
    print(f"阈值过滤(≥{threshold})后保留 {len(kept)}/{len(hits)} 条"
          f"——空结果是合法状态，应显式处理而非硬答")


def main() -> None:
    demo_mini_embedding()
    demo_random_scale()


if __name__ == "__main__":
    main()
