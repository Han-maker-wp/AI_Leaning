"""L03：自注意力最小实现（含因果掩码、softmax 饱和实验、多头）。

运行：python L03_attention_demo.py
对应讲义：03_nlp_transformer/L03_Attention机制精讲.md
"""
import torch


def softmax(x, dim=-1):
    """数值稳定版 softmax：先减最大值防上溢。"""
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def self_attention(x, W_q, W_k, W_v, mask=None):
    """单头自注意力（20 行核心版）。

    x: [n, d_model] 一个序列的 embedding
    返回 out: [n, d_v] 新表示；attn: [n, n] 注意力权重（观察用）
    """
    Q = x @ W_q                                # [n, d_k] 查询：我想找什么
    K = x @ W_k                                # [n, d_k] 索引：怎样被找到
    V = x @ W_v                                # [n, d_v] 内容：交付什么

    scores = Q @ K.T / Q.size(-1) ** 0.5       # 打分 + 除 sqrt(d_k) 防饱和
    if mask is not None:                       # 因果掩码：未来位置 -> -inf -> softmax 后为 0
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = softmax(scores, dim=-1)             # 每行归一化成分布（和为 1）
    return attn @ V, attn                      # 加权汇总全序列


def multi_head_attention(x, W_q, W_k, W_v, W_o, h, mask=None):
    """多头：把投影结果拆成 h 份并行计算，拼接后过输出投影。"""
    n, d_model = x.shape
    d_head = d_model // h
    Q = (x @ W_q).view(n, h, d_head).transpose(0, 1)   # [h, n, d_head]
    K = (x @ W_k).view(n, h, d_head).transpose(0, 1)
    V = (x @ W_v).view(n, h, d_head).transpose(0, 1)

    scores = Q @ K.transpose(-2, -1) / d_head ** 0.5   # [h, n, n]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = softmax(scores, dim=-1)
    out = attn @ V                                      # [h, n, d_head]
    out = out.transpose(0, 1).reshape(n, d_model)       # 拼接回 [n, d_model]
    return out @ W_o, attn


def main():
    torch.manual_seed(0)
    n, d_model = 6, 8

    # 玩具数据：假装 6 个 token 已变成 8 维 embedding
    x = torch.randn(n, d_model)
    W_q, W_k, W_v = (torch.randn(d_model, d_model) for _ in range(3))

    # 1) 双向自注意力（BERT 式：可看全句）
    out, attn = self_attention(x, W_q, W_k, W_v)
    print("== 双向自注意力 ==")
    print("输出形状:", out.shape)                        # [6, 8]，与输入同形
    print("注意力矩阵（每行和=1）:\n", attn.round(decimals=2))

    # 2) 因果掩码（GPT 式：只准看左边）
    causal = torch.tril(torch.ones(n, n))               # 下三角=1
    _, attn_c = self_attention(x, W_q, W_k, W_v, mask=causal)
    print("\n== 因果掩码注意力（上三角应全为 0）==")
    print(attn_c.round(decimals=2))

    # 3) softmax 饱和实验：为什么必须除 sqrt(d_k)
    print("\n== softmax 饱和实验 ==")
    big = torch.tensor([1.0, 2.0, 3.0]) * 20            # 大分数（模拟高维点积）
    print("无缩放的大分数 softmax:", softmax(big).round(decimals=4))   # 近似 one-hot
    print("除 sqrt 后再 softmax:  ", softmax(big / 20 ** 0.5).round(decimals=2))

    # 4) 多头（4 头，每头 2 维）
    W_o = torch.randn(d_model, d_model)
    out_m, _ = multi_head_attention(x, W_q, W_k, W_v, W_o, h=4, mask=causal)
    print("\n== 多头注意力 ==")
    print("输出形状:", out_m.shape)                      # [6, 8]


if __name__ == "__main__":
    main()
