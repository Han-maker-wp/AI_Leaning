# 第 3 课：Attention 机制精讲

> 🎬 开场钩子：2017 年，Google 八位研究员在论文标题里写下一句狂话——「注意力就是你所需要的一切」（Attention Is All You Need）。当时没人想到，这句话会在之后几年重塑整个 AI 行业。今天这节课，我们把这句狂话拆到每一项：注意力到底是什么？为什么「它就是一切」？

⏱ 建议学时：3 小时（**本阶段核心课**，值得慢）

## 本课目标

1. 用「查字典」直觉解释 Q / K / V 三者的角色；
2. 逐项拆解 Attention(Q,K,V) = softmax(QK^T/√d_k)V，说清为什么除 √d_k、为什么用 softmax；
3. 解释自注意力、因果掩码、多头机制各自解决什么问题；
4. 会推导 O(n²) 复杂度并说出它对长文本的后果；
5. 用 torch 写出 20 行自注意力（带因果掩码），逐行注释。

## 3.1 查字典直觉：Q、K、V

软件里的字典是**精确匹配**：`dict["apple"]` 命中一个 key，取回它的 value。但语言的问题是模糊的——翻译「苹果很好吃」时，「苹果」应该关注（attend to）句子里哪些词？没有唯一答案，需要**按相似度加权**。

Attention 把「模糊查字典」数学化。每个 token 准备三样东西：

- **Query（查询）**：我想找什么（「苹果」带着「我在找可食用相关的属性」这样的查询）；
- **Key（索引）**：我能被怎样找到（每个词挂出自己的标签）；
- **Value（内容）**：如果你选中我，我实际交付的信息。

流程：拿 Query 与所有 Key 逐个算相似度 → 相似度经 softmax 变成一组权重 → 用权重对所有 Value 加权求和，得到本次查询的结果。

一句话：**字典返回一个 value；注意力返回所有 value 的加权平均，权重 = 查询与索引的匹配程度**。

> 🍿 彩蛋｜Q/K/V 是检索黑话：Query / Key / Value 这套词汇借自信息检索系统——你调 Elasticsearch 时的检索三元组就是它：查询去匹配索引、命中后取回内容。区别在于数据库是硬匹配（只命中一个 key），注意力是软匹配（所有 key 都按相似度分到权重）。

## 3.2 公式逐项拆解

先看懂形状，再看懂数值。记序列长 n、每个向量 d_k 维。

```
Attention(Q, K, V) = softmax( Q K^T / √d_k ) V
```

**第一项：QK^T —— 打分**

```
Q: [n, d_k]     每行是一个 token 的查询
K: [n, d_k]     每行是一个 token 的索引
Q @ K.T: [n, n] 第 (i,j) 元 = 第 i 个 query 与第 j 个 key 的点积
```

为什么用点积衡量匹配度（线代回顾）：两向量点积 = 模长 × cos 夹角，方向越一致值越大。Q 和 K 是模型学出来的投影（见 3.3），训练会自动把「该相关的方向」对齐。得到的 [n,n] 矩阵就是「注意力打分表」：第 i 行是 token i 对所有 token 的兴趣度。

**第二项：除以 √d_k —— 控制方差**

点积是 d_k 个乘积的和，维度越大数值方差越大（各分量独立时，点积方差约与 d_k 成正比）。d_k=64 时分数动辄正负几十，而 softmax 对大数值极其敏感：

```
softmax([1, 2, 3])    = [0.09, 0.24, 0.67]        分布还算平滑
softmax([10, 20, 30]) = [约 0, 约 0, 约 1]          近似 one-hot！
```

softmax 一旦饱和成 one-hot，梯度几乎为 0，训练就死了。除以 √d_k 恰好把方差拉回约 1，保持分布「软」。这是论文里少数纯数值稳定性的技巧，记住「除以 √维度 = 防饱和」。

**第三项：softmax —— 归一化成分布**

对每行独立做 softmax：把 n 个任意实数分数变成 n 个正数、和为 1 的概率分布。作用：可解释（权重即注意力百分比）、可微（能端到端训练）、有竞争性（总注意力恒为 1，多关注这里就少关注别处）。

**第四项：乘 V —— 加权汇总**

```
softmax 后: [n, n]（每行和为 1） @ V: [n, d_v] -> 输出 [n, d_v]
第 i 行输出 = Σ_j 权重_ij × V_j —— token i 的新表示 = 全序列的加权平均
```

输出与输入同形状：每个 token 带着从全序列收集来的信息，替换原来「只含自己」的表示。

## 3.3 自注意力：序列自己看自己

「自」（self）指 Q、K、V 都来自**同一个序列**的三个线性投影：

```
Q = X @ W_Q      K = X @ W_K      V = X @ W_V
X: [n, d_model]（embedding 序列），W_*: [d_model, d_k]
```

为什么不直接拿 X 当 Q/K/V？因为「拿什么去查」「怎样被查到」「交付什么内容」是三种不同角色，需要三组独立可学的投影。经典直觉例子（英语）：

```
The animal didn't cross the street because it was too tired.
                                              ↑
                                    "it" 的注意力学会强烈关注 "animal"
```

指代消解没有被显式编程，而是从训练任务中自发涌现——这是注意力机制最出名的可解释案例。

> 🍿 彩蛋｜祖传例句：The animal didn't cross the street because it was too tired——这个例句几乎出现在每一篇讲注意力的科普里，堪称本领域的「Hello, World」。你以后在别处再见到它，请会心一笑，然后默背一遍：it 关注的是 animal。

对比 RNN（上一课）：RNN 把历史压进固定维度 h；自注意力让每个位置直接、并行地查看全部历史，容量随 n 增长。

## 3.4 因果掩码：只准看左边

自注意力有个 bug：算第 t 个 token 的表示时，它也看到了第 t+1 个 token——未来。理解任务（BERT 式）没问题；但**生成式语言模型**不行：预测第 t 个词的依据只能是前 t-1 个词，否则训练目标「预测下一词」直接泄漏答案。

修法：在 softmax **之前**，把打分矩阵的上三角（j > i 的位置）填成 -inf：

```
打分矩阵              掩码后                softmax 后（每行）
1  2  3  4        1  -inf -inf -inf      1.00  0     0     0
5  6  7  8   →     5   6  -inf -inf  →   0.27  0.73  0     0
9  ...             ...
```

-inf 经 softmax 变成精确的 0——位置 i 对 j > i 的注意力为 0，只能看左边。代码一行（见 3.7）。这就是「因果」（causal）：新表示不含任何未来信息，自回归生成（第 2 课的循环）才合法。

## 3.5 多头：多组视角

一组 Q/K/V 只能学一种「关注模式」。但语言里同时存在多种关系：语法（主谓）、指代（it→animal）、位置相邻、语义相似……于是并行做 h 组（如 8 头），每组独立的 W_Q/W_K/W_V 和更小的维度（d_head = d_model / h，总计算量与单头大致持平）：

```
head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)       各头 [n, d_model/h]
输出   = Concat(head_1 .. head_h) @ W_O             拼回 [n, d_model]
```

类比：8 个专家同时读同一句话，各自摘出自己关心的关系，最后汇总。后续大量可解释性工作确实发现某些头专管指代、某些管相邻位置——名字了解即可，不必深究。

## 3.6 复杂度 O(n²) 及其后果

打分矩阵 QK^T 是 n×n：**计算量与显存都随序列长度平方增长**。

```
n=128:    128²   ≈ 1.6 万个分数
n=2048:   2048²  ≈ 420 万        （GPT-3 上下文窗口的时代）
n=100000: 100k²  = 100 亿        —— 贵到离谱
```

序列长 10 倍，注意力开销 100 倍——这就是「长上下文贵」的原因，也是 FlashAttention（IO 优化、不改数学）、稀疏/滑窗注意力、RoPE 长度外推等一整个研究方向的存在理由（后续阶段还会遇到）。现在记住结论：**注意力贵在 n²，不在参数量**。

## 3.7 动手：20 行自注意力

完整代码在 `code/L03_attention_demo.py`，核心如下（torch，逐行注释）：

```python
import torch

def self_attention(x, W_q, W_k, W_v, mask=None):
    """x: [n, d_model] 一个序列的 embedding，返回 (新表示, 注意力权重)。"""
    Q = x @ W_q                                  # [n, d_k] 每个词的查询
    K = x @ W_k                                  # [n, d_k] 每个词的索引
    V = x @ W_v                                  # [n, d_v] 每个词的内容

    scores = Q @ K.T / Q.size(-1) ** 0.5         # 打分 + 除 sqrt(d_k) 防饱和

    if mask is not None:                         # 因果掩码：未来位置填 -inf
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)         # 每行归一化成分布
    return attn @ V, attn                        # 加权汇总


if __name__ == "__main__":
    torch.manual_seed(0)
    n, d = 4, 8
    x = torch.randn(n, d)                        # 4 个 token 的 embedding
    Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))

    out, attn = self_attention(x, Wq, Wk, Wv)    # 双向（BERT 式）
    print("输出形状:", out.shape)                  # [4, 8]，与输入同形
    print("注意力矩阵:\n", attn.round(decimals=2))

    causal = torch.tril(torch.ones(n, n))        # 下三角=1：只准看左边
    out_c, attn_c = self_attention(x, Wq, Wk, Wv, mask=causal)
    print("因果掩码注意力（上三角应为 0）:\n", attn_c.round(decimals=2))
```

跑起来重点观察两件事：

1. 输出形状与输入相同——注意力是「表示替换器」，可以无限堆叠（第 4 课的 Transformer 就是把它当积木）；
2. 因果掩码后注意力矩阵上三角精确为 0。

（demo 文件里还有多头版：reshape 成 [h, n, d_head] 批量算 h 个头再拼接，以及 softmax 饱和实验。）

## 练习

1. 把 d_k 改成 512、去掉 √d_k 缩放，打印 softmax 前的分数与 softmax 后的分布。
   提示：观察分布是否近似 one-hot（饱和）。
2. 把掩码从下三角改成上三角（只看右边），解释「预测下一词」任务是否还成立。
   提示：第 t 个位置看到了 t+1 的信息——答案泄漏。
3. 手写多头版：4 个头各 2 维，验证拼接后形状。
   提示：`Q.view(n, h, d_head).transpose(0, 1)` 变 [h, n, d_head] 批量算。
4. 用 numpy 重写 self_attention（softmax 手写：先减最大值防上溢）。
   提示：`e = np.exp(s - s.max(-1, keepdims=True)); e / e.sum(-1, keepdims=True)`。

## 小结与自测

1. Q/K/V 各自的角色？
   答：查询（我想找什么）、索引（怎样被找到）、内容（选中后交付什么）。
2. 为什么除 √d_k？
   答：点积方差随维度增大，softmax 饱和梯度消失；缩放把方差拉回约 1。
3. 自注意力「自」在哪？
   答：Q/K/V 由同一序列投影而来——序列内部互相注意。
4. 因果掩码怎么实现、为什么必须在 softmax 之前？
   答：未来位置填 -inf，softmax 后变 0；若在 softmax 后置零，行和不为 1。
5. O(n²) 指什么的平方？
   答：打分矩阵 QK^T 是 n×n（n 为序列长度），计算与显存都随 n² 增长。

## 下一课预告

零件齐了：多头注意力、FFN（就是 MLP）、残差、LayerNorm、位置编码。第 4 课把它们组装成 Transformer——精读 arXiv 1706.03762，顺带学会「怎么读论文」。
