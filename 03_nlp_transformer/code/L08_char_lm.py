"""L08：字符级语言模型实战——bigram 基线 + makemore 式神经语言模型。

任务：从内置英文名字数据学习「名字的分布」，采样生成新名字。
运行：python L08_char_lm.py（CPU 几秒钟）
对应讲义：03_nlp_transformer/L08_字符级语言模型实战.md
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NAMES = ["emma", "olivia", "sophia", "isabella", "mia", "charlotte",
         "amelia", "harper", "evelyn", "abigail", "james", "william",
         "benjamin", "lucas", "henry", "jack", "liam", "noah", "aiden",
         "logan", "samuel", "miles", "adam", "nina", "luna"]
BLOCK = 3        # 上下文窗口：用前 3 个字符预测第 4 个


# ---------- 1. bigram 基线：数频次的查表模型 ----------
def build_bigram(names):
    chars = sorted(set("".join(names)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0                                   # 0 同时当起始与结束符
    itos = {i: s for s, i in stoi.items()}
    N = torch.zeros(len(stoi), len(stoi), dtype=torch.long)
    for name in names:
        ws = ["."] + list(name) + ["."]
        for c1, c2 in zip(ws, ws[1:]):              # 滑动统计相邻字符对
            N[stoi[c1], stoi[c2]] += 1
    P = N.float() / N.float().sum(dim=1, keepdim=True)   # 行归一化 -> 条件分布
    return P, stoi, itos


def bigram_loss(P, names, stoi):
    """bigram 查表的交叉熵：-mean log P(下一字符 | 当前字符)。"""
    log_likelihood = 0.0
    n = 0
    for name in names:
        ws = ["."] + list(name) + ["."]
        for c1, c2 in zip(ws, ws[1:]):
            log_likelihood += torch.log(P[stoi[c1], stoi[c2]])
            n += 1
    return (-log_likelihood / n).item()


def sample_bigram(P, itos, n=10, seed=42):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for _ in range(n):
        ix, chars = 0, []
        while True:
            ix = torch.multinomial(P[ix], num_samples=1, generator=g).item()
            if ix == 0:
                break                               # 采到结束符
            chars.append(itos[ix])
        outs.append("".join(chars))
    return outs


# ---------- 2. 神经语言模型：embedding + 上下文拼接 + 线性层 ----------
def build_dataset(names, stoi, block=BLOCK):
    """滑窗切片：每个名字产出多个 (上下文3字符, 目标1字符) 样本。"""
    X, Y = [], []
    for name in names:
        ctx = [0] * block                           # 起始用 '.'(=0) 填满窗口
        for ch in name + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]                    # 窗口右滑
    return torch.tensor(X), torch.tensor(Y)


class CharLM(nn.Module):
    def __init__(self, vocab, emb=10, hidden=200, block=BLOCK):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)         # 查表（第 1 课）
        self.fc1 = nn.Linear(block * emb, hidden)
        self.fc2 = nn.Linear(hidden, vocab)

    def forward(self, idx):                         # idx: [B, block]
        e = self.emb(idx)                           # [B, block, emb]
        e = e.view(e.size(0), -1)                   # [B, block*emb] 拼接上下文
        h = torch.tanh(self.fc1(e))                 # [B, hidden]
        return self.fc2(h)                          # [B, vocab] logits


def train(model, X, Y, steps=2000, batch=32, lr=0.1, seed=42):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        ix = torch.randint(len(X), (batch,), generator=g)   # 随机小批
        xb, yb = X[ix], Y[ix]
        opt.zero_grad()                             # ① 清梯度
        logits = model(xb)                          # ② 前向
        loss = F.cross_entropy(logits, yb)          # ③ 损失
        loss.backward()                             # ④ 反向
        opt.step()                                  # ⑤ 更新
        if step % 500 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def sample(model, itos, n=10, block=BLOCK, seed=42):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    outs = []
    for _ in range(n):
        ctx = [0] * block                           # 从起始符开始
        chars = []
        while True:
            logits = model(torch.tensor([ctx]))     # [1, vocab]
            probs = F.softmax(logits, dim=-1)[0]
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break                               # 结束符
            chars.append(itos[ix])
            ctx = ctx[1:] + [ix]                    # 窗口右滑
        outs.append("".join(chars))
    return outs


def main():
    torch.manual_seed(42)
    P, stoi, itos = build_bigram(NAMES)
    vocab = len(stoi)
    print(f"词表大小: {vocab}（a-z 共 26 个字母 + 起止符 '.'）")

    # ---------- bigram 基线 ----------
    print("\n== 1. bigram 基线（纯统计，零参数） ==")
    print(f"bigram 训练集交叉熵: {bigram_loss(P, NAMES, stoi):.4f}")
    print("bigram 采样:", sample_bigram(P, itos))

    # ---------- 神经语言模型 ----------
    print("\n== 2. 神经语言模型（makemore MLP：3 字符上下文） ==")
    X, Y = build_dataset(NAMES, stoi)
    print(f"样本数: {X.shape[0]}，输入形状 {tuple(X.shape)}，输出形状 {tuple(Y.shape)}")

    model = CharLM(vocab)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    model = train(model, X, Y)

    # 训练集整体 loss（与 bigram 对比）
    with torch.no_grad():
        loss = F.cross_entropy(model(X), Y)
    print(f"神经模型训练集交叉熵: {loss.item():.4f}（对比 bigram：上下文更长，loss 更低）")

    print("神经模型采样:", sample(model, itos))
    print("（多换 seed 看随机性；把 BLOCK 改成 5 再跑，观察 loss 与生成质量）")


if __name__ == "__main__":
    main()
