"""L06：完整训练循环——MLP 玩 MNIST。

运行：python L06_mlp_mnist.py
依赖：uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
      uv pip install matplotlib
产物：mnist_curves.png（训练曲线）、mnist_mlp.pt（模型权重）
预计耗时（CPU）：首次下载数据约 10MB，之后每 epoch 约 10~30 秒，共 5 个 epoch。
对应讲义：02_pytorch_dl/L06_完整训练循环MNIST.md
"""
import matplotlib
matplotlib.use("Agg")          # 无显示器服务器用 Agg 后端；本地桌面可删除此行改 plt.show()
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 超参数（Checkpoint 的对比实验就改这里） ----------
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"


def build_loaders():
    """下载并包装 MNIST：ToTensor 缩放到 [0,1]，再按全集均值/标准差归一化。"""
    transform = transforms.Compose([
        transforms.ToTensor(),                        # -> [0,1]，形状 [1,28,28]
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 的均值/标准差
    ])
    train_set = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)  # 评估无需打乱
    return train_loader, test_loader


def build_mlp():
    return nn.Sequential(
        nn.Flatten(),                # [B,1,28,28] -> [B,784]
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),          # 输出 logits（softmax 交给 CrossEntropyLoss）
    )


def train_one_epoch(model, loader, loss_fn, optimizer):
    """训练循环八股文：train -> to(device) -> zero_grad/forward/loss/backward/step。"""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()                     # ① 清梯度
        logits = model(x)                         # ② 前向
        loss = loss_fn(logits, y)                 # ③ 损失
        loss.backward()                           # ④ 反向
        optimizer.step()                          # ⑤ 更新
        total_loss += loss.item() * y.size(0)     # 统计（批均还原成批总再累加）
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()                                   # 评估不建图：省内存、更快
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def plot_history(h, path="mnist_curves.png"):
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, h["train_loss"], label="train")
    axes[0].plot(epochs, h["test_loss"], label="test")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].plot(epochs, h["train_acc"], label="train")
    axes[1].plot(epochs, h["test_acc"], label="test")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"训练曲线已保存: {path}")


def main():
    torch.manual_seed(42)
    train_loader, test_loader = build_loaders()

    model = build_mlp().to(DEVICE)
    print(model)
    print("参数量:", sum(p.numel() for p in model.parameters()))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn)
        for k, v in zip(history, (tr_loss, tr_acc, te_loss, te_acc)):
            history[k].append(v)
        print(f"epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}")

    plot_history(history)

    # 保存权重（L07 checkpoint 的基础）
    torch.save(model.state_dict(), "mnist_mlp.pt")
    print("模型权重已保存: mnist_mlp.pt")

    # 单张预测演示
    model.eval()
    x0, y0 = next(iter(test_loader))
    with torch.no_grad():
        pred = model(x0[:1].to(DEVICE)).argmax(1).item()
    print(f"测试集第 0 张: 预测={pred}, 真实={y0[0].item()}")


if __name__ == "__main__":
    main()
